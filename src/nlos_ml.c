#include "nlos_ml.h"

#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <wchar.h>

#ifdef NLOS_ONNX_RUNTIME
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmacro-redefined"
#include <onnxruntime_c_api.h>
#pragma clang diagnostic pop
#endif

#define NLOS_RT_MAX_MEMORY_SEC 300
#define NLOS_RT_MIN_OPEN_SKY   36.0f
#define NLOS_RT_MAX_SEQ_LEN    256

/* Proof in trace that this translation unit was rebuilt (change if you touch this file). */
#define NLOS_ML_BUILD_STAMP "nlos_ml.c " __DATE__ " " __TIME__

/* ONNX Runtime shared state (global singleton). */
#ifdef NLOS_ONNX_RUNTIME
typedef struct {
    float snr_hist[NLOS_RT_MAX_MEMORY_SEC];
    int snr_pos;
    int snr_count;
    float last_snr_norm;
    float last_receiver_state;

    float window[NLOS_RT_MAX_SEQ_LEN][2]; /* [snr_norm, receiver_state] */
    int win_pos;
    int win_count;
    int last_epoch;
    float last_input_snr_dbhz;
    int last_input_receiver_state;
} nlos_tracker_t;

static nlos_tracker_t trackers[MAXSAT][NFREQ];

static int cache_epoch = -1;
static uint8_t cache_valid[MAXSAT][NFREQ];
static double cache_pnlos[MAXSAT][NFREQ];

static const OrtApi *ort_api = NULL;
static OrtEnv *ort_env = NULL;
static OrtSessionOptions *ort_session_options = NULL;
static OrtSession *ort_session = NULL;
static OrtAllocator *ort_allocator = NULL;
static OrtMemoryInfo *ort_memory_info = NULL;
static char ort_input_name[64] = {0};
static char ort_output_name[64] = {0};
static int ort_refcount = 0;
static int ort_seq_len = 1;

static int ort_check(OrtStatus *status)
{
    if (!status) return 1;
    const char *msg = ort_api ? ort_api->GetErrorMessage(status) : "unknown error";
    /* Use stderr because trace() lives in rtklib runtime code. */
    if (msg) fprintf(stderr, "ONNX Runtime error: %s\n", msg);
    ort_api->ReleaseStatus(status);
    return 0;
}

static void reset_trackers(void)
{
    memset(trackers, 0, sizeof(trackers));
    for (int sat = 0; sat < MAXSAT; ++sat) {
        for (int f = 0; f < NFREQ; ++f) {
            trackers[sat][f].last_epoch = -1;
            trackers[sat][f].last_input_snr_dbhz = -1.0f;
            trackers[sat][f].last_input_receiver_state = -1;
        }
    }
    memset(cache_valid, 0, sizeof(cache_valid));
    cache_epoch = -1;
}

static float tracker_max_snr(const nlos_tracker_t *tr)
{
    float m = 0.0f;
    for (int i = 0; i < tr->snr_count; ++i) {
        if (tr->snr_hist[i] > m) m = tr->snr_hist[i];
    }
    return m;
}

static void tracker_push_window(nlos_tracker_t *tr, float snr_norm, float receiver_state)
{
    tr->window[tr->win_pos][0] = snr_norm;
    tr->window[tr->win_pos][1] = receiver_state;
    tr->win_pos = (tr->win_pos + 1) % NLOS_RT_MAX_SEQ_LEN;
    if (tr->win_count < NLOS_RT_MAX_SEQ_LEN) tr->win_count++;
}

static void tracker_update_once_per_epoch(nlos_tracker_t *tr, int epoch, float snr_dbhz, int receiver_state)
{
    if (epoch == tr->last_epoch) return;

    tr->snr_hist[tr->snr_pos] = snr_dbhz;
    tr->snr_pos = (tr->snr_pos + 1) % NLOS_RT_MAX_MEMORY_SEC;
    if (tr->snr_count < NLOS_RT_MAX_MEMORY_SEC) tr->snr_count++;

    {
        float dynamic_max = tracker_max_snr(tr);
        float safe_open_sky = dynamic_max > NLOS_RT_MIN_OPEN_SKY ? dynamic_max : NLOS_RT_MIN_OPEN_SKY;
        tr->last_snr_norm = snr_dbhz - safe_open_sky;
    }
    tr->last_receiver_state = (float)receiver_state;

    tracker_push_window(tr, tr->last_snr_norm, tr->last_receiver_state);
    tr->last_epoch = epoch;
    tr->last_input_snr_dbhz = snr_dbhz;
    tr->last_input_receiver_state = receiver_state;
}

static int tracker_build_model_input(const nlos_tracker_t *tr, float *out_seq, int seq_len)
{
    int i, pad;
    int history_count = tr->win_count;
    if (seq_len <= 0 || seq_len > NLOS_RT_MAX_SEQ_LEN) return 0;
    if (history_count <= 0) return 0;
    if (history_count > seq_len) history_count = seq_len;

    pad = seq_len - history_count;
    for (i = 0; i < pad; ++i) {
        out_seq[i * 2 + 0] = tr->last_snr_norm;
        out_seq[i * 2 + 1] = tr->last_receiver_state;
    }

    {
        int start = tr->win_pos - history_count;
        if (start < 0) start += NLOS_RT_MAX_SEQ_LEN;
        for (i = 0; i < history_count; ++i) {
            int idx = (start + i) % NLOS_RT_MAX_SEQ_LEN;
            out_seq[(pad + i) * 2 + 0] = tr->window[idx][0];
            out_seq[(pad + i) * 2 + 1] = tr->window[idx][1];
        }
    }
    return 1;
}

static void resolve_model_seq_len(void)
{
    OrtTypeInfo *type_info = NULL;
    const OrtTensorTypeAndShapeInfo *tensor_info = NULL;
    size_t dim_count = 0;
    int64_t dims[8];
    OrtStatus *st;

    ort_seq_len = 1;
    st = ort_api->SessionGetInputTypeInfo(ort_session, 0, &type_info);
    if (!ort_check(st) || !type_info) goto cleanup;

    st = ort_api->CastTypeInfoToTensorInfo(type_info, &tensor_info);
    if (!ort_check(st) || !tensor_info) goto cleanup;

    st = ort_api->GetDimensionsCount(tensor_info, &dim_count);
    if (!ort_check(st) || dim_count < 2 || dim_count > 8) goto cleanup;

    st = ort_api->GetDimensions(tensor_info, dims, dim_count);
    if (!ort_check(st)) goto cleanup;

    if (dims[1] > 0 && dims[1] <= NLOS_RT_MAX_SEQ_LEN) {
        ort_seq_len = (int)dims[1];
    }

cleanup:
    if (type_info) ort_api->ReleaseTypeInfo(type_info);
}
#endif

static double clamp01(double x)
{
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

int nlos_ml_onnx_init(const prcopt_t *opt)
{
    if (!opt || !opt->nlos_onnx_enabled) return 0;
    if (!opt->nlos_onnx_model[0]) return 0;

#ifndef NLOS_ONNX_RUNTIME
    /* Build without ONNX runtime support. */
    trace(1, "NLOS-ONNX init: runtime not compiled in\n");
    return 0;
#else
    if (ort_refcount++ > 0) return 1;

    /* Load API. Allow fallback to older runtime API versions if DLL is older. */
    {
        const OrtApiBase *api_base = OrtGetApiBase();
        uint32_t req_ver = ORT_API_VERSION;
        uint32_t selected_ver = 0;
        if (!api_base) {
            trace(1, "NLOS-ONNX init failed: OrtGetApiBase() returned NULL\n");
            ort_refcount--;
            return 0;
        }
        for (uint32_t v = req_ver; v >= 1; --v) {
            const OrtApi *candidate = api_base->GetApi(v);
            if (candidate) {
                ort_api = candidate;
                selected_ver = v;
                break;
            }
            if (v == 1) break;
        }
        if (!ort_api) {
            trace(1, "NLOS-ONNX init failed: no compatible ORT API found (requested=%u)\n",
                  (unsigned)req_ver);
            ort_refcount--;
            return 0;
        }
        if (selected_ver != req_ver) {
            trace(1, "NLOS-ONNX init: ORT API fallback requested=%u selected=%u\n",
                  (unsigned)req_ver, (unsigned)selected_ver);
        }
    }

    OrtStatus *st = ort_api->GetAllocatorWithDefaultOptions(&ort_allocator);
    if (!ort_check(st) || !ort_allocator) {
        trace(1, "NLOS-ONNX init failed: default allocator unavailable\n");
        ort_refcount--;
        ort_api = NULL;
        return 0;
    }

    st = ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ort_memory_info);
    if (!ort_check(st) || !ort_memory_info) goto fail;

    /* Create environment and session options. */
    st = ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "nlos_ml", &ort_env);
    if (!ort_check(st)) goto fail;

    st = ort_api->CreateSessionOptions(&ort_session_options);
    if (!ort_check(st)) goto fail;

    /* Create session from ONNX model. */
    wchar_t model_path_w[MAXSTRPATH];
    model_path_w[0] = L'\0';
    mbstowcs(model_path_w, opt->nlos_onnx_model, MAXSTRPATH - 1);
    model_path_w[MAXSTRPATH - 1] = L'\0';

    st = ort_api->CreateSession(ort_env, model_path_w, ort_session_options, &ort_session);
    if (!ort_check(st)) goto fail;

    /* Cache input/output names (assume single input/output). */
    size_t input_count = 0, output_count = 0;
    st = ort_api->SessionGetInputCount(ort_session, &input_count);
    if (!ort_check(st)) goto fail;
    st = ort_api->SessionGetOutputCount(ort_session, &output_count);
    if (!ort_check(st)) goto fail;
    if (input_count < 1 || output_count < 1) goto fail;

    {
        char *in_name = NULL;
        st = ort_api->SessionGetInputName(ort_session, 0, ort_allocator, &in_name);
        if (!ort_check(st) || !in_name) goto fail;
        strncpy(ort_input_name, in_name, sizeof(ort_input_name) - 1);
        ort_api->AllocatorFree(ort_allocator, in_name);
    }
    {
        char *out_name = NULL;
        st = ort_api->SessionGetOutputName(ort_session, 0, ort_allocator, &out_name);
        if (!ort_check(st) || !out_name) goto fail;
        strncpy(ort_output_name, out_name, sizeof(ort_output_name) - 1);
        ort_api->AllocatorFree(ort_allocator, out_name);
    }
    resolve_model_seq_len();
    reset_trackers();
    trace(1, "NLOS-ONNX init: enabled model=%s seq_len=%d\n", opt->nlos_onnx_model, ort_seq_len);

    return 1;

fail:
    trace(1, "NLOS-ONNX init failed: model=%s\n", opt->nlos_onnx_model);
    nlos_ml_onnx_shutdown();
    return 0;
#endif
}

void nlos_ml_onnx_shutdown(void)
{
#ifdef NLOS_ONNX_RUNTIME
    if (ort_refcount <= 0) return;
    ort_refcount--;
    if (ort_refcount != 0) return;

    if (ort_api) {
        if (ort_session) {
            ort_api->ReleaseSession(ort_session);
            ort_session = NULL;
        }
        if (ort_memory_info) {
            ort_api->ReleaseMemoryInfo(ort_memory_info);
            ort_memory_info = NULL;
        }
        if (ort_session_options) {
            ort_api->ReleaseSessionOptions(ort_session_options);
            ort_session_options = NULL;
        }
        if (ort_env) {
            ort_api->ReleaseEnv(ort_env);
            ort_env = NULL;
        }
    }
    ort_input_name[0] = 0;
    ort_output_name[0] = 0;
    ort_seq_len = 1;
    reset_trackers();
    ort_api = NULL;
#endif
}

double getNlosProbability(const nlos_features_t *features)
{
#ifndef NLOS_ONNX_RUNTIME
    (void)features;
    return 0.0;
#else
    static int logged_build_stamp = 0;
    static int warned_inactive = 0;
    static int warned_bad_seq = 0;
    static int warned_no_input = 0;
    int sat_idx, freq_idx, receiver_state;
    nlos_tracker_t *tr;
    int seq_len;
    float input_data[NLOS_RT_MAX_SEQ_LEN * 2];
    int64_t input_shape[3];
    OrtStatus *st;
    OrtValue *input_tensor = NULL;
    const char *input_names[1] = {ort_input_name};
    const char *output_names[1] = {ort_output_name};
    OrtValue *input_tensors[1];
    OrtValue *output_tensors[1] = {NULL};

    if (!features) return 0.0;
    if (!logged_build_stamp) {
        trace(1, "NLOS-ONNX build: %s\n", NLOS_ML_BUILD_STAMP);
        logged_build_stamp = 1;
    }
    if (ort_refcount <= 0 || !ort_session || !ort_memory_info) {
        if (!warned_inactive) {
            trace(1, "NLOS-ONNX inactive: ref=%d session=%d mem=%d "
                     "(if ref>0 with null session: partial init or stale state; "
                     "restart app or rtkfree/rtkinit after rebuild)\n",
                  ort_refcount, ort_session ? 1 : 0, ort_memory_info ? 1 : 0);
            warned_inactive = 1;
        }
        return 0.0;
    }

    sat_idx = features->sat_no - 1;
    freq_idx = features->freq_idx;
    if (sat_idx < 0 || sat_idx >= MAXSAT || freq_idx < 0 || freq_idx >= NFREQ) return 0.0;
    receiver_state = features->receiver_state;
    if (receiver_state < 0) receiver_state = 0;
    if (receiver_state > 7) receiver_state = 7;

    if (cache_epoch != features->epoch) {
        memset(cache_valid, 0, sizeof(cache_valid));
        cache_epoch = features->epoch;
    }
    if (cache_valid[sat_idx][freq_idx]) {
        trace(2, "NLOS-ONNX cache: sat=%d f=%d P_nlos=%.6f\n",
              features->sat_no, features->freq_idx + 1, cache_pnlos[sat_idx][freq_idx]);
        return cache_pnlos[sat_idx][freq_idx];
    }

    tr = &trackers[sat_idx][freq_idx];
    {
        const float snr_in = (float)features->snr_dbhz;
        const int rs_in = receiver_state;
        const int epoch = features->epoch;
        const int need_update =
            (tr->last_epoch != epoch) ||
            (snr_in != tr->last_input_snr_dbhz) ||
            (rs_in != tr->last_input_receiver_state);
        if (need_update) {
            if (tr->last_epoch == epoch) {
                /* Same epoch but inputs changed: rewind epoch guard to allow one update. */
                tr->last_epoch = epoch - 1;
            }
            tracker_update_once_per_epoch(tr, epoch, snr_in, rs_in);
        }
    }

    seq_len = ort_seq_len;
    if (seq_len <= 0 || seq_len > NLOS_RT_MAX_SEQ_LEN) {
        if (!warned_bad_seq) {
            trace(1, "NLOS-ONNX invalid seq_len=%d\n", seq_len);
            warned_bad_seq = 1;
        }
        return 0.0;
    }
    if (!tracker_build_model_input(tr, input_data, seq_len)) {
        if (!warned_no_input) {
            trace(1, "NLOS-ONNX no input window: sat=%d f=%d win_count=%d seq_len=%d\n",
                  features->sat_no, features->freq_idx + 1, tr->win_count, seq_len);
            warned_no_input = 1;
        }
        return 0.0;
    }

    input_shape[0] = 1;
    input_shape[1] = seq_len;
    input_shape[2] = 2;

    st = ort_api->CreateTensorWithDataAsOrtValue(
        ort_memory_info,
        input_data,
        (size_t)(seq_len * 2 * (int)sizeof(float)),
        input_shape,
        3,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor);
    if (!ort_check(st) || !input_tensor) return 0.0;

    input_tensors[0] = input_tensor;

    st = ort_api->Run(ort_session,
                      NULL /* run options */,
                      input_names,
                      (const OrtValue *const *)input_tensors,
                      1,
                      output_names,
                      1,
                      output_tensors);
    if (!ort_check(st) || !output_tensors[0]) {
        ort_api->ReleaseValue(input_tensor);
        return 0.0;
    }

    {
        float *out_data = NULL;
        size_t out_count = 0;
        OrtTensorTypeAndShapeInfo *out_shape = NULL;
        float p = 0.0f;
        double pnlos;

        st = ort_api->GetTensorMutableData(output_tensors[0], (void **)&out_data);
        if (!ort_check(st)) {
            ort_api->ReleaseValue(output_tensors[0]);
            ort_api->ReleaseValue(input_tensor);
            return 0.0;
        }
        st = ort_api->GetTensorTypeAndShape(output_tensors[0], &out_shape);
        if (ort_check(st) && out_shape) {
            st = ort_api->GetTensorShapeElementCount(out_shape, &out_count);
            if (!ort_check(st)) out_count = 0;
            ort_api->ReleaseTensorTypeAndShapeInfo(out_shape);
        }
        if (out_data && out_count > 0) {
            p = out_data[out_count - 1]; /* Supports scalar or sequence output. */
        }
        p = (float)clamp01((double)p);

        ort_api->ReleaseValue(output_tensors[0]);
        ort_api->ReleaseValue(input_tensor);
        pnlos = (double)(1.0f - p); /* Model predicts P(LOS); return P(NLOS). */
        trace(2, "NLOS-ONNX out: sat=%d f=%d P_los=%.6f P_nlos=%.6f\n",
              features->sat_no, features->freq_idx + 1, (double)p, pnlos);
        cache_pnlos[sat_idx][freq_idx] = pnlos;
        cache_valid[sat_idx][freq_idx] = 1;
        return pnlos;
    }
#endif
}

