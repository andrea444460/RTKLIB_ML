#ifndef NLOS_ML_H
#define NLOS_ML_H

#include "rtklib.h" /* for MAXSTRPATH, prcopt_t */

typedef struct {
    /* Satellite/frequency identity for per-channel causal memory. */
    int sat_no;        /* 1..MAXSAT */
    int freq_idx;      /* 0..NFREQ-1 */
    int epoch;         /* RTK epoch counter */

    /* Raw rover SNR in dBHz (same unit as obsd_t::SNR / ssat_t::snr_rover). */
    double snr_dbhz;

    /* Receiver state in [0,7] as defined by model training. */
    int receiver_state;
} nlos_features_t;

/* Initialize ONNX Runtime session once per process (ref-counted internally). */
int nlos_ml_onnx_init(const prcopt_t *opt);
/* Shutdown ONNX Runtime session when last user calls shutdown. */
void nlos_ml_onnx_shutdown(void);

/* Return P(NLOS) in [0,1] (1 => strongly NLOS). */
double getNlosProbability(const nlos_features_t *features);

#endif /* NLOS_ML_H */

