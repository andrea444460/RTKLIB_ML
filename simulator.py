import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
import os
from pathlib import Path
import klt_eval_dashboard as ked
import onnxruntime as ort

# ==========================================
# CONFIGURAZIONE DEL SIMULATORE
# ==========================================
MODEL_PATH = "klt_rt_2feat.keras"
ONNX_MODEL_PATH = "klt_rt_2feat.onnx"
USE_ONNX_RUNTIME = True
CSV_TEST_PATH = "klt_eval_predictions_from_gnss_train.csv"

MAX_MEMORY_SEC = 300        # 5 minuti di memoria per il massimo dinamico (Rolling Max)
MIN_OPEN_SKY = 36.0         # Soglia di sicurezza per evitare falsi positivi nel cold-start in canyon
CNR_MASK_VALUE = -999.0
STATE_PAD_IDX = 8
N_CTX_FEATURES = 4
LOG_EVERY_TICKS = 200       # Stampa un log ogni N secondi/tick
LOG_SAMPLE_PREDS = 3        # Quante predizioni campione mostrare per tick
USE_STABLE_MASK_FOR_METRICS = True
METRICS_CSV_OUT = "simulator_metrics.csv"


def _build_causal_valid_mask(m: tf.Tensor) -> tf.Tensor:
    valid_k = tf.cast(tf.expand_dims(m, axis=1), tf.bool)
    t = tf.shape(m)[1]
    causal = tf.linalg.band_part(tf.ones((t, t), dtype=tf.bool), -1, 0)
    causal = tf.expand_dims(causal, axis=0)
    return tf.logical_and(valid_k, causal)


# ==========================================
# CLASSE CHE SIMULA LA RAM DEL MICROCONTROLLORE (C)
# ==========================================
class SatelliteFirmwareTracker:
    """
    Questa classe è l'esatto equivalente della struct C che dovrai 
    creare per gestire la memoria di ogni singolo satellite tracciato.
    """
    def __init__(self, prn, seq_len):
        self.prn = prn
        self.seq_len = seq_len
        # Ring buffer a lungo termine per calcolare il cielo aperto dinamico
        self.snr_history = deque(maxlen=MAX_MEMORY_SEC)
        
        # Ring buffer a breve termine per dare in pasto la sequenza causale al modello
        self.causal_window = deque(maxlen=seq_len)

    def process_epoch(self, raw_snr, receiver_state):
        # 1. Aggiorna la memoria a lungo termine
        self.snr_history.append(raw_snr)
        
        # 2. Calcola il massimo dinamico (Rolling Max)
        current_dynamic_max = max(self.snr_history)
        
        # 3. Protezione dal Cold-Start in zone coperte
        safe_open_sky = max(current_dynamic_max, MIN_OPEN_SKY)
        
        # 4. Calcolo della feature 1: Normalizzazione
        snr_norm = raw_snr - safe_open_sky
        
        # 5. Vettore feature attuale: [snr_norm, receiver_state]
        current_features = (float(snr_norm), float(receiver_state))
        
        # 6. Aggiorna la finestra temporale del modello
        self.causal_window.append(current_features)
        
        # 7. Gestione Padding per i primissimi secondi (quando il buffer non è ancora pieno)
        # In C, all'avvio riempirai l'array con duplicati della prima lettura
        if len(self.causal_window) < self.seq_len:
            pad_size = self.seq_len - len(self.causal_window)
            # Duplichiamo la lettura corrente per riempire il passato vuoto
            window_to_return = [current_features] * pad_size + list(self.causal_window)
        else:
            window_to_return = list(self.causal_window)
        return np.asarray(window_to_return, dtype=np.float32)

# ==========================================
# LOOP PRINCIPALE DEL SIMULATORE
# ==========================================
def run_simulation():
    use_onnx = bool(USE_ONNX_RUNTIME and os.path.exists(ONNX_MODEL_PATH))
    if use_onnx:
        print(f"[1] Caricamento modello ONNX da {ONNX_MODEL_PATH}...")
        sess = ort.InferenceSession(
            ONNX_MODEL_PATH,
            providers=["CPUExecutionProvider"],
        )
        onnx_input_name = sess.get_inputs()[0].name
        onnx_output_name = sess.get_outputs()[0].name
        # modello ONNX esportato con seq_len fisso
        seq_len = int(sess.get_inputs()[0].shape[1])
        model_kind = "onnx_2_input"
        model = None
        print(f"  ONNX Runtime attivo. input={onnx_input_name}, output={onnx_output_name}, SEQ_LEN={seq_len}")
    else:
        print(f"[1] Caricamento modello da {MODEL_PATH}...")
        if USE_ONNX_RUNTIME and not os.path.exists(ONNX_MODEL_PATH):
            print(f"  [WARN] ONNX non trovato ({ONNX_MODEL_PATH}), uso Keras fallback.")
        if not os.path.exists(MODEL_PATH):
        # Fallback legacy: usa i pesi del modello 4-input ricostruendo architettura.
            legacy_weights = "klt_causal_rt_weights.weights.h5"
            if not os.path.exists(legacy_weights):
                raise FileNotFoundError(
                    f"Modello non trovato: {MODEL_PATH} (e nemmeno fallback {legacy_weights})."
                )
            model = ked.build_model()
            model.load_weights(legacy_weights)
            seq_len = int(model.inputs[0].shape[1])
            model_kind = "legacy_4_input"
            print(f"  Fallback legacy attivo. SEQ_LEN modello = {seq_len}")
        else:
            # Modello consigliato real-time: input unico (B, T, 2).
            model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
            seq_len = int(model.inputs[0].shape[1])
            model_kind = "realtime_2_input"
            print(f"  Modello 2-feature caricato. SEQ_LEN modello = {seq_len}")
        
    print(f"[2] Caricamento dati da {CSV_TEST_PATH}...")
    # Sostituisci questo blocco con la tua vera funzione di lettura dati
    try:
        df = pd.read_csv(CSV_TEST_PATH)
    except FileNotFoundError:
        print("File CSV non trovato, genero dati fittizi per la simulazione...")
        # Genera dati dummy per mostrare come funziona
        tempo = np.arange(0, 200)
        df = pd.DataFrame({
            "GPS_Time(s)": np.repeat(tempo, 2),
            "Satellite": np.tile(["G01", "C20"], 200),
            "snr": np.random.normal(40, 2, 400),
            "Receiver_State": np.ones(400),
            "Label": np.ones(400) # 1 = LOS
        })
        # Simuliamo un canyon per C20 dal sec 50 al 100
        mask_canyon = (df["Satellite"] == "C20") & (df["GPS_Time(s)"] >= 50) & (df["GPS_Time(s)"] <= 100)
        df.loc[mask_canyon, "snr"] = np.random.normal(20, 3, mask_canyon.sum())
        df.loc[mask_canyon, "Label"] = 0

    # ASSOLUTAMENTE CRITICO: I dati devono scorrere in ordine temporale!
    # Normalizza nomi colonne tra dataset.
    if "sat_id" in df.columns and "Satellite" not in df.columns:
        df["Satellite"] = df["sat_id"].astype(str)
    if "obs_time_utc" in df.columns and "GPS_Time(s)" not in df.columns:
        df["GPS_Time(s)"] = pd.to_numeric(df["obs_time_utc"], errors="coerce")
    if "snr" not in df.columns:
        if "Cnr_L1" in df.columns:
            df["snr"] = pd.to_numeric(df["Cnr_L1"], errors="coerce")
        elif "CNR" in df.columns:
            df["snr"] = pd.to_numeric(df["CNR"], errors="coerce")
        else:
            raise KeyError("Colonna snr/Cnr_L1/CNR non trovata nel CSV.")
    if "Receiver_State" not in df.columns:
        print("Receiver_State non trovato: imposto valore neutro 7.")
        df["Receiver_State"] = 7

    df["GPS_Time(s)"] = pd.to_numeric(df["GPS_Time(s)"], errors="coerce")
    df["snr"] = pd.to_numeric(df["snr"], errors="coerce").fillna(0.0)
    df["Receiver_State"] = pd.to_numeric(df["Receiver_State"], errors="coerce").fillna(7).clip(0, 7).astype(int)
    df = df.dropna(subset=["GPS_Time(s)", "Satellite"]).copy()
    df = df.sort_values(by="GPS_Time(s)")
    print(
        f"  Dati pronti: {len(df)} righe, "
        f"{df['Satellite'].nunique()} satelliti, "
        f"{df['GPS_Time(s)'].nunique()} tick temporali."
    )

    # La "RAM" del nostro sistema embedded
    firmware_memory = {}
    pred_rt = np.full(len(df), np.nan, dtype=np.float32)
    
    print("[3] Avvio simulazione Real-Time (streaming a 1 Hz)...")
    # Iterazione per tick più efficiente (evita groupby/iterrows nel loop caldo).
    tvals = df["GPS_Time(s)"].to_numpy(dtype=np.float64)
    sat_vals = df["Satellite"].astype(str).to_numpy()
    snr_vals = df["snr"].to_numpy(dtype=np.float32)
    rs_vals = df["Receiver_State"].to_numpy(dtype=np.int32)
    row_ids = np.arange(len(df), dtype=np.int64)

    split_pos = np.flatnonzero(np.diff(tvals) != 0) + 1
    starts = np.concatenate(([0], split_pos))
    ends = np.concatenate((split_pos, [len(df)]))
    n_ticks = len(starts)
    tick_idx = 0

    # Streaming per tick temporale.
    for st, en in zip(starts, ends):
        tick_idx += 1
        current_time = tvals[st]
        n_sat = int(en - st)
        X_tensor = np.empty((n_sat, seq_len, 2), dtype=np.float32)
        row_indices = row_ids[st:en]

        # Processiamo ogni satellite visibile in QUESTO secondo
        for j in range(n_sat):
            i = st + j
            prn = sat_vals[i]
            snr = float(snr_vals[i])
            state = float(rs_vals[i])

            # Se è la prima volta che vediamo questo satellite, allochiamo la memoria
            if prn not in firmware_memory:
                firmware_memory[prn] = SatelliteFirmwareTracker(prn, seq_len=seq_len)

            # Chiamata alla funzione firmware
            tracker = firmware_memory[prn]
            X_tensor[j] = tracker.process_epoch(snr, state)

        # Inferenza hardware (Batch prediction per tutti i satelliti attivi nel tick)
        if n_sat > 0:
            if model_kind == "onnx_2_input":
                pred = sess.run([onnx_output_name], {onnx_input_name: X_tensor})[0]
                if pred.ndim == 3:
                    preds = pred[:, -1, 0]
                else:
                    preds = pred[:, 0]
            elif model_kind == "realtime_2_input":
                # Più rapido di model.predict nel loop streaming.
                pred = model(X_tensor, training=False).numpy()
                # Supporta output (N,1) o (N,T,1)
                if pred.ndim == 3:
                    preds = pred[:, -1, 0]
                else:
                    preds = pred[:, 0]
            else:
                # Fallback legacy 4-input
                x_cnr = X_tensor[:, :, 0:1]
                x_state = np.clip(np.rint(X_tensor[:, :, 1]), 0, 7).astype(np.int32)
                x_ctx = np.zeros((X_tensor.shape[0], seq_len, N_CTX_FEATURES), dtype=np.float32)
                x_gap = np.zeros((X_tensor.shape[0], seq_len, 1), dtype=np.float32)
                pred_seq = model([x_cnr, x_state, x_ctx, x_gap], training=False).numpy()
                preds = pred_seq[:, -1, 0]

            # Salvataggio vettoriale (pre-allocato).
            pred_rt[row_indices] = preds.astype(np.float32)

            # Log periodico di avanzamento con alcune predizioni di esempio.
            if (tick_idx % LOG_EVERY_TICKS) == 0 or tick_idx == 1 or tick_idx == n_ticks:
                sample_parts = []
                smax = min(LOG_SAMPLE_PREDS, n_sat)
                for j in range(smax):
                    sample_parts.append(
                        f"{sat_vals[st + j]}@{current_time:.0f}s p={float(preds[j]):.3f} "
                        f"snr={float(snr_vals[st + j]):.1f} rs={int(rs_vals[st + j])}"
                    )
                sample_txt = " | ".join(sample_parts) if sample_parts else "no-samples"
                print(
                    f"  [Tick {tick_idx}/{n_ticks}] sats={n_sat} "
                    f"processed_rows={int(np.isfinite(pred_rt).sum())} :: {sample_txt}"
                )

    print("[4] Simulazione completata! Preparazione risultati...")
    
    # Scrivi direttamente le predizioni pre-allocate nel dataframe.
    df["Pred_Prob_RT"] = pred_rt
    miss_pred = int(df["Pred_Prob_RT"].isna().sum()) if "Pred_Prob_RT" in df.columns else len(df)
    print(
        f"  Join completato: predizioni={int(np.isfinite(pred_rt).sum())}, "
        f"righe_senza_pred={miss_pred}."
    )

    # Metriche aggregate (se Label disponibile).
    if "Label" in df.columns:
        if "eval_satellite" in df.columns:
            m_eval = df["eval_satellite"].astype(bool)
        else:
            m_eval = pd.Series(True, index=df.index)

        if USE_STABLE_MASK_FOR_METRICS and {"tripId", "Satellite", "Label"}.issubset(df.columns):
            stable = ked.klt_stable_metrics_mask(df, ked.KLT_METRICS_TRANSITION_BUFFER_SEC)
        else:
            stable = pd.Series(True, index=df.index)

        m = m_eval & stable & df["Pred_Prob_RT"].notna()
        y = pd.to_numeric(df.loc[m, "Label"], errors="coerce").fillna(0).astype(int).to_numpy()
        p = pd.to_numeric(df.loc[m, "Pred_Prob_RT"], errors="coerce").fillna(0.5).to_numpy(float)
        yb = (p >= 0.5).astype(int)

        tp = int(((y == 1) & (yb == 1)).sum())
        tn = int(((y == 0) & (yb == 0)).sum())
        fp = int(((y == 0) & (yb == 1)).sum())
        fn = int(((y == 1) & (yb == 0)).sum())
        n = len(y)
        acc = (tp + tn) / max(n, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = (2 * prec * rec) / max(prec + rec, 1e-12)
        brier = float(np.mean((p - y.astype(float)) ** 2)) if n else float("nan")

        print("[4b] Metriche simulazione:")
        print(
            f"  n={n} tp={tp} tn={tn} fp={fp} fn={fn} "
            f"acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f} brier={brier:.4f}"
        )

        out_metrics = Path(__file__).resolve().parent / METRICS_CSV_OUT
        pd.DataFrame(
            [
                {
                    "n_eval": n,
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "brier": brier,
                    "stable_mask_enabled": bool(USE_STABLE_MASK_FOR_METRICS),
                }
            ]
        ).to_csv(out_metrics, index=False)
        print(f"  Metriche salvate: {out_metrics}")
    else:
        print("[4b] Label non presente: metriche aggregate non calcolate.")
    
    # Grafico di test per un satellite specifico
    plot_satellite_rt(df, satellite_to_plot=df["Satellite"].unique()[0])
    
    return df

def plot_satellite_rt(df, satellite_to_plot):
    df_sat = df[df["Satellite"] == satellite_to_plot].sort_values("GPS_Time(s)")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Subplot 1: La Fisica (SNR)
    ax1.plot(df_sat["GPS_Time(s)"], df_sat["snr"], color='gray', linewidth=1, label="Raw SNR")
    
    # Se abbiamo la label di ground truth, la usiamo per colorare i punti
    if "Label" in df_sat.columns:
        los_mask = df_sat["Label"] == 1
        nlos_mask = df_sat["Label"] == 0
        ax1.scatter(df_sat[los_mask]["GPS_Time(s)"], df_sat[los_mask]["snr"], color='green', s=10, label="True LOS")
        ax1.scatter(df_sat[nlos_mask]["GPS_Time(s)"], df_sat[nlos_mask]["snr"], color='red', s=10, label="True NLOS")
        
    ax1.set_ylabel("SNR (dB-Hz)")
    ax1.set_title(f"Simulazione Real-Time Causale - Satellite {satellite_to_plot}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Le Predizioni Real-Time vs Label Reale
    if "Label" in df_sat.columns:
        ax2.plot(df_sat["GPS_Time(s)"], df_sat["Label"], 'bo', label="True Label", alpha=0.5)
        
    ax2.plot(df_sat["GPS_Time(s)"], df_sat["Pred_Prob_RT"], color='orange', label="Real-Time Probability", linewidth=2)
    ax2.axhline(0.5, color='gray', linestyle='--')
    ax2.set_ylabel("Probabilità LOS")
    ax2.set_xlabel("GPS Time (s)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_png = Path(__file__).resolve().parent / f"simulator_rt_{satellite_to_plot}.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[5] Plot salvato: {out_png}")

if __name__ == "__main__":
    df_finale = run_simulation()