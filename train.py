import numpy as np
from preprocessing import normalize, build_sequences, save_scaler
from model import ResidualLSTM, train_lstm, save_lstm, SARIMAForecaster
from config import SCALER_PATH, MODEL_PATH, SEQ_LEN, INPUT_DIM, INPUT_DIM_EE, FEATURES
from pso import fetch_data_from_victoriametrics

def generate_training_data_with_sarima(data):
    print(f"[DEBUG] data shape: {data.shape}")
    print(f"[DEBUG] SEQ_LEN: {SEQ_LEN}")

    if data.shape[0] < SEQ_LEN + 1:
        raise ValueError(f"Not enough data for sequence building: need at least {SEQ_LEN+1}, got {data.shape[0]}")
    data_scaled, scaler = normalize(data, method="z-score", fit=True)

    print("[CHECK] data_scaled shape:", data_scaled.shape)
    print("[CHECK] data_scaled sample (first 10 rows):", data_scaled[:10])

    X_raw, y = build_sequences(data_scaled, seq_len=SEQ_LEN)
    print("[CHECK] X_raw shape:", X_raw.shape)
    print("[CHECK] y shape:", y.shape)

    if X_raw is None or y is None or X_raw.shape[0] == 0:
        raise ValueError("Failed to build sequences! Check data and SEQ_LEN.")
    sarima_forecasts = []
    sarima_residuals = []
    for i in range(INPUT_DIM):
        sarima = SARIMAForecaster()
        sarima.fit(data_scaled[:, i])
        preds = []
        resids = []
        for start in range(len(X_raw)):
            try:
                pred, _ = sarima.predict(steps=1, start=start + SEQ_LEN - 1)
                pred = pred.values[0] if hasattr(pred, "values") else pred[0]
            except Exception:
                pred = 0.0
            preds.append(pred)
            resids.append(X_raw[start, -1, i] - pred)
        sarima_forecasts.append(np.array(preds))
        sarima_residuals.append(np.array(resids))
    sarima_forecasts = np.stack(sarima_forecasts, axis=1)
    sarima_residuals = np.stack(sarima_residuals, axis=1)
    X_combined = []
    for i in range(X_raw.shape[0]):
        X_sample = np.concatenate([
            X_raw[i],
            np.tile(sarima_forecasts[i], (SEQ_LEN, 1)),
            np.tile(sarima_residuals[i], (SEQ_LEN, 1)),
        ], axis=1)
        X_combined.append(X_sample)
    X_combined = np.stack(X_combined, axis=0)
    if np.isnan(X_combined).any() or np.isinf(X_combined).any():
        raise ValueError("NaN or Inf found in training data!")
    return X_combined, y, scaler

try:
    print("🔄 Fetching data from VictoriaMetrics...")
    data = fetch_data_from_victoriametrics("now-1d", "now", "30s")
    print(f"Data shape: {data.shape}, NaNs: {np.isnan(data).sum()}, Std: {np.std(data, axis=0)}")
except Exception as e:
    print(f"❌ Failed to fetch data: {e}. Using random data...")
    data = np.random.rand(1000, INPUT_DIM)

if np.isnan(data).any():
    print("⚠️ Filling NaNs with mean...")
    data = np.where(np.isnan(data), np.nanmean(data, axis=0), data)
if np.std(data, axis=0).min() < 1e-8:
    print("⚠️ Low variance detected. Adding noise...")
    data += np.random.normal(0, 1e-4, data.shape)

X, y, scaler = generate_training_data_with_sarima(data)
print("X is None?", X is None)
print("y is None?", y is None)
print("scaler is None?", scaler is None)
print(f"Training LSTM: X shape={X.shape}, y shape={y.shape}")

lstm_model = ResidualLSTM(input_dim=INPUT_DIM_EE)
lstm_model = train_lstm(lstm_model, X, y)
save_lstm(lstm_model, MODEL_PATH)
save_scaler(scaler, SCALER_PATH)
print("✅ LSTM and scaler trained & saved!")
