import numpy as np
from preprocessing import normalize, build_sequences, save_scaler
from model import ResidualLSTM, train_lstm, save_lstm
from config import SCALER_PATH, MODEL_PATH, SEQ_LEN, INPUT_DIM, FEATURES
from pso import fetch_data_from_victoriametrics

def generate_training_data(data):
    """
    Generate training data for LSTM (without SARIMA residuals for simplicity).
    
    Args:
        data: np.ndarray (n_samples, n_features) - Raw time series data
    
    Returns:
        X: np.ndarray (n_samples, seq_len, input_dim*2) - Sequences with padded zeros
        y: np.ndarray (n_samples, input_dim) - Targets
        scaler: Scaler - Fitted scaler
    """
    # نرمال‌سازی داده‌ها
    data_scaled, scaler = normalize(data, method="z-score", fit=True)
    
    # ساخت sequence‌ها
    X, y = build_sequences(data_scaled, seq_len=SEQ_LEN)
    
    # پد کردن برای تطبیق با input_dim=16 (8 فیچر + 8 صفر به جای residuals)
    X_padded = np.concatenate([X, np.zeros_like(X)], axis=2)
    
    return X_padded, y, scaler

# دریافت داده از VictoriaMetrics
try:
    print("🔄 Fetching data from VictoriaMetrics...")
    data = fetch_data_from_victoriametrics("now-1d", "now", "30s")
    print(f"Data shape: {data.shape}, NaNs: {np.isnan(data).sum()}, Std: {np.std(data, axis=0)}")
except Exception as e:
    print(f"❌ Failed to fetch data: {e}. Using random data...")
    data = np.random.rand(1000, INPUT_DIM)

# بررسی داده‌ها
if np.isnan(data).any():
    print("⚠️ Filling NaNs with mean...")
    data = np.where(np.isnan(data), np.nanmean(data, axis=0), data)
if np.std(data, axis=0).min() < 1e-8:
    print("⚠️ Low variance detected. Adding noise...")
    data += np.random.normal(0, 1e-4, data.shape)

X, y, scaler = generate_training_data(data)
print(f"Training LSTM: X shape={X.shape}, y shape={y.shape}")  # باید (n_samples, SEQ_LEN, 16) باشد

# آموزش مدل
lstm_model = ResidualLSTM(input_dim=INPUT_DIM * 2)  # 8 فیچر + 8 پد
lstm_model = train_lstm(lstm_model, X, y)
save_lstm(lstm_model, MODEL_PATH)
save_scaler(scaler, SCALER_PATH)

print("✅ LSTM and scaler trained & saved!")