# preprocessing.py
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from config import INPUT_DIM, SEQ_LEN  # اضافه شده برای اعتبارسنجی

def normalize(data, method="z-score", scaler=None, fit=False, scaler_path=None):
    """
    نرمال‌سازی داده‌ها با استفاده از StandardScaler یا MinMaxScaler.
    
    Args:
        data: np.ndarray (n_samples, n_features) - داده‌های ورودی
        method: str - نوع نرمال‌سازی ("z-score" یا "minmax")
        scaler: Scaler object - اگر ارائه شود، از این scaler استفاده می‌شود
        fit: bool - اگر True باشد، scaler روی داده‌ها فیت می‌شود
        scaler_path: str - مسیر ذخیره/بارگذاری scaler
    
    Returns:
        tuple: (normalized_data, fitted_scaler)
    
    Raises:
        ValueError: اگر شکل داده‌ها نامعتبر باشد یا تعداد فیچرها با INPUT_DIM سازگار نباشد
    """
    # اعتبارسنجی اولیه
    if data is None or not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError(f"Invalid input data: Expected 2D numpy array, got {type(data)} or shape {data.shape if data is not None else None}")
    
    if data.shape[1] != INPUT_DIM:
        raise ValueError(f"Number of features ({data.shape[1]}) does not match INPUT_DIM ({INPUT_DIM})")
    
    if data.shape[0] < 1:
        raise ValueError("Input data must have at least one sample")

    # مدیریت مقادیر نان
    nan_count = np.isnan(data).sum()
    if nan_count > 0:
        print(f"⚠️ Warning: Found {nan_count} NaN values in input data. Applying forward-fill and mean imputation...")
        # Forward-fill برای حفظ پیوستگی سری زمانی
        for j in range(data.shape[1]):
            col = data[:, j]
            mask = np.isnan(col)
            if mask.any():
                # Forward-fill
                for i in range(1, len(col)):
                    if mask[i] and not mask[i-1]:
                        col[i] = col[i-1]
                # پر کردن باقی‌مانده نان‌ها با میانگین ستون
                mask = np.isnan(col)
                if mask.any():
                    col[mask] = np.nanmean(col)
        # بررسی نهایی برای نان‌ها
        if np.isnan(data).any():
            raise ValueError("Failed to impute all NaN values in input data")

    # مقداردهی scaler
    if scaler is None:
        scaler = StandardScaler() if method == "z-score" else MinMaxScaler()
        if fit:
            scaler.fit(data)
            if scaler_path:
                joblib.dump(scaler, scaler_path)
    else:
        if fit:
            scaler.fit(data)
            if scaler_path:
                joblib.dump(scaler, scaler_path)
        else:
            if scaler_path:
                scaler = joblib.load(scaler_path)

    # نرمال‌سازی داده‌ها
    data_scaled = scaler.transform(data)
    return data_scaled, scaler

def inverse_transform(data_scaled, scaler):
    """
    تبدیل داده‌های نرمال‌شده به مقیاس اصلی.
    
    Args:
        data_scaled: np.ndarray - داده‌های نرمال‌شده
        scaler: Scaler object - اسکیلر استفاده‌شده برای نرمال‌سازی
    
    Returns:
        np.ndarray - داده‌های در مقیاس اصلی
    """
    return scaler.inverse_transform(data_scaled)

def build_sequences(data, seq_len=SEQ_LEN):
    """
    ساخت sequence‌های sliding window برای آموزش LSTM.
    
    Args:
        data: np.ndarray, shape (n_samples, n_features)
        seq_len: int - طول هر sequence ورودی
    
    Returns:
        X: np.ndarray, shape (n_sequences, seq_len, n_features)
        y: np.ndarray, shape (n_sequences, n_features)
    """
    if data.shape[0] < seq_len + 1:
        print(f"⚠️ Warning: Not enough samples ({data.shape[0]}) for seq_len={seq_len}. Returning empty sequences.")
        return np.array([]), np.array([])

    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])  # پیش‌بینی گام بعدی
    return np.array(X), np.array(y)

def split_data(data, ratio=0.7):
    """
    تقسیم داده‌ها به مجموعه‌های آموزشی و تست (به‌صورت ترتیبی برای سری‌های زمانی).
    
    Args:
        data: np.ndarray - داده‌های ورودی
        ratio: float - نسبت تقسیم برای داده‌های آموزشی
    
    Returns:
        tuple: (train_data, test_data)
    """
    n = int(len(data) * ratio)
    return data[:n], data[n:]

def save_scaler(scaler, path):
    joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)

# Example: Usage in pipeline
if __name__ == "__main__":
    # داده نمونه
    data = np.random.rand(100, INPUT_DIM)  # (timesteps, features)
    data_scaled, scaler = normalize(data, method="z-score", fit=True, scaler_path="scaler.pkl")
    X, y = build_sequences(data_scaled, seq_len=SEQ_LEN)
    train_X, test_X = split_data(X, ratio=0.7)
    train_y, test_y = split_data(y, ratio=0.7)
    print("X shape:", X.shape, "y shape:", y.shape)