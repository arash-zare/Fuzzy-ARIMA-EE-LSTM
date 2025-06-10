# preprocessing.py
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

def normalize(data, method="z-score", scaler=None, fit=False, scaler_path=None):
    """
    data: np.ndarray (n_samples, n_features)
    method: "z-score" or "minmax"
    scaler: if provided, use this fitted scaler
    fit: if True, fit scaler on data and save
    scaler_path: path to save/load scaler (pickle)
    Returns: (normalized_data, fitted_scaler)
    """
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

    data_scaled = scaler.transform(data)
    return data_scaled, scaler

def inverse_transform(data_scaled, scaler):
    """
    Converts normalized data back to original scale.
    """
    return scaler.inverse_transform(data_scaled)

def build_sequences(data, seq_len):
    """
    Converts array to sliding windows for LSTM training.
    Args:
        data: np.ndarray, shape (n_samples, n_features)
        seq_len: int, length of each input sequence
    Returns:
        X: np.ndarray, shape (n_sequences, seq_len, n_features)
        y: np.ndarray, shape (n_sequences, n_features)
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])  # next-step prediction
    return np.array(X), np.array(y)

def split_data(data, ratio=0.7):
    """
    Splits data into train/test sets (sequentially, for time series).
    """
    n = int(len(data) * ratio)
    return data[:n], data[n:]


def save_scaler(scaler, path):
    joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)

# Example: Usage in pipeline
if __name__ == "__main__":
    # Sample data
    data = np.random.rand(100, 8)  # (timesteps, features)
    data_scaled, scaler = normalize(data, method="z-score", fit=True, scaler_path="scaler.pkl")
    X, y = build_sequences(data_scaled, seq_len=16)
    train_X, test_X = split_data(X, ratio=0.7)
    train_y, test_y = split_data(y, ratio=0.7)
    print("X shape:", X.shape, "y shape:", y.shape)
