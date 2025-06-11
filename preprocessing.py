import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from config import SEQ_LEN, INPUT_DIM, SCALER_PATH

def handle_nan(data, fill_value="mean"):
    data = np.array(data)
    if fill_value == "mean":
        means = np.nanmean(data, axis=0)
        inds = np.where(np.isnan(data))
        data[inds] = np.take(means, inds[1])
    elif fill_value == "ffill":
        for i in range(1, data.shape[0]):
            mask = np.isnan(data[i])
            data[i, mask] = data[i-1, mask]
    return data

def normalize(X, method="z-score", scaler=None, fit=False):
    """
    - Training: fit=True, method="z-score"
    - Inference: scaler only
    """
    if fit:
        if method == "z-score":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError("Unknown normalization method")
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler
    elif scaler is not None:
        X_scaled = scaler.transform(X)
        return X_scaled
    else:
        raise ValueError("Either fit=True with method or provide a fitted scaler.")

def build_sequences(X, seq_len=SEQ_LEN):
    X_seq, y_seq = [], []
    if len(X) <= seq_len:
        return np.array([]), np.array([])
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(X[i+seq_len])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    return X_seq, y_seq

def save_scaler(scaler, path=SCALER_PATH):
    import joblib
    joblib.dump(scaler, path)

def load_scaler(path=SCALER_PATH):
    import joblib
    return joblib.load(path)

def check_shape(X, seq_len=SEQ_LEN, input_dim=INPUT_DIM):
    assert X.shape[1] == seq_len, f"X seq_len mismatch: {X.shape[1]} != {seq_len}"
    assert X.shape[2] == input_dim, f"X input_dim mismatch: {X.shape[2]} != {input_dim}"
