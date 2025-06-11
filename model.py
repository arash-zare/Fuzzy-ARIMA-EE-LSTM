import torch
import torch.nn as nn
import numpy as np
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from config import SARIMA_ORDER, SARIMA_SEASONAL_ORDER, DEVICE, SEQ_LEN, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, LSTM_LEARNING_RATE, LSTM_EPOCHS, LSTM_BATCH_SIZE, MODEL_PATH, SCALER_PATH

# ------ SARIMA Forecaster ------
class SARIMAForecaster:
    def __init__(self, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.results = None

    def fit(self, data):
        if data.ndim > 1:
            raise ValueError("SARIMA only supports 1D (univariate) input")
        self.model = SARIMAX(data, order=self.order, seasonal_order=self.seasonal_order,
                             enforce_stationarity=False, enforce_invertibility=False)
        self.results = self.model.fit(disp=False)
        return self

    def predict(self, steps, start=None):
        if self.results is None:
            raise ValueError("Model is not fitted yet!")
        if start is not None:
            forecast = self.results.get_prediction(start=start, end=start+steps-1)
            mean = forecast.predicted_mean
            conf_int = forecast.conf_int(alpha=0.05)
        else:
            forecast = self.results.get_forecast(steps=steps)
            mean = forecast.predicted_mean
            conf_int = forecast.conf_int(alpha=0.05)
        return mean, conf_int

    def in_sample_residuals(self):
        if self.results is None:
            raise ValueError("Model is not fitted yet!")
        return self.results.resid

# ------ EE-LSTM Model ------
class ResidualLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=INPUT_DIM):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# ------ LSTM Training ------
def train_lstm(model, X_train, y_train, lr=LSTM_LEARNING_RATE, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE):
    model = model.to(DEVICE)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(X_train.size(0))
        epoch_loss = 0
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            optimizer.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
    return model

def predict_lstm(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        preds = model(X_tensor)
        return preds.cpu().numpy()

# ------ ذخیره و بارگذاری مدل و اسکیلر ------
def save_lstm(model, path=MODEL_PATH):
    torch.save(model.state_dict(), path)

def load_lstm(model_class, path=MODEL_PATH):
    model = model_class()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model = model.to(DEVICE)
    return model

def save_scaler(scaler, path=SCALER_PATH):
    joblib.dump(scaler, path)

def load_scaler(path=SCALER_PATH):
    return joblib.load(path)

# ------ Hybrid Forecast با EE-LSTM ------
def hybrid_forecast(
    sarima_models, lstm_model, input_sequence, scaler, forecast_steps=1, seq_len=None
):
    """
    EE-LSTM hybrid forecast: [raw | SARIMA forecast | residual] → LSTM input
    input_sequence: np.ndarray (seq_len, n_features)  # Normalized
    Returns:
        forecasts, upper_bounds, lower_bounds, residuals, lstm_pred
    """
    seq_len = seq_len or input_sequence.shape[0]
    n_features = input_sequence.shape[1]
    
    # --- SARIMA step for each feature over the entire sequence ---
    sarima_forecasts = np.zeros_like(input_sequence)
    sarima_residuals = np.zeros_like(input_sequence)
    for i in range(n_features):
        univariate_seq = input_sequence[:, i]
        preds = []
        for t in range(seq_len):
            try:
                pred, _ = sarima_models[i].predict(steps=1, start=t)
                preds.append(pred.values[0] if hasattr(pred, "values") else pred[0])
            except Exception:
                preds.append(0.0)
        sarima_forecasts[:, i] = preds
        sarima_residuals[:, i] = univariate_seq - sarima_forecasts[:, i]

    # --- EE-LSTM input: concatenate along features axis ---
    ee_seq = np.concatenate([input_sequence, sarima_forecasts, sarima_residuals], axis=1)  # (seq_len, 3*n_features)

    # --- LSTM forecast ---
    lstm_in = ee_seq[-seq_len:].reshape(1, seq_len, ee_seq.shape[1])
    lstm_pred = predict_lstm(lstm_model, lstm_in)[0]  # (n_features,)

    # --- SARIMA 1-step forecast for fuzzy logic (latest window) ---
    forecasts, upper_bounds, lower_bounds, residuals = [], [], [], []
    for i in range(n_features):
        mean_pred, conf_int = sarima_models[i].predict(steps=forecast_steps)
        mean_pred = mean_pred.values[0] if hasattr(mean_pred, "values") else mean_pred[0]
        if hasattr(conf_int, "iloc"):
            upper = conf_int.iloc[0, 1]
            lower = conf_int.iloc[0, 0]
        else:
            upper = conf_int[0, 1]
            lower = conf_int[0, 0]
        residual = input_sequence[-1, i] - mean_pred
        forecasts.append(mean_pred)
        upper_bounds.append(upper)
        lower_bounds.append(lower)
        residuals.append(residual)
    # Outputs:
    # - forecasts: SARIMA forecast for each feature (for fuzzy)
    # - upper/lower bounds: confidence intervals
    # - residuals: difference real-last-point vs SARIMA forecast
    # - lstm_pred: EE-LSTM output (can use for evaluation or fusion if needed)
    return (
        np.array(forecasts),
        np.array(upper_bounds),
        np.array(lower_bounds),
        np.array(residuals),
        lstm_pred
    )


# ---------- Shape check (for debug) ----------
def check_shape(X, seq_len=SEQ_LEN, input_dim=INPUT_DIM):
    assert X.shape[1] == seq_len, f"X seq_len mismatch: {X.shape[1]} != {seq_len}"
    assert X.shape[2] == input_dim, f"X input_dim mismatch: {X.shape[2]} != {input_dim}"
