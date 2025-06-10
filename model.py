import torch
import torch.nn as nn
import numpy as np
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from config import SARIMA_ORDER, SARIMA_SEASONAL_ORDER, DEVICE, SEQ_LEN, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, LSTM_LEARNING_RATE, LSTM_EPOCHS, LSTM_BATCH_SIZE, MODEL_PATH, SCALER_PATH

# ---------- Utility Functions ----------
def to_device(tensor, device=DEVICE):
    """Convert tensor to specified device."""
    return torch.tensor(tensor, dtype=torch.float32).to(device)

def validate_input_sequence(input_sequence, expected_shape=(SEQ_LEN, INPUT_DIM)):
    """Validate input sequence shape and contents."""
    if input_sequence is None or not isinstance(input_sequence, np.ndarray) or input_sequence.ndim != 2:
        raise ValueError(f"Expected 2D numpy array, got {type(input_sequence)} or shape {input_sequence.shape if input_sequence is not None else None}")
    if input_sequence.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {input_sequence.shape}")
    if np.isnan(input_sequence).any():
        raise ValueError("Input sequence contains NaN values")

def preprocess_sarima_data(data):
    """Preprocess data for SARIMA: remove NaN, check length, and variance."""
    if data.ndim != 1:
        raise ValueError("SARIMA requires 1D array")
    if np.isnan(data).any():
        data = np.where(np.isnan(data), np.nanmean(data), data)
    if len(data) < 5:  # حداقل طول افزایش یافت
        print("⚠️ Data too short. Using mean as fallback.")
        return np.full_like(data, np.mean(data))
    if np.std(data) < 1e-8:
        print("⚠️ Constant data detected. Adding noise.")
        data += np.random.normal(0, 1e-4, data.shape)
    return data

# ---------- SARIMA Forecaster ----------
class SARIMAForecaster:
    def __init__(self, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.results = None

    def fit(self, data):
        data = preprocess_sarima_data(data)
        try:
            self.model = SARIMAX(data, order=self.order, seasonal_order=self.seasonal_order,
                                 enforce_stationarity=True, enforce_invertibility=False)
            self.results = self.model.fit(disp=False, maxiter=100)  # افزایش maxiter
        except Exception as e:
            print(f"⚠️ SARIMA fit failed: {e}. Using mean as fallback.")
            self.results = type('MockResults', (), {
                'predict': lambda *args, **kwargs: np.mean(data),
                'get_forecast': lambda steps: type('MockForecast', (), {
                    'predicted_mean': np.full(steps, np.mean(data)),
                    'conf_int': lambda alpha: np.array([[np.mean(data), np.mean(data)]] * steps)
                })(),
                'resid': np.zeros_like(data)
            })()
        return self

    def predict(self, steps, start=None):
        if self.results is None:
            raise ValueError("Model not fitted")
        try:
            if start is not None:
                forecast = self.results.get_prediction(start=start, end=start+steps-1)
            else:
                forecast = self.results.get_forecast(steps=steps)
            mean = forecast.predicted_mean
            conf_int = forecast.conf_int(alpha=0.05)
            mean = mean[0] if isinstance(mean, np.ndarray) else mean
            if isinstance(conf_int, np.ndarray):
                upper, lower = conf_int[0, 1], conf_int[0, 0]
            else:
                upper, lower = conf_int.iloc[0, 1], conf_int.iloc[0, 0]
            return mean, (lower, upper)
        except Exception as e:
            print(f"⚠️ SARIMA predict failed: {e}")
            return np.nan, (np.nan, np.nan)

    def in_sample_residuals(self):
        if self.results is None:
            raise ValueError("Model not fitted")
        return self.results.resid

# ---------- LSTM Model ----------
class ResidualLSTM(nn.Module):
    def __init__(self, input_dim=INPUT_DIM*2, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=INPUT_DIM):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# ---------- LSTM Train & Predict ----------
def train_lstm(model, X_train, y_train, lr=LSTM_LEARNING_RATE, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE):
    model = model.to(DEVICE)
    X_train = to_device(X_train)
    y_train = to_device(y_train)
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
        X_tensor = to_device(X)
        preds = model(X_tensor)
        return preds.cpu().numpy()

# ---------- Model Save & Load ----------
def save_lstm(model, path=MODEL_PATH):
    torch.save(model.state_dict(), path)

def load_lstm(model_class, path=MODEL_PATH):
    model = model_class()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE)

def save_scaler(scaler, path=SCALER_PATH):
    joblib.dump(scaler, path)

def load_scaler(path=SCALER_PATH):
    return joblib.load(path)

# ---------- Hybrid Forecasting ----------
def hybrid_forecast(sarima_models, lstm_model, input_sequence, scaler, forecast_steps=1, use_ee_lstm=True):
    validate_input_sequence(input_sequence, expected_shape=(SEQ_LEN, INPUT_DIM))
    
    forecasts, upper_bounds, lower_bounds, residuals = [], [], [], []
    
    sarima_residuals = []
    for i in range(input_sequence.shape[1]):
        univariate_seq = input_sequence[:, i]
        mean_pred, (lower, upper) = sarima_models[i].predict(steps=forecast_steps)
        if np.isnan(mean_pred):
            mean_pred, lower, upper = 0.0, 0.0, 0.0
        residual = univariate_seq[-1] - mean_pred
        sarima_residuals.append(residual)
        forecasts.append(mean_pred)
        upper_bounds.append(upper)
        lower_bounds.append(lower)
        residuals.append(residual)
    
    lstm_in = input_sequence[-forecast_steps:].reshape(1, forecast_steps, input_sequence.shape[1])
    if use_ee_lstm:
        sarima_residuals = np.array(sarima_residuals).reshape(1, 1, -1)
        lstm_in = np.concatenate([lstm_in, sarima_residuals], axis=2)
        if lstm_in.shape[-1] != INPUT_DIM * 2:
            raise ValueError(f"LSTM input size mismatch: expected {INPUT_DIM * 2}, got {lstm_in.shape[-1]}")
    
    print(f"DEBUG: lstm_in shape={lstm_in.shape}")
    lstm_preds = predict_lstm(lstm_model, lstm_in)[0]
    
    forecasts = np.array(forecasts) * 0.5 + lstm_preds * 0.5
    
    return (
        np.array(forecasts),
        np.array(upper_bounds),
        np.array(lower_bounds),
        np.array(residuals),
    )