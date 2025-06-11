import numpy as np
from model import hybrid_forecast
from fuzzy import evaluate_fuzzy_anomaly
from config import FEATURES, FUZZY_THRESHOLD, INPUT_DIM, INPUT_DIM_EE, SEQ_LEN

def detect_anomaly_per_feature(input_sequence, sarima_models, lstm_model, scaler, fuzzy_system, fuzzy_threshold=FUZZY_THRESHOLD):
    """
    اجرای تشخیص ناهنجاری برای همه featureها در یک sequence.
    input_sequence: np.ndarray (seq_len, input_dim) - sequence نرمال شده فقط داده خام
    sarima_models: لیست مدل SARIMAForecaster برای هر feature
    lstm_model: مدل ResidualLSTM آموزش‌دیده (input_dim=INPUT_DIM_EE)
    scaler: fitted scaler object
    fuzzy_system: شی آماده FuzzySystem با پارامترهای نهایی
    fuzzy_threshold: آستانه خروجی فازی (پیش‌فرض از config)
    """
    # --- ترکیب sequence برای EE-LSTM: ---
    # داده خام را به همراه forecast و residual تبدیل کن به (seq_len, input_dim*3)
    seq_len, n_features = input_sequence.shape
    if n_features != INPUT_DIM or seq_len != SEQ_LEN:
        raise ValueError(f"input_sequence shape expected ({SEQ_LEN}, {INPUT_DIM}), got {input_sequence.shape}")

    sarima_forecasts = []
    sarima_residuals = []
    for i in range(INPUT_DIM):
        preds = []
        resids = []
        for t in range(seq_len):
            try:
                pred, _ = sarima_models[i].predict(steps=1, start=t)
                pred = pred.values[0] if hasattr(pred, "values") else pred[0]
            except Exception:
                pred = 0.0
            preds.append(pred)
            resids.append(input_sequence[t, i] - pred)
        sarima_forecasts.append(preds)
        sarima_residuals.append(resids)
    sarima_forecasts = np.array(sarima_forecasts).T  # (seq_len, n_features)
    sarima_residuals = np.array(sarima_residuals).T  # (seq_len, n_features)

    X_EE = np.concatenate([input_sequence, sarima_forecasts, sarima_residuals], axis=1)[None, :, :]  # (1, seq_len, input_dim*3)

    # اجرای پیش‌بینی ترکیبی EE-LSTM
    forecast_values, upper_bounds, lower_bounds, residuals = hybrid_forecast(
        sarima_models, lstm_model, X_EE[0], scaler
    )
    actual = input_sequence[-1]  # مقدار واقعی آخرین نقطه (در مقیاس نرمال)
    anomalies = {}
    mse_per_feature = {}
    fuzzy_risk_per_feature = {}

    for i, feature in enumerate(FEATURES):
        upper_diff = actual[i] - upper_bounds[i]
        lower_diff = actual[i] - lower_bounds[i]
        residual = residuals[i]
        risk_score = evaluate_fuzzy_anomaly(
            residual=residual,
            upper_diff=upper_diff,
            lower_diff=lower_diff,
            fuzzy_system=fuzzy_system
        )
        fuzzy_risk_per_feature[feature] = risk_score
        is_anomaly = int(risk_score > fuzzy_threshold)
        anomalies[feature] = is_anomaly
        mse = float((actual[i] - forecast_values[i]) ** 2)
        mse_per_feature[feature] = mse

    return anomalies, mse_per_feature, fuzzy_risk_per_feature, forecast_values, upper_bounds, lower_bounds
