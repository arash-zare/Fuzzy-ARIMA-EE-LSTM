import numpy as np
from model import hybrid_forecast
from fuzzy import evaluate_fuzzy_anomaly
from config import FEATURES, FUZZY_THRESHOLD

def detect_anomaly_per_feature(input_sequence, sarima_models, lstm_model, scaler, fuzzy_system, fuzzy_threshold=FUZZY_THRESHOLD):
    """
    اجرای تشخیص ناهنجاری برای همه featureها در یک sequence.
    input_sequence: np.ndarray (seq_len, n_features) - نرمال شده (ورودی sliding window)
    sarima_models: لیست مدل SARIMAForecaster برای هر feature
    lstm_model: مدل ResidualLSTM آموزش‌دیده
    scaler: fitted scaler object
    fuzzy_system: شی آماده FuzzySystem با پارامترهای نهایی
    fuzzy_threshold: آستانه خروجی فازی (پیش‌فرض از config)
    Returns:
        anomalies: dict {feature: 0/1}
        mse_per_feature: dict {feature: mse}
        fuzzy_risk_per_feature: dict {feature: [risk_score]}
        forecast_values, upper_bounds, lower_bounds: np.ndarray (features,)
    """
    # اجرای پیش‌بینی ترکیبی
    forecast_values, upper_bounds, lower_bounds, residuals = hybrid_forecast(
        sarima_models, lstm_model, input_sequence, scaler
    )
    actual = input_sequence[-1]  # مقدار واقعی آخرین نقطه (در مقیاس نرمال)
    anomalies = {}
    mse_per_feature = {}
    fuzzy_risk_per_feature = {}

    for i, feature in enumerate(FEATURES):
        # محاسبه انحراف از حدود بالا و پایین
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
        # آستانه‌بندی تصمیم فازی
        is_anomaly = int(risk_score > fuzzy_threshold)
        anomalies[feature] = is_anomaly
        # MSE (فقط برای معیار جانبی)
        mse = float((actual[i] - forecast_values[i]) ** 2)
        mse_per_feature[feature] = mse

    return anomalies, mse_per_feature, fuzzy_risk_per_feature, forecast_values, upper_bounds, lower_bounds

