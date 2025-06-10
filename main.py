from flask import Flask, Response
from prometheus_client import Gauge, generate_latest, REGISTRY
import prometheus_client as prom_client
import threading
import time
import numpy as np
import joblib  # اضافه شده برای بارگذاری پارامترهای PSO

from config import FEATURES, FETCH_INTERVAL, SEQ_LEN, MODEL_PATH, SCALER_PATH, FUZZY_THRESHOLD, DEVICE
from data_fetcher import get_data
from preprocessing import normalize, load_scaler, build_sequences
from model import SARIMAForecaster, ResidualLSTM, load_lstm, load_scaler as model_load_scaler
from fuzzy import FuzzySystem, default_membership_params, default_rule_list, default_rule_weights
from detect_anomalies import detect_anomaly_per_feature
from pso import extract_membership_params, extract_rule_weights  # اضافه شده برای استفاده از توابع PSO

app = Flask(__name__)

def sanitize_feature_name(feature):
    import re
    return re.sub(r'[^a-zA-Z0-9_]', '_', feature).lower()

def safe_gauge(name, documentation):
    try:
        metric = Gauge(name, documentation, registry=None)
        REGISTRY.unregister(metric)
    except KeyError:
        pass
    return Gauge(name, documentation)

# --------- Prometheus Metrics ----------
anomaly_gauges = {feature: safe_gauge(f"{sanitize_feature_name(feature)}_anomaly", f"Anomaly detection for {feature}") for feature in FEATURES}
mse_gauges = {feature: safe_gauge(f"{sanitize_feature_name(feature)}_mse", f"MSE error for {feature}") for feature in FEATURES}
forecast_gauges = {feature: safe_gauge(f"{sanitize_feature_name(feature)}_forecast", f"Forecasted value for {feature}") for feature in FEATURES}
upper_gauges = {feature: safe_gauge(f"{sanitize_feature_name(feature)}_upper", f"Upper bound for {feature}") for feature in FEATURES}
lower_gauges = {feature: safe_gauge(f"{sanitize_feature_name(feature)}_lower", f"Lower bound for {feature}") for feature in FEATURES}
fuzzy_risk_gauges = {feature: safe_gauge(f"{sanitize_feature_name(feature)}_fuzzy_risk", f"Fuzzy risk score for {feature}") for feature in FEATURES}

# --------- بارگذاری مدل‌ها ---------
print("📦 Loading scaler and models...")
scaler = model_load_scaler(SCALER_PATH)
lstm_model = load_lstm(ResidualLSTM, MODEL_PATH)

# SARIMA برای هر feature
sarima_models = [SARIMAForecaster() for _ in FEATURES]
sarima_fitted = [False for _ in FEATURES]

# ایجاد سیستم فازی و تنظیم با پارامترهای بهینه PSO
print("📦 Loading optimized fuzzy parameters...")
fuzzy_system = FuzzySystem(default_membership_params, default_rule_weights, default_rule_list)
try:
    pso_params = joblib.load("optimized_fuzzy_params.pkl")
    mem_params = extract_membership_params(pso_params, fuzzy_system)
    rule_weights = extract_rule_weights(pso_params, fuzzy_system)
    fuzzy_system.set_params(membership_params=mem_params, rule_weights=rule_weights)
    print("✅ Loaded optimized fuzzy parameters")
except FileNotFoundError:
    print("⚠️ No optimized fuzzy parameters found. Using default parameters.")
except Exception as e:
    print(f"❌ Error loading PSO parameters: {e}. Using default parameters.")

sequence_buffer = []

def monitor():
    global sequence_buffer, sarima_models, sarima_fitted

    while True:
        try:
            latest_data = get_data()  # (n_features, )
            sequence_buffer.append(latest_data)
            if len(sequence_buffer) > SEQ_LEN:
                sequence_buffer = sequence_buffer[-SEQ_LEN:]

            if len(sequence_buffer) == SEQ_LEN:
                # نرمال‌سازی با همان scaler ذخیره‌شده
                input_seq = np.array(sequence_buffer, dtype=np.float32)
                input_seq_norm = scaler.transform(input_seq)
                
                # مدل SARIMA را (برای هر feature) فقط یک بار روی داده denorm آموزش بده
                for i, feature in enumerate(FEATURES):
                    if not sarima_fitted[i]:
                        sarima_models[i].fit(input_seq[:, i])
                        sarima_fitted[i] = True

                # تشخیص ناهنجاری
                anomalies, mse_per_feature, fuzzy_risk_per_feature, forecast_values, upper_bounds, lower_bounds = detect_anomaly_per_feature(
                    input_seq_norm, sarima_models, lstm_model, scaler, fuzzy_system, fuzzy_threshold=FUZZY_THRESHOLD
                )

                # آپدیت متریک‌های Prometheus
                for i, feature in enumerate(FEATURES):
                    anomaly_gauges[feature].set(anomalies[feature])
                    mse_gauges[feature].set(mse_per_feature[feature])
                    forecast_gauges[feature].set(forecast_values[i])
                    upper_gauges[feature].set(upper_bounds[i])
                    lower_gauges[feature].set(lower_bounds[i])
                    fuzzy_risk_gauges[feature].set(fuzzy_risk_per_feature[feature])

                    if anomalies[feature]:
                        print(f"🚨 {feature}: Anomaly Detected! (Risk={fuzzy_risk_per_feature[feature]:.2f})")
                    else:
                        print(f"✅ {feature}: Normal. (Risk={fuzzy_risk_per_feature[feature]:.2f})")

        except Exception as e:
            print(f"[monitor] ❌ Error: {e}")

        time.sleep(FETCH_INTERVAL)

# -- حذف collectorهای اضافی Prometheus
try:
    REGISTRY.unregister(prom_client.GC_COLLECTOR)
    REGISTRY.unregister(prom_client.PLATFORM_COLLECTOR)
    REGISTRY.unregister(prom_client.PROCESS_COLLECTOR)
except Exception as e:
    print(f"ℹ️ Could not unregister some collectors: {e}")

# --- Flask route برای metrics ---
@app.route('/metrics')
def metrics():
    return Response(generate_latest(REGISTRY), mimetype='text/plain')

if __name__ == '__main__':
    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.daemon = True
    monitor_thread.start()
    app.run(host='0.0.0.0', port=8000)