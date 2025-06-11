import os
import pickle
import threading
import time
import numpy as np
import torch

from flask import Flask, Response
from prometheus_client import Gauge, generate_latest, REGISTRY

from config import *
from data_fetcher import fetch_latest_data
from preprocessing import normalize, load_scaler
from model import load_lstm, SARIMAForecaster, hybrid_forecast, ResidualLSTM
from fuzzy import FuzzySystem, default_membership_params, default_rule_list, default_rule_weights, evaluate_fuzzy_anomaly

# ---------- Flask App ----------
app = Flask(__name__)

# ---------- Prometheus Metric Utilities ----------
def sanitize_feature_name(feature):
    import re
    return re.sub(r'[^a-zA-Z0-9_]', '_', feature).lower()

def safe_gauge(name, documentation):
    """Create Gauge and unregister old one if it exists."""
    try:
        metric = Gauge(name, documentation, registry=None)
        REGISTRY.unregister(metric)
    except Exception:
        pass
    return Gauge(name, documentation)

def build_metric_dict(name_suffix, doc_suffix):
    return {
        f: safe_gauge(f"{sanitize_feature_name(f)}_{name_suffix}", f"{doc_suffix} for {f}")
        for f in FEATURES
    }

anomaly_gauges   = build_metric_dict("anomaly",   "Anomaly detection")
mse_gauges       = build_metric_dict("mse",       "MSE error")
forecast_gauges  = build_metric_dict("forecast",  "Forecasted value")
upper_gauges     = build_metric_dict("upper",     "Upper bound")
lower_gauges     = build_metric_dict("lower",     "Lower bound")
fuzzy_gauges     = build_metric_dict("fuzzy_risk","Fuzzy risk score")

# ---------- Model and Buffer Initialization ----------
scaler = load_scaler(SCALER_PATH)
lstm_model = load_lstm(lambda: ResidualLSTM(input_dim=INPUT_DIM_EE), MODEL_PATH)
sarima_models = [SARIMAForecaster() for _ in FEATURES]

sequence_buffer = []
BUFFER_PATH = "sequence_buffer.pkl"
if os.path.exists(BUFFER_PATH):
    with open(BUFFER_PATH, "rb") as f:
        sequence_buffer = pickle.load(f)

# ---------- Fuzzy System ----------
fuzzy_system = FuzzySystem(default_membership_params, default_rule_weights, default_rule_list)

def refit_sarima_models(buffer, models):
    for i, feature in enumerate(FEATURES):
        try:
            univariate = np.array(buffer)[:, i]
            models[i].fit(univariate)
        except Exception as e:
            print(f"[SARIMA][{feature}] Error: {e}")

def save_sequence_buffer(buffer):
    with open(BUFFER_PATH, "wb") as f:
        pickle.dump(buffer, f)

def anomaly_monitor():
    global sequence_buffer, sarima_models, fuzzy_system
    sarima_refit_interval = 3600  # seconds
    last_refit_time = time.time()
    sarima_initialized = False

    while True:
        try:
            data = fetch_latest_data()
            # NaN handling
            if any(np.isnan(x) for x in data):
                if sequence_buffer:
                    data = [d if not np.isnan(d) else sequence_buffer[-1][i] for i, d in enumerate(data)]
                else:
                    data = [0 if np.isnan(d) else d for d in data]

            sequence_buffer.append(data)
            if len(sequence_buffer) > SEQ_LEN:
                sequence_buffer = sequence_buffer[-SEQ_LEN:]

            if len(sequence_buffer) == SEQ_LEN:
                # First-time SARIMA fit
                if not sarima_initialized:
                    refit_sarima_models(sequence_buffer, sarima_models)
                    last_refit_time = time.time()
                    sarima_initialized = True

                # Periodic SARIMA re-fit
                if time.time() - last_refit_time > sarima_refit_interval:
                    refit_sarima_models(sequence_buffer, sarima_models)
                    last_refit_time = time.time()

                # Hybrid forecasting
                input_seq = np.array(sequence_buffer, dtype=np.float32)
                input_seq_norm = normalize(input_seq, scaler=scaler)

                forecast, upper, lower, residual, lstm_pred = hybrid_forecast(
                    sarima_models, lstm_model, input_seq_norm, scaler, forecast_steps=1
                )

                # Update Prometheus metrics
                for i, feature in enumerate(FEATURES):
                    risk = evaluate_fuzzy_anomaly(
                        forecast[i], input_seq[-1, i], upper[i], lower[i], fuzzy_system=fuzzy_system
                    )
                    anomaly = int(risk > FUZZY_THRESHOLD)
                    mse = float((input_seq[-1, i] - forecast[i]) ** 2)
                    anomaly_gauges[feature].set(anomaly)
                    mse_gauges[feature].set(mse)
                    forecast_gauges[feature].set(forecast[i])
                    upper_gauges[feature].set(upper[i])
                    lower_gauges[feature].set(lower[i])
                    fuzzy_gauges[feature].set(risk)
                    print(f"[{feature}] Risk={risk:.2f} | Status: {'🚨 ANOMALY' if anomaly else '✅ Normal'}")

            if len(sequence_buffer) % 100 == 0:
                save_sequence_buffer(sequence_buffer)

        except Exception as e:
            print(f"[monitor] ❌ Error: {e}")
        time.sleep(FETCH_INTERVAL)

# ---------- Flask Prometheus endpoint ----------
@app.route("/metrics")
def metrics():
    return Response(generate_latest(REGISTRY), mimetype="text/plain")

if __name__ == "__main__":
    monitor_thread = threading.Thread(target=anomaly_monitor, daemon=True)
    monitor_thread.start()
    app.run(host="0.0.0.0", port=8000)
