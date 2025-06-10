# 🔍 Fuzzy-ARIMA-EE-LSTM-PSO Anomaly Detection System

This project is a **hybrid anomaly detection system** combining statistical forecasting, deep learning, fuzzy logic, and optimization techniques for accurate and explainable anomaly detection in time-series data (such as network or system metrics). The key components include:

- **ARIMA (SARIMA)** for univariate trend forecasting.
- **LSTM** for capturing temporal dependencies.
- **Fuzzy Logic System** for explainable decision-making.
- **PSO (Particle Swarm Optimization)** for tuning fuzzy parameters.
- **Prometheus Integration** for live metrics monitoring.

---

## 🚀 Features

- Real-time anomaly detection with time series forecasting.
- Prometheus-compatible `/metrics` endpoint for observability.
- Explainable fuzzy decision system with rule-based logic.
- Self-tunable via PSO (optional training module).
- Modular architecture: easy to swap components (SARIMA/LSTM/etc).

---

## 📁 Project Structure

```bash
.
├── main.py                 # Prometheus Flask app entrypoint
├── data_fetcher.py         # Fetches latest feature values from VictoriaMetrics
├── preprocessing.py        # Data normalization, sequence building, etc.
├── model.py                # SARIMA forecaster & Residual LSTM model
├── fuzzy.py                # Fuzzy inference engine
├── pso.py                  # PSO optimizer for fuzzy tuning
├── detect_anomalies.py     # Combines predictions & fuzzy logic for detection
├── train.py                # LSTM model training script
├── config.py               # Centralized configuration
├── requirements.txt        # Python dependencies
└── README.md               # 📘 This file


flowchart TD
    A[Start: Fetch Metrics] --> B[Buffer Sequence (SEQ_LEN)]
    B --> C[Normalize with Saved Scaler]
    C --> D1[SARIMA Forecast]
    C --> D2[LSTM Forecast]
    D1 --> E1[Mean, Upper, Lower]
    D2 --> E2[LSTM Preds]
    E1 & E2 --> F[Compute Residuals & Diffs]
    F --> G[Fuzzy Inference System]
    G --> H[Risk Score > Threshold?]
    H -- Yes --> I[Flag as Anomaly]
    H -- No --> J[Mark as Normal]
    I & J --> K[Export Metrics to Prometheus]
