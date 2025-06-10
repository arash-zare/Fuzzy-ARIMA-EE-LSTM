🔍 Fuzzy-ARIMA-EE-LSTM-PSO Anomaly Detection System
A hybrid anomaly detection system for real-time monitoring of time-series data (e.g., network metrics, system performance). It integrates SARIMA for statistical forecasting, Ensemble-Enhanced LSTM (EE-LSTM) for temporal modeling, Fuzzy Logic for interpretable anomaly scoring, and Particle Swarm Optimization (PSO) for tuning fuzzy parameters. The system exposes results via Prometheus for seamless integration with monitoring tools like Grafana.
🚀 Features

Real-Time Anomaly Detection: Processes time-series data with sliding windows for continuous monitoring.
Hybrid Modeling: Combines SARIMA (univariate forecasting) and LSTM (multivariate temporal patterns).
Explainable Decisions: Fuzzy logic provides interpretable risk scores based on residuals and confidence bounds.
PSO Optimization: Tunes fuzzy membership functions and rule weights for optimal performance.
Prometheus Integration: Exposes metrics (anomalies, risk scores, forecasts) for observability.
Modular Design: Easily extensible with swappable components (e.g., replace SARIMA with another model).

📚 Table of Contents

Architecture
Logic and Components
Installation
Usage
Configuration
Directory Structure
Flowchart
Contributing
License

🏗 Architecture
The system follows a modular pipeline, with PSO as an offline optimization step:
graph TD
    A[Data Source: VictoriaMetrics/Mock] --> B[Data Fetcher]
    B --> C[Preprocessing]
    C --> D[Hybrid Model]
    D --> D1[SARIMA]
    D --> D2[LSTM]
    D --> D3[Fuzzy Logic]
    D3 --> D3a[PSO Optimizer<br>(Offline)]
    D1 & D2 & D3 --> E[Anomaly Detection]
    E --> F[Prometheus Metrics]
    F --> G[Flask Server]


Data Fetcher: Retrieves real-time metrics from VictoriaMetrics or generates mock data.
Preprocessing: Normalizes data and builds sliding window sequences.
Hybrid Model:
SARIMA: Forecasts per feature, computes residuals and confidence bounds.
LSTM: Predicts next timesteps for multivariate data.
Fuzzy Logic: Evaluates anomaly risk using residuals and bounds.


PSO Optimizer (Offline): Optimizes fuzzy parameters before deployment.
Anomaly Detection: Combines model outputs to flag anomalies.
Prometheus Metrics: Exposes results via a Flask server.

🧠 Logic and Components

Data Fetching (data_fetcher.py):

Queries VictoriaMetrics for metrics (e.g., rate(node_network_receive_bytes_total[1m])) or uses mock data.
Handles errors by returning np.nan for missing values.


Preprocessing (preprocessing.py):

Normalizes data using StandardScaler (z-score) or MinMaxScaler.
Creates sliding window sequences (SEQ_LEN) for LSTM.


Hybrid Forecasting (model.py):

SARIMA: Fits a SARIMAX model per feature, producing forecasts, confidence intervals, and residuals.
LSTM: A ResidualLSTM predicts the next timestep for all features.
Hybrid Forecast: Combines SARIMA residuals, bounds, and LSTM predictions.


Fuzzy Logic (fuzzy.py):

Uses triangular membership functions for inputs (residual, upper_diff, lower_diff).
Applies rules (e.g., IF residual IS High AND upper_diff IS Low THEN risk IS High) to compute a risk score [0,1].
Parameters are tunable via PSO.


PSO Optimization (pso.py):

Optimizes fuzzy membership parameters and rule weights to minimize α(1-Accuracy) + β*FPR.
Runs offline and saves optimized parameters for use in the main pipeline.


Anomaly Detection (detect_anomalies.py):

Combines SARIMA and LSTM outputs with fuzzy risk scores.
Flags anomalies per feature using FUZZY_THRESHOLD.


Monitoring (main.py):

Runs a Flask server with a /metrics endpoint for Prometheus.
Maintains a sliding window buffer and updates metrics in real-time.


Configuration (config.py):

Centralizes parameters for data fetching, models, fuzzy logic, and PSO.



🛠 Installation
Prerequisites

Python 3.8+
Virtualenv (recommended)
VictoriaMetrics (optional, for real data)
Prometheus and Grafana (for monitoring)

Steps

Clone the repository:
git clone https://github.com/your-username/fuzzy-arima-ee-lstm-pso.git
cd fuzzy-arima-ee-lstm-pso


Set up a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:
pip install -r requirements.txt

Example requirements.txt:
flask==2.0.1
prometheus-client==0.14.1
numpy==1.23.5
torch==2.0.1
scikit-learn==1.2.2
statsmodels==0.14.0
requests==2.28.1
joblib==1.2.0


Configure VictoriaMetrics:

Update VICTORIA_METRICS_URL in config.py to your VictoriaMetrics instance.
Set USE_MOCK_DATA = True in data_fetcher.py for testing without real data.



🚀 Usage
Training Models

Train the LSTM model:
python train.py


Uses random data (replace with real data in train.py if needed).
Saves the trained LSTM model and scaler to MODEL_PATH and SCALER_PATH.


Optimize Fuzzy Parameters with PSO:
python pso.py


Optimizes fuzzy parameters and saves them (e.g., to optimized_fuzzy_params.pkl).
Update main.py to load optimized parameters:import joblib
pso_params = joblib.load("optimized_fuzzy_params.pkl")
mem_params = extract_membership_params(pso_params, fuzzy_system)
rule_weights = extract_rule_weights(pso_params, fuzzy_system)
fuzzy_system.set_params(membership_params=mem_params, rule_weights=rule_weights)





Running the Application

Start the Flask server:
python main.py


Fetches data every FETCH_INTERVAL seconds.
Detects anomalies and updates Prometheus metrics at http://localhost:8000/metrics.


Monitor with Prometheus/Grafana:

Configure Prometheus to scrape http://localhost:8000/metrics.
Visualize metrics in Grafana (e.g., network_receive_rate_anomaly, cpu_system_usage_fuzzy_risk).



Example Output
✅ network_receive_rate: Normal. (Risk=0.05)
🚨 cpu_system_usage: Anomaly Detected! (Risk=0.82)

⚙ Configuration
Key parameters in config.py:

Data Fetching:
VICTORIA_METRICS_URL: VictoriaMetrics endpoint (default: http://192.168.1.98:8428).
FEATURE_QUERIES: Prometheus queries for metrics.


Model:
SEQ_LEN: Sequence length for LSTM (default: 16).
SARIMA_ORDER: (p,d,q) for SARIMA (default: (1,1,1)).
HIDDEN_DIM, NUM_LAYERS: LSTM architecture (default: 64, 2).


Fuzzy Logic:
FUZZY_THRESHOLD: Anomaly risk threshold (default: 0.1).
FUZZY_RULES_COUNT: Number of fuzzy rules (default: 6).


PSO:
PSO_PARTICLES, PSO_ITER: Swarm size and iterations (default: 30, 40).
PSO_ALPHA, PSO_BETA: Objective function weights (default: 0.7, 0.3).



Tune these based on your data characteristics.
📊 Flowchart
Below is the pipeline for real-time anomaly detection:
graph TD
    A[Fetch Metrics<br>VictoriaMetrics/Mock] --> B[Buffer Sequence<br>SEQ_LEN]
    B --> C[Normalize with<br>Saved Scaler]
    C --> D1[SARIMA Forecast]
    C --> D2[LSTM Forecast]
    D1 --> E1[Mean, Upper, Lower]
    D2 --> E2[LSTM Predictions]
    E1 & E2 --> F[Compute Residuals<br>& Bound Diffs]
    F --> G[Fuzzy Inference<br>System]
    G --> H[Risk Score > FUZZY_THRESHOLD?]
    H -->|Yes| I[Flag as Anomaly]
    H -->|No| J[Mark as Normal]
    I & J --> K[Export Metrics<br>to Prometheus]

📁 Directory Structure
.
├── config.py               # Centralized configuration
├── data_fetcher.py         # Fetches metrics from VictoriaMetrics/mock
├── detect_anomalies.py     # Anomaly detection logic
├── fuzzy.py                # Fuzzy inference system
├── main.py                 # Flask server and monitoring loop
├── model.py                # SARIMA, LSTM, and hybrid forecasting
├── pso.py                  # PSO optimization for fuzzy parameters
├── preprocessing.py        # Data normalization and sequence building
├── train.py                # LSTM training script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

🤝 Contributing

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m 'Add your feature').
Push to the branch (git push origin feature/your-feature).
Open a Pull Request.

Please include tests and update documentation for new features.


Developed by [Arash Zare]For questions, contact [https://arashzare,ir] or open an issue on GitHub.
