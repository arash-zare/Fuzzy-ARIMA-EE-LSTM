# config.py
# Central configuration for Fuzzy ARIMA-EE-LSTM-PSO anomaly detection platform

import torch

### Data Fetching
VICTORIA_METRICS_URL = "http://192.168.1.98:8428"
FETCH_INTERVAL = 30  # seconds

FEATURE_QUERIES = {
    "network_receive_rate": 'rate(node_network_receive_bytes_total[1m])',
    "network_transmit_rate": 'rate(node_network_transmit_bytes_total[1m])',
    "active_connections": 'node_nf_conntrack_entries',
    "receive_packets_rate": 'rate(node_network_receive_packets_total[1m])',
    "receive_errors_rate": 'rate(node_network_receive_errs_total[1m])',
    "cpu_system_usage": 'rate(node_cpu_seconds_total{mode="system"}[1m])',
    "memory_available_ratio": 'node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes',
    "receive_drops_rate": 'rate(node_network_receive_drop_total[1m])',
}
FEATURES = list(FEATURE_QUERIES.keys())
INPUT_DIM = len(FEATURES)

### Model parameters
SEQ_LEN = 16                   # Length of input sequence for LSTM (timesteps)
FORECAST_STEPS = 2             # Number of steps to forecast ahead
MODEL_PATH = "sarima_eelstm_model.pth"
SCALER_PATH = "sarima_eelstm_scaler.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### SARIMA config
SARIMA_ORDER = (1, 1, 1)              # (p, d, q)
SARIMA_SEASONAL_ORDER = (0, 0, 0, 0)  # (P, D, Q, s) -- adjust for your data

### LSTM config
HIDDEN_DIM = 64
NUM_LAYERS = 2
LSTM_LEARNING_RATE = 1e-3
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32

### Fuzzy logic config
FUZZY_MEMBERSHIP_TYPE = "triangular"        # 'triangular', 'trapezoidal', ...
FUZZY_RULES_COUNT = 6                       # Number of fuzzy rules (example)
FUZZY_INPUTS = ["residual", "upper_diff", "lower_diff"]
FUZZY_OUTPUT = "anomaly_risk"
FUZZY_THRESHOLD = 0.1                       # Default threshold for anomaly risk

### PSO (Particle Swarm Optimization) config
PSO_PARTICLES = 30          # Number of particles in swarm
PSO_ITER = 40               # Maximum PSO iterations
PSO_INERTIA = 0.7           # Inertia weight (w)
PSO_COGNITIVE = 1.5         # Cognitive component (c1)
PSO_SOCIAL = 1.5            # Social component (c2)
PSO_ALPHA = 0.7             # Weight for (1-Accuracy) in loss
PSO_BETA = 0.3              # Weight for False Positive Rate in loss

### Anomaly threshold for each feature (use fuzzy threshold for main output)
THRESHOLDS = {
    'network_receive_rate': 7200,
    'network_transmit_rate': 18000,
    'active_connections': 1,
    'receive_packets_rate': 50,
    'receive_errors_rate': 10,
    'cpu_system_usage': 0.01,
    'memory_available_ratio': 0.8,
    'receive_drops_rate': 10,
}

### Evaluation metrics
METRICS = ["F1", "Precision", "Recall", "RMSE"]

# For reproducibility (optional)
SEED = 42
torch.manual_seed(SEED)
