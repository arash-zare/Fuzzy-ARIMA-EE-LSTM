# config.py
import torch

# -------- Data Fetching ---------
VICTORIA_METRICS_URL = "http://192.168.1.98:8428"
FETCH_INTERVAL = 5

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
INPUT_DIM_EE = INPUT_DIM * 3  # برای EE-LSTM

# ---------- Model ----------
SEQ_LEN = 16      # input length to LSTM
FORECAST_STEPS = 2
MODEL_PATH = "sarima_eelstm_model.pth"
SCALER_PATH = "sarima_eelstm_scaler.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- SARIMA ----------
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (0, 0, 0, 0)

# ---------- LSTM ----------
HIDDEN_DIM = 64
NUM_LAYERS = 2
LSTM_LEARNING_RATE = 1e-3
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32

# ---------- Fuzzy Logic ----------
FUZZY_MEMBERSHIP_TYPE = "triangular"
FUZZY_RULES_COUNT = 6
FUZZY_INPUTS = ["residual", "upper_diff", "lower_diff"]
FUZZY_OUTPUT = "anomaly_risk"
FUZZY_THRESHOLD = 0.5  # Default threshold, adjust based on validation!
FUZZY_LEVELS = [0.0, 0.5, 1.0]  # For "low", "medium", "high" output mapping

# ---------- PSO ----------
PSO_PARTICLES = 30
PSO_ITER = 40
PSO_INERTIA = 0.7
PSO_COGNITIVE = 1.5
PSO_SOCIAL = 1.5
PSO_ALPHA = 0.7
PSO_BETA = 0.3
PSO_BOUNDS_MEMBERSHIP = [0, 2]  # [min, max] for all fuzzy params (customize if needed)

# ---------- Anomaly Thresholds ----------
THRESHOLDS = {
    'network_receive_rate': 7200,
    'network_transmit_rate': 18000,
    'active_connections': 40,  # update if needed
    'receive_packets_rate': 50,
    'receive_errors_rate': 10,
    'cpu_system_usage': 0.01,
    'memory_available_ratio': 0.8,
    'receive_drops_rate': 10,
}

# ---------- Evaluation ----------
METRICS = ["F1", "Precision", "Recall", "RMSE"]

# ---------- Random seed ----------
SEED = 42
torch.manual_seed(SEED)
