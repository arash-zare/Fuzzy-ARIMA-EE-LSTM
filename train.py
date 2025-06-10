import numpy as np
from preprocessing import normalize, build_sequences, save_scaler
from model import ResidualLSTM, train_lstm, save_lstm
from config import SCALER_PATH, MODEL_PATH, SEQ_LEN, INPUT_DIM

# فرض: داده خام را از هر جایی (مثلاً CSV یا ... یا mock) می‌گیری
data = np.random.rand(1000, INPUT_DIM)  # یا load_from_csv('your_data.csv')
data_scaled, scaler = normalize(data, method="z-score", fit=True, scaler_path=SCALER_PATH)

X, y = build_sequences(data_scaled, seq_len=SEQ_LEN)
print("Training LSTM: ", X.shape, y.shape)

lstm_model = ResidualLSTM()
lstm_model = train_lstm(lstm_model, X, y)
save_lstm(lstm_model, MODEL_PATH)
save_scaler(scaler, SCALER_PATH)

print("✅ LSTM and scaler trained & saved!")
