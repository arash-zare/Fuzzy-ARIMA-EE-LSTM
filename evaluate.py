import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model import SARIMAForecaster, ResidualLSTM, load_lstm, load_scaler, hybrid_forecast
from pso import fetch_data_from_victoriametrics, build_sequences
from fuzzy import FuzzySystem, default_membership_params, default_rule_list, default_rule_weights
from config import FEATURES, SEQ_LEN, INPUT_DIM, MODEL_PATH, SCALER_PATH, THRESHOLDS, INPUT_DIM_EE
import joblib

def calculate_regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2
    }

def calculate_classification_metrics(y_true, y_pred, y_scores=None):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp + 1e-8)
    roc_auc = roc_auc_score(y_true, y_scores) if y_scores is not None else None
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Specificity": specificity,
        "ROC-AUC": roc_auc
    }

def evaluate_model(start_time="now-1d", end_time="now", step="30s"):
    print("📦 Loading models and scaler...")
    scaler = load_scaler()
    lstm_model = load_lstm(lambda: ResidualLSTM(input_dim=INPUT_DIM_EE), MODEL_PATH)
    sarima_models = [SARIMAForecaster() for _ in FEATURES]

    print("📦 Loading fuzzy system...")
    try:
        fuzzy_params = joblib.load("optimized_fuzzy_params.pkl")
        fuzzy_system = FuzzySystem(fuzzy_params, default_rule_weights, default_rule_list)
    except FileNotFoundError:
        print("⚠️ No optimized fuzzy parameters found. Using default parameters.")
        fuzzy_system = FuzzySystem(default_membership_params, default_rule_weights, default_rule_list)

    print("🔄 Fetching test data from VictoriaMetrics...")
    try:
        data = fetch_data_from_victoriametrics(start_time, end_time, step)
        print(f"Test data shape: {data.shape}, NaNs: {np.isnan(data).sum()}, Std: {np.std(data, axis=0)}")
    except Exception as e:
        print(f"❌ Failed to fetch data: {e}. Using random data...")
        data = np.random.rand(1000, INPUT_DIM)

    # Preprocessing
    if np.isnan(data).any():
        print("⚠️ Filling NaNs with mean...")
        data = np.where(np.isnan(data), np.nanmean(data, axis=0), data)
    if np.std(data, axis=0).min() < 1e-8:
        print("⚠️ Low variance detected. Adding noise...")
        data += np.random.normal(0, 1e-4, data.shape)

    data_scaled = scaler.transform(data)
    X, y = build_sequences(data_scaled, seq_len=SEQ_LEN)
    print(f"Test sequences: X shape={X.shape}, y shape={y.shape}")

    # Forecasting and Anomaly Detection
    all_forecasts = []
    all_actuals = []
    all_anomaly_preds = []
    all_anomaly_true = []
    all_anomaly_scores = []

    for i, input_seq in enumerate(X):
        try:
            # Fit SARIMA for each feature
            for j, feature in enumerate(FEATURES):
                sarima_models[j].fit(input_seq[:, j])

            # Hybrid forecast (!!! THIS LINE WAS CHANGED !!!)
            forecasts, upper_bounds, lower_bounds, residuals, _ = hybrid_forecast(
                sarima_models, lstm_model, input_seq, scaler, forecast_steps=1
            )

            actual = y[i]
            all_forecasts.append(forecasts)
            all_actuals.append(actual)

            for j, feature in enumerate(FEATURES):
                input_dict = {
                    "residual": residuals[j],
                    "upper_diff": actual[j] - upper_bounds[j],
                    "lower_diff": actual[j] - lower_bounds[j]
                }
                anomaly_score = fuzzy_system.infer(input_dict)
                anomaly_pred = 1 if anomaly_score > 0.5 else 0
                threshold = THRESHOLDS.get(feature, 1.0)
                anomaly_true = 1 if actual[j] > threshold else 0

                all_anomaly_preds.append(anomaly_pred)
                all_anomaly_true.append(anomaly_true)
                all_anomaly_scores.append(anomaly_score)

        except Exception as e:
            print(f"❌ Error processing sequence {i}: {e}")
            continue

    # Convert to arrays
    all_forecasts = np.array(all_forecasts)
    all_actuals = np.array(all_actuals)
    all_anomaly_preds = np.array(all_anomaly_preds)
    all_anomaly_true = np.array(all_anomaly_true)
    all_anomaly_scores = np.array(all_anomaly_scores)

    # Metrics
    regression_metrics = calculate_regression_metrics(all_actuals, all_forecasts)
    classification_metrics = calculate_classification_metrics(all_anomaly_true, all_anomaly_preds, all_anomaly_scores)

    results = {
        "Regression Metrics": regression_metrics,
        "Classification Metrics": classification_metrics,
        "Data Summary": {
            "Test Sequences": len(X),
            "Features": FEATURES,
            "NaNs in Data": int(np.isnan(data).sum()),
            "Low Variance Features": bool(np.std(data, axis=0).min() < 1e-8)
        }
    }

    # Print results
    print("\n📊 Evaluation Results:")
    print("Regression Metrics:")
    for key, value in regression_metrics.items():
        print(f"  {key}: {value:.4f}")
    print("Classification Metrics:")
    for key, value in classification_metrics.items():
        if value is not None:
            print(f"  {key}: {value:.4f}")

    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("✅ Results saved to 'evaluation_results.json'")

    return results

if __name__ == "__main__":
    evaluate_model()
