import numpy as np
import joblib
import requests
from model import SARIMAForecaster, ResidualLSTM, load_lstm, load_scaler, hybrid_forecast
from preprocessing import normalize, build_sequences
from fuzzy import FuzzySystem, default_membership_params, default_rule_list, default_rule_weights
from config import FEATURES, FEATURE_QUERIES, VICTORIA_METRICS_URL, INPUT_DIM, SEQ_LEN, MODEL_PATH, SCALER_PATH, THRESHOLDS

class Particle:
    def __init__(self, dim, bounds):
        self.position = np.random.uniform(bounds[:,0], bounds[:,1], size=dim)
        self.velocity = np.zeros(dim)
        self.best_position = np.copy(self.position)
        self.best_score = np.inf

class PSO:
    def __init__(self, objective_func, dim, bounds, num_particles=30, max_iter=40, w=0.7, c1=1.5, c2=1.5):
        self.objective_func = objective_func
        self.dim = dim
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w, self.c1, self.c2 = w, c1, c2

    def optimize(self):
        swarm = [Particle(self.dim, self.bounds) for _ in range(self.num_particles)]
        gbest_position = np.copy(swarm[0].position)
        gbest_score = np.inf
        for t in range(self.max_iter):
            for particle in swarm:
                score = self.objective_func(particle.position)
                if score < particle.best_score:
                    particle.best_score = score
                    particle.best_position = np.copy(particle.position)
                if score < gbest_score:
                    gbest_score = score
                    gbest_position = np.copy(particle.position)
            for particle in swarm:
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = self.c1 * r1 * (particle.best_position - particle.position)
                social = self.c2 * r2 * (gbest_position - particle.position)
                particle.velocity = self.w * particle.velocity + cognitive + social
                particle.position += particle.velocity
                particle.position = np.clip(particle.position, self.bounds[:,0], self.bounds[:,1])
            if (t+1) % 10 == 0 or t == 0:
                print(f"PSO iter {t+1}/{self.max_iter} | Best score: {gbest_score:.2f}")
        
        joblib.dump(gbest_position, "optimized_fuzzy_params.pkl")
        print("✅ Optimized fuzzy parameters saved to 'optimized_fuzzy_params.pkl'")
        
        return gbest_position, gbest_score

def fuzzy_objective(param_vector, fuzzy_system_template, train_X, train_labels, alpha=0.7, beta=0.3):
    mem_params = extract_membership_params(param_vector, fuzzy_system_template)
    rule_weights = extract_rule_weights(param_vector, fuzzy_system_template)
    fuzzy_system = fuzzy_system_template
    fuzzy_system.set_params(membership_params=mem_params, rule_weights=rule_weights)
    preds = []
    for i in range(len(train_X)):
        input_dict = {
            "residual": train_X[i,0],
            "upper_diff": train_X[i,1],
            "lower_diff": train_X[i,2]
        }
        preds.append(fuzzy_system.infer(input_dict))
    preds = np.array(preds)
    preds_bin = (preds > 0.5).astype(int)
    acc = np.mean(preds_bin == train_labels)
    fpr = np.sum((preds_bin == 1) & (train_labels == 0)) / (np.sum(train_labels == 0) + 1e-8)
    return alpha * (1 - acc) + beta * fpr

def extract_membership_params(param_vector, fuzzy_system_template):
    num_inputs = len(fuzzy_system_template.input_names)
    mem_params = {}
    idx = 0
    for var in fuzzy_system_template.input_names:
        mem_params[var] = []
        for _ in range(3):
            mem_params[var].append(tuple(param_vector[idx:idx+3]))
            idx += 3
    return mem_params

def extract_rule_weights(param_vector, fuzzy_system_template):
    rule_count = len(fuzzy_system_template.rule_list)
    num_inputs = len(fuzzy_system_template.input_names)
    mem_params_len = num_inputs * 3 * 3
    return param_vector[mem_params_len:mem_params_len+rule_count]

def fetch_data_from_victoriametrics(start_time, end_time, step="30s"):
    data = []
    for feature in FEATURES:
        query = FEATURE_QUERIES[feature]
        url = f"{VICTORIA_METRICS_URL}/api/v1/query_range"
        params = {
            "query": query,
            "start": start_time,
            "end": end_time,
            "step": step
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            result = resp.json()['data']['result']
            if result:
                values = [float(v[1]) for v in result[0]['values']]
                data.append(values)
            else:
                print(f"⚠️ No data for {feature}. Filling with zeros.")
                data.append([0.0] * (len(data[0]) if data else 100))
        except Exception as e:
            print(f"❌ Error fetching {feature}: {e}")
            data.append([0.0] * (len(data[0]) if data else 100))
    
    data = np.array(data).T
    nan_count = np.isnan(data).sum()
    if nan_count > 0:
        print(f"⚠️ Found {nan_count} NaN values. Applying forward-fill and mean imputation...")
        for j in range(data.shape[1]):
            col = data[:, j]
            mask = np.isnan(col)
            if mask.any():
                col[mask] = np.nanmean(col)  # پر کردن با میانگین
                if np.isnan(col).any():  # اگر هنوز NaN باقی ماند
                    col[np.isnan(col)] = 0.0
    # بررسی واریانس
    stds = np.std(data, axis=0)
    for j, (std, feature) in enumerate(zip(stds, FEATURES)):
        if std < 1e-8:
            print(f"⚠️ Low variance in {feature}. Setting to zeros.")
            data[:, j] = 0.0
    return data

def generate_training_data_from_victoriametrics(start_time="now-1d", end_time="now", step="30s", max_sequences=100):
    data = fetch_data_from_victoriametrics(start_time, end_time, step)
    
    if data.shape[0] < SEQ_LEN + 1:
        raise ValueError(f"Not enough samples ({data.shape[0]}) for SEQ_LEN={SEQ_LEN}")
    
    scaler = load_scaler(SCALER_PATH)
    data_scaled = scaler.transform(data)
    
    X, _ = build_sequences(data_scaled, seq_len=SEQ_LEN)
    
    if len(X) > max_sequences:
        indices = np.random.choice(len(X), max_sequences, replace=False)
        X = X[indices]
    
    lstm_model = load_lstm(ResidualLSTM, MODEL_PATH)
    sarima_models = [SARIMAForecaster() for _ in FEATURES]
    
    train_X = []
    train_labels = []
    
    for i, input_seq in enumerate(X):
        try:
            if np.isnan(input_seq).any() or np.std(input_seq, axis=0).min() < 1e-8:
                print(f"Skipping sequence {i}: Invalid data (NaN or constant)")
                continue
            
            for j, feature in enumerate(FEATURES):
                try:
                    sarima_models[j].fit(input_seq[:, j])
                except Exception as e:
                    print(f"⚠️ SARIMA fit failed for {feature} in sequence {i}: {e}")
                    continue
            
            forecasts, upper_bounds, lower_bounds, residuals = hybrid_forecast(
                sarima_models, lstm_model, input_seq, scaler, forecast_steps=1, use_ee_lstm=True
            )
            
            actual = input_seq[-1]
            for j, feature in enumerate(FEATURES):
                upper_diff = actual[j] - upper_bounds[j]
                lower_diff = actual[j] - lower_bounds[j]
                train_X.append([residuals[j], upper_diff, lower_diff])
                threshold = THRESHOLDS.get(feature, 1.0)
                label = 1 if actual[j] > threshold else 0
                train_labels.append(label)
        
        except Exception as e:
            print(f"❌ Error processing sequence {i}: {e}")
            continue
    
    if not train_X:
        raise ValueError("No valid sequences processed. Check data or models.")
    
    train_X = np.array(train_X)
    train_labels = np.array(train_labels)
    
    joblib.dump({"train_X": train_X, "train_labels": train_labels}, "pso_training_data.pkl")
    print("✅ PSO training data saved to 'pso_training_data.pkl'")
    
    return train_X, train_labels

if __name__ == "__main__":
    fuzzy_system = FuzzySystem(default_membership_params, default_rule_weights, default_rule_list)
    
    try:
        data = joblib.load("pso_training_data.pkl")
        train_X = data["train_X"]
        train_labels = data["train_labels"]
        print("✅ Loaded PSO training data")
    except FileNotFoundError:
        print("🔄 Fetching data from VictoriaMetrics...")
        train_X, train_labels = generate_training_data_from_victoriametrics()
    
    dim = 3*3*3 + len(default_rule_list)
    bounds = np.array([[0,2]] * dim)
    pso = PSO(
        objective_func=lambda params: fuzzy_objective(params, fuzzy_system, train_X, train_labels),
        dim=dim,
        bounds=bounds,
        num_particles=20,
        max_iter=10
    )
    best_params, best_score = pso.optimize()
    print("Best params:", best_params)
    print("Best score:", best_score)