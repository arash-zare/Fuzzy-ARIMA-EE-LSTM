import numpy as np

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
        self.bounds = bounds  # np.array shape (dim, 2)
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
                # Clip to bounds
                particle.position = np.clip(particle.position, self.bounds[:,0], self.bounds[:,1])
            if (t+1) % 10 == 0 or t == 0:
                print(f"PSO iter {t+1}/{self.max_iter} | Best score: {gbest_score:.4f}")
        return gbest_position, gbest_score

# ==== تابع هدف برای بهینه‌سازی فازی ====
def fuzzy_objective(param_vector, fuzzy_system_template, train_X, train_labels, alpha=0.7, beta=0.3):
    """
    param_vector: آرایه شامل مقادیر توابع عضویت و وزن قواعد
    fuzzy_system_template: شی پایه FuzzySystem برای ست کردن پارامترها
    train_X: آرایه ورودی‌های آموزشی (n_samples, 3)
    train_labels: برچسب‌های واقعی ریسک (۰ یا ۱ یا مقدار پیوسته)
    alpha, beta: ضرایب وزن‌دهی
    خروجی: مقدار J = α(1-Acc) + β*FPR
    """
    # فرض: membership_params و rule_weights در یک وکتور concatenated هستند
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
    # پیاده‌سازی: بخش مربوط به توابع عضویت را از param_vector جدا کن و به قالب dict برای fuzzy_system بده
    # بستگی به ساختار membership_params و ترتیب پارامترها دارد
    # مثلاً:
    # فرض: هر ورودی ۳ تابع عضویت و هر تابع سه پارامتر (a,b,c)
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
    # فرض: بقیه پارامترها وزن قواعد هستند
    rule_count = len(fuzzy_system_template.rule_list)
    num_inputs = len(fuzzy_system_template.input_names)
    mem_params_len = num_inputs * 3 * 3  # هر ورودی ۳ تابع عضویت، هرکدام سه پارامتر
    return param_vector[mem_params_len:mem_params_len+rule_count]

# ---- مثال نحوه استفاده ----
if __name__ == "__main__":
    from fuzzy import FuzzySystem, default_membership_params, default_rule_list, default_rule_weights
    # آماده‌سازی داده فرضی و سیستم پایه
    fuzzy_system = FuzzySystem(default_membership_params, default_rule_weights, default_rule_list)
    train_X = np.random.rand(100, 3)
    train_labels = np.random.randint(0, 2, 100)
    dim = 3*3*3 + len(default_rule_list)  # تعداد کل پارامترها
    bounds = np.array([[0,2]] * (3*3*3) + [[0,2]]*len(default_rule_list))  # باید بر اساس واقعیت مسئله اصلاح شود
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
