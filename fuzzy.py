import numpy as np
from config import FUZZY_MEMBERSHIP_TYPE, FUZZY_INPUTS, FUZZY_OUTPUT, FUZZY_RULES_COUNT

# --- توابع عضویت مثلثی ---
def triangular_membership(x, a, b, c):
    return np.maximum(np.minimum((x - a) / (b - a + 1e-8), (c - x) / (c - b + 1e-8)), 0)

# --- سیستم فازی پایه ---
class FuzzySystem:
    def __init__(self, membership_params, rule_weights, rule_list=None):
        """
        membership_params: dict مثل {"residual": [(a1,b1,c1), (a2,b2,c2), (a3,b3,c3)], ...}
        rule_weights: وزن هر قاعده (np.array)
        rule_list: لیست قواعد [("High", "Low", "Medium") ...]
        """
        self.membership_params = membership_params
        self.rule_weights = rule_weights
        self.rule_list = rule_list if rule_list is not None else []
        self.input_names = FUZZY_INPUTS
        self.output_name = FUZZY_OUTPUT

    def set_params(self, membership_params=None, rule_weights=None):
        if membership_params is not None:
            self.membership_params = membership_params
        if rule_weights is not None:
            self.rule_weights = rule_weights

    def compute_memberships(self, input_dict):
        """
        input_dict: {"residual": val, "upper_diff": val, "lower_diff": val}
        خروجی: dict {"residual": [μ_low, μ_medium, μ_high], ...}
        """
        memberships = {}
        for var in self.input_names:
            params = self.membership_params[var]
            vals = [triangular_membership(input_dict[var], *p) for p in params]
            memberships[var] = vals
        return memberships

    def infer(self, input_dict):
        """
        خروجی: anomaly_risk در بازه [0,1]
        """
        memberships = self.compute_memberships(input_dict)
        # یک مثال ساده: فرض کنیم rule_list به شکل [("High","Low","Medium"), ...] و output سطح risk (مثلاً [0, 0.5, 1])
        output_levels = [0.0, 0.5, 1.0]  # [Low, Medium, High]
        rule_outputs = []
        for i, rule in enumerate(self.rule_list):
            # مقدار عضویت هر ورودی بر اساس سطح متناظر قاعده
            μs = [memberships[var][level_idx(label)] for var, label in zip(self.input_names, rule)]
            rule_strength = np.prod(μs)
            rule_outputs.append(rule_strength * output_levels[level_idx(rule[-1])] * self.rule_weights[i])
        # جمع وزنی قواعد
        if sum(rule_outputs) > 0:
            risk = sum(rule_outputs) / (sum([abs(w) for w in self.rule_weights]) + 1e-8)
        else:
            risk = 0.0
        return np.clip(risk, 0, 1)

def level_idx(level):
    return {"Low": 0, "Medium": 1, "High": 2}[level]

# --- مقداردهی نمونه ---
default_membership_params = {
    "residual":   [(0, 0, 1), (0, 1, 2), (1, 2, 2.5)],
    "upper_diff": [(0, 0, 0.5), (0, 0.5, 1), (0.5, 1, 1.5)],
    "lower_diff": [(0, 0, 0.5), (0, 0.5, 1), (0.5, 1, 1.5)],
}
default_rule_list = [
    ("High", "Low", "Low"),     # Rule 1
    ("Medium", "Medium", "Medium"), # Rule 2
    ("Low", "High", "Low"),     # Rule 3
    ("High", "High", "High"),   # Rule 4
    ("Low", "Low", "Low"),      # Rule 5
    ("Medium", "Low", "High")   # Rule 6
]
default_rule_weights = np.ones(len(default_rule_list))

# ایجاد شی فازی نمونه (قابل تغییر توسط PSO)
fuzzy_system = FuzzySystem(
    membership_params=default_membership_params,
    rule_weights=default_rule_weights,
    rule_list=default_rule_list
)

# --- API برای main/detect_anomalies ---
def evaluate_fuzzy_anomaly(residual, upper_diff, lower_diff, fuzzy_system=fuzzy_system):
    input_dict = {"residual": residual, "upper_diff": upper_diff, "lower_diff": lower_diff}
    return fuzzy_system.infer(input_dict)
