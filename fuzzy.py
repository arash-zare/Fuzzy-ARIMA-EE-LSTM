import numpy as np
from config import FUZZY_INPUTS, FUZZY_LEVELS

class FuzzySystem:
    def __init__(self, membership_params, rule_weights, rule_list, input_names=FUZZY_INPUTS, output_levels=FUZZY_LEVELS):
        self.input_names = input_names
        self.membership_params = membership_params  # dict: input -> list of tuples (a,b,c) for each level
        self.rule_weights = rule_weights            # list: وزن هر قاعده
        self.rule_list = rule_list                  # list of dict: {'if': ..., 'then': ...}
        self.output_levels = output_levels          # [0, 0.5, 1.0] ← mapping به خروجی crisp

    def triangular(self, x, a, b, c):
        # تابع عضویت مثلثی
        if a == b == c:
            return 1.0 if x == a else 0.0
        return max(min((x-a)/(b-a+1e-8), (c-x)/(c-b+1e-8)), 0.0)

    def fuzzify(self, x, varname):
        # مقداردهی عضویت برای یک متغیر
        params = self.membership_params[varname]
        memberships = [self.triangular(x, *abc) for abc in params]
        return memberships

    def infer(self, input_dict):
        # مرحله فازی و rule evaluation
        rule_strengths = []
        for i, rule in enumerate(self.rule_list):
            cond_strength = 1.0
            for var in self.input_names:
                val = input_dict[var]
                level = rule['if'][var]
                cond_strength *= self.fuzzify(val, var)[level]
            rule_strengths.append(cond_strength * self.rule_weights[i])
        if sum(rule_strengths) == 0:
            # هیچ قاعده‌ای فعال نشد، خروجی پیش‌فرض
            return np.mean(self.output_levels)
        # defuzzification: weighted average
        risk = sum(s * lvl for s, lvl in zip(rule_strengths, self.output_levels)) / (sum(rule_strengths)+1e-8)
        return risk

    def set_params(self, membership_params=None, rule_weights=None):
        if membership_params is not None:
            self.membership_params = membership_params
        if rule_weights is not None:
            self.rule_weights = rule_weights

# ---------- پارامترها و قواعد پیش‌فرض ----------
default_membership_params = {
    "residual":   [(0, 0, 1), (0, 1, 2), (1, 2, 2)],
    "upper_diff": [(0, 0, 1), (0, 1, 2), (1, 2, 2)],
    "lower_diff": [(0, 0, 1), (0, 1, 2), (1, 2, 2)],
}
# به ترتیب: Low, Med, High ← index=0,1,2

default_rule_list = [
    {"if": {"residual": 2, "upper_diff": 0, "lower_diff": 1}, "then": 2},  # High risk
    {"if": {"residual": 1, "upper_diff": 1, "lower_diff": 1}, "then": 1},  # Med
    {"if": {"residual": 0, "upper_diff": 2, "lower_diff": 0}, "then": 1},
    {"if": {"residual": 2, "upper_diff": 2, "lower_diff": 2}, "then": 2},
    {"if": {"residual": 1, "upper_diff": 1, "lower_diff": 2}, "then": 1},
    {"if": {"residual": 2, "upper_diff": 1, "lower_diff": 1}, "then": 2},
]
default_rule_weights = [1, 1, 1, 1, 1, 1]

# ---------- Fuzzy Evaluation برای main ----------
def evaluate_fuzzy_anomaly(pred, actual, upper, lower, fuzzy_system=None):
    # pred, actual, upper, lower: float
    if fuzzy_system is None:
        fuzzy_system = FuzzySystem(default_membership_params, default_rule_weights, default_rule_list)
    inputs = {
        "residual": actual - pred,
        "upper_diff": actual - upper,
        "lower_diff": actual - lower,
    }
    risk = fuzzy_system.infer(inputs)
    return risk

