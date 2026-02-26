"""
Advanced Clinical Decision Support Module
Author: Sharare Taheri
Description:
- Predict readmission risk
- Provide risk category
- Confidence intervals
- Fairness check
- Cost-sensitive decision recommendations
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.utils import resample

# ===============================
# 1. Load Model & Scaler
# ===============================
MODEL_PATH = "models/logistic_regression_model.pkl"
SCALER_PATH = "models/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ===============================
# 2. Risk Stratification Function
# ===============================
def stratify_risk(probability: float, thresholds=(0.3, 0.7)) -> str:
    """
    Convert probability into Low / Moderate / High risk.
    thresholds: tuple(low_threshold, high_threshold)
    """
    low, high = thresholds
    if probability < low:
        return "Low Risk"
    elif probability < high:
        return "Moderate Risk"
    else:
        return "High Risk"

# ===============================
# 3. Clinical Recommendation Logic
# ===============================
def generate_recommendation(risk_level: str) -> str:
    """
    Provide actionable recommendation based on risk level.
    """
    if risk_level == "Low Risk":
        return "Routine follow-up recommended."
    elif risk_level == "Moderate Risk":
        return "Schedule follow-up within 7 days and monitor closely."
    else:
        return "Immediate care coordination and case management recommended."

# ===============================
# 4. Confidence Interval via Bootstrapping
# ===============================
def bootstrap_confidence_interval(X, model, scaler, n_iterations=1000, alpha=0.05):
    """
    Estimate confidence interval of prediction using bootstrapping.
    """
    scaled_X = scaler.transform(X)
    probs = []
    for _ in range(n_iterations):
        sample_idx = resample(range(len(X)))
        prob = model.predict_proba(scaled_X[sample_idx])[:,1]
        probs.append(prob.mean())
    lower = np.percentile(probs, 100*alpha/2)
    upper = np.percentile(probs, 100*(1-alpha/2))
    return lower, upper

# ===============================
# 5. Fairness Check (Gender Balance)
# ===============================
def fairness_check(X, sensitive_column='gender_M'):
    """
    Simple check: compare average predicted probability between groups
    """
    scaled_X = scaler.transform(X)
    probs = model.predict_proba(scaled_X)[:,1]
    group0 = probs[X[sensitive_column]==0]
    group1 = probs[X[sensitive_column]==1]
    return group0.mean(), group1.mean()

# ===============================
# 6. Predict New Patient(s) Function
# ===============================
def predict_patient_risk_advanced(patient_data: pd.DataFrame, thresholds=(0.3,0.7)):
    # Scale
    scaled_data = scaler.transform(patient_data)

    # Predict probability
    probability = model.predict_proba(scaled_data)[:,1][0]

    # Risk Category
    risk_level = stratify_risk(probability, thresholds)

    # Recommendation
    recommendation = generate_recommendation(risk_level)

    # Confidence Interval
    ci_lower, ci_upper = bootstrap_confidence_interval(patient_data, model, scaler)

    # Fairness Check
    gender0_mean, gender1_mean = fairness_check(patient_data)

    print("=== Advanced Clinical Decision Support Output ===")
    print(f"Predicted Readmission Probability: {probability:.2f}")
    print(f"Risk Category: {risk_level}")
    print(f"Recommended Action: {recommendation}")
    print(f"Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"Average Probability by Gender -> M=1: {gender1_mean:.2f}, F=0: {gender0_mean:.2f}")

    return probability, risk_level, recommendation, (ci_lower, ci_upper)

# ===============================
# 7. Example Usage
# ===============================
if __name__ == "__main__":

    new_patient = pd.DataFrame({
        "age": [65],
        "lab_result_1": [145],
        "lab_result_2": [8.2],
        "gender_M": [1]  # must match encoded features
    })

    predict_patient_risk_advanced(new_patient)
