"""
Clinical Readmission Prediction
Author: Sharare Taheri
Description: Predict 30-day hospital readmission risk using structured clinical data.
Models: Logistic Regression, Random Forest
Metrics: AUC, Precision, Recall, F1-score
"""

# ===============================
# 1. Import Required Libraries
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

# ===============================
# 2. Load and Explore Dataset
# ===============================
# Replace 'clinical_data.csv' with your dataset path
df = pd.read_csv('clinical_data.csv')

# Quick overview
print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())  # Check missing values

# ===============================
# 3. Data Preprocessing
# ===============================
# Example preprocessing steps (adjust according to your dataset)

# Fill missing numerical values with median
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill missing categorical values with mode
cat_cols = df.select_dtypes(include='object').columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Feature-target split
X = df.drop('readmitted_30days', axis=1)  # replace with your target column
y = df['readmitted_30days']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 4. Model Training
# ===============================

# ---- Logistic Regression ----
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)

# ---- Random Forest (baseline) ----
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ===============================
# 5. Model Evaluation
# ===============================
def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n=== {model_name} Performance ===")
    print(f"AUC: {auc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Evaluate Logistic Regression
evaluate_model(logreg, X_test_scaled, y_test, "Logistic Regression")

# Evaluate Random Forest
evaluate_model(rf, X_test, y_test, "Random Forest")

# ===============================
# 6. Feature Importance Analysis
# ===============================

# Logistic Regression coefficients
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': logreg.coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\nTop 10 Important Features (Logistic Regression):")
print(coef_df.head(10))

# Random Forest feature importance
rf_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Important Features (Random Forest):")
print(rf_importances.head(10))

# Visualize feature importance
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=rf_importances.head(15))
plt.title('Top 15 Feature Importances (Random Forest)')
plt.tight_layout()
plt.show()

# ===============================
# 7. Save Models for Reuse
# ===============================
import joblib

joblib.dump(logreg, 'logistic_regression_model.pkl')
joblib.dump(rf, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Models and scaler saved successfully!")

# ===============================
# End of Project
# ===============================
