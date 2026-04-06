# ==========================================================
# LAB 07 – CKD ENSEMBLE MODELS (Bagging, Boosting, Stacking)
# ==========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensemble models
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier

# Base models for stacking
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ----------------------------------------------------------
# CREATE OUTPUT FOLDER
# ----------------------------------------------------------
OUTPUT_FOLDER = "lab07"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ----------------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------------
df = pd.read_csv("kidney_disease.csv")

# ----------------------------------------------------------
# DATA CLEANING
# ----------------------------------------------------------
df.replace("?", np.nan, inplace=True)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")

df.dropna(inplace=True)

# ----------------------------------------------------------
# TARGET CONVERSION
# ----------------------------------------------------------
df["classification"] = df["classification"].map({
    "ckd": 1,
    "notckd": 0
})

df_num = df.select_dtypes(include=["int64", "float64"])

# ----------------------------------------------------------
# FEATURE & TARGET
# ----------------------------------------------------------
X = df_num.drop("classification", axis=1)
y = df_num["classification"]

# ----------------------------------------------------------
# TRAIN TEST SPLIT
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ----------------------------------------------------------
# SCALING (for SVM + stacking)
# ----------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================================
# 1. BAGGING → RANDOM FOREST
# ==========================================================
print("\n===== RANDOM FOREST =====")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

cm = confusion_matrix(y_test, rf_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.savefig(os.path.join(OUTPUT_FOLDER, "rf_cm.png"))
plt.show()

# ==========================================================
# 2. BOOSTING → ADABOOST
# ==========================================================
print("\n===== ADABOOST =====")

ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_model.fit(X_train, y_train)

ada_pred = ada_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, ada_pred))
print(classification_report(y_test, ada_pred))

cm = confusion_matrix(y_test, ada_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("AdaBoost Confusion Matrix")
plt.savefig(os.path.join(OUTPUT_FOLDER, "ada_cm.png"))
plt.show()

# ==========================================================
# 3. STACKING
# ==========================================================
print("\n===== STACKING CLASSIFIER =====")

estimators = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('dt', DecisionTreeClassifier(max_depth=5)),
    ('svm', SVC(probability=True))
]

stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

stack_model.fit(X_train_scaled, y_train)

stack_pred = stack_model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, stack_pred))
print(classification_report(y_test, stack_pred))

cm = confusion_matrix(y_test, stack_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
plt.title("Stacking Confusion Matrix")
plt.savefig(os.path.join(OUTPUT_FOLDER, "stack_cm.png"))
plt.show()

# ----------------------------------------------------------
# END
# ----------------------------------------------------------
print(f"\n📁 Outputs saved in '{OUTPUT_FOLDER}' folder")
print("\n✅ Lab 07 Completed Successfully")

