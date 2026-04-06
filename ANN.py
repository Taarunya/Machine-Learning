# ==========================================================
# LAB 11 – CKD ANN (MLP) CLASSIFICATION WITH SAVED PLOTS
# ==========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ----------------------------------------------------------
# CREATE OUTPUT FOLDER
# ----------------------------------------------------------
OUTPUT_FOLDER = "lab09_ann"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ----------------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------------
df = pd.read_csv("kidney_disease.csv")

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Info ---")
print(df.info())

# ----------------------------------------------------------
# DATA CLEANING
# ----------------------------------------------------------
df.replace("?", np.nan, inplace=True)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")

df.dropna(inplace=True)

print("\nClean dataset shape:", df.shape)

# ----------------------------------------------------------
# CONVERT TARGET → CLASSIFICATION
# ----------------------------------------------------------
df["classification"] = df["classification"].map({
    "ckd": 1,
    "notckd": 0
})

# Keep only numeric columns
df_num = df.select_dtypes(include=["int64", "float64"])

# ----------------------------------------------------------
# FEATURE & TARGET SPLIT
# ----------------------------------------------------------
X = df_num.drop("classification", axis=1)
y = df_num["classification"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# ----------------------------------------------------------
# TRAIN TEST SPLIT
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ----------------------------------------------------------
# FEATURE SCALING
# ----------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------------------------------
# BUILD ANN MODEL (MLP)
# ----------------------------------------------------------
model = MLPClassifier(
    hidden_layer_sizes=(32, 16),   # 2 hidden layers
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

# ----------------------------------------------------------
# TRAIN MODEL
# ----------------------------------------------------------
model.fit(X_train, y_train)

# ----------------------------------------------------------
# PREDICTIONS
# ----------------------------------------------------------
y_pred = model.predict(X_test)

# ----------------------------------------------------------
# MODEL EVALUATION
# ----------------------------------------------------------
print("\n========== MODEL PERFORMANCE ==========")

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------------------------------------
# CONFUSION MATRIX (SAVE)
# ----------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - ANN (MLP)")

plt.savefig(os.path.join(OUTPUT_FOLDER, "confusion_matrix.png"))
plt.show()

# ----------------------------------------------------------
# LOSS CURVE (IMPORTANT)
# ----------------------------------------------------------
plt.figure()
plt.plot(model.loss_curve_)
plt.title("Training Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")

plt.savefig(os.path.join(OUTPUT_FOLDER, "loss.png"))
plt.show()

# ----------------------------------------------------------
# END
# ----------------------------------------------------------
print(f"\n📁 Outputs saved inside '{OUTPUT_FOLDER}' folder")
print("\n✅ ANN Lab Completed Successfully")

