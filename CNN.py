# ==========================================================
# LAB 12 – CKD CNN (SIMULATED USING MLP) CLASSIFICATION
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
OUTPUT_FOLDER = "lab10_cnn"
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
# SCALING
# ----------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------------------------------
# RESHAPE DATA (SIMULATING CNN INPUT)
# ----------------------------------------------------------
# CNN expects 2D/3D data → we reshape
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# ----------------------------------------------------------
# "CNN-LIKE" MODEL USING MLP
# ----------------------------------------------------------
model = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),  # deeper network (like CNN layers)
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
# CONFUSION MATRIX
# ----------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - CNN (Simulated)")

plt.savefig(os.path.join(OUTPUT_FOLDER, "confusion_matrix.png"))
plt.show()

# ----------------------------------------------------------
# LOSS CURVE
# ----------------------------------------------------------
plt.figure()
plt.plot(model.loss_curve_)
plt.title("Training Loss Curve (CNN-like)")
plt.xlabel("Iterations")
plt.ylabel("Loss")

plt.savefig(os.path.join(OUTPUT_FOLDER, "loss.png"))
plt.show()

# ----------------------------------------------------------
# END
# ----------------------------------------------------------
print(f"\n📁 Outputs saved in '{OUTPUT_FOLDER}' folder")
print("\n✅ CNN Lab Completed Successfully")

