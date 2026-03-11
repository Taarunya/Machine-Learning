# ==========================================================
# CHRONIC KIDNEY DISEASE CLASSIFICATION
# SVM + DECISION TREE
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ----------------------------------------------------------
# 1. LOAD DATASET
# ----------------------------------------------------------

df = pd.read_csv("kidney_disease.csv")

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Info ---")
print(df.info())


# ----------------------------------------------------------
# 2. DATA CLEANING
# ----------------------------------------------------------

# Replace ? with NaN
df.replace("?", np.nan, inplace=True)

# Convert numeric columns
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")

# Drop missing values
df.dropna(inplace=True)

print("\nClean dataset shape:", df.shape)


# ----------------------------------------------------------
# 3. CONVERT TARGET TO NUMERIC
# ----------------------------------------------------------

df["classification"] = df["classification"].map({
    "ckd": 1,
    "notckd": 0
})


# ----------------------------------------------------------
# 4. KEEP ONLY NUMERIC DATA
# ----------------------------------------------------------

df_num = df.select_dtypes(include=["int64", "float64"])


# ----------------------------------------------------------
# 5. CORRELATION HEATMAP
# ----------------------------------------------------------

plt.figure(figsize=(10,7))
sns.heatmap(df_num.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# ----------------------------------------------------------
# 6. FEATURE AND TARGET
# ----------------------------------------------------------

X = df_num.drop("classification", axis=1)
y = df_num["classification"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)


# ----------------------------------------------------------
# 7. TRAIN TEST SPLIT
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# ----------------------------------------------------------
# 8. FEATURE SCALING (IMPORTANT FOR SVM)
# ----------------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==========================================================
# 9. SUPPORT VECTOR MACHINE MODEL
# ==========================================================

print("\n==============================")
print("SVM MODEL")
print("==============================")

svm_model = SVC(kernel='linear')

svm_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, svm_predictions))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, svm_predictions)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, svm_predictions))


# Plot Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ==========================================================
# 10. DECISION TREE MODEL
# ==========================================================

print("\n==============================")
print("DECISION TREE MODEL")
print("==============================")

tree_model = DecisionTreeClassifier(random_state=42)

tree_model.fit(X_train, y_train)

tree_predictions = tree_model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, tree_predictions))

print("\nConfusion Matrix:")
cm2 = confusion_matrix(y_test, tree_predictions)
print(cm2)

print("\nClassification Report:")
print(classification_report(y_test, tree_predictions))


# Plot Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens")
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


print("\n✅ Program Finished Successfully")
