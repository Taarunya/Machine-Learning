# ==========================================================
# LAB 10 – CKD CLUSTERING (K-MEANS & HIERARCHICAL)
# ==========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

# ----------------------------------------------------------
# CREATE OUTPUT FOLDER
# ----------------------------------------------------------
OUTPUT_FOLDER = "lab08"
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

# Keep numeric only (no target needed)
df_num = df.select_dtypes(include=["int64", "float64"])

print("Dataset shape:", df_num.shape)

# ----------------------------------------------------------
# FEATURE SCALING
# ----------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_num)

# ==========================================================
# 1. K-MEANS CLUSTERING
# ==========================================================
print("\n===== K-MEANS CLUSTERING =====")

# Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.savefig(os.path.join(OUTPUT_FOLDER, "elbow.png"))
plt.show()

# Final model (choose k=3 or based on elbow)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df_num["Cluster"] = clusters

# Scatter plot (use 2 features for visualization)
plt.figure()
plt.scatter(df_num.iloc[:, 0], df_num.iloc[:, 1], c=clusters, cmap='viridis')
plt.title("K-Means Clustering")
plt.xlabel(df_num.columns[0])
plt.ylabel(df_num.columns[1])
plt.savefig(os.path.join(OUTPUT_FOLDER, "kmeans_clusters.png"))
plt.show()

# ==========================================================
# 2. HIERARCHICAL CLUSTERING
# ==========================================================
print("\n===== HIERARCHICAL CLUSTERING =====")

linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(linked)
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.savefig(os.path.join(OUTPUT_FOLDER, "dendrogram.png"))
plt.show()

# ----------------------------------------------------------
# END
# ----------------------------------------------------------
print(f"\n📁 Outputs saved in '{OUTPUT_FOLDER}' folder")
print("\n✅ Lab 08 Completed Successfully")


