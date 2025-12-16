import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import numpy as np

NORMAL_CSV = "data/features/normal.scaled.csv"
BOTNET_CSV = "data/features/botnet.scaled.csv"

print("[+] Loading datasets...")
df_normal = pd.read_csv(NORMAL_CSV)
df_botnet = pd.read_csv(BOTNET_CSV)

# Xóa cột label nếu có
df_normal = df_normal.drop(columns=["label"], errors="ignore")
df_botnet = df_botnet.drop(columns=["label"], errors="ignore")

# Train chỉ trên NORMAL
print("[+] Training IsolationForest on NORMAL traffic...")
model = IsolationForest(
    n_estimators=300,
    contamination=0.01,      # 1% anomalous expected
    random_state=42,
    n_jobs=-1
)

model.fit(df_normal)

# Predict
print("[+] Testing on NORMAL + BOTNET...")
df_all = pd.concat([df_normal, df_botnet], axis=0)
y_true = np.array([0]*len(df_normal) + [1]*len(df_botnet))   # 0=normal, 1=botnet

scores = model.decision_function(df_all)  # anomaly score
pred_raw = model.predict(df_all)          # 1=normal, -1=anomaly

# Convert to 0/1
y_pred = (pred_raw == -1).astype(int)

# Show metrics
print("\n===== METRICS =====")
print(classification_report(y_true, y_pred, digits=4))

# Show examples
print("\n===== SAMPLE OUTPUT =====")
sample = df_all.copy()
sample["score"] = scores
sample["pred"] = y_pred
print(sample.head(10))

# Print score ranges
print("\nScore stats:")
print(sample["score"].describe())
