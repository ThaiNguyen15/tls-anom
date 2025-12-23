import pandas as pd
import joblib
import yaml

# -----------------------------
# Load config
# -----------------------------
cfg = yaml.safe_load(open("config/default.yaml"))

IFOREST_PATH = "data/models/iforest_lgbm.iforest.iforest.joblib"
SCALER_PATH = "data/models/iforest_lgbm.scaler.joblib"
LGBM_PATH = "data/models/iforest_lgbm.iforest.lgbm.joblib"

iforest = joblib.load(IFOREST_PATH)
scaler = joblib.load(SCALER_PATH)
lgbm = joblib.load(LGBM_PATH)

# -----------------------------
# Load features
# -----------------------------
df = pd.read_csv(cfg["input"]["feature_csv"]).fillna(0)

X = df.drop(columns=["label"], errors="ignore")

# -----------------------------
# Stage 1: Isolation Forest
# -----------------------------
X_scaled = scaler.transform(X)
anomaly_score = -iforest.decision_function(X_scaled)

# 👉 GÁN NGAY VÀO df
df["anomaly_score"] = anomaly_score

# -----------------------------
# Stage 2: LightGBM
# -----------------------------
X2 = X.copy()
X2["anomaly_score"] = anomaly_score

c2_score = lgbm.predict_proba(X2)[:, 1]

# 👉 GÁN NGAY VÀO df
df["c2_score"] = c2_score

# -----------------------------
# SOC-style alerting
# -----------------------------
alerts = df[df["c2_score"] >= cfg["alert"]["threshold"]]

print("\n=== 🚨 TLS C2 ALERTS ===")
print(
    alerts.sort_values("c2_score", ascending=False)
    .head(cfg["alert"]["top_k"])[
        [
            "c2_score",
            "anomaly_score",
            "iat_cv",
            "iat_cnt",
            "ja3_reuse_cnt",
        ]
    ]
)

print(f"\nTotal flows: {len(df)}")
print(f"Alerts raised: {len(alerts)}")