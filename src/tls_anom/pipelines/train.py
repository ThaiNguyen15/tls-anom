import os
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


# ==========================================================
# Stage 1: Isolation Forest
# ==========================================================

def train_iforest(
    feature_csv: str,
    model_path: str,
    scaler_path: str,
    contamination: float,
):
    print(f"[train][iforest] loading {feature_csv}")
    X = pd.read_csv(feature_csv).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_scaled)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"[train][iforest] saved model -> {model_path}")
    print(f"[train][iforest] saved scaler -> {scaler_path}")

    return model, scaler


def compute_iforest_score(df, model, scaler):
    X_scaled = scaler.transform(df.fillna(0))
    return -model.decision_function(X_scaled)


# ==========================================================
# Stage 2: LightGBM
# ==========================================================

def train_lgbm(df: pd.DataFrame, model_path: str):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import numpy as np
    import lightgbm as lgb
    import joblib

    y = df["label"].astype(int)
    X = df.drop(columns=["label"])

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight={0: 1, 1: 2000},  # quan trọng cho imbalance
        random_state=42,
    )

    model.fit(X_train, y_train)

    # ✅ SAVE MODEL LUÔN – KHÔNG PHỤ THUỘC METRIC
    joblib.dump(model, model_path)
    print(f"[train][lgbm] model saved -> {model_path}")

    # -----------------------------
    # OPTIONAL EVALUATION (SAFE)
    # -----------------------------
    print("\n[train][lgbm] validation report:")
    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred, digits=4, zero_division=0))

    # AUC chỉ tính nếu đủ 2 class
    if len(np.unique(y_val)) > 1:
        from sklearn.metrics import roc_auc_score
        y_prob = model.predict_proba(X_val)[:, 1]
        print("AUC:", roc_auc_score(y_val, y_prob))
    else:
        print("[train][lgbm] AUC skipped (only one class in validation)")



# ==========================================================
# Main entry (CLI calls this)
# ==========================================================

def run(ctx, feature_csv: str, lgbm_csv: str, model_path: str, scaler_path: str):
    cfg = ctx.cfg
    model_kind = cfg["model"]["kind"]
    contamination = cfg["model"].get("contamination", 0.01)

    models_dir = cfg["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)

    # ==================================================
    # IF ONLY
    # ==================================================
    if model_kind == "iforest":
        train_iforest(
            feature_csv=feature_csv,   # normal.features.csv
            model_path=model_path,
            scaler_path=scaler_path,
            contamination=contamination,
        )
        return

    # ==================================================
    # IF → LGBM (requires labeled dataset)
    # ==================================================
    if model_kind == "iforest_lgbm":
        print(f"[train][iforest_lgbm] loading labeled data from {lgbm_csv}")
        df = pd.read_csv(lgbm_csv).fillna(0)

        if "label" not in df.columns:
            raise RuntimeError(
                "iforest_lgbm requires labeled dataset "
                "(normal + c2 with label column)"
            )

        iforest, scaler = train_iforest(
            feature_csv=feature_csv,
            model_path=model_path.replace(".joblib", ".iforest.joblib"),
            scaler_path=scaler_path,
            contamination=contamination,
        )

        # ---- compute anomaly score ----
        df["anomaly_score"] = compute_iforest_score(
            df.drop(columns=["label"]), iforest, scaler
        )

        # ---- train LGBM ----
        train_lgbm(
            df=df,
            model_path=model_path.replace(".joblib", ".lgbm.joblib"),
        )
        return

