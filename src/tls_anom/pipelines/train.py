import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

def train_isolation_forest(normal_csv: str, model_path: str, contamination: float = 0.01):
    """
    Train IsolationForest model using ONLY normal traffic.
    
    Args:
        normal_csv (str): path to normal.scaled.csv
        model_path (str): output model file (joblib)
        contamination (float): expected anomaly percentage (~0.01 = 1%)

    Returns:
        model (IsolationForest): trained model
    """

    print(f"[+] Loading NORMAL dataset: {normal_csv}")
    df = pd.read_csv(normal_csv)

    # Remove label column if exists
    df = df.drop(columns=["label"], errors="ignore")

    print(f"[+] Training IsolationForest (contamination={contamination})")
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    model.fit(df)

    print(f"[+] Saving model → {model_path}")
    joblib.dump(model, model_path)

    print("[+] DONE: Model trained successfully.")
    return model

def run(ctx, scaled_csv: str, model_path: str):
    logger = ctx.logger
    logger.info(f"[train] loading {scaled_csv}")

    df = pd.read_csv(scaled_csv)

    # label nếu có (supervised)
    if "label" in df.columns:
        y = df["label"].astype(int)
        X = df.drop(columns=["label"])
    else:
        y = None
        X = df

    model_kind = ctx.cfg["model"]["kind"]


    # ==========================================================
    # 1) TRAIN ISOLATION FOREST (unsupervised)
    # ==========================================================
    if model_kind == "iforest":
        model = train_isolation_forest(
            normal_csv=scaled_csv, 
            model_path=model_path,
            contamination=ctx.cfg["model"].get("contamination", 0.01)
        )
        return

    # ==========================================================
    # 2) TRAIN LIGHTGBM (supervised)
    # ==========================================================
    if model_kind == "lightgbm":
        if y is None:
            raise RuntimeError("Supervised model requires labels.")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        params = ctx.cfg["model"].get("params", {})

        gbm = lgb.LGBMClassifier(
            n_estimators=params.get("n_estimators", 500),
            learning_rate=params.get("learning_rate", 0.05),
            num_leaves=params.get("num_leaves", 64),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            objective="binary",
            boosting_type="gbdt",
            random_state=42
        )

        logger.info("[train] training LightGBM...")

        gbm.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[
                early_stopping(stopping_rounds=50),
                log_evaluation(period=20),
            ]
        )

        joblib.dump(gbm, model_path)
        logger.info(f"[train] LightGBM saved to {model_path}")
        return

    raise RuntimeError(f"Unknown model kind: {model_kind}")
