import pandas as pd
import joblib
from sklearn.metrics import classification_report


def run(
    ctx,
    feature_csv: str,
    model_path: str,
    scaler_path: str,
):
    """
    Predict anomaly scores and PRINT results to console.
    """

    logger = ctx.logger

    logger.info(f"[predict] loading model from {model_path}")
    model = joblib.load(model_path)

    logger.info(f"[predict] loading scaler from {scaler_path}")
    scaler = joblib.load(scaler_path)

    logger.info(f"[predict] loading features from {feature_csv}")
    df = pd.read_csv(feature_csv)

    # -----------------------------
    # Separate label if exists
    # -----------------------------
    y_true = None
    if "label" in df.columns:
        y_true = df["label"].astype(int)
        X = df.drop(columns=["label"])
    else:
        X = df

    # safety
    X = X.fillna(0)

    # scale
    X_scaled = scaler.transform(X)

    logger.info("[predict] running inference...")

    # decision_function: higher = more normal
    normal_score = model.decision_function(X_scaled)

    # anomaly score: higher = more anomalous
    anomaly_score = -normal_score

    # predict: 1 = normal, -1 = anomaly
    raw_preds = model.predict(X_scaled)
    preds = (raw_preds == -1).astype(int)   # 1 = anomaly

    # -----------------------------
    # SUMMARY
    # -----------------------------
    total = len(preds)
    anomaly_cnt = int(preds.sum())
    anomaly_rate = anomaly_cnt / total if total > 0 else 0.0

    logger.info(
        f"[predict] SUMMARY | total={total}, "
        f"anomaly={anomaly_cnt}, "
        f"rate={anomaly_rate:.4%}"
    )

    # -----------------------------
    # OPTIONAL: classification report
    # -----------------------------
    if y_true is not None:
        logger.info(
            "\n" + classification_report(
                y_true, preds, digits=4, zero_division=0
            )
        )

    # -----------------------------
    # PRINT TOP ANOMALIES
    # -----------------------------
    df_dbg = df.copy()
    df_dbg["anomaly_score"] = anomaly_score
    df_dbg["pred"] = preds

    logger.info("[predict] TOP 10 anomalies:")
    print(
        df_dbg.sort_values("anomaly_score", ascending=False)
        .head(10)[
            [
                "ja3_reuse_cnt",
                "iat_mean",
                "iat_std",
                "iat_cv",
                "iat_cnt",
                "anomaly_score",
            ]
        ]
    )

    logger.info("[predict] DONE.")
    return df_dbg
