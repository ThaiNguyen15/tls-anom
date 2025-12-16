import pandas as pd
import joblib
from sklearn.metrics import classification_report


def run(ctx, scaled_csv: str, model_path: str, output_path: str = None):
    """
    Predict anomaly scores on PREPROCESSED (scaled) dataset.
    Assumes model was trained on the same feature space.
    """
    logger = ctx.logger

    logger.info(f"[predict] loading model from {model_path}")
    model = joblib.load(model_path)

    logger.info(f"[predict] loading data from {scaled_csv}")
    df = pd.read_csv(scaled_csv)

    # -----------------------------
    # Separate label if exists
    # -----------------------------
    y_true = None
    if "label" in df.columns:
        y_true = df["label"].astype(int)
        X = df.drop(columns=["label"])
    else:
        X = df

    logger.info("[predict] running inference...")

    # IsolationForest:
    # decision_function → higher = more normal
    # predict → 1 = normal, -1 = anomaly
    scores = model.decision_function(X)
    raw_preds = model.predict(X)

    # Convert to anomaly label:
    # 1 = anomaly, 0 = normal
    preds = (raw_preds == -1).astype(int)

    # -----------------------------
    # Attach results
    # -----------------------------
    df_out = df.copy()
    df_out["anomaly_score"] = scores
    df_out["pred"] = preds

    # -----------------------------
    # Optional evaluation
    # -----------------------------
    if y_true is not None:
        logger.info(
            "\n" + classification_report(
                y_true, preds, digits=4, zero_division=0
            )
        )

    # -----------------------------
    # Save
    # -----------------------------
    if output_path:
        df_out.to_csv(output_path, index=False)
        logger.info(f"[predict] result saved to {output_path}")

    logger.info("[predict] DONE.")
    return df_out
