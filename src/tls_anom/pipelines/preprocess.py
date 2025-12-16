import pandas as pd
import numpy as np
import joblib
from tls_anom.utils.io import write_csv
from sklearn.preprocessing import StandardScaler


def run(ctx, feat_csv: str, out_scaled_csv: str):
    logger = ctx.logger
    logger.info(f"[preprocess] loading {feat_csv}")

    df = pd.read_csv(feat_csv)

    # Separate label
    if "label" in df.columns:
        y = df["label"].astype(int)
        X = df.drop(columns=["label"])
    else:
        y = None
        X = df

    # Detect numeric cols (non-string)
    numeric_cols = [c for c in X.columns if X[c].dtype != object]

    # Fill NA
    X = X.fillna(0)

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[numeric_cols])

    # Save scaler
    model_dir = ctx.cfg["paths"]["models_dir"]
    joblib.dump(scaler, f"{model_dir}/scaler.joblib")
    joblib.dump(numeric_cols, f"{model_dir}/feature_list.joblib")

    # Output scaled CSV
    out = pd.DataFrame(X_scaled, columns=numeric_cols)
    if y is not None:
        out["label"] = y.values

    write_csv(out, out_scaled_csv)
    logger.info(f"[preprocess] saved scaled file -> {out_scaled_csv}")
