import pandas as pd
from tls_anom.models import iforest
from tls_anom.utils.io import write_csv

def run(ctx, X_csv: str, model_path: str, out_pred_csv: str):
    logger = ctx.logger
    model = iforest.load(model_path)
    df = pd.read_csv(X_csv)
    preds = model.predict(df.values)
    df['pred'] = preds  # 1=inlier, -1=outlier
    write_csv(df, out_pred_csv)
    logger.info(f"[eval] Predictions -> {out_pred_csv}")
