import json
from pathlib import Path
from sklearn import metrics as M


def save_metrics(path_dir, name, y_true, scores):
    p = Path(path_dir) / f"{name}.json"
    out = {}
    if y_true is not None:
        try:
            out["roc_auc"] = M.roc_auc_score(y_true, scores)
        except Exception:
            pass
        try:
            prec, rec, _ = M.precision_recall_curve(y_true, scores)
            out["pr_auc"] = M.auc(rec, prec)
        except Exception:
            pass
        # F1@K = nhị phân hoá theo ngưỡng top-k
        k = min(len(scores), max(1, int(0.01 * len(scores))))
        idx = scores.argsort()[::-1][:k]
        import numpy as np

        y_pred = np.zeros_like(scores, dtype=int)
        y_pred[idx] = 1
        out["f1_at_k"] = M.f1_score(y_true, y_pred)
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    return out
