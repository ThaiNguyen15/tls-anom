import pandas as pd
import numpy as np
import math
from collections import Counter
from tls_anom.utils.io import write_csv
import pandas as pd
import joblib
import os

def fit_frequency(df):
    return {
        "ja3_freq": df["ja3"].value_counts().to_dict(),
        "sni_freq": df["sni"].value_counts().to_dict(),
    }

def apply_frequency(df, freq_model):
    df["ja3_freq"] = df["ja3"].apply(
        lambda x: freq_model["ja3_freq"].get(x, 0)
    )
    df["sni_freq"] = df["sni"].apply(
        lambda x: freq_model["sni_freq"].get(x, 0)
    )
    return df

# ================================
# Utility functions
# ================================

def hash_str(s, bits=18):
    """Hash string to fixed integer space."""
    mod = 2 ** bits
    if pd.isna(s):
        return 0
    return hash(str(s)) % mod


def entropy(s: str) -> float:
    """Shannon entropy of a string (SNI domain)."""
    if not isinstance(s, str) or len(s) == 0:
        return 0.0
    freq = Counter(s)
    total = len(s)
    return -sum((c/total) * math.log2(c/total) for c in freq.values())


def rarity(series: pd.Series) -> pd.Series:
    """Rarity score: 1 / frequency."""
    counts = series.value_counts()
    return series.map(lambda x: 1.0 / counts[x] if x in counts else 0.0)

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Beaconing features per (src_ip, ja3)
    """

    required_cols = {"ts_ssl", "id.orig_h_ssl", "ja3"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"[featurize] missing columns: {missing}")

    df = df.sort_values(["id.orig_h_ssl", "ja3", "ts_ssl"])

    df["iat"] = df.groupby(["id.orig_h_ssl", "ja3"])["ts_ssl"].diff()
    df["iat"] = df["iat"].fillna(0)

    agg = (
        df.groupby(["id.orig_h_ssl", "ja3"])["iat"]
        .agg(
            iat_mean="mean",
            iat_std="std",
            iat_cnt="count"
        )
        .reset_index()
    )

    agg["iat_cv"] = agg["iat_std"] / (agg["iat_mean"] + 1e-6)

    return df.merge(agg, on=["id.orig_h_ssl", "ja3"], how="left")

# ================================
# Main featurizer
# ================================

def run(
    ctx,
    csv_in: str,
    out_features_csv: str,
    mode: str = "train",              # train | predict
    freq_model_path: str = "models/freq_model.joblib",
):
    logger = ctx.logger
    logger.info(f"[featurize] mode={mode} input={csv_in}")

    df = pd.read_csv(csv_in)

    # =====================
    # basic normalization
    # =====================
    df = df.fillna({"server_name": "", "ja3": "", "ja3s": ""})
    df["sni"] = df["server_name"].astype(str)

    # =====================
    # TLS identity
    # =====================
    df["ja3_int"] = df["ja3"].apply(hash_str)
    df["ja3s_int"] = df["ja3s"].apply(hash_str)

    # =====================
    # SNI behavior
    # =====================
    df["sni_len"] = df["sni"].str.len()
    df["sni_entropy"] = df["sni"].apply(entropy)
    df["sni_is_ip"] = df["sni"].apply(lambda x: x.replace(".", "").isdigit())

    # =====================
    # TEMPORAL (beaconing)
    # =====================
    df = add_temporal_features(df)

    # =====================
    # FREQUENCY (PHASED)
    # =====================
    if mode == "train":
        logger.info("[featurize] fitting frequency model (NORMAL)")
        freq_model = fit_frequency(df)
        os.makedirs(os.path.dirname(freq_model_path), exist_ok=True)
        joblib.dump(freq_model, freq_model_path)
        logger.info(f"[featurize] saved freq model -> {freq_model_path}")

        # during training, freq not used directly
        df["ja3_freq"] = 0
        df["sni_freq"] = 0

    elif mode == "predict":
        logger.info("[featurize] applying frequency model")
        freq_model = joblib.load(freq_model_path)
        df = apply_frequency(df, freq_model)

    else:
        raise ValueError("mode must be 'train' or 'predict'")

    # =====================
    # JA3 reuse
    # =====================
    df["ja3_reuse_cnt"] = (
        df.groupby(["id.orig_h_ssl", "ja3"])["ja3"].transform("count")
    )

    # =====================
    # SELECT FEATURES
    # =====================
    feature_cols = [
        "ja3_int",
        "ja3s_int",
        "sni_len",
        "sni_entropy",
        "sni_is_ip",
        "ja3_freq",
        "sni_freq",
        "ja3_reuse_cnt",
        "iat_mean",
        "iat_std",
        "iat_cv",
        "iat_cnt",
    ]

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0

    out = df[feature_cols]
    write_csv(out, out_features_csv)
    logger.info(f"[featurize] saved -> {out_features_csv} ({len(out)} rows)")
