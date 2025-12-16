import pandas as pd
import numpy as np
import math
from collections import Counter
from tls_anom.utils.io import write_csv


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


# ================================
# Main featurizer
# ================================

def run(ctx, csv_in: str, out_features_csv: str):
    logger = ctx.logger
    logger.info(f"[featurize] loading {csv_in}")

    df = pd.read_csv(csv_in)

    # Normalize missing
    df = df.fillna({"server_name": "", "ja3": "", "ja3s": ""})

    # ====================================
    # FLOW-LEVEL FEATURES
    # ====================================
    df["duration"] = df.get("duration", 0).astype(float)

    df["orig_bytes"] = df.get("orig_bytes", 0).astype(float)
    df["resp_bytes"] = df.get("resp_bytes", 0).astype(float)
    df["byte_ratio"] = df["orig_bytes"] / (df["resp_bytes"] + 1)

    df["orig_pkts"] = df.get("orig_pkts", 0).astype(float)
    df["resp_pkts"] = df.get("resp_pkts", 0).astype(float)

    # ====================================
    # TLS METADATA FEATURES
    # ====================================

    # Convert JA3 / JA3S to hashed ints
    df["ja3_int"] = df["ja3"].apply(lambda x: hash_str(x, bits=18))
    df["ja3s_int"] = df["ja3s"].apply(lambda x: hash_str(x, bits=18))

    # Version (convert to numeric)
    df["tls_version_int"] = df.get("version", "").apply(hash_str)

    # Cipher list length
    if "ciphers" in df.columns:
        df["cipher_count"] = df["ciphers"].apply(
            lambda x: len(str(x).split(",")) if isinstance(x, str) else 0
        )
    else:
        df["cipher_count"] = 0

    # TLS extensions count
    if "client_extensions" in df.columns:
        df["ext_count"] = df["client_extensions"].apply(
            lambda x: len(str(x).split(",")) if isinstance(x, str) else 0
        )
    else:
        df["ext_count"] = 0

    # ====================================
    # SNI FEATURES
    # ====================================

    df["sni"] = df.get("server_name", "").astype(str)
    df["sni_len"] = df["sni"].apply(lambda x: len(x))
    df["sni_entropy"] = df["sni"].apply(entropy)
    df["sni_is_ip"] = df["sni"].apply(lambda x: x.replace(".", "").isdigit())

    # SNI rarity
    df["sni_rarity"] = rarity(df["sni"])

    # JA3 rarity (quan trọng!)
    df["ja3_rarity"] = rarity(df["ja3"])

    # ====================================
    # CERTIFICATE FEATURES (optional)
    # ====================================

    if "certificate_chain_fuids" in df.columns:
        df["cert_cnt"] = df["certificate_chain_fuids"].apply(
            lambda x: len(str(x).split(",")) if isinstance(x, str) else 0
        )
    else:
        df["cert_cnt"] = 0

    # ====================================
    # SELECT OUTPUT FEATURES
    # ====================================

    feature_cols = [
        # Flow-level
        "duration", "orig_bytes", "resp_bytes",
        "byte_ratio", "orig_pkts", "resp_pkts",

        # TLS metadata
        "ja3_int", "ja3s_int",
        "tls_version_int", "cipher_count", "ext_count",

        # SNI behavior
        "sni_len", "sni_entropy", "sni_is_ip",
        "sni_rarity",

        # JA3 behavior
        "ja3_rarity",

        # Cert features
        "cert_cnt",
    ]

    # Check missing columns
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0

    out = df[feature_cols].copy()  # keep label for next stage

    write_csv(out, out_features_csv)
    logger.info(f"[featurize] saved -> {out_features_csv} ({len(out)} rows)")
