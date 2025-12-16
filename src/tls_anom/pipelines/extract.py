import pandas as pd
from pathlib import Path
from tls_anom.utils.io import write_csv


def load_zeek_json(path: Path) -> pd.DataFrame:
    """
    Load Zeek JSON log (JSON-lines).
    Zeek JSON logs không có header, mỗi dòng là 1 JSON object → không bị lệch cột.
    """
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        df = pd.read_json(path, lines=True)
    except ValueError as e:
        raise RuntimeError(f"Failed to parse JSON log {path}: {e}")
    return df


def merge_zeek_logs(dataset: Path) -> pd.DataFrame:
    """
    Ghép conn.log + ssl.log + x509.log theo UID.
    """
    conn = load_zeek_json(dataset / "conn.log")
    ssl  = load_zeek_json(dataset / "ssl.log")
    x509 = load_zeek_json(dataset / "x509.log")

    if conn.empty:
        raise RuntimeError(f"conn.log not found or empty: {dataset}")
    if ssl.empty:
        raise RuntimeError(f"ssl.log not found or empty: {dataset}")

    # Merge ssl + conn
    merged = ssl.merge(conn, on="uid", how="left", suffixes=("_ssl", "_conn"))

    # Merge x509 (Zeek x509 logs sometimes use certificate.fuid as key)
    if not x509.empty and "certificate_chain_fuids" in ssl.columns:
        merged = merged.merge(
            x509,
            left_on="certificate_chain_fuids",
            right_on="id",
            how="left",
            suffixes=("", "_x509")
        )

    return merged


def run(ctx, dataset: str, out_csv: str):
    """
    Extract TLS-related features từ Zeek JSON logs.
    Input: folder chứa file JSON (.log)
    Output: CSV hợp nhất.
    """
    logger = ctx.logger
    dataset = Path(dataset)
    out_csv = Path(out_csv)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"[extract] from Zeek JSON logs: {dataset}")

    df = merge_zeek_logs(dataset)

    # Clean nested types (list/dict) → convert to string
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df[col] = df[col].astype(str)

    write_csv(df, out_csv)
    logger.info(f"[done] saved merged log -> {out_csv}")