import pandas as pd
from tls_anom.utils.io import write_csv

# Map label string → int (dễ mở rộng sau này)
LABEL_MAP = {
    "normal": 0,
    "botnet": 1
}

def run(ctx, csv_in: str, label_name: str, out_csv: str):
    df = pd.read_csv(csv_in)

    # Validate label
    if label_name not in LABEL_MAP:
        raise RuntimeError(f"Unknown label '{label_name}'. Expected: {list(LABEL_MAP.keys())}")

    # Miss step add column "label"
    
    # Assign numeric label
    df["label"] = LABEL_MAP[label_name]

    write_csv(df, out_csv)
