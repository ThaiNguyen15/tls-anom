import pandas as pd

def read_csv(path):
    return pd.read_csv(path)

def write_csv(df, path):
    df.to_csv(path, index=False)
