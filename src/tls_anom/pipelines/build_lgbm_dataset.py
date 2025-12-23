import pandas as pd

normal = pd.read_csv("data/features/normal.features.csv")
normal["label"] = 0

c2 = pd.read_csv("data/features/c2.features.csv")
c2["label"] = 1

df = pd.concat([normal, c2], ignore_index=True)
df.to_csv("data/features/stage2.lgbm.csv", index=False)
