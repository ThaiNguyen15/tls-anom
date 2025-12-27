import pandas as pd

normal = pd.read_csv("data/features/normal.features.csv")
normal["label"] = 0

botnet = pd.read_csv("data/features/botnet.features.csv")
botnet["label"] = 1

df = pd.concat([normal, botnet], ignore_index=True)
df.to_csv("data/features/stage2.lgbm.csv", index=False)

