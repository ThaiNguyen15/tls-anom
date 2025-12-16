import pandas as pd
import matplotlib.pyplot as plt

def plot(normal_csv, botnet_csv, out_png="score_dist.png"):
    df_n = pd.read_csv(normal_csv)
    df_b = pd.read_csv(botnet_csv)

    plt.figure(figsize=(8, 5))

    plt.hist(df_n["anomaly_score"], bins=100, alpha=0.6, label="Normal", density=True)
    plt.hist(df_b["anomaly_score"], bins=100, alpha=0.6, label="Botnet", density=True)

    plt.xlabel("Anomaly Score (higher = more anomalous)")
    plt.ylabel("Density")
    plt.legend()
    plt.title("IsolationForest Anomaly Score Distribution")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.show()

if __name__ == "__main__":
    plot(
        "outputs/predictions/normal.pred.csv",
        "outputs/predictions/botnet.pred.csv"
    )
