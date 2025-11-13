import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

os.makedirs("plots", exist_ok=True)

def evaluate():
    df = pd.read_csv("data/test_predictions.csv")
    y = df["actual"]
    y_pred = df["pred_rf"]  # choose RF predictions
    cm = confusion_matrix(y, y_pred)
    print("Confusion matrix:\n", cm)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Confusion Matrix (Random Forest)")
    plt.savefig("plots/confusion_matrix.png", bbox_inches="tight", dpi=200)
    plt.show()
    print("Saved plots/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()
