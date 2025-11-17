import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Create plots folder
os.makedirs("plots", exist_ok=True)

def evaluate_model():
    # Load data
    test_df = pd.read_csv("data/test_predictions.csv")
    y_true = test_df["actual"]
    y_pred_rf = test_df["pred_rf"]

    # ------------------------------
    # Confusion Matrix
    # ------------------------------
    cm = confusion_matrix(y_true, y_pred_rf)

    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                annot_kws={"size": 14}, linewidths=0.5)
    plt.title("Confusion Matrix (Random Forest)", fontsize=18)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("Actual", fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png")
    plt.show()
    print("Saved: plots/confusion_matrix.png")

    # ------------------------------
    # Accuracy
    # ------------------------------
    acc = accuracy_score(y_true, y_pred_rf)
    print(f"\nRandom Forest Accuracy: {acc:.4f}")

    # ------------------------------
    # Probability Distribution Plot
    # ------------------------------
    model = joblib.load("models/random_forest.joblib")
    prob_rf = model.predict_proba(test_df[["rsi", "sma10", "sma20", "macd", "volatility"]])[:, 1]

    plt.figure(figsize=(8,5))
    plt.hist(prob_rf, bins=20, color="teal", edgecolor="black")
    plt.title("Prediction Probability Distribution (RF)", fontsize=16)
    plt.xlabel("Probability of Class 1 (Up)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("plots/probability_distribution.png")
    plt.show()
    print("Saved: plots/probability_distribution.png")

    print("\n✅ Evaluation complete.")

if __name__ == "__main__":
    evaluate_model()
