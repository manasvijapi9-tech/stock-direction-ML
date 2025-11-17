import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import matplotlib.pyplot as plt

# Create folders if missing
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

def train_models():
    # Load prepared dataset
    X = pd.read_csv("data/X.csv")
    y = pd.read_csv("data/y.csv").values.ravel()

    # Split (time-series safe: no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print("Train samples:", X_train.shape[0], "Test samples:", X_test.shape[0])

    # ------------------------------
    # Logistic Regression Model
    # ------------------------------
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print("\n📌 Logistic Regression Accuracy:", acc_lr)
    print(classification_report(y_test, y_pred_lr))
    joblib.dump(lr, "models/logistic_model.joblib")

    # ------------------------------
    # Random Forest Model
    # ------------------------------
    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print("\n🌲 Random Forest Accuracy:", acc_rf)
    print(classification_report(y_test, y_pred_rf))
    joblib.dump(rf, "models/random_forest.joblib")

    # ------------------------------
    # Feature Importance Plot
    # ------------------------------
    importances = rf.feature_importances_
    features = ["rsi", "sma10", "sma20", "macd", "volatility"]

    plt.figure(figsize=(8,5))
    plt.bar(features, importances, color="purple")
    plt.title("Feature Importance (Random Forest)", fontsize=16)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png")
    plt.show()
    print("Saved: plots/feature_importance.png")

    # ------------------------------
    # Save Predictions for Evaluation
    # ------------------------------
    test_df = X_test.copy()
    test_df["actual"] = y_test
    test_df["pred_lr"] = y_pred_lr
    test_df["pred_rf"] = y_pred_rf
    test_df.to_csv("data/test_predictions.csv", index=False)

    print("\n✅ Models trained and saved successfully.")
    print("   Saved predictions → data/test_predictions.csv\n")

if __name__ == "__main__":
    train_models()
