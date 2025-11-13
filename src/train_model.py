import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

def train_models():
    X = pd.read_csv("data/X.csv")
    y = pd.read_csv("data/y.csv").values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print("Train samples:", X_train.shape[0], "Test samples:", X_test.shape[0])

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print("Logistic Regression Acc:", acc_lr)
    print(classification_report(y_test, y_pred_lr))
    joblib.dump(lr, "models/logistic_model.joblib")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print("Random Forest Acc:", acc_rf)
    print(classification_report(y_test, y_pred_rf))
    joblib.dump(rf, "models/random_forest.joblib")

    # Save test predictions
    test_df = X_test.copy()
    test_df["actual"] = y_test
    test_df["pred_lr"] = y_pred_lr
    test_df["pred_rf"] = y_pred_rf
    test_df.to_csv("data/test_predictions.csv", index=False)
    print("Saved models and data/test_predictions.csv")

if __name__ == "__main__":
    train_models()
