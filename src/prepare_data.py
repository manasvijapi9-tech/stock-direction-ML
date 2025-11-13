import pandas as pd
import os

def prepare_dataset():
    # Read dataset WITH indicators (clean file)
    df = pd.read_csv("data/data_with_indicators.csv")

    # Create label: 1 if next day close is higher else 0
    df["future_close"] = df["close"].shift(-1)
    df["direction"] = (df["future_close"] > df["close"]).astype(int)
    df.dropna(inplace=True)

    # Features
    features = df[["rsi", "sma10", "sma20", "macd", "volatility"]]
    labels = df["direction"]

    # Save outputs
    features.to_csv("data/X.csv", index=False)
    labels.to_csv("data/y.csv", index=False)
    df.to_csv("data/full_dataset.csv", index=False)

    print("Saved: data/X.csv, data/y.csv, data/full_dataset.csv")

if __name__ == "__main__":
    prepare_dataset()
