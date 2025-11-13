import pandas as pd
import os

os.makedirs("data", exist_ok=True)

def prepare_dataset(ticker="TCS_NS_raw"):
    src = f"data/{ticker}.csv"
    df = pd.read_csv(src, index_col=0, parse_dates=True)
    # Add indicators (if not already present) - expects indicators.py used beforehand
    if "rsi" not in df.columns:
        from src.indicators import add_all_indicators
        df = add_all_indicators(df)
    # label: 1 if next day close higher else 0
    df["future_close"] = df["close"].shift(-1)
    df["direction"] = (df["future_close"] > df["close"]).astype(int)
    df.dropna(inplace=True)
    features = df[["rsi", "sma10", "sma20", "macd", "volatility"]]
    labels = df["direction"]
    features.to_csv("data/X.csv", index=False)
    labels.to_csv("data/y.csv", index=False)
    df.to_csv("data/full_dataset.csv")
    print("Saved data/X.csv, data/y.csv and data/full_dataset.csv")

if __name__ == "__main__":
    prepare_dataset("TCS_NS_raw")
