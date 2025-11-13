import pandas as pd

def prepare_dataset():
    # Load the dataset that already contains technical indicators
    df = pd.read_csv("data/data_with_indicators.csv")

    # Create target: 1 if next day's close is higher than today
    df["future_close"] = df["close"].shift(-1)
    df["direction"] = (df["future_close"] > df["close"]).astype(int)

    df.dropna(inplace=True)

    # Select ML features
    X = df[["rsi", "sma10", "sma20", "macd", "volatility"]]
    y = df["direction"]

    # Save all output files
    X.to_csv("data/X.csv", index=False)
    y.to_csv("data/y.csv", index=False)
    df.to_csv("data/full_dataset.csv", index=False)

    print("Saved: data/X.csv, data/y.csv, data/full_dataset.csv")

if __name__ == "__main__":
    prepare_dataset()
