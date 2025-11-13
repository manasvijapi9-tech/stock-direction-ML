import pandas as pd

def RSI(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, 1e-9))
    return 100 - (100 / (1 + rs))

def SMA(series, period=10):
    return series.rolling(period, min_periods=period).mean()

def MACD(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    return ema12 - ema26

def add_all_indicators(df):
    df = df.copy()
    df["rsi"] = RSI(df["close"])
    df["sma10"] = SMA(df["close"], 10)
    df["sma20"] = SMA(df["close"], 20)
    df["macd"] = MACD(df["close"])
    df["returns"] = df["close"].pct_change()
    df["volatility"] = df["returns"].rolling(10, min_periods=1).std()
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/TCS_NS_raw.csv", index_col=0, parse_dates=True)
    df = add_all_indicators(df)
    df.to_csv("data/data_with_indicators.csv")
    print("Saved data/data_with_indicators.csv")
