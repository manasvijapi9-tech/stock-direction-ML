import yfinance as yf
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

def fetch_stock_data(ticker="TRENT.NS", start="2025-01-01", end="2025-10-31"):
    print(f"Fetching data for {ticker} from {start} to {end} ...")
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if "Close" in data.columns:
        df = data[["Close"]].rename(columns={"Close": "close"})
    else:
        df = data.iloc[:, -1].to_frame(name="close")
    df.dropna(inplace=True)
    out = f"data/{ticker.replace('.', '_')}_raw.csv"
    df.to_csv(out)
    print("Saved:", out)
    return out

if __name__ == "__main__":
    fetch_stock_data("TCS.NS")
