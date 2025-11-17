import yfinance as yf
import pandas as pd
import os

def fetch_data(ticker="TRENT.NS",
               start_date="2015-01-01",
               end_date="2025-10-31"):
    
    os.makedirs("data", exist_ok=True)

    print(f"Fetching data for {ticker} from {start_date} to {end_date} ...")

    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        raise ValueError("No data fetched. Check the ticker or date range.")

    # Reset index
    data.reset_index(inplace=True)

    # FLATTEN MultiIndex columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip().lower() for col in data.columns]
    else:
        data.columns = [col.lower().replace(" ", "_") for col in data.columns]

    # Output filename
    output_name = ticker.replace(".", "_") + "_raw.csv"
    output_path = f"data/{output_name}"

    data.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    fetch_data()
