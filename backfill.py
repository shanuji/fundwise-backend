import yfinance as yf
import json

def backfill():
    print("Downloading 10 years of Nifty 50 data...")
    ticker = yf.Ticker("^NSEI")
    df = ticker.history(period="10y")
    
    data = {}
    for index, row in df.iterrows():
        data[index.strftime("%Y-%m-%d")] = round(float(row["Close"]), 2)
        
    with open("nifty50_history.json", "w") as f:
        json.dump(data, f, indent=2)
        
    print(f"Success! Backfilled {len(data)} rows into nifty50_history.json.")

if __name__ == "__main__":
    backfill()
