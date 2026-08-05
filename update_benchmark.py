import yfinance as yf
import json
import os

def update():
    file_path = "nifty50_history.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = {}
        
    print("Fetching recent benchmark prices...")
    ticker = yf.Ticker("^NSEI")
    df = ticker.history(period="5d")
    
    for index, row in df.iterrows():
        data[index.strftime("%Y-%m-%d")] = round(float(row["Close"]), 2)
        
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
        
    print("Benchmark successfully updated.")

if __name__ == "__main__":
    update()
