import yfinance as yf
import json
import os

def update():
    file_path = "nifty50_history.json"
    
    # We load existing data just in case, but we will overwrite with full history
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}
        
    print("Fetching full historical benchmark prices...")
    
    try:
        ticker = yf.Ticker("^NSEI")
        # Fetch MAX history. This guarantees we have data for 10+ year old CAS statements.
        # The download takes < 2 seconds and the JSON is < 150KB.
        df = ticker.history(period="max")
        
        if df.empty:
            print("Warning: No data received from Yahoo Finance.")
            return

        for index, row in df.iterrows():
            # Standardize date format to match app.py expectations
            date_str = index.strftime("%Y-%m-%d")
            data[date_str] = round(float(row["Close"]), 2)
            
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
            
        print(f"Benchmark successfully updated with {len(data)} trading days.")
        
    except Exception as e:
        print(f"Failed to update benchmark: {str(e)}")

if __name__ == "__main__":
    update()
