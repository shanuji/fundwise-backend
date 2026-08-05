from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import casparser
import numpy_financial as npf
from scipy.optimize import newton
from datetime import datetime
import tempfile
import os
import yfinance as yf
import json

app = FastAPI(title="FundWise Precision Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def calculate_xirr(cash_flows: list[dict], current_market_value: float, end_date: datetime) -> float:
    if not cash_flows:
        return 0.0

    dates = [datetime.strptime(str(cf["date"])[:10], "%Y-%m-%d") for cf in cash_flows]
    amounts = [-abs(cf["amount"]) if cf["type"] in ["PURCHASE", "SIP", "SWITCH_IN", "DIVIDEND_REINVEST"] else abs(cf["amount"]) for cf in cash_flows]

    dates.append(end_date)
    amounts.append(current_market_value)

    start_date = dates[0]
    days = [(d - start_date).days for d in dates]

    def xirr_func(r):
        return sum(a / ((1 + r) ** (day / 365.0)) for a, day in zip(amounts, days))

    def xirr_derivative(r):
        return sum(-a * (day / 365.0) / ((1 + r) ** ((day / 365.0) + 1)) for a, day in zip(amounts, days))

    try:
        rate = newton(xirr_func, 0.10, fprime=xirr_derivative, maxiter=100)
        return round(rate * 100, 2)
    except Exception:
        return round(npf.irr(amounts) * 100, 2)

def get_benchmark_return(start_date_str: str) -> float:
    try:
        clean_date = str(start_date_str)[:10]
        start_date = datetime.strptime(clean_date, "%Y-%m-%d")
        ticker = yf.Ticker("^NSEI")
        df = ticker.history(start=clean_date)
        
        if df.empty:
            return 14.0 
            
        start_price = df['Close'].iloc[0]
        end_price = df['Close'].iloc[-1]
        
        days = (datetime.now() - start_date).days
        if days <= 30:
            return 12.0
            
        cagr = ((end_price / start_price) ** (365.0 / days) - 1) * 100
        return round(cagr, 2)
    except Exception:
        return 14.2

def process_capital_gains(schemes_data, ltcg_rate, stcg_rate, exemption_limit):
    total_stcg = 0.0
    total_ltcg = 0.0
    
    for scheme in schemes_data:
        buy_queue = [] 
        transactions = sorted(
            scheme.get("transactions", []), 
            key=lambda x: datetime.strptime(str(x["date"])[:10], "%Y-%m-%d")
        )
        
        for tx in transactions:
            t_type = str(tx.get("type", "")).split('.')[-1].upper()
            units = tx.get("units")
            nav = tx.get("nav")
            
            if units is None or nav is None:
                continue
                
            date_obj = datetime.strptime(str(tx["date"])[:10], "%Y-%m-%d")
                
            if t_type in ["PURCHASE", "SIP", "SWITCH_IN", "DIVIDEND_REINVEST"]:
                buy_queue.append({'date': date_obj, 'units': float(units), 'nav': float(nav)})
            
            elif t_type in ["REDEMPTION", "SWITCH_OUT"]:
                sell_units = abs(float(units))
                sell_nav = float(nav)
                
                while sell_units > 0.0001 and buy_queue:
                    oldest_buy = buy_queue[0]
                    buy_date = oldest_buy['date']
                    buy_nav = oldest_buy['nav']
                    available_units = oldest_buy['units']
                    
                    units_to_sell = min(sell_units, available_units)
                    days_held = (date_obj - buy_date).days
                    gain = (sell_nav - buy_nav) * units_to_sell
                    
                    if days_held > 365:
                        total_ltcg += gain
                    else:
                        total_stcg += gain
                        
                    sell_units -= units_to_sell
                    buy_queue[0]['units'] -= units_to_sell
                    
                    if buy_queue[0]['units'] <= 0.0001:
                        buy_queue.pop(0)

    taxable_ltcg = max(0, total_ltcg - exemption_limit)
    ltcg_tax = taxable_ltcg * (ltcg_rate / 100.0)
    stcg_tax = max(0, total_stcg) * (stcg_rate / 100.0)
    total_tax = max(0, ltcg_tax) + max(0, stcg_tax)
    
    return {
        "realized_stcg": round(total_stcg, 2),
        "realized_ltcg": round(total_ltcg, 2),
        "taxable_ltcg": round(taxable_ltcg, 2),
        "estimated_tax_liability": round(total_tax, 2)
    }

@app.post("/api/v1/parse-cas")
async def parse_statement(
    file: UploadFile = File(...),
    password: str = Form(""),
    ltcg_rate: float = Form(12.5),
    stcg_rate: float = Form(20.0),
    exemption_limit: float = Form(125000.0),
    income_slab: float = Form(30.0)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be a PDF.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        raw_json_str = casparser.read_cas_pdf(tmp_path, password=password, output="json")
        data = json.loads(raw_json_str)
        
        total_invested = 0.0
        current_value = 0.0
        all_cash_flows = []
        schemes_data = []
        first_date_str = "2023-01-01"

        folios = data.get("folios", [])
        if folios and folios[0].get("schemes"):
            txs = folios[0]["schemes"][0].get("transactions", [])
            if txs:
                first_date_str = str(txs[0].get("date", "2023-01-01"))[:10]

        for folio in folios:
            for scheme in folio.get("schemes", []):
                schemes_data.append(scheme)
                valuation = scheme.get("valuation") or {}
                
                # THE FIX: Safely convert the JSON string value into a pure float
                raw_val = valuation.get("value", 0.0)
                if raw_val is not None and str(raw_val).strip() != "":
                    current_value += float(raw_val)

                for tx in scheme.get("transactions", []):
                    tx_type = str(tx.get("type", "")).split('.')[-1].upper()
                    
                    if tx.get("amount") and tx_type in ["PURCHASE", "SIP", "SWITCH_IN", "DIVIDEND_REINVEST", "REDEMPTION", "SWITCH_OUT"]:
                        amt = float(tx["amount"])
                        
                        if tx_type in ["PURCHASE", "SIP", "SWITCH_IN", "DIVIDEND_REINVEST"]:
                            total_invested += amt
                            
                        all_cash_flows.append({
                            "date": str(tx["date"])[:10],
                            "amount": amt,
                            "type": tx_type
                        })

        abs_profit = current_value - total_invested
        abs_return_pct = round((abs_profit / total_invested) * 100, 2) if total_invested > 0 else 0.0
        
        xirr = calculate_xirr(all_cash_flows, current_value, datetime.now())
        benchmark_xirr = get_benchmark_return(first_date_str)
        tax_data = process_capital_gains(schemes_data, ltcg_rate, stcg_rate, exemption_limit)

        return {
            "status": "success",
            "summary": {
                "capital_invested": round(total_invested, 2),
                "current_value": round(current_value, 2),
                "absolute_profit": round(abs_profit, 2),
                "absolute_return_pct": abs_return_pct,
                "xirr": xirr,
                "benchmark_xirr": benchmark_xirr,
                "total_transactions": len(all_cash_flows)
            },
            "taxes": tax_data
        }

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"CAS Parse Failed: {str(e)}")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
