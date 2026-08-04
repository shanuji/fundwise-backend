from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import casparser
import numpy_financial as npf
from scipy.optimize import newton
from datetime import datetime
import tempfile
import os

app = FastAPI(title="FundWise Precision Engine")

# Enable CORS for Flutter app requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def calculate_xirr(cash_flows: list[dict], current_market_value: float, end_date: datetime) -> float:
    """
    Computes exact XIRR using SciPy Newton-Raphson method.
    """
    if not cash_flows:
        return 0.0

    dates = [datetime.strptime(cf["date"], "%Y-%m-%d") for cf in cash_flows]
    amounts = [-abs(cf["amount"]) if cf["type"] in ["PURCHASE", "SIP", "SWITCH_IN", "DIVIDEND_REINVEST"] else abs(cf["amount"]) for cf in cash_flows]

    # Add terminal portfolio value as a positive cash flow today
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
        # Fallback to numpy-financial if Newton-Raphson fails to converge
        return round(npf.irr(amounts) * 100, 2)

def process_capital_gains(schemes_data, ltcg_rate, stcg_rate, exemption_limit):
    """
    Core FIFO engine to calculate realized gains based on holding periods.
    """
    total_stcg = 0.0
    total_ltcg = 0.0
    
    for scheme in schemes_data:
        buy_queue = [] 
        
        # Sort transactions chronologically
        transactions = sorted(scheme.get("transactions", []), key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"))
        
        for tx in transactions:
            t_type = tx.get("type")
            units = tx.get("units")
            nav = tx.get("nav")
            
            if units is None or nav is None:
                continue
                
            date_obj = datetime.strptime(tx["date"], "%Y-%m-%d")
                
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
                    
                    # Calculate gain for this specific batch
                    gain = (sell_nav - buy_nav) * units_to_sell
                    
                    # 365 days cutoff for equity funds
                    if days_held > 365:
                        total_ltcg += gain
                    else:
                        total_stcg += gain
                        
                    # Deduct units from queue
                    sell_units -= units_to_sell
                    buy_queue[0]['units'] -= units_to_sell
                    
                    if buy_queue[0]['units'] <= 0.0001:
                        buy_queue.pop(0)

    # Apply Tax Rules parameters passed from Flutter
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
    ltcg_rate: float = Form(12.5),
    stcg_rate: float = Form(20.0),
    exemption_limit: float = Form(125000.0),
    income_slab: float = Form(30.0)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be a PDF.")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Execute Deterministic casparser
        data = casparser.read_cas_pdf(tmp_path, password="", output="dict")
        
        total_invested = 0.0
        current_value = 0.0
        all_cash_flows = []
        schemes_data = []

        for folio in data.get("folios", []):
            for scheme in folio.get("schemes", []):
                schemes_data.append(scheme)
                valuation = scheme.get("valuation", {})
                current_value += valuation.get("value", 0.0)

                for tx in scheme.get("transactions", []):
                    # Filter out micro-deductions like Stamp Duty / STT for XIRR cash flows
                    if tx.get("amount") and tx.get("type") in ["PURCHASE", "SIP", "SWITCH_IN", "DIVIDEND_REINVEST", "REDEMPTION", "SWITCH_OUT"]:
                        amt = float(tx["amount"])
                        if tx["type"] in ["PURCHASE", "SIP", "SWITCH_IN", "DIVIDEND_REINVEST"]:
                            total_invested += amt
                            
                        all_cash_flows.append({
                            "date": tx["date"],
                            "amount": amt,
                            "type": tx["type"]
                        })

        # Calculations
        abs_profit = current_value - total_invested
        abs_return_pct = round((abs_profit / total_invested) * 100, 2) if total_invested > 0 else 0.0
        xirr = calculate_xirr(all_cash_flows, current_value, datetime.now())
        
        # Run FIFO Capital Gains Tax Engine
        tax_data = process_capital_gains(schemes_data, ltcg_rate, stcg_rate, exemption_limit)

        return {
            "status": "success",
            "summary": {
                "capital_invested": round(total_invested, 2),
                "current_value": round(current_value, 2),
                "absolute_profit": round(abs_profit, 2),
                "absolute_return_pct": abs_return_pct,
                "xirr": xirr,
                "total_transactions": len(all_cash_flows)
            },
            "taxes": tax_data
        }

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Parsing error: {str(e)}")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
