from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import casparser
from scipy.optimize import newton, brentq
from datetime import datetime, timedelta
import tempfile
import os
import yfinance as yf
import json
import requests

app = FastAPI(title="FundWise Fault-Tolerant Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# GLOBAL CACHES
# ---------------------------------------------------------
NAV_CACHE = {}

# ---------------------------------------------------------
# TRANSACTION MAPPING & DATE PARSING
# ---------------------------------------------------------
TXN_MAP = {
    "PURCHASE": 1,
    "ADDITIONAL PURCHASE": 1,
    "FRESH PURCHASE": 1,
    "SIP": 1,
    "LUMPSUM": 1,
    "SWITCH IN": 1,
    "STP IN": 1,
    "DIVIDEND REINVESTMENT": 1,
    "DIVIDEND REINVEST": 1,
    
    "REDEMPTION": -1,
    "SWITCH OUT": -1,
    "STP OUT": -1,
    "SWP": -1,
    "DIVIDEND PAYOUT": -1,
    
    "SEGREGATION": 0,
    "MERGER": 0,
    "REVERSE MERGER": 0,
    "BONUS": 0,
    "BONUS UNITS": 0,
    "CORPORATE ACTION": 0,
    "REVERSAL": 0,
    "STAMP DUTY": 0,
    "TDS": 0,
    "TAX": 0
}

def normalize_txn_type(txn_type: str) -> int:
    clean_type = str(txn_type).replace("_", " ").replace("-", " ").strip().upper()
    for key, val in TXN_MAP.items():
        if key in clean_type:
            return val
    return 0

def parse_flexible_date(date_str: str) -> str:
    if not date_str:
        return "2025-04-01"
    cleaned = str(date_str).strip()
    if len(cleaned) == 10 and cleaned.endswith("202"):
        cleaned += "5"
    elif len(cleaned) == 9 and cleaned.endswith("20"):
        cleaned += "25"

    formats = ["%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y", "%d-%b-%y"]
    for fmt in formats:
        try:
            dt = datetime.strptime(cleaned[:11], fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return "2025-04-01"

# ---------------------------------------------------------
# DETERMINISTIC NAV LOOKUP
# ---------------------------------------------------------
def fetch_historical_nav_by_amfi(amfi_code: str, date_str: str) -> float:
    """Strictly fetches historical NAV using the official AMFI code. No text matching."""
    if not amfi_code or str(amfi_code).strip().lower() == "none":
        return None
        
    scheme_code = str(amfi_code).strip()
    target_dt = datetime.strptime(date_str, "%Y-%m-%d")
    
    if scheme_code not in NAV_CACHE:
        try:
            resp = requests.get(f"https://api.mfapi.in/mf/{scheme_code}", timeout=5)
            if resp.status_code == 200:
                data = resp.json().get('data', [])
                NAV_CACHE[scheme_code] = {
                    datetime.strptime(entry['date'], "%d-%m-%Y").strftime("%Y-%m-%d"): float(entry['nav'])
                    for entry in data
                }
            else:
                NAV_CACHE[scheme_code] = {}
        except Exception:
            return None
            
    scheme_navs = NAV_CACHE.get(scheme_code, {})
    
    # 7-Day Lookback for weekends and holidays
    for i in range(8):
        check_str = (target_dt - timedelta(days=i)).strftime("%Y-%m-%d")
        if check_str in scheme_navs:
            return scheme_navs[check_str]
            
    return None

# ---------------------------------------------------------
# MATHEMATICAL SOLVER (-99% to +200%)
# ---------------------------------------------------------
def solve_annualized_rate(cf_data: list[dict], closing_market_value: float) -> float:
    if not cf_data or closing_market_value <= 0:
        return 0.0

    def return_func(r):
        return sum(item["amount"] * ((1.0 + r) ** (item["days"] / 365.0)) for item in cf_data) - closing_market_value

    def return_derivative(r):
        deriv = 0.0
        for item in cf_data:
            t = item["days"] / 365.0
            if t > 0:
                deriv += item["amount"] * t * ((1.0 + r) ** (t - 1.0))
        return deriv

    # 1. Newton-Raphson Optimization
    try:
        rate = newton(return_func, 0.10, fprime=return_derivative, maxiter=500)
        if not isinstance(rate, complex) and -0.99 <= rate <= 2.0:
            return round(float(rate) * 100, 2)
    except Exception:
        pass

    # 2. Brent's Method Fallback
    try:
        rate = brentq(return_func, -0.99, 2.0, maxiter=500)
        return round(float(rate) * 100, 2)
    except Exception:
        pass

    # 3. Binary Search Fallback
    low, high = -0.99, 2.0
    mid = 0.0
    for _ in range(100):
        mid = (low + high) / 2.0
        val = return_func(mid)
        if abs(val) < 1e-5:
            return round(mid * 100, 2)
        if val > 0:
            high = mid
        else:
            low = mid
            
    return round(mid * 100, 2)

def get_benchmark_return(start_date_str: str, end_date_str: str) -> float:
    try:
        start_date = datetime.strptime(parse_flexible_date(start_date_str), "%Y-%m-%d")
        end_date = datetime.strptime(parse_flexible_date(end_date_str), "%Y-%m-%d")
        end_fetch_date = end_date + timedelta(days=1)
        
        ticker = yf.Ticker("^NSEI")
        df = ticker.history(start=start_date.strftime("%Y-%m-%d"), end=end_fetch_date.strftime("%Y-%m-%d"))
        
        if df.empty or len(df) < 2:
            return 14.0 
            
        start_price = df['Close'].iloc[0]
        end_price = df['Close'].iloc[-1]
        
        days = (end_date - start_date).days
        if days <= 0:
            return 0.0
            
        cagr = ((end_price / start_price) ** (365.0 / days) - 1) * 100
        return round(cagr, 2)
    except Exception:
        return 14.2

# ---------------------------------------------------------
# API ENDPOINT
# ---------------------------------------------------------
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
        
        statement_start_str = "2025-04-01"
        statement_end_dt = datetime.now()
        
        period_info = data.get("statement_period")
        if isinstance(period_info, dict):
            if period_info.get("from"):
                statement_start_str = parse_flexible_date(period_info.get("from"))
            if period_info.get("to"):
                statement_end_dt = datetime.strptime(parse_flexible_date(period_info.get("to")), "%Y-%m-%d")

        statement_end_str = statement_end_dt.strftime("%Y-%m-%d")

        portfolio_opening_value = 0.0
        portfolio_total_investments = 0.0
        portfolio_total_redemptions = 0.0
        portfolio_current_value = 0.0
        portfolio_cash_flows = []
        
        funds_breakdown = []
        folios = data.get("folios", [])
        
        for folio in folios:
            for scheme in folio.get("schemes", []):
                scheme_name = scheme.get("scheme", "Unknown Fund")
                valuation = scheme.get("valuation") or {}
                closing_value = float(valuation.get("value", 0.0) or 0.0)
                closing_cost = float(valuation.get("cost", 0.0) or 0.0)
                open_units = float(scheme.get("open", 0.0))
                
                # First pass: Aggregate transactions (needed for fallbacks)
                fund_investments = 0.0
                fund_redemptions = 0.0
                tx_list = []
                first_tx_nav = None
                
                for tx in scheme.get("transactions", []):
                    tx_date_str = parse_flexible_date(str(tx.get("date", "")))
                    
                    if statement_start_str <= tx_date_str <= statement_end_str:
                        amt_val = tx.get("amount")
                        if amt_val is not None:
                            amt_abs = abs(float(amt_val))
                            tx_dir = normalize_txn_type(tx.get("type", ""))
                            
                            if tx_dir == 1:
                                fund_investments += amt_abs
                                tx_list.append({"date": tx_date_str, "amount": amt_abs})
                            elif tx_dir == -1:
                                fund_redemptions += amt_abs
                                tx_list.append({"date": tx_date_str, "amount": -amt_abs})
                        
                        # Capture earliest NAV in period as a fallback
                        if first_tx_nav is None and tx.get("nav"):
                            try:
                                nav_val = float(tx["nav"])
                                if nav_val > 0:
                                    first_tx_nav = nav_val
                            except ValueError:
                                pass

                # ---------------------------------------------------------
                # RESOLUTION WATERFALL: Determine Opening Market Value
                # ---------------------------------------------------------
                opening_value = None
                resolution_path = ""

                # Condition 0: No starting units
                if open_units == 0:
                    opening_value = 0.0
                    resolution_path = "Zero Opening Balance"

                # Condition 1: CAS explicitly provides opening value
                if opening_value is None and scheme.get("opening_value") is not None:
                    opening_value = float(scheme["opening_value"])
                    resolution_path = "CAS Explicit Opening Value"

                # Condition 2: CAS explicitly provides starting NAV
                if opening_value is None and scheme.get("open_nav"):
                    opening_value = open_units * float(scheme["open_nav"])
                    resolution_path = f'CAS Explicit Opening NAV ({scheme["open_nav"]})'

                # Condition 3: Official AMFI Code API Lookup
                if opening_value is None and scheme.get("amfi"):
                    amfi_code = str(scheme.get("amfi")).strip()
                    fetched_nav = fetch_historical_nav_by_amfi(amfi_code, statement_start_str)
                    if fetched_nav:
                        opening_value = open_units * fetched_nav
                        resolution_path = f"MFAPI Official AMFI Lookup ({fetched_nav})"

                # Condition 4: Fallback to earliest transaction NAV
                if opening_value is None and first_tx_nav is not None:
                    opening_value = open_units * first_tx_nav
                    resolution_path = f"Fallback 1: Earliest Transaction NAV ({first_tx_nav})"

                # Condition 5: Fallback to reconstructed historical cost (Guaranteed calculation continuity)
                if opening_value is None:
                    estimated_cost = closing_cost - (fund_investments - fund_redemptions)
                    opening_value = max(0.0, estimated_cost)
                    resolution_path = "Fallback 2: Reconstructed Historical Cost (ESTIMATED)"
                
                # Server Logging for transparency
                print(f"[FundWise Resolver] Scheme: {scheme_name}")
                print(f"[FundWise Resolver] - Resolution Path: {resolution_path}")
                print(f"[FundWise Resolver] - Final Derived Opening Value: {opening_value}\n")

                # ---------------------------------------------------------
                # CALCULATIONS
                # ---------------------------------------------------------
                fund_cash_flows = []
                if opening_value > 0:
                    fund_cash_flows.append({
                        "date": statement_start_str,
                        "amount": opening_value
                    })
                fund_cash_flows.extend(tx_list)

                capital_deployed = opening_value + fund_investments - fund_redemptions
                absolute_profit = closing_value - capital_deployed
                absolute_return_pct = round((absolute_profit / capital_deployed) * 100, 2) if capital_deployed > 0 else 0.0
                
                cf_data = []
                for cf in fund_cash_flows:
                    cf_date = datetime.strptime(cf["date"], "%Y-%m-%d")
                    holding_days = max(0, (statement_end_dt - cf_date).days)
                    cf_data.append({"amount": cf["amount"], "days": holding_days})
                statement_annualized_return = solve_annualized_rate(cf_data, closing_value)

                funds_breakdown.append({
                    "scheme_name": scheme_name,
                    "opening_value": round(opening_value, 2),
                    "capital_deployed": round(capital_deployed, 2),
                    "current_value": round(closing_value, 2),
                    "absolute_profit": round(absolute_profit, 2),
                    "absolute_return_pct": absolute_return_pct,
                    "statement_annualized_return": statement_annualized_return,
                    "resolution_path": resolution_path # Exposing path in JSON internally
                })

                portfolio_opening_value += opening_value
                portfolio_total_investments += fund_investments
                portfolio_total_redemptions += fund_redemptions
                portfolio_current_value += closing_value
                portfolio_cash_flows.extend(fund_cash_flows)

        # Portfolio Aggregation
        total_capital_deployed = portfolio_opening_value + portfolio_total_investments - portfolio_total_redemptions
        total_profit = portfolio_current_value - total_capital_deployed
        portfolio_abs_return_pct = round((total_profit / total_capital_deployed) * 100, 2) if total_capital_deployed > 0 else 0.0
        
        port_cf_data = []
        for cf in portfolio_cash_flows:
            cf_date = datetime.strptime(cf["date"], "%Y-%m-%d")
            holding_days = max(0, (statement_end_dt - cf_date).days)
            port_cf_data.append({"amount": cf["amount"], "days": holding_days})
            
        portfolio_annualized_return = solve_annualized_rate(port_cf_data, portfolio_current_value)
        benchmark_annualized = get_benchmark_return(statement_start_str, statement_end_str)

        return {
            "status": "success",
            "statement_period": {
                "from": statement_start_str,
                "to": statement_end_str
            },
            "portfolio_summary": {
                "opening_portfolio_value": round(portfolio_opening_value, 2),
                "total_capital_deployed": round(total_capital_deployed, 2),
                "current_portfolio_value": round(portfolio_current_value, 2),
                "total_profit": round(total_profit, 2),
                "absolute_return_pct": portfolio_abs_return_pct,
                "statement_annualized_return": portfolio_annualized_return,
                "benchmark_annualized_return": benchmark_annualized
            },
            "funds_breakdown": funds_breakdown
        }

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"CAS Parse Failed: {str(e)}")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
