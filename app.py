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
import re
import threading

app = FastAPI(title="FundWise Strict Math Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# GLOBAL CACHES & THREAD LOCKS
# ---------------------------------------------------------
SCHEME_CACHE_FILE = "scheme_cache.json"
SCHEME_NAME_TO_CODE = {}
AMFI_MASTER_LIST = []
AMFI_LOCK = threading.Lock()
NAV_CACHE = {}
BENCHMARK_PRICE_CACHE = {}

if os.path.exists(SCHEME_CACHE_FILE):
    try:
        with open(SCHEME_CACHE_FILE, "r") as f:
            SCHEME_NAME_TO_CODE = json.load(f)
    except Exception:
        pass

def save_scheme_cache():
    try:
        with open(SCHEME_CACHE_FILE, "w") as f:
            json.dump(SCHEME_NAME_TO_CODE, f, indent=4)
    except Exception:
        pass

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
    
    "REDEMPTION": -1,
    "SWITCH OUT": -1,
    "STP OUT": -1,
    "SWP": -1,
    "DIVIDEND PAYOUT": -1,
    
    # Neutral events that do not inject or remove external capital
    "DIVIDEND REINVESTMENT": 0,
    "DIVIDEND REINVEST": 0,
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
# DETERMINISTIC AMFI KEYWORD MATCHING
# ---------------------------------------------------------
def get_amfi_master():
    global AMFI_MASTER_LIST
    with AMFI_LOCK:
        if AMFI_MASTER_LIST:
            return AMFI_MASTER_LIST
        try:
            resp = requests.get("https://www.amfiindia.com/spages/NAVAll.txt", timeout=10)
            if resp.status_code == 200:
                for line in resp.text.split('\n'):
                    parts = line.split(';')
                    if len(parts) >= 6 and parts[0].strip().isdigit():
                        AMFI_MASTER_LIST.append({
                            "code": parts[0].strip(),
                            "name": parts[3].strip()
                        })
        except Exception:
            pass
        return AMFI_MASTER_LIST

def get_base_keywords(text: str) -> set:
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    stopwords = {'direct', 'regular', 'plan', 'growth', 'option', 'idcw', 'non', 'demat', 'mutual', 'fund', 'advisor', 'dp', 'gr', 'dividend', 'payout', 'reinvestment'}
    return {w for w in text.split() if w not in stopwords and len(w) > 1}

def find_scheme_code(cas_scheme_name: str, amfi_hint: str = "") -> str:
    if cas_scheme_name in SCHEME_NAME_TO_CODE:
        return SCHEME_NAME_TO_CODE[cas_scheme_name]

    if amfi_hint and amfi_hint.isdigit():
        SCHEME_NAME_TO_CODE[cas_scheme_name] = amfi_hint
        save_scheme_cache()
        return amfi_hint

    master = get_amfi_master()
    if not master:
        return None

    cas_lower = cas_scheme_name.lower()
    base_keywords = get_base_keywords(cas_scheme_name)
    if not base_keywords: return None

    candidates = []
    for item in master:
        amfi_lower = item["name"].lower()
        if all(kw in amfi_lower for kw in base_keywords):
            candidates.append(item)

    if len(candidates) > 1:
        if 'direct' in cas_lower or 'dir' in cas_lower.split() or 'dp' in cas_lower.split():
            filtered = [c for c in candidates if 'direct' in c['name'].lower()]
            if filtered: candidates = filtered
        elif 'regular' in cas_lower or 'reg' in cas_lower.split():
            filtered = [c for c in candidates if 'regular' in c['name'].lower()]
            if filtered: candidates = filtered

    if len(candidates) > 1:
        if 'growth' in cas_lower or 'gr' in cas_lower.split():
            filtered = [c for c in candidates if 'growth' in c['name'].lower()]
            if filtered: candidates = filtered
        elif 'idcw' in cas_lower or 'dividend' in cas_lower:
            filtered = [c for c in candidates if 'idcw' in c['name'].lower() or 'dividend' in c['name'].lower()]
            if filtered: candidates = filtered

    if len(candidates) == 1:
        best_code = candidates[0]["code"]
        SCHEME_NAME_TO_CODE[cas_scheme_name] = best_code
        save_scheme_cache()
        return best_code

    return None

def fetch_historical_nav(scheme_name: str, date_str: str, amfi_hint: str = "") -> float:
    scheme_code = find_scheme_code(scheme_name, amfi_hint)
    if not scheme_code: return None
        
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
    
    for i in range(8):
        check_str = (target_dt - timedelta(days=i)).strftime("%Y-%m-%d")
        if check_str in scheme_navs:
            return scheme_navs[check_str]
            
    return None

# ---------------------------------------------------------
# MONEY-WEIGHTED BENCHMARK SIMULATOR
# ---------------------------------------------------------
def fetch_benchmark_prices(start_date_str: str, end_date_str: str) -> dict:
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d") - timedelta(days=10)
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=5)
    cache_key = f"{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}"
    
    if cache_key in BENCHMARK_PRICE_CACHE:
        return BENCHMARK_PRICE_CACHE[cache_key]

    # Using NIFTYBEES.NS (ETF) to properly account for Total Return (Dividends)
    ticker = yf.Ticker("NIFTYBEES.NS")
    df = ticker.history(start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"))
    
    if df.empty or len(df) == 0:
        raise ValueError(f"Yahoo Finance returned 0 prices for NIFTYBEES.NS between {start_dt.date()} and {end_dt.date()}.")
        
    price_dict = {}
    for index, row in df.iterrows():
        price_dict[index.strftime("%Y-%m-%d")] = float(row['Close'])
        
    BENCHMARK_PRICE_CACHE[cache_key] = price_dict
    print(f"[Benchmark Setup] Successfully downloaded {len(price_dict)} daily prices from YFinance.")
    return price_dict

def get_closest_benchmark_price(date_str: str, price_dict: dict) -> float:
    target_dt = datetime.strptime(date_str, "%Y-%m-%d")
    
    # 1. Walk FORWARD to simulate Next-Business-Day execution (Standard MF behavior)
    for i in range(5):
        check_str = (target_dt + timedelta(days=i)).strftime("%Y-%m-%d")
        if check_str in price_dict:
            return price_dict[check_str]
            
    # 2. Walk BACKWARD only as a fallback (e.g., if target is today and markets haven't closed)
    for i in range(1, 10):
        check_str = (target_dt - timedelta(days=i)).strftime("%Y-%m-%d")
        if check_str in price_dict:
            return price_dict[check_str]
    
    raise ValueError(f"Could not find any benchmark price near {date_str}. Missing market data.")

# ---------------------------------------------------------
# STRICT MATHEMATICAL SOLVER (-99% to +200%)
# ---------------------------------------------------------
def solve_annualized_rate(cf_data: list[dict], closing_market_value: float, context="Solver") -> float:
    if not cf_data:
        raise ValueError(f"[{context}] Failed: No cash flows provided to the solver.")
    if closing_market_value <= 0:
        raise ValueError(f"[{context}] Failed: Invalid closing market value ({closing_market_value}).")

    def return_func(r):
        return sum(item["amount"] * ((1.0 + r) ** (item["days"] / 365.0)) for item in cf_data) - closing_market_value

    def return_derivative(r):
        deriv = 0.0
        for item in cf_data:
            t = item["days"] / 365.0
            if t > 0:
                deriv += item["amount"] * t * ((1.0 + r) ** (t - 1.0))
        return deriv

    # 1. Newton
    try:
        rate = newton(return_func, 0.10, fprime=return_derivative, maxiter=500)
        if not isinstance(rate, complex) and -0.99 <= rate <= 2.0:
            return round(float(rate) * 100, 2)
    except Exception as e:
        print(f"[{context}] Newton solver failed to converge: {e}")

    # 2. Brent
    try:
        rate = brentq(return_func, -0.99, 2.0, maxiter=500)
        return round(float(rate) * 100, 2)
    except Exception as e:
        print(f"[{context}] Brentq solver failed to converge: {e}")

    # 3. Binary Search
    low, high = -0.99, 2.0
    mid = 0.0
    val = 0.0
    for _ in range(100):
        mid = (low + high) / 2.0
        val = return_func(mid)
        if abs(val) < 1e-4:
            return round(mid * 100, 2)
        if val > 0:
            high = mid
        else:
            low = mid
            
    if abs(val) > 1.0: # If the mathematical residual is off by more than 1 Rupee
        error_msg = f"[{context}] Binary search failed to converge. Final residual gap: {val}, Rate bounded between {low} and {high}."
        print(error_msg)
        raise ValueError(error_msg)
        
    return round(mid * 100, 2)

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
                        
                        if first_tx_nav is None and tx.get("nav"):
                            try:
                                nav_val = float(tx["nav"])
                                if nav_val > 0:
                                    first_tx_nav = nav_val
                            except ValueError:
                                pass

                # RESOLUTION WATERFALL
                opening_value = None
                resolution_path = ""

                if open_units == 0:
                    opening_value = 0.0
                    resolution_path = "Zero Opening Balance"

                if opening_value is None and scheme.get("opening_value") is not None:
                    opening_value = float(scheme["opening_value"])
                    resolution_path = "CAS Explicit Opening Value"

                if opening_value is None and scheme.get("open_nav"):
                    opening_value = open_units * float(scheme["open_nav"])
                    resolution_path = f'CAS Explicit Opening NAV ({scheme["open_nav"]})'

                if opening_value is None:
                    amfi_code = str(scheme.get("amfi", "")).strip()
                    fetched_nav = fetch_historical_nav(scheme_name, statement_start_str, amfi_code)
                    if fetched_nav:
                        opening_value = open_units * fetched_nav
                        resolution_path = f"MFAPI Official AMFI Lookup ({fetched_nav})"

                if opening_value is None and first_tx_nav is not None:
                    opening_value = open_units * first_tx_nav
                    resolution_path = f"Fallback 1: Earliest Transaction NAV ({first_tx_nav})"

                if opening_value is None:
                    estimated_cost = closing_cost - (fund_investments - fund_redemptions)
                    opening_value = max(0.0, estimated_cost)
                    resolution_path = "Fallback 2: Reconstructed Historical Cost (ESTIMATED)"
                
                print(f"[FundWise Resolver] Scheme: {scheme_name}")
                print(f"[FundWise Resolver] - Resolution Path: {resolution_path}")
                print(f"[FundWise Resolver] - Final Derived Opening Value: {opening_value}\n")

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
                statement_annualized_return = solve_annualized_rate(cf_data, closing_value, context=f"Fund: {scheme_name}")

                funds_breakdown.append({
                    "scheme_name": scheme_name,
                    "opening_value": round(opening_value, 2),
                    "capital_deployed": round(capital_deployed, 2),
                    "current_value": round(closing_value, 2),
                    "absolute_profit": round(absolute_profit, 2),
                    "absolute_return_pct": absolute_return_pct,
                    "statement_annualized_return": statement_annualized_return,
                    "resolution_path": resolution_path 
                })

                portfolio_opening_value += opening_value
                portfolio_total_investments += fund_investments
                portfolio_total_redemptions += fund_redemptions
                portfolio_current_value += closing_value
                portfolio_cash_flows.extend(fund_cash_flows)

        total_capital_deployed = portfolio_opening_value + portfolio_total_investments - portfolio_total_redemptions
        total_profit = portfolio_current_value - total_capital_deployed
        portfolio_abs_return_pct = round((total_profit / total_capital_deployed) * 100, 2) if total_capital_deployed > 0 else 0.0
        
        port_cf_data = []
        for cf in portfolio_cash_flows:
            cf_date = datetime.strptime(cf["date"], "%Y-%m-%d")
            holding_days = max(0, (statement_end_dt - cf_date).days)
            port_cf_data.append({"amount": cf["amount"], "days": holding_days})
            
        portfolio_annualized_return = solve_annualized_rate(port_cf_data, portfolio_current_value, context="Total Portfolio")
        
        # ---------------------------------------------------------
        # TRUE MONEY-WEIGHTED BENCHMARK SIMULATION
        # ---------------------------------------------------------
        print("\n=== STARTING BENCHMARK SIMULATION ===")
        print(f"Total Portfolio Cash Flows: {len(portfolio_cash_flows)}")
        
        benchmark_prices = fetch_benchmark_prices(statement_start_str, statement_end_str)
        benchmark_units = 0.0
        benchmark_cf_data = []
        
        for cf in portfolio_cash_flows:
            cf_date = cf["date"]
            cf_amount = cf["amount"] 
            
            if cf_amount == 0:
                continue
                
            nav = get_closest_benchmark_price(cf_date, benchmark_prices)
            units_transacted = cf_amount / nav
            realized_cf_amount = cf_amount
            
            # Prevent Negative Benchmark Bankruptcy
            if units_transacted < 0 and abs(units_transacted) > benchmark_units:
                units_transacted = -benchmark_units 
                realized_cf_amount = units_transacted * nav 
                print(f"[Warning] Benchmark capital depleted. Capping withdrawal on {cf_date}.")
                
            benchmark_units += units_transacted
            
            cf_date_obj = datetime.strptime(cf_date, "%Y-%m-%d")
            holding_days = max(0, (statement_end_dt - cf_date_obj).days)
            benchmark_cf_data.append({"amount": realized_cf_amount, "days": holding_days})
            
            direction = "BOUGHT" if cf_amount > 0 else "SOLD"
            print(f"[Benchmark] Date: {cf_date} | {direction} | Nifty50 NAV: {nav} | Units: {units_transacted} | Balance: {benchmark_units}")
                
        final_benchmark_nav = get_closest_benchmark_price(statement_end_str, benchmark_prices)
        benchmark_simulated_closing_value = max(0.001, benchmark_units * final_benchmark_nav)
        
        print(f"=== SIMULATION COMPLETE ===")
        print(f"Final Benchmark Units: {benchmark_units}")
        print(f"Final Benchmark NAV: {final_benchmark_nav}")
        print(f"Benchmark Simulated Closing Value: {benchmark_simulated_closing_value}")

        benchmark_annualized = solve_annualized_rate(benchmark_cf_data, benchmark_simulated_closing_value, context="Nifty 50 Benchmark")
        print(f"Benchmark Annualized Return Successfully Solved: {benchmark_annualized}%\n")

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

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"CAS Parse Failed: {str(e)}")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
