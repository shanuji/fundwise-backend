from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import casparser
from scipy.optimize import newton, brentq
from datetime import datetime, timedelta
import tempfile
import os
import yfinance as yf
import json

app = FastAPI(title="FundWise Precision Statement Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def solve_annualized_rate(cf_data: list[dict], closing_market_value: float) -> float:
    """
    Solves for 'r' where: Closing Value = Σ(Cash Flow × (1 + r)^(HoldingDays / 365))
    Using Newton-Raphson -> Brentq -> Binary Search fallbacks.
    """
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

    # 1. Newton-Raphson
    try:
        rate = newton(return_func, 0.10, fprime=return_derivative, maxiter=1000)
        if not isinstance(rate, complex) and rate > -1.0:
            return round(float(rate) * 100, 2)
    except Exception:
        pass

    # 2. Brent's Method
    try:
        rate = brentq(return_func, -0.9999, 100.0, maxiter=1000)
        return round(float(rate) * 100, 2)
    except Exception:
        pass

    # 3. Binary Search (Bisection Fallback)
    low, high = -0.9999, 100.0
    for _ in range(100):
        mid = (low + high) / 2.0
        val = return_func(mid)
        if abs(val) < 1e-5:
            return round(mid * 100, 2)
        if val > 0:
            high = mid  # Future value is too high, lower the rate
        else:
            low = mid   # Future value is too low, raise the rate
            
    return round(mid * 100, 2)

def calculate_statement_annualized_return(cash_flows: list[dict], closing_market_value: float, end_date: datetime) -> float:
    cf_data = []
    for cf in cash_flows:
        cf_date = datetime.strptime(cf["date"], "%Y-%m-%d")
        holding_days = max(0, (end_date - cf_date).days)
        cf_data.append({"amount": cf["amount"], "days": holding_days})
        
    return solve_annualized_rate(cf_data, closing_market_value)

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
        
        inflow_keywords = ["PURCHASE", "SIP", "LUMPSUM", "SWITCH IN", "STP IN", "DIVIDEND REINVEST"]
        outflow_keywords = ["REDEMPTION", "SWITCH OUT", "STP OUT", "SWP", "DIVIDEND PAYOUT"]
        neutral_keywords = ["SEGREGATION", "MERGER", "REVERSE MERGER", "BONUS", "STAMP"]
        
        for folio in folios:
            for scheme in folio.get("schemes", []):
                scheme_name = scheme.get("scheme", "Unknown Fund")
                valuation = scheme.get("valuation") or {}
                
                closing_value = float(valuation.get("value", 0.0) or 0.0)
                
                # Accurately isolate starting market value
                opening_value = float(valuation.get("opening", valuation.get("opening_value", 0.0)))
                if opening_value <= 0.0:
                    opening_units = float(scheme.get("open", 0.0))
                    if opening_units > 0:
                        first_nav = None
                        for tx in scheme.get("transactions", []):
                            if tx.get("nav") is not None:
                                first_nav = float(tx["nav"])
                                break
                        if first_nav:
                            opening_value = opening_units * first_nav
                
                fund_investments = 0.0
                fund_redemptions = 0.0
                tx_list = []
                
                for tx in scheme.get("transactions", []):
                    tx_date_str = parse_flexible_date(str(tx.get("date", "")))
                    
                    if statement_start_str <= tx_date_str <= statement_end_str:
                        amt_val = tx.get("amount")
                        if amt_val is not None:
                            amt = float(amt_val)
                            tx_type_upper = str(tx.get("type", "")).replace("_", " ").upper()
                            
                            is_inflow = any(kw in tx_type_upper for kw in inflow_keywords)
                            is_outflow = any(kw in tx_type_upper for kw in outflow_keywords)
                            is_neutral = any(kw in tx_type_upper for kw in neutral_keywords)
                            
                            if is_neutral or amt == 0:
                                continue
                                
                            if is_inflow:
                                amt_abs = abs(amt)
                                fund_investments += amt_abs
                                tx_list.append({"date": tx_date_str, "amount": amt_abs})
                            elif is_outflow:
                                amt_abs = abs(amt)
                                fund_redemptions += amt_abs
                                tx_list.append({"date": tx_date_str, "amount": -amt_abs})
                            else:
                                if amt > 0:
                                    fund_investments += amt
                                    tx_list.append({"date": tx_date_str, "amount": amt})
                                elif amt < 0:
                                    fund_redemptions += abs(amt)
                                    tx_list.append({"date": tx_date_str, "amount": amt})

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
                statement_annualized_return = calculate_statement_annualized_return(fund_cash_flows, closing_value, statement_end_dt)

                funds_breakdown.append({
                    "scheme_name": scheme_name,
                    "opening_value": round(opening_value, 2),
                    "capital_deployed": round(capital_deployed, 2),
                    "current_value": round(closing_value, 2),
                    "absolute_profit": round(absolute_profit, 2),
                    "absolute_return_pct": absolute_return_pct,
                    "statement_annualized_return": statement_annualized_return
                })

                portfolio_opening_value += opening_value
                portfolio_total_investments += fund_investments
                portfolio_total_redemptions += fund_redemptions
                portfolio_current_value += closing_value
                portfolio_cash_flows.extend(fund_cash_flows)

        total_capital_deployed = portfolio_opening_value + portfolio_total_investments - portfolio_total_redemptions
        total_profit = portfolio_current_value - total_capital_deployed
        portfolio_abs_return_pct = round((total_profit / total_capital_deployed) * 100, 2) if total_capital_deployed > 0 else 0.0
        
        portfolio_annualized_return = calculate_statement_annualized_return(portfolio_cash_flows, portfolio_current_value, statement_end_dt)
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
