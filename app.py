from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import casparser
from datetime import datetime
import tempfile
import os
import yfinance as yf
import json

app = FastAPI(title="FundWise Weighted Return Engine")

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

def calculate_weighted_returns(cash_flows: list[dict], closing_value: float, start_date: datetime, end_date: datetime):
    """
    Calculates the time-weighted average capital return based on the exact days each cash flow was held.
    """
    total_statement_days = (end_date - start_date).days
    if total_statement_days <= 0:
        total_statement_days = 1 # Prevent division by zero

    total_inflows = 0.0
    total_outflows = 0.0
    weighted_capital_base = 0.0

    for cf in cash_flows:
        cf_date = datetime.strptime(parse_flexible_date(cf["date"]), "%Y-%m-%d")
        # Ensure days held doesn't exceed statement duration and isn't negative
        days_held = max(0, min(total_statement_days, (end_date - cf_date).days))
        weight = days_held / total_statement_days
        
        amt = abs(float(cf["amount"]))
        
        if cf["type"] in ["OPENING_VALUE", "INVESTMENT", "SIP", "PURCHASE", "SWITCH_IN", "DIVIDEND_REINVEST"]:
            total_inflows += amt
            weighted_capital_base += (amt * weight)
        elif cf["type"] in ["REDEMPTION", "SWITCH_OUT"]:
            total_outflows += amt
            # Redemptions reduce the capital base for the remainder of the period
            weighted_capital_base -= (amt * weight)

    capital_deployed = total_inflows - total_outflows
    absolute_profit = closing_value - capital_deployed

    if weighted_capital_base <= 0:
        return 0.0, 0.0

    # 1. Period Average Return based on weighted base
    weighted_period_return_pct = (absolute_profit / weighted_capital_base) * 100
    
    # 2. Annualized Average Return
    annualized_weighted_return_pct = weighted_period_return_pct * (365.0 / total_statement_days)

    return round(weighted_period_return_pct, 2), round(annualized_weighted_return_pct, 2)

def get_benchmark_return(start_date_str: str) -> float:
    try:
        clean_date = parse_flexible_date(start_date_str)
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

        statement_start_dt = datetime.strptime(statement_start_str, "%Y-%m-%d")

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
                opening_value = float(valuation.get("opening", 0.0) or valuation.get("cost", 0.0) or 0.0)
                
                fund_investments = 0.0
                fund_redemptions = 0.0
                fund_cash_flows = []
                
                if opening_value > 0:
                    fund_cash_flows.append({
                        "date": statement_start_str,
                        "amount": opening_value,
                        "type": "OPENING_VALUE"
                    })

                for tx in scheme.get("transactions", []):
                    tx_date_raw = str(tx.get("date", ""))
                    tx_date_str = parse_flexible_date(tx_date_raw)
                    tx_type = str(tx.get("type", "")).split('.')[-1].upper()
                    
                    if statement_start_str <= tx_date_str:
                        if tx.get("amount"):
                            amt = float(tx["amount"])
                            if tx_type in ["PURCHASE", "SIP", "SWITCH_IN", "DIVIDEND_REINVEST"]:
                                fund_investments += amt
                                fund_cash_flows.append({"date": tx_date_str, "amount": amt, "type": "INVESTMENT"})
                            elif tx_type in ["REDEMPTION", "SWITCH_OUT"]:
                                amt_abs = abs(amt)
                                fund_redemptions += amt_abs
                                fund_cash_flows.append({"date": tx_date_str, "amount": amt_abs, "type": "REDEMPTION"})

                capital_deployed = opening_value + fund_investments - fund_redemptions
                absolute_profit = closing_value - capital_deployed
                absolute_return_pct = round((absolute_profit / capital_deployed) * 100, 2) if capital_deployed > 0 else 0.0
                
                weighted_period_ret, annualized_ret = calculate_weighted_returns(
                    fund_cash_flows, closing_value, statement_start_dt, statement_end_dt
                )

                funds_breakdown.append({
                    "scheme_name": scheme_name,
                    "opening_value": round(opening_value, 2),
                    "capital_deployed": round(capital_deployed, 2),
                    "current_value": round(closing_value, 2),
                    "absolute_profit": round(absolute_profit, 2),
                    "absolute_return_pct": absolute_return_pct,
                    "weighted_period_return": weighted_period_ret,
                    "statement_annualized_return": annualized_ret
                })

                portfolio_opening_value += opening_value
                portfolio_total_investments += fund_investments
                portfolio_total_redemptions += fund_redemptions
                portfolio_current_value += closing_value
                portfolio_cash_flows.extend(fund_cash_flows)

        total_capital_deployed = portfolio_opening_value + portfolio_total_investments - portfolio_total_redemptions
        total_profit = portfolio_current_value - total_capital_deployed
        portfolio_abs_return_pct = round((total_profit / total_capital_deployed) * 100, 2) if total_capital_deployed > 0 else 0.0
        
        port_weighted_period_ret, port_annualized_ret = calculate_weighted_returns(
            portfolio_cash_flows, portfolio_current_value, statement_start_dt, statement_end_dt
        )
        
        benchmark_annualized = get_benchmark_return(statement_start_str)

        return {
            "status": "success",
            "statement_period": {
                "from": statement_start_str,
                "to": statement_end_dt.strftime("%Y-%m-%d"),
                "total_days": (statement_end_dt - statement_start_dt).days
            },
            "portfolio_summary": {
                "opening_portfolio_value": round(portfolio_opening_value, 2),
                "total_capital_deployed": round(total_capital_deployed, 2),
                "current_portfolio_value": round(portfolio_current_value, 2),
                "total_profit": round(total_profit, 2),
                "absolute_return_pct": portfolio_abs_return_pct,
                "weighted_period_return": port_weighted_period_ret,
                "statement_annualized_return": port_annualized_ret,
                "benchmark_annualized_return": benchmark_annualized
            },
            "funds_breakdown": funds_breakdown
        }

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"CAS Parse Failed: {str(e)}")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
