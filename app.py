from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import date, datetime, timedelta
import casparser
from pyxirr import xirr
import json
import os
import httpx

app = FastAPI(title="FundWise API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEBUG_LOGGING = os.getenv("FUNDWISE_DEBUG", "True").lower() in ("true", "1")

# ==========================================
# 1. STRICT RESPONSE MODELS
# ==========================================
class FundBreakdown(BaseModel):
    scheme_name: str
    opening_value: Optional[float]
    fund_investments: float
    fund_redemptions: float
    capital_deployed: Optional[float]
    current_value: float
    units: float
    latest_nav: float
    absolute_profit: Optional[float]
    absolute_return_pct: Optional[float]
    statement_annualized_return: Optional[float]
    resolution_path: str
    is_fully_redeemed: bool

class DataQualityMetrics(BaseModel):
    status: str
    total_funds: int
    resolved_funds: int
    coverage_percentage: float

class PortfolioSummary(BaseModel):
    statement_period: dict
    opening_portfolio_value: Optional[float]
    total_capital_deployed: Optional[float]
    current_portfolio_value: float
    total_profit: Optional[float]
    statement_annualized_return: Optional[float]
    benchmark_annualized_return: Optional[float]
    benchmark_status: str
    data_quality: DataQualityMetrics

class CASResponse(BaseModel):
    portfolio_summary: PortfolioSummary
    funds_breakdown: List[FundBreakdown]
    transactions: list


# ==========================================
# 2. ASYNC & CACHED HELPER FUNCTIONS
# ==========================================
def calculate_xirr(cashflows: list) -> Optional[float]:
    if len(cashflows) < 2:
        return None
    has_pos = any(amt > 0 for _, amt in cashflows)
    has_neg = any(amt < 0 for _, amt in cashflows)
    if not (has_pos and has_neg):
        return None
    try:
        dates = [cf[0] for cf in cashflows]
        amounts = [cf[1] for cf in cashflows]
        result = xirr(dates, amounts)
        if result is not None:
            return min(max(result * 100, -99.99), 10000.0) 
        return None
    except Exception:
        return None

def get_nifty_benchmark(cashflows: list) -> Optional[float]:
    if not os.path.exists("nifty50_history.json"):
        raise FileNotFoundError("nifty50_history.json missing")
    with open("nifty50_history.json", "r") as f:
        nifty_data = json.load(f)
    
    benchmark_cashflows = []
    total_benchmark_units = 0.0

    for dt, amount in cashflows:
        search_date = dt
        attempts = 0
        while search_date.strftime("%Y-%m-%d") not in nifty_data and attempts < 30:
            search_date -= timedelta(days=1)
            attempts += 1
            
        nav_date_str = search_date.strftime("%Y-%m-%d")
        if nav_date_str not in nifty_data:
            return None
            
        nifty_price = nifty_data[nav_date_str]
        if amount < 0:
            units_bought = abs(amount) / nifty_price
            total_benchmark_units += units_bought
            benchmark_cashflows.append((dt, amount))
        elif amount > 0 and dt != cashflows[-1][0]:
            units_sold = amount / nifty_price
            total_benchmark_units -= units_sold
            benchmark_cashflows.append((dt, amount))
        else:
            final_value = total_benchmark_units * nifty_price
            benchmark_cashflows.append((dt, final_value))

    return calculate_xirr(benchmark_cashflows)

def normalize_txn_type(tx_desc: str, tx_type_raw: str) -> str:
    desc = (tx_desc or "").upper()
    raw_type = (tx_type_raw or "").upper()
    combined = f"{desc} {raw_type}"
    
    if "REINVESTMENT" in combined or "DIVIDEND" in combined:
        return "DIVIDEND_REINVESTMENT"
    if "SIP" in combined:
        return "SIP"
    if "SWP" in combined:
        return "SWP"
    if "STP" in combined:
        return "SWITCH_IN" if "IN" in combined or "PURCHASE" in combined else "SWITCH_OUT"
    if "SWITCH" in combined:
        return "SWITCH_IN" if "IN" in combined else "SWITCH_OUT"
    if "PURCHASE" in combined or "LUMPSUM" in combined or "ADDITIONAL" in combined:
        return "PURCHASE"
    if "REDEMPTION" in combined or "SELL" in combined:
        return "REDEMPTION"
    return raw_type

async def fetch_amfi_nav_async(client: httpx.AsyncClient, amfi_code: str, target_date: date, cache: dict) -> Optional[float]:
    """Asynchronous fetcher with request-scoped memoization caching."""
    if not amfi_code:
        return None
    
    cache_key = (str(amfi_code), target_date.strftime("%Y-%m-%d"))
    if cache_key in cache:
        return cache[cache_key]

    try:
        response = await client.get(f"https://api.mfapi.in/mf/{amfi_code}", timeout=5.0)
        if response.status_code != 200:
            return None
        data = response.json()
        nav_history = data.get("data", [])
        search_date = target_date
        for _ in range(7):
            date_str = search_date.strftime("%d-%m-%Y")
            for entry in nav_history:
                if entry["date"] == date_str:
                    val = float(entry["nav"])
                    cache[cache_key] = val
                    return val
            search_date -= timedelta(days=1)
    except Exception as e:
        if DEBUG_LOGGING: print(f"AMFI API error for {amfi_code}: {e}")
    return None

async def resolve_opening_value_async(scheme: dict, stmt_from: date, scheme_name: str, client: httpx.AsyncClient, cache: dict) -> tuple[Optional[float], str]:
    opening_units = scheme.get('open', 0.0)
    if opening_units == 0.0:
        return 0.0, "Zero opening units"
        
    val = scheme.get('opening_value')
    if val is not None and float(val) > 0:
        return float(val), "Explicit opening_value from CAS"
        
    nav = scheme.get('open_nav')
    if nav is not None and float(nav) > 0:
        return opening_units * float(nav), "Calculated via open_nav * units"
        
    amfi_code = scheme.get('amfi')
    if amfi_code:
        amfi_nav = await fetch_amfi_nav_async(client, amfi_code, stmt_from, cache)
        if amfi_nav and amfi_nav > 0:
            return opening_units * amfi_nav, "Resolved via AMFI historical NAV lookup"
    
    for tx in scheme.get('transactions', []):
        tx_nav = tx.get('nav')
        if tx_nav is not None and float(tx_nav) > 0:
            return opening_units * float(tx_nav), "Resolved via earliest transaction NAV proxy"

    return None, "Opening market value unavailable"


# ==========================================
# 3. MAIN API ENDPOINT
# ==========================================
@app.post("/api/v1/parse-cas", response_model=CASResponse)
async def parse_cas_file(file: UploadFile = File(...), password: str = Form("")):
    temp_path = f"/tmp/{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())
            
        raw_parsed = casparser.read_cas_pdf(temp_path, password)
        parsed_data = raw_parsed if isinstance(raw_parsed, dict) else raw_parsed.model_dump()
        
        stmt_from = datetime.strptime(parsed_data['statement_period']['from'], "%b %Y").date() if len(parsed_data['statement_period']['from']) == 8 else parsed_data['statement_period']['from']
        if isinstance(stmt_from, str):
            stmt_from = datetime.strptime(stmt_from, "%Y-%m-%d").date()
        
        raw_to = parsed_data['statement_period']['to']
        if isinstance(raw_to, str):
            stmt_to = datetime.strptime(raw_to, "%Y-%m-%d").date()
        else:
            stmt_to = raw_to
            
        portfolio_opening = 0.0
        portfolio_investments = 0.0
        portfolio_redemptions = 0.0
        portfolio_current = 0.0
        
        total_funds_count = 0
        resolved_funds_count = 0
        resolved_current_value = 0.0
        
        funds_breakdown_list = []
        resolved_xirr_cashflows = []
        all_transactions = []
        amfi_request_cache = {}

        async with httpx.AsyncClient() as client:
            for folio in parsed_data.get('folios', []):
                for scheme in folio.get('schemes', []):
                    total_funds_count += 1
                    scheme_name = scheme.get('scheme', 'Unknown Scheme')
                    current_value = scheme.get('valuation', {}).get('value', 0.0)
                    units = scheme.get('valuation', {}).get('balance', 0.0)
                    latest_nav = scheme.get('valuation', {}).get('nav', 0.0)
                    
                    opening_value, resolution_path = await resolve_opening_value_async(scheme, stmt_from, scheme_name, client, amfi_request_cache)
                    
                    fund_investments = 0.0
                    fund_redemptions = 0.0
                    fund_xirr_cashflows = []
                    
                    is_resolved = opening_value is not None
                    if is_resolved:
                        resolved_funds_count += 1
                        resolved_current_value += current_value
                        if opening_value > 0:
                            fund_xirr_cashflows.append((stmt_from - timedelta(days=1), -opening_value))
                            resolved_xirr_cashflows.append((stmt_from - timedelta(days=1), -opening_value))

                    for tx in scheme.get('transactions', []):
                        tx_date_str = tx['date']
                        if isinstance(tx_date_str, str):
                            tx_date = datetime.strptime(tx_date_str, "%Y-%m-%d").date()
                        else:
                            tx_date = tx_date_str
                            tx['date'] = tx_date.strftime("%Y-%m-%d")

                        if not (stmt_from <= tx_date <= stmt_to):
                            continue

                        tx_desc = tx.get('description', '')
                        tx_type_raw = tx.get('type', '')
                        tx_type = normalize_txn_type(tx_desc, tx_type_raw)
                        
                        tx_amt = abs(tx.get('amount', 0.0))
                        tx['scheme_name'] = scheme_name
                        tx['normalized_type'] = tx_type
                        all_transactions.append(tx)
                        
                        if tx_type in ['PURCHASE', 'SIP', 'SWITCH_IN']:
                            fund_investments += tx_amt
                            if is_resolved:
                                resolved_xirr_cashflows.append((tx_date, -tx_amt))
                        elif tx_type in ['REDEMPTION', 'SWP', 'SWITCH_OUT']:
                            fund_redemptions += tx_amt
                            if is_resolved:
                                resolved_xirr_cashflows.append((tx_date, tx_amt))

                    if is_resolved:
                        capital_deployed = opening_value + fund_investments - fund_redemptions
                        absolute_profit = current_value - capital_deployed
                        absolute_return_pct = (absolute_profit / capital_deployed * 100) if capital_deployed > 0 else 0.0
                        fund_xirr_cashflows.append((stmt_to, current_value))
                        statement_annualized_return = calculate_xirr(fund_xirr_cashflows)
                        
                        portfolio_opening += opening_value
                        portfolio_investments += fund_investments
                        portfolio_redemptions += fund_redemptions
                    else:
                        capital_deployed = None
                        absolute_profit = None
                        absolute_return_pct = None
                        statement_annualized_return = None

                    portfolio_current += current_value
                    is_fully_redeemed = True if (units == 0.0 or current_value == 0.0) else False

                    funds_breakdown_list.append(FundBreakdown(
                        scheme_name=scheme_name,
                        opening_value=opening_value,
                        fund_investments=fund_investments,
                        fund_redemptions=fund_redemptions,
                        capital_deployed=capital_deployed,
                        current_value=current_value,
                        units=units,
                        latest_nav=latest_nav,
                        absolute_profit=absolute_profit,
                        absolute_return_pct=absolute_return_pct,
                        statement_annualized_return=statement_annualized_return,
                        resolution_path=resolution_path,
                        is_fully_redeemed=is_fully_redeemed
                    ))

        coverage_percentage = (resolved_current_value / portfolio_current * 100) if portfolio_current > 0 else 0.0
        if resolved_funds_count == total_funds_count:
            quality_status = "complete"
        elif resolved_funds_count > 0:
            quality_status = "partial"
        else:
            quality_status = "unresolved"

        data_quality = DataQualityMetrics(
            status=quality_status,
            total_funds=total_funds_count,
            resolved_funds=resolved_funds_count,
            coverage_percentage=round(coverage_percentage, 2)
        )

        total_capital_deployed = (portfolio_opening + portfolio_investments - portfolio_redemptions) if resolved_funds_count > 0 else None
        total_profit = (resolved_current_value - total_capital_deployed) if total_capital_deployed is not None else None
        
        resolved_xirr_cashflows.append((stmt_to, resolved_current_value))
        portfolio_annualized_return = calculate_xirr(resolved_xirr_cashflows) if resolved_funds_count > 0 else None
        
        benchmark_annualized_return = None
        benchmark_status = "Available"
        try:
            if resolved_funds_count > 0:
                benchmark_annualized_return = get_nifty_benchmark(resolved_xirr_cashflows)
            else:
                benchmark_status = "Unavailable: No resolved funds for benchmark mapping"
        except Exception as e:
            benchmark_status = f"Unavailable: {str(e)}"

        portfolio_summary = PortfolioSummary(
            statement_period={"from": str(stmt_from), "to": str(stmt_to)},
            opening_portfolio_value=portfolio_opening if resolved_funds_count > 0 else None,
            total_capital_deployed=total_capital_deployed,
            current_portfolio_value=portfolio_current,
            total_profit=total_profit,
            statement_annualized_return=portfolio_annualized_return,
            benchmark_annualized_return=benchmark_annualized_return,
            benchmark_status=benchmark_status,
            data_quality=data_quality
        )

        return CASResponse(
            portfolio_summary=portfolio_summary,
            funds_breakdown=funds_breakdown_list,
            transactions=all_transactions
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
