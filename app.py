from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import date, datetime, timedelta
from decimal import Decimal
import casparser
from pyxirr import xirr
import json
import os
import uuid
import httpx
from collections import defaultdict
import traceback

app = FastAPI(title="FundWise Analytics Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. STRICT RESPONSE MODELS
# ==========================================
class FundBreakdown(BaseModel):
    scheme_name: str
    opening_market_value: Optional[float]
    statement_investments: float
    statement_redemptions: float
    dividend_payouts: float
    stamp_duty_costs: float
    ending_market_value: float
    units: float
    latest_nav: float
    net_wealth_gain: Optional[float]
    statement_return_pct: Optional[float]
    statement_annualized_return: Optional[float]
    nifty_statement_return_pct: Optional[float]
    nifty_annualized_return: Optional[float]
    resolution_path: str
    is_fully_redeemed: bool
    diagnostic_info: dict

class DataQualityMetrics(BaseModel):
    status: str
    total_funds: int
    resolved_funds: int
    coverage_percentage: float

class PortfolioSummary(BaseModel):
    statement_period: dict
    opening_portfolio_value: Optional[float]
    total_statement_investments: float
    total_statement_redemptions: float
    total_dividend_payouts: float
    total_stamp_duty_costs: float
    ending_portfolio_value: float
    net_wealth_gain: Optional[float]
    statement_return_pct: Optional[float]
    statement_annualized_return: Optional[float]
    portfolio_return_status: str
    nifty_statement_return_pct: Optional[float]
    nifty_annualized_return: Optional[float]
    benchmark_status: str
    data_quality: DataQualityMetrics

class CASResponse(BaseModel):
    portfolio_summary: PortfolioSummary
    funds_breakdown: List[FundBreakdown]
    transactions: list


# ==========================================
# 2. ANALYTICS & MATH HELPERS
# ==========================================
def to_float(val, default=0.0) -> float:
    if val is None:
        return default
    if isinstance(val, Decimal):
        return float(val)
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def parse_statement_date(date_str: str) -> date:
    if not date_str:
        raise ValueError("Statement date is missing.")
    date_str_clean = str(date_str).strip()
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%Y-%m-%d", "%b %Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(date_str_clean, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unable to parse statement date explicitly: {date_str}")

def parse_tx_date(date_str: str) -> date:
    if not date_str:
        raise ValueError("Transaction date is missing.")
    date_str_clean = str(date_str).strip()
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%B-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(date_str_clean, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unable to parse transaction date explicitly: {date_str}")

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
            return float(result * 100.0)
        return None
    except Exception:
        return None

def calculate_period_return(cashflows: list) -> Optional[float]:
    annualized = calculate_xirr(cashflows)
    if annualized is None or len(cashflows) < 2:
        return None
        
    days = (cashflows[-1][0] - cashflows[0][0]).days
    if days <= 0:
        return 0.0
        
    r = annualized / 100.0
    period_ret = (((1.0 + r) ** (days / 365.0)) - 1.0) * 100.0
    return float(period_ret)

def replay_nifty_tri_cashflows(transaction_cashflows: list, is_fully_redeemed: bool, valuation_date: date) -> tuple[Optional[float], Optional[float]]:
    if not os.path.exists("nifty50_history.json"):
        return None, None
        
    with open("nifty50_history.json", "r") as f:
        nifty_data = json.load(f)
    
    benchmark_cashflows = []
    total_benchmark_units = 0.0

    def get_nifty_price(target_date: date) -> Optional[float]:
        search_date = target_date
        for _ in range(30):
            nav_date_str = search_date.strftime("%Y-%m-%d")
            if nav_date_str in nifty_data:
                return to_float(nifty_data[nav_date_str])
            search_date -= timedelta(days=1)
        return None

    for dt, amount in transaction_cashflows:
        nifty_price = get_nifty_price(dt)
        if nifty_price is None:
            return None, None
            
        if amount < 0:
            units_bought = abs(amount) / nifty_price
            total_benchmark_units += units_bought
            benchmark_cashflows.append((dt, amount))
        elif amount > 0:
            units_sold = amount / nifty_price
            total_benchmark_units -= units_sold
            benchmark_cashflows.append((dt, amount))

    if not is_fully_redeemed:
        final_price = get_nifty_price(valuation_date)
        if final_price is not None:
            final_value = total_benchmark_units * final_price
            benchmark_cashflows.append((valuation_date, final_value))

    ann_ret = calculate_xirr(benchmark_cashflows)
    per_ret = calculate_period_return(benchmark_cashflows)
    return per_ret, ann_ret

def normalize_txn_type(tx_desc: str, tx_type_raw: str) -> str:
    desc = (tx_desc or "").upper()
    raw_type = (tx_type_raw or "").upper()
    combined = f"{desc} {raw_type}"
    tokens = set(combined.split())
    
    if "STAMP" in combined and "DUTY" in combined:
        return "STAMP_DUTY"
    if "REINVESTMENT" in combined:
        return "DIVIDEND_REINVESTMENT"
    if "DIVIDEND" in combined and ("PAYOUT" in combined or "ISSUED" in combined or "TRANSFER" in combined):
        return "DIVIDEND_PAYOUT"
    if "LATERAL" in combined and "SHIFT" in combined:
        if "IN" in tokens:
            return "SWITCH_IN"
        if "OUT" in tokens:
            return "SWITCH_OUT"
    if "SIP" in tokens:
        return "SIP"
    if "SWP" in tokens:
        return "SWP"
    if "SYSTEMATIC" in combined and "TRANSFER" in combined or "STP" in tokens:
        if "IN" in tokens or "PURCHASE" in tokens:
            return "SWITCH_IN"
        if "OUT" in tokens or "REDEMPTION" in tokens or "SELL" in tokens:
            return "SWITCH_OUT"
    if "SWITCH" in combined:
        if "IN" in tokens:
            return "SWITCH_IN"
        if "OUT" in tokens:
            return "SWITCH_OUT"
    if "PURCHASE" in combined or "LUMPSUM" in combined or "ADDITIONAL" in combined:
        return "PURCHASE"
    if "REDEMPTION" in combined or "SELL" in combined:
        return "REDEMPTION"
    return raw_type

async def fetch_amfi_nav_async(client: httpx.AsyncClient, amfi_code: str, target_date: date, cache: dict) -> Optional[float]:
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
                    val = to_float(entry.get("nav"))
                    cache[cache_key] = val
                    return val
            search_date -= timedelta(days=1)
    except Exception:
        pass
    return None

async def resolve_opening_market_value(scheme: dict, stmt_from: date, scheme_name: str, client: httpx.AsyncClient, cache: dict) -> tuple[Optional[float], str]:
    opening_units = to_float(scheme.get('open', 0.0))
    if opening_units == 0.0:
        return 0.0, "Zero opening units"
        
    val = scheme.get('opening_value')
    if val is not None and to_float(val) > 0:
        return to_float(val), "Explicit opening_value from CAS"
        
    nav = scheme.get('open_nav')
    if nav is not None and to_float(nav) > 0:
        return opening_units * to_float(nav), "Calculated via open_nav * units"
        
    amfi_code = scheme.get('amfi')
    if amfi_code:
        amfi_nav = await fetch_amfi_nav_async(client, amfi_code, stmt_from, cache)
        if amfi_nav and amfi_nav > 0:
            return opening_units * amfi_nav, "Resolved via AMFI historical NAV lookup"

    return None, "Opening market value unresolved: earliest-transaction NAV proxy disabled."


# ==========================================
# 3. MAIN ANALYTICS ENDPOINT 
# ==========================================
@app.post("/api/v1/parse-cas", response_model=CASResponse)
async def parse_cas_file(file: UploadFile = File(...), password: str = Form("")):
    sanitized_filename = f"{uuid.uuid4()}_{os.path.basename(file.filename or 'statement.pdf')}"
    temp_path = os.path.join("/tmp", sanitized_filename)
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())
            
        raw_parsed = casparser.read_cas_pdf(temp_path, password)
        parsed_data = raw_parsed if isinstance(raw_parsed, dict) else raw_parsed.model_dump()
        
        raw_period = parsed_data.get('statement_period', {})
        from_val = raw_period.get('from') or raw_period.get('start_date')
        if not from_val:
            raise HTTPException(status_code=400, detail="Statement start date missing from CAS data.")
        stmt_from = parse_statement_date(from_val)
            
        to_val = raw_period.get('to') or raw_period.get('end_date')
        if not to_val:
            raise HTTPException(status_code=400, detail="Statement end date missing from CAS data.")
        stmt_to = parse_statement_date(to_val)
            
        total_funds_count = 0
        resolved_funds_count = 0
        resolved_current_value = 0.0
        portfolio_current_value = 0.0
        
        total_portfolio_investments = 0.0
        total_portfolio_redemptions = 0.0
        total_portfolio_dividends = 0.0
        total_portfolio_stamp_duty = 0.0
        portfolio_opening_val = 0.0
        
        funds_breakdown_list = []
        portfolio_daily_cashflows = defaultdict(float)
        resolved_fund_units = []
        resolved_valuation_dates = set()
        all_transactions = []
        amfi_request_cache = {}

        async with httpx.AsyncClient() as client:
            for folio in parsed_data.get('folios', []):
                for scheme in folio.get('schemes', []):
                    total_funds_count += 1
                    scheme_name = scheme.get('scheme', 'Unknown Scheme')
                    valuation_data = scheme.get('valuation', {})
                    ending_market_value = to_float(valuation_data.get('value', 0.0))
                    units = to_float(valuation_data.get('balance', 0.0))
                    latest_nav = to_float(valuation_data.get('nav', 0.0))
                    
                    val_date_raw = valuation_data.get('date')
                    if not val_date_raw:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Valuation date missing for scheme: {scheme_name}"
                        )
                    valuation_date = parse_statement_date(val_date_raw)
                    
                    opening_market_value, resolution_path = await resolve_opening_market_value(scheme, stmt_from, scheme_name, client, amfi_request_cache)
                    
                    statement_investments = 0.0
                    statement_redemptions = 0.0
                    dividend_payouts = 0.0
                    stamp_duty_costs = 0.0
                    
                    fund_transaction_cashflows = []
                    is_resolved = opening_market_value is not None
                    
                    if is_resolved:
                        resolved_funds_count += 1
                        resolved_current_value += ending_market_value
                        resolved_fund_units.append(units)
                        resolved_valuation_dates.add(valuation_date)
                        
                        if opening_market_value > 0:
                            open_dt = stmt_from - timedelta(days=1)
                            fund_transaction_cashflows.append((open_dt, -opening_market_value))
                            portfolio_daily_cashflows[open_dt] -= opening_market_value
                            portfolio_opening_val += opening_market_value

                    for tx in scheme.get('transactions', []):
                        tx_date_str = tx.get('date')
                        if not tx_date_str:
                            raise HTTPException(status_code=400, detail=f"Transaction date missing in scheme {scheme_name}")
                        tx_date = parse_tx_date(tx_date_str)

                        if not (stmt_from <= tx_date <= stmt_to):
                            continue

                        tx_desc = tx.get('description', '')
                        tx_type_raw = tx.get('type', '')
                        tx_type = normalize_txn_type(tx_desc, tx_type_raw)
                        tx_amt = abs(to_float(tx.get('amount', 0.0)))
                        
                        normalized_tx = tx.copy()
                        normalized_tx['date'] = tx_date.strftime("%Y-%m-%d")
                        normalized_tx['scheme_name'] = scheme_name
                        normalized_tx['normalized_type'] = tx_type
                        all_transactions.append(normalized_tx)

                        if tx_type == 'DIVIDEND_REINVESTMENT':
                            continue
                        elif tx_type == 'STAMP_DUTY':
                            stamp_duty_costs += tx_amt
                            if is_resolved:
                                fund_transaction_cashflows.append((tx_date, -tx_amt))
                                portfolio_daily_cashflows[tx_date] -= tx_amt
                        elif tx_type in ['PURCHASE', 'SIP', 'SWITCH_IN']:
                            statement_investments += tx_amt
                            if is_resolved:
                                fund_transaction_cashflows.append((tx_date, -tx_amt))
                                portfolio_daily_cashflows[tx_date] -= tx_amt
                        elif tx_type in ['REDEMPTION', 'SWP', 'SWITCH_OUT']:
                            statement_redemptions += tx_amt
                            if is_resolved:
                                fund_transaction_cashflows.append((tx_date, tx_amt))
                                portfolio_daily_cashflows[tx_date] += tx_amt
                        elif tx_type == 'DIVIDEND_PAYOUT':
                            dividend_payouts += tx_amt
                            if is_resolved:
                                fund_transaction_cashflows.append((tx_date, tx_amt))
                                portfolio_daily_cashflows[tx_date] += tx_amt

                    is_fully_redeemed = units < 0.001

                    if is_resolved:
                        fund_xirr_cashflows = list(fund_transaction_cashflows)
                        
                        if not is_fully_redeemed:
                            fund_xirr_cashflows.append((valuation_date, ending_market_value))
                        
                        net_wealth_gain = (ending_market_value + statement_redemptions + dividend_payouts) - (opening_market_value + statement_investments + stamp_duty_costs)
                        statement_return_pct = calculate_period_return(fund_xirr_cashflows)
                        statement_annualized_return = calculate_xirr(fund_xirr_cashflows)
                        
                        nifty_per_ret, nifty_ann_ret = replay_nifty_tri_cashflows(fund_transaction_cashflows, is_fully_redeemed, valuation_date)
                        
                        total_portfolio_investments += statement_investments
                        total_portfolio_redemptions += statement_redemptions
                        total_portfolio_dividends += dividend_payouts
                        total_portfolio_stamp_duty += stamp_duty_costs
                    else:
                        net_wealth_gain = None
                        statement_return_pct = None
                        statement_annualized_return = None
                        nifty_per_ret = None
                        nifty_ann_ret = None

                    portfolio_current_value += ending_market_value

                    diagnostic_info = {
                        "opening_date": str(stmt_from - timedelta(days=1)),
                        "opening_value": opening_market_value,
                        "valuation_date": str(valuation_date),
                        "valuation_value": ending_market_value,
                        "transactions": [
                            {"date": str(d), "amount": a} for d, a in fund_transaction_cashflows
                        ],
                        "final_xirr_cashflow_array": [
                            {"date": str(d), "amount": a} for d, a in (fund_xirr_cashflows if is_resolved else [])
                        ]
                    }

                    funds_breakdown_list.append(FundBreakdown(
                        scheme_name=scheme_name,
                        opening_market_value=opening_market_value,
                        statement_investments=statement_investments,
                        statement_redemptions=statement_redemptions,
                        dividend_payouts=dividend_payouts,
                        stamp_duty_costs=stamp_duty_costs,
                        ending_market_value=ending_market_value,
                        units=units,
                        latest_nav=latest_nav,
                        net_wealth_gain=net_wealth_gain,
                        statement_return_pct=statement_return_pct,
                        statement_annualized_return=statement_annualized_return,
                        nifty_statement_return_pct=nifty_per_ret,
                        nifty_annualized_return=nifty_ann_ret,
                        resolution_path=resolution_path,
                        is_fully_redeemed=is_fully_redeemed,
                        diagnostic_info=diagnostic_info
                    ))

        coverage_percentage = (resolved_current_value / portfolio_current_value * 100) if portfolio_current_value > 0 else 0.0
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

        if resolved_funds_count == total_funds_count and total_funds_count > 0:
            portfolio_return_status = "Complete"
        elif resolved_funds_count > 0:
            portfolio_return_status = "Partial (Resolved Subset Only)"
        else:
            portfolio_return_status = "Unavailable"

        if resolved_funds_count > 0:
            if len(resolved_valuation_dates) > 1:
                raise HTTPException(status_code=400, detail="All resolved funds must share the same valuation date for portfolio calculation.")
            portfolio_valuation_date = list(resolved_valuation_dates)[0] if resolved_valuation_dates else stmt_to

            portfolio_transaction_cashflows = []
            for dt in sorted(portfolio_daily_cashflows.keys()):
                amt = portfolio_daily_cashflows[dt]
                if abs(amt) > 0.001:
                    portfolio_transaction_cashflows.append((dt, amt))

            port_is_fully_redeemed = (
                resolved_funds_count == total_funds_count and 
                total_funds_count > 0 and 
                all(u < 0.001 for u in resolved_fund_units)
            )
            
            portfolio_xirr_cashflows = list(portfolio_transaction_cashflows)
            if not port_is_fully_redeemed:
                portfolio_xirr_cashflows.append((portfolio_valuation_date, resolved_current_value))
            
            portfolio_net_wealth_gain = (resolved_current_value + total_portfolio_redemptions + total_portfolio_dividends) - (portfolio_opening_val + total_portfolio_investments + total_portfolio_stamp_duty)
            portfolio_statement_return = calculate_period_return(portfolio_xirr_cashflows)
            portfolio_annualized_return = calculate_xirr(portfolio_xirr_cashflows)
            
            nifty_port_per, nifty_port_ann = replay_nifty_tri_cashflows(portfolio_transaction_cashflows, port_is_fully_redeemed, portfolio_valuation_date)
        else:
            portfolio_net_wealth_gain = None
            portfolio_statement_return = None
            portfolio_annualized_return = None
            nifty_port_per = None
            nifty_port_ann = None

        if resolved_funds_count == 0:
            benchmark_status = "Unavailable: No resolved funds for benchmark mapping"
        elif nifty_port_per is None or nifty_port_ann is None:
            benchmark_status = "Unavailable: Nifty price data could not fully resolve the transaction stream"
        else:
            benchmark_status = "Available"

        portfolio_summary = PortfolioSummary(
            statement_period={"from": str(stmt_from), "to": str(stmt_to)},
            opening_portfolio_value=portfolio_opening_val if resolved_funds_count > 0 else None,
            total_statement_investments=total_portfolio_investments,
            total_statement_redemptions=total_portfolio_redemptions,
            total_dividend_payouts=total_portfolio_dividends,
            total_stamp_duty_costs=total_portfolio_stamp_duty,
            ending_portfolio_value=portfolio_current_value,
            net_wealth_gain=portfolio_net_wealth_gain,
            statement_return_pct=portfolio_statement_return,
            statement_annualized_return=portfolio_annualized_return,
            portfolio_return_status=portfolio_return_status,
            nifty_statement_return_pct=nifty_port_per,
            nifty_annualized_return=nifty_port_ann,
            benchmark_status=benchmark_status,
            data_quality=data_quality
        )

        response_obj = CASResponse(
            portfolio_summary=portfolio_summary,
            funds_breakdown=funds_breakdown_list,
            transactions=all_transactions
        )

        return response_obj

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=400,
            content={"message": str(e), "detail": str(e)}
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
