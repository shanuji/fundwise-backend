from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import date, datetime, timedelta
from decimal import Decimal
import casparser
from pyxirr import xirr
import json
import os
import httpx

app = FastAPI(title="FundWise Analytics Engine")

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
    opening_market_value: Optional[float]
    statement_investments: float
    statement_redemptions: float
    dividend_payouts: float
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
    ending_portfolio_value: float
    net_wealth_gain: Optional[float]
    statement_return_pct: Optional[float]
    statement_annualized_return: Optional[float]
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

def calculate_period_return(cashflows: list, start_date: date, end_date: date) -> Optional[float]:
    annualized = calculate_xirr(cashflows)
    if annualized is None:
        return None
    days = (end_date - start_date).days
    if days <= 0:
        return 0.0
    r = annualized / 100.0
    period_ret = (((1.0 + r) ** (days / 365.0)) - 1.0) * 100.0
    return min(max(period_ret, -99.99), 100000.0)

def replay_nifty_tri_cashflows(cashflows: list, start_date: date, end_date: date) -> tuple[Optional[float], Optional[float]]:
    if not os.path.exists("nifty50_history.json"):
        return None, None
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
            return None, None
            
        nifty_price = to_float(nifty_data[nav_date_str])
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

    ann_ret = calculate_xirr(benchmark_cashflows)
    per_ret = calculate_period_return(benchmark_cashflows, start_date, end_date)
    return per_ret, ann_ret

def normalize_txn_type(tx_desc: str, tx_type_raw: str) -> str:
    desc = (tx_desc or "").upper()
    raw_type = (tx_type_raw or "").upper()
    combined = f"{desc} {raw_type}"
    
    if "REINVESTMENT" in combined:
        return "DIVIDEND_REINVESTMENT"
    if "DIVIDEND" in combined and ("PAYOUT" in combined or "ISSUED" in combined or "TRANSFER" in combined):
        return "DIVIDEND_PAYOUT"
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
    
    for tx in scheme.get('transactions', []):
        tx_nav = tx.get('nav')
        if tx_nav is not None and to_float(tx_nav) > 0:
            return opening_units * to_float(tx_nav), "Resolved via earliest transaction NAV proxy"

    return None, "Opening market value unavailable with mathematical certainty"


# ==========================================
# 3. MAIN ANALYTICS ENDPOINT WITH DIAGNOSTICS
# ==========================================
@app.post("/api/v1/parse-cas", response_model=CASResponse)
async def parse_cas_file(file: UploadFile = File(...), password: str = Form("")):
    temp_path = f"/tmp/{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())
            
        raw_parsed = casparser.read_cas_pdf(temp_path, password)
        parsed_data = raw_parsed if isinstance(raw_parsed, dict) else raw_parsed.model_dump()
        
        raw_period = parsed_data.get('statement_period', {})
        from_val = raw_period.get('from') or raw_period.get('start_date') or "01-Apr-2026"
        if isinstance(from_val, str):
            try:
                stmt_from = datetime.strptime(from_val, "%b %Y").date()
            except ValueError:
                try:
                    stmt_from = datetime.strptime(from_val, "%Y-%m-%d").date()
                except ValueError:
                    stmt_from = date(2026, 4, 1)
        else:
            stmt_from = from_val
            
        to_val = raw_period.get('to') or raw_period.get('end_date') or "05-Aug-2026"
        if isinstance(to_val, str):
            try:
                stmt_to = datetime.strptime(to_val, "%b %Y").date()
            except ValueError:
                try:
                    stmt_to = datetime.strptime(to_val, "%Y-%m-%d").date()
                except ValueError:
                    stmt_to = date(2026, 8, 5)
        else:
            stmt_to = to_val
            
        total_funds_count = 0
        resolved_funds_count = 0
        resolved_current_value = 0.0
        portfolio_current_value = 0.0
        
        total_portfolio_investments = 0.0
        total_portfolio_redemptions = 0.0
        total_portfolio_dividends = 0.0
        portfolio_opening_val = 0.0
        
        funds_breakdown_list = []
        resolved_portfolio_cashflows = []
        all_transactions = []
        amfi_request_cache = {}

        async with httpx.AsyncClient() as client:
            for folio in parsed_data.get('folios', []):
                for scheme in folio.get('schemes', []):
                    total_funds_count += 1
                    scheme_name = scheme.get('scheme', 'Unknown Scheme')
                    ending_market_value = to_float(scheme.get('valuation', {}).get('value', 0.0))
                    units = to_float(scheme.get('valuation', {}).get('balance', 0.0))
                    latest_nav = to_float(scheme.get('valuation', {}).get('nav', 0.0))
                    
                    # ==========================================
                    # CHECKPOINT 1: casparser Extraction Log
                    # ==========================================
                    print("======== FUND TRIALS (CHECKPOINT 1) ========")
                    print(f"Scheme name: {scheme_name}")
                    print(f"Opening units: {scheme.get('open')}")
                    print(f"Opening NAV: {scheme.get('open_nav')}")
                    print(f"Opening market value: {scheme.get('opening_value')}")
                    print(f"Closing units: {units}")
                    print(f"Closing NAV: {latest_nav}")
                    print(f"Closing market value: {ending_market_value}")
                    print(f"AMFI code: {scheme.get('amfi')}")
                    print(f"Transaction count: {len(scheme.get('transactions', []))}")
                    print("============================================")

                    opening_market_value, resolution_path = await resolve_opening_market_value(scheme, stmt_from, scheme_name, client, amfi_request_cache)
                    
                    statement_investments = 0.0
                    statement_redemptions = 0.0
                    dividend_payouts = 0.0
                    fund_cashflows = []
                    
                    is_resolved = opening_market_value is not None
                    
                    if is_resolved:
                        resolved_funds_count += 1
                        resolved_current_value += ending_market_value
                        if opening_market_value > 0:
                            fund_cashflows.append((stmt_from - timedelta(days=1), -opening_market_value))
                            resolved_portfolio_cashflows.append((stmt_from - timedelta(days=1), -opening_market_value))
                            portfolio_opening_val += opening_market_value

                    for tx in scheme.get('transactions', []):
                        tx_date_str = tx.get('date')
                        if not tx_date_str:
                            continue
                        if isinstance(tx_date_str, str):
                            try:
                                tx_date = datetime.strptime(tx_date_str, "%Y-%m-%d").date()
                            except ValueError:
                                continue
                        else:
                            tx_date = tx_date_str
                            tx['date'] = tx_date.strftime("%Y-%m-%d")

                        if not (stmt_from <= tx_date <= stmt_to):
                            continue

                        tx_desc = tx.get('description', '')
                        tx_type_raw = tx.get('type', '')
                        tx_type = normalize_txn_type(tx_desc, tx_type_raw)
                        
                        tx_amt = abs(to_float(tx.get('amount', 0.0)))
                        tx['scheme_name'] = scheme_name
                        tx['normalized_type'] = tx_type
                        all_transactions.append(tx)
                        
                        if tx_type in ['PURCHASE', 'SIP', 'SWITCH_IN']:
                            statement_investments += tx_amt
                            if is_resolved:
                                fund_cashflows.append((tx_date, -tx_amt))
                                resolved_portfolio_cashflows.append((tx_date, -tx_amt))
                        elif tx_type in ['REDEMPTION', 'SWP', 'SWITCH_OUT']:
                            statement_redemptions += tx_amt
                            if is_resolved:
                                fund_cashflows.append((tx_date, tx_amt))
                                resolved_portfolio_cashflows.append((tx_date, tx_amt))
                        elif tx_type == 'DIVIDEND_PAYOUT':
                            dividend_payouts += tx_amt
                            if is_resolved:
                                fund_cashflows.append((tx_date, tx_amt))
                                resolved_portfolio_cashflows.append((tx_date, tx_amt))

                    if is_resolved:
                        fund_cashflows.append((stmt_to, ending_market_value))
                        
                        net_wealth_gain = (ending_market_value + statement_redemptions + dividend_payouts) - (opening_market_value + statement_investments)
                        statement_return_pct = calculate_period_return(fund_cashflows, stmt_from, stmt_to)
                        statement_annualized_return = calculate_xirr(fund_cashflows)
                        
                        nifty_per_ret, nifty_ann_ret = replay_nifty_tri_cashflows(fund_cashflows, stmt_from, stmt_to)
                        
                        total_portfolio_investments += statement_investments
                        total_portfolio_redemptions += statement_redemptions
                        total_portfolio_dividends += dividend_payouts
                    else:
                        net_wealth_gain = None
                        statement_return_pct = None
                        statement_annualized_return = None
                        nifty_per_ret = None
                        nifty_ann_ret = None

                    # ==========================================
                    # CHECKPOINT 2: Analytics Output Log
                    # ==========================================
                    print(f"======== FUND CALCULATIONS (CHECKPOINT 2): {scheme_name} =======")
                    print(f"Opening Market Value : {opening_market_value}")
                    print(f"Statement Investments: {statement_investments}")
                    print(f"Statement Redemptions: {statement_redemptions}")
                    print(f"Dividend Payouts     : {dividend_payouts}")
                    print(f"Ending Market Value  : {ending_market_value}")
                    print(f"Net Wealth Gain      : {net_wealth_gain}")
                    print(f"Statement Return     : {statement_return_pct}")
                    print(f"Annualized Return    : {statement_annualized_return}")
                    print(f"Nifty Statement Return: {nifty_per_ret}")
                    print("----------------------------------------------------------------\n")

                    portfolio_current_value += ending_market_value
                    is_fully_redeemed = True if (units == 0.0 or ending_market_value == 0.0) else False

                    funds_breakdown_list.append(FundBreakdown(
                        scheme_name=scheme_name,
                        opening_market_value=opening_market_value,
                        statement_investments=statement_investments,
                        statement_redemptions=statement_redemptions,
                        dividend_payouts=dividend_payouts,
                        ending_market_value=ending_market_value,
                        units=units,
                        latest_nav=latest_nav,
                        net_wealth_gain=net_wealth_gain,
                        statement_return_pct=statement_return_pct,
                        statement_annualized_return=statement_annualized_return,
                        nifty_statement_return_pct=nifty_per_ret,
                        nifty_annualized_return=nifty_ann_ret,
                        resolution_path=resolution_path,
                        is_fully_redeemed=is_fully_redeemed
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

        if resolved_funds_count > 0:
            resolved_portfolio_cashflows.append((stmt_to, resolved_current_value))
            portfolio_net_wealth_gain = (resolved_current_value + total_portfolio_redemptions + total_portfolio_dividends) - (portfolio_opening_val + total_portfolio_investments)
            portfolio_statement_return = calculate_period_return(resolved_portfolio_cashflows, stmt_from, stmt_to)
            portfolio_annualized_return = calculate_xirr(resolved_portfolio_cashflows)
            nifty_port_per, nifty_port_ann = replay_nifty_tri_cashflows(resolved_portfolio_cashflows, stmt_from, stmt_to)
        else:
            portfolio_net_wealth_gain = None
            portfolio_statement_return = None
            portfolio_annualized_return = None
            nifty_port_per = None
            nifty_port_ann = None

        benchmark_status = "Available"
        try:
            if resolved_funds_count == 0:
                benchmark_status = "Unavailable: No resolved funds for benchmark mapping"
        except Exception as e:
            benchmark_status = f"Unavailable: {str(e)}"

        portfolio_summary = PortfolioSummary(
            statement_period={"from": str(stmt_from), "to": str(stmt_to)},
            opening_portfolio_value=portfolio_opening_val if resolved_funds_count > 0 else None,
            total_statement_investments=total_portfolio_investments,
            total_statement_redemptions=total_portfolio_redemptions,
            total_dividend_payouts=total_portfolio_dividends,
            ending_portfolio_value=portfolio_current_value,
            net_wealth_gain=portfolio_net_wealth_gain,
            statement_return_pct=portfolio_statement_return,
            statement_annualized_return=portfolio_annualized_return,
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

        # ==========================================
        # CHECKPOINT 3: API Response JSON & Keys Log (Resilient)
        # ==========================================
        payload = response_obj.model_dump()
        print("========== CHECKPOINT 3: API RESPONSE ==========")
        print("Top-level keys:", list(payload.keys()))
        if "portfolio_summary" in payload:
            print("Portfolio keys:", list(payload["portfolio_summary"].keys()))
        if payload.get("funds_breakdown"):
            print("Fund keys:", list(payload["funds_breakdown"][0].keys()))
            print("Sample fund:", json.dumps(payload["funds_breakdown"][0], indent=2, default=str))
        else:
            print("No funds returned.")
        print(json.dumps(payload, indent=2, default=str))
        print("================================================")

        return response_obj

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
