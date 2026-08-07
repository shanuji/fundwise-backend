from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import date, datetime, timedelta
import casparser
from pyxirr import xirr
import json
import os

app = FastAPI(title="FundWise API")

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
    opening_value: float
    fund_investments: float
    fund_redemptions: float
    capital_deployed: float
    current_value: float
    units: float
    latest_nav: float
    absolute_profit: float
    absolute_return_pct: float
    statement_annualized_return: Optional[float]
    resolution_path: str
    is_fully_redeemed: bool

class PortfolioSummary(BaseModel):
    statement_period: dict
    opening_portfolio_value: float
    total_capital_deployed: float
    current_portfolio_value: float
    total_profit: float
    statement_annualized_return: Optional[float]
    benchmark_annualized_return: Optional[float]
    benchmark_status: str

class CASResponse(BaseModel):
    portfolio_summary: PortfolioSummary
    funds_breakdown: List[FundBreakdown]
    transactions: list


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def calculate_xirr(cashflows: list) -> Optional[float]:
    """Calculates XIRR given a list of (date, amount) tuples."""
    if len(cashflows) < 2:
        return None
    
    # Check if we have both positive and negative cashflows
    has_pos = any(amt > 0 for _, amt in cashflows)
    has_neg = any(amt < 0 for _, amt in cashflows)
    if not (has_pos and has_neg):
        return None

    try:
        dates = [cf[0] for cf in cashflows]
        amounts = [cf[1] for cf in cashflows]
        result = xirr(dates, amounts)
        # Convert to percentage and cap wild values
        if result is not None:
            return min(max(result * 100, -99.99), 10000.0) 
        return None
    except Exception:
        return None

def get_nifty_benchmark(cashflows: list) -> Optional[float]:
    """Calculates benchmark return using historical Nifty 50 data."""
    if not os.path.exists("nifty50_history.json"):
        raise FileNotFoundError("nifty50_history.json missing")
    
    with open("nifty50_history.json", "r") as f:
        nifty_data = json.load(f)
        
    benchmark_cashflows = []
    total_benchmark_units = 0.0

    for dt, amount in cashflows:
        date_str = dt.strftime("%Y-%m-%d")
        
        # Fallback to nearest available previous date if exact date is weekend/holiday
        search_date = dt
        while search_date.strftime("%Y-%m-%d") not in nifty_data and search_date > dt - timedelta(days=7):
            search_date -= timedelta(days=1)
            
        nav_date_str = search_date.strftime("%Y-%m-%d")
        if nav_date_str not in nifty_data:
            raise ValueError(f"No benchmark data near {date_str}")
            
        nifty_price = nifty_data[nav_date_str]
        
        if amount < 0:  # Investment
            units_bought = abs(amount) / nifty_price
            total_benchmark_units += units_bought
            benchmark_cashflows.append((dt, amount))
        elif amount > 0 and dt != cashflows[-1][0]:  # Redemption
            units_sold = amount / nifty_price
            total_benchmark_units -= units_sold
            benchmark_cashflows.append((dt, amount))
        else: # Final evaluation
            final_value = total_benchmark_units * nifty_price
            benchmark_cashflows.append((dt, final_value))

    return calculate_xirr(benchmark_cashflows)


# ==========================================
# 3. MAIN API ENDPOINT
# ==========================================
@app.post("/api/v1/parse-cas", response_model=CASResponse)
async def parse_cas_file(file: UploadFile = File(...), password: str = Form("")):
    temp_path = f"/tmp/{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())
            
        parsed_data = casparser.read_cas_pdf(temp_path, password)
        
        stmt_from = datetime.strptime(parsed_data['statement_period']['from'], "%b %Y").date() if len(parsed_data['statement_period']['from']) == 8 else parsed_data['statement_period']['from']
        
        # Determine exact statement boundaries (handling varying casparser date formats)
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
        
        funds_breakdown_list = []
        portfolio_xirr_cashflows = []
        all_transactions = []
        
        # 1. PARSE EACH FUND
        for folio in parsed_data.get('folios', []):
            for scheme in folio.get('schemes', []):
                scheme_name = scheme.get('scheme', 'Unknown Scheme')
                current_value = scheme.get('valuation', {}).get('value', 0.0)
                units = scheme.get('valuation', {}).get('balance', 0.0)
                latest_nav = scheme.get('valuation', {}).get('nav', 0.0)
                
                opening_value = scheme.get('open', 0.0)
                
                fund_investments = 0.0
                fund_redemptions = 0.0
                fund_xirr_cashflows = []
                
                # Treat opening balance as initial capital deployment
                if opening_value > 0:
                    fund_xirr_cashflows.append((stmt_from - timedelta(days=1), -opening_value))
                    portfolio_xirr_cashflows.append((stmt_from - timedelta(days=1), -opening_value))

                # Process Transactions strictly for the statement period
                for tx in scheme.get('transactions', []):
                    tx_date = datetime.strptime(tx['date'], "%Y-%m-%d").date()
                    tx_amt = tx.get('amount', 0.0)
                    tx_type = tx.get('type', '')
                    
                    tx['scheme_name'] = scheme_name
                    all_transactions.append(tx)
                    
                    # IGNORE Dividend Reinvestments for Capital Deployed
                    if tx_type in ['PURCHASE', 'PURCHASE_SIP', 'SIP', 'SWITCH_IN', 'LUMPSUM']:
                        fund_investments += tx_amt
                        fund_xirr_cashflows.append((tx_date, -tx_amt))
                        portfolio_xirr_cashflows.append((tx_date, -tx_amt))
                        
                    elif tx_type in ['REDEMPTION', 'SWP', 'SWITCH_OUT']:
                        fund_redemptions += abs(tx_amt)
                        fund_xirr_cashflows.append((tx_date, abs(tx_amt)))
                        portfolio_xirr_cashflows.append((tx_date, abs(tx_amt)))

                # Fund Level Math
                capital_deployed = opening_value + fund_investments - fund_redemptions
                absolute_profit = current_value - capital_deployed
                absolute_return_pct = (absolute_profit / capital_deployed * 100) if capital_deployed > 0 else 0.0
                
                # Fund XIRR evaluation
                fund_xirr_cashflows.append((stmt_to, current_value))
                fund_annualized_return = calculate_xirr(fund_xirr_cashflows)

                # Fully Redeemed Flag
                is_fully_redeemed = True if (units < 0.001 or current_value < 1.0) else False

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
                    statement_annualized_return=fund_annualized_return,
                    resolution_path="Strict Parser",
                    is_fully_redeemed=is_fully_redeemed
                ))
                
                # Aggregate for Portfolio
                portfolio_opening += opening_value
                portfolio_investments += fund_investments
                portfolio_redemptions += fund_redemptions
                portfolio_current += current_value

        # 2. PORTFOLIO LEVEL MATH
        total_capital_deployed = portfolio_opening + portfolio_investments - portfolio_redemptions
        total_profit = portfolio_current - total_capital_deployed
        
        portfolio_xirr_cashflows.append((stmt_to, portfolio_current))
        portfolio_annualized_return = calculate_xirr(portfolio_xirr_cashflows)
        
        # 3. BENCHMARK LOGIC
        benchmark_annualized_return = None
        benchmark_status = "Available"
        try:
            benchmark_annualized_return = get_nifty_benchmark(portfolio_xirr_cashflows)
        except Exception as e:
            benchmark_status = f"Unavailable: {str(e)}"

        portfolio_summary = PortfolioSummary(
            statement_period={"from": str(stmt_from), "to": str(stmt_to)},
            opening_portfolio_value=portfolio_opening,
            total_capital_deployed=total_capital_deployed,
            current_portfolio_value=portfolio_current,
            total_profit=total_profit,
            statement_annualized_return=portfolio_annualized_return,
            benchmark_annualized_return=benchmark_annualized_return,
            benchmark_status=benchmark_status
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
