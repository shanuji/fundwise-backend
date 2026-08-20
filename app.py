from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import date, datetime, timedelta
from decimal import Decimal
import casparser
import json
import os
import uuid
import httpx
from collections import defaultdict
import traceback

app = FastAPI(title="FundWise Analytics Engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class FundBreakdown(BaseModel):
    scheme_name: str
    opening_market_value: Optional[float]
    statement_investments: float = 0
    statement_redemptions: float = 0
    dividend_payouts: float = 0
    stamp_duty_costs: float = 0
    ending_market_value: float
    units: float = 0
    latest_nav: float = 0
    net_wealth_gain: Optional[float] = None
    statement_return_pct: Optional[float] = None
    statement_annualized_return: Optional[float] = None
    nifty_statement_return_pct: Optional[float] = None
    nifty_annualized_return: Optional[float] = None
    resolution_path: str
    is_fully_redeemed: bool
    diagnostic_info: dict = {}

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
    net_wealth_gain: Optional[float] = None
    statement_return_pct: Optional[float] = None
    statement_annualized_return: Optional[float] = None
    portfolio_return_status: str
    nifty_statement_return_pct: Optional[float] = None
    nifty_annualized_return: Optional[float] = None
    benchmark_status: str
    data_quality: DataQualityMetrics

class CASResponse(BaseModel):
    portfolio_summary: PortfolioSummary
    funds_breakdown: List[FundBreakdown]
    transactions: list

def num(v, default=0.0):
    if v is None: return default
    if isinstance(v, Decimal): return float(v)
    try: return float(v)
    except (ValueError, TypeError): return default

def parse_date(v):
    s = str(v or '').strip()
    for fmt in ('%d-%b-%Y','%d-%B-%Y','%Y-%m-%d','%b %Y','%d/%m/%Y'):
        try: return datetime.strptime(s, fmt).date()
        except ValueError: pass
    raise ValueError(f'Unable to parse date: {v}')

def normalize(desc, raw):
    c = f'{desc or ""} {raw or ""}'.upper()
    t = set(c.split())
    if 'STAMP' in c and 'DUTY' in c: return 'STAMP_DUTY'
    if 'REINVESTMENT' in c: return 'DIVIDEND_REINVESTMENT'
    if 'DIVIDEND' in c and any(x in c for x in ('PAYOUT','ISSUED','TRANSFER')): return 'DIVIDEND_PAYOUT'
    if 'LATERAL' in c and 'SHIFT' in c: return 'SWITCH_IN' if 'IN' in t else 'SWITCH_OUT' if 'OUT' in t else raw
    if 'SIP' in t: return 'SIP'
    if 'SWP' in t: return 'SWP'
    if 'STP' in t or ('SYSTEMATIC' in c and 'TRANSFER' in c): return 'SWITCH_IN' if ('IN' in t or 'PURCHASE' in t) else 'SWITCH_OUT' if ('OUT' in t or 'REDEMPTION' in t or 'SELL' in t) else raw
    if 'SWITCH' in c: return 'SWITCH_IN' if 'IN' in t else 'SWITCH_OUT' if 'OUT' in t else raw
    if any(x in c for x in ('PURCHASE','LUMPSUM','ADDITIONAL')): return 'PURCHASE'
    if any(x in c for x in ('REDEMPTION','SELL')): return 'REDEMPTION'
    return raw

async def opening_value(scheme, stmt_from, client):
    units = num(scheme.get('open'))
    if units == 0: return 0.0, 'Zero opening units'
    if num(scheme.get('opening_value')) > 0: return num(scheme.get('opening_value')), 'Explicit opening_value from CAS'
    if num(scheme.get('open_nav')) > 0: return units * num(scheme.get('open_nav')), 'Calculated via open_nav * units'
    code = scheme.get('amfi')
    if code:
        try:
            r = await client.get(f'https://api.mfapi.in/mf/{code}', timeout=2.0)
            if r.status_code == 200:
                history = r.json().get('data', [])
                d = stmt_from
                for _ in range(7):
                    key = d.strftime('%d-%m-%Y')
                    for row in history:
                        if row.get('date') == key:
                            v = num(row.get('nav'))
                            if v > 0: return units * v, 'Resolved via AMFI historical NAV lookup'
                    d -= timedelta(days=1)
        except Exception:
            pass
    return None, 'Opening market value unresolved'

@app.get('/health')
async def health():
    return {'status':'ok','service':'fundwise-backend','timestamp':datetime.utcnow().isoformat()}

@app.post('/api/v1/parse-cas', response_model=CASResponse)
async def parse_cas(file: UploadFile = File(...), password: str = Form('')):
    path = f'/tmp/{uuid.uuid4()}_{os.path.basename(file.filename or "statement.pdf")}'
    try:
        with open(path, 'wb') as f: f.write(await file.read())
        raw = casparser.read_cas_pdf(path, password)
        data = raw if isinstance(raw, dict) else raw.model_dump()
        period = data.get('statement_period', {})
        start = parse_date(period.get('from') or period.get('start_date'))
        end = parse_date(period.get('to') or period.get('end_date'))
        funds=[]; txs=[]; opening_total=0.0; ending_total=0.0; inv=red=div=stamp=0.0; resolved=0
        async with httpx.AsyncClient() as client:
            for folio in data.get('folios', []):
                for scheme in folio.get('schemes', []):
                    name=scheme.get('scheme','Unknown Scheme'); val=scheme.get('valuation',{})
                    ending=num(val.get('value')); units=num(val.get('balance')); nav=num(val.get('nav'))
                    val_date=parse_date(val.get('date'))
                    opening,path_name=await opening_value(scheme,start,client)
                    ok=opening is not None
                    if ok: resolved+=1; opening_total+=opening; ending_total+=ending
                    si=sr=dp=sd=0.0
                    fund_txs=[]
                    for tx in scheme.get('transactions',[]):
                        if not tx.get('date'): continue
                        dt=parse_date(tx['date'])
                        if not (start <= dt <= end): continue
                        typ=normalize(tx.get('description',''),tx.get('type','')); amt=abs(num(tx.get('amount')))
                        nt=dict(tx); nt.update({'date':dt.isoformat(),'scheme_name':name,'normalized_type':typ}); txs.append(nt)
                        if typ in ('PURCHASE','SIP','SWITCH_IN'): si+=amt
                        elif typ in ('REDEMPTION','SWP','SWITCH_OUT'): sr+=amt
                        elif typ=='DIVIDEND_PAYOUT': dp+=amt
                        elif typ=='STAMP_DUTY': sd+=amt
                    inv+=si; red+=sr; div+=dp; stamp+=sd
                    funds.append(FundBreakdown(scheme_name=name,opening_market_value=opening,statement_investments=si,statement_redemptions=sr,dividend_payouts=dp,stamp_duty_costs=sd,ending_market_value=ending,units=units,latest_nav=nav,resolution_path=path_name,is_fully_redeemed=units < .001,diagnostic_info={'valuation_date':val_date.isoformat(),'opening_resolved':ok}))
        coverage=(sum(f.ending_market_value for f in funds if f.opening_market_value is not None)/ending_total*100) if ending_total else 0
        quality='complete' if resolved==len(funds) else 'partial' if resolved else 'unresolved'
        portfolio=PortfolioSummary(statement_period={'from':start.isoformat(),'to':end.isoformat()},opening_portfolio_value=opening_total if resolved else None,total_statement_investments=inv,total_statement_redemptions=red,total_dividend_payouts=div,total_stamp_duty_costs=stamp,ending_portfolio_value=sum(f.ending_market_value for f in funds),portfolio_return_status='Complete' if resolved==len(funds) else 'Partial (Resolved Subset Only)' if resolved else 'Unavailable',benchmark_status='Pending client-side calculation',data_quality=DataQualityMetrics(status=quality,total_funds=len(funds),resolved_funds=resolved,coverage_percentage=round(coverage,2)))
        return CASResponse(portfolio_summary=portfolio,funds_breakdown=funds,transactions=txs)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=400,content={'message':str(e),'detail':str(e)})
    finally:
        if os.path.exists(path): os.remove(path)

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='0.0.0.0',port=int(os.environ.get('PORT',10000)))
