import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

OUT = Path(__file__).with_name('quarterly_scores.csv')
WEIGHTS = {
    'EPS Growth': 0.16,
    'Revenue Growth': 0.13,
    'Net Margin': 0.10,
    'ROIC / Capital Efficiency': 0.13,
    'Forward P/E': 0.12,
    'Analyst Conviction': 0.08,
    'Institutional Ownership': 0.05,
    'Insider Activity': 0.04,
    '12M Momentum': 0.09,
    'Chart Health': 0.10,
}


def clean_num(v):
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def pct(v):
    v = clean_num(v)
    return None if v is None else v * 100


def normalize(v, low, high, reverse=False):
    v = clean_num(v)
    if v is None:
        return None
    s = max(0.0, min(100.0, (v - low) / (high - low) * 100))
    return 100.0 - s if reverse else s



def curve_score(value, points):
    value = clean_num(value)
    if value is None:
        return None
    points = sorted(points)
    if value <= points[0][0]: return float(points[0][1])
    if value >= points[-1][0]: return float(points[-1][1])
    for (x0,y0),(x1,y1) in zip(points, points[1:]):
        if x0 <= value <= x1:
            t=(value-x0)/(x1-x0) if x1 != x0 else 1.0
            return float(y0+t*(y1-y0))
    return None


def eps_growth_score(value):
    return curve_score(value,[(-30,0),(-20,10),(-10,22),(0,35),(5,45),(10,55),(15,65),(20,73),(30,82),(40,89),(60,95),(100,98),(150,100)])


def revenue_growth_score(value):
    return curve_score(value,[(-20,0),(-10,15),(0,35),(5,45),(10,58),(15,68),(20,77),(30,88),(40,94),(60,98),(80,100)])


def margin_score(value):
    return curve_score(value,[(-10,0),(0,30),(5,42),(10,55),(15,65),(20,74),(25,82),(30,88),(40,95),(50,100)])


def roic_score(value):
    return curve_score(value,[(-5,0),(0,20),(5,35),(10,50),(15,65),(20,77),(25,86),(30,92),(40,97),(50,100)])


def valuation_score(forward_pe, eps_growth):
    pe=clean_num(forward_pe); growth=clean_num(eps_growth)
    if pe is None or pe <= 0: return None
    pe_score=curve_score(pe,[(8,95),(12,92),(18,84),(25,72),(35,55),(45,38),(60,20),(90,5)])
    if growth is None or growth <= 0: return min(70.0, pe_score)
    peg=pe/max(growth,1.0)
    peg_score=curve_score(peg,[(0.3,98),(0.6,94),(0.9,86),(1.2,76),(1.5,65),(2.0,50),(3.0,30),(5.0,10)])
    return round(.65*peg_score+.35*pe_score,1)


def institutional_score(value):
    value=clean_num(value)
    if value is None: return None
    return curve_score(value,[(0,42),(20,46),(40,50),(60,56),(75,61),(90,65),(100,66)])


def insider_activity_score(value):
    return curve_score(value,[(-5,30),(-3,38),(-1,46),(0,50),(1,61),(2,73),(3,84),(4,93),(5,100)])


def momentum_score(m):
    m3=clean_num(m.get('momentum_3m')); m6=clean_num(m.get('momentum_6m')); m12=clean_num(m.get('momentum'))
    vals=[]
    if m3 is not None: vals.append((curve_score(m3,[(-25,5),(-10,25),(0,45),(5,55),(10,65),(20,78),(35,90),(55,97),(80,100)]),.25))
    if m6 is not None: vals.append((curve_score(m6,[(-35,5),(-15,25),(0,45),(8,57),(15,68),(30,82),(50,93),(75,98),(110,100)]),.35))
    if m12 is not None: vals.append((curve_score(m12,[(-50,0),(-20,20),(0,42),(10,55),(20,66),(35,78),(55,89),(80,96),(120,100)]),.40))
    if not vals: return None
    score=sum(v*w for v,w in vals)/sum(w for _,w in vals)
    raw=[x for x in (m3,m6,m12) if x is not None]
    if len(raw)>=2:
        positives=sum(x>0 for x in raw)
        if positives==len(raw): score += 3
        elif positives<=1: score -= 5
    return round(max(0,min(100,score)),1)

def safe_row(df, names):
    if df is None or df.empty:
        return None
    for name in names:
        if name in df.index:
            vals = pd.to_numeric(df.loc[name], errors='coerce').dropna()
            if len(vals):
                return float(vals.iloc[0])
    return None


def calc_roic(financials, balance_sheet):
    ebit = safe_row(financials, ['EBIT', 'Operating Income'])
    pretax = safe_row(financials, ['Pretax Income', 'Income Before Tax'])
    tax = safe_row(financials, ['Tax Provision', 'Income Tax Expense'])
    equity = safe_row(balance_sheet, ['Stockholders Equity', 'Total Stockholder Equity'])
    debt = safe_row(balance_sheet, ['Total Debt'])
    cash = safe_row(balance_sheet, ['Cash Cash Equivalents And Short Term Investments', 'Cash And Cash Equivalents'])
    if ebit is None or equity is None:
        return None
    tax_rate = 0.21
    if pretax not in (None, 0) and tax is not None:
        tax_rate = max(0.0, min(0.40, tax / pretax))
    invested = equity + (debt or 0.0) - (cash or 0.0)
    if invested <= 0:
        return None
    return ebit * (1 - tax_rate) / invested * 100


def insider_score(df):
    if df is None or df.empty:
        return None
    cols = [c for c in df.columns if any(k in str(c).lower() for k in ['transaction', 'text', 'type'])]
    if not cols:
        return None
    text = df[cols].astype(str).agg(' '.join, axis=1).str.lower()
    buys = text.str.contains('buy|purchase|acquisition').sum()
    sells = text.str.contains('sale|sell|disposition').sum()
    total = buys + sells
    if total == 0:
        return 0.0
    return float(max(-5, min(5, (buys - sells) / total * 5)))


def momentum_12m(history):
    if history is None or history.empty or 'Close' not in history:
        return None
    closes = history['Close'].dropna()
    if len(closes) < 2:
        return None
    return (float(closes.iloc[-1]) / float(closes.iloc[0]) - 1) * 100


def momentum_period(history, trading_days):
    if history is None or history.empty or 'Close' not in history:
        return None
    closes=history['Close'].dropna()
    if len(closes) <= trading_days: return None
    return (float(closes.iloc[-1])/float(closes.iloc[-trading_days])-1)*100



def chart_health(history):
    if history is None or history.empty or 'Close' not in history: return None
    closes=history['Close'].dropna()
    if len(closes)<200: return None
    price=float(closes.iloc[-1]); sma50=float(closes.tail(50).mean()); sma200=float(closes.tail(200).mean()); high52=float(closes.max())
    sma50_20d_ago=float(closes.iloc[-70:-20].mean()) if len(closes)>=70 else sma50
    score=0.0
    if price>sma50: score += 15
    if price>sma200: score += 20
    if sma50>sma200: score += 20
    mom3=momentum_period(history,63)
    if mom3 is not None: score += curve_score(mom3,[(-20,0),(-5,25),(0,45),(8,65),(15,80),(25,95),(40,100)])*.20
    if high52>0:
        drawdown=(price/high52-1)*100
        score += curve_score(drawdown,[(-40,0),(-25,20),(-15,45),(-10,60),(-5,80),(0,100)])*.15
    if sma50_20d_ago>0:
        slope=(sma50/sma50_20d_ago-1)*100
        score += curve_score(slope,[(-8,0),(-3,20),(0,45),(2,65),(5,85),(8,100)])*.10
    if price<sma200: score=min(score,48)
    elif price<sma50: score=min(score,64)
    if sma50<sma200: score=min(score,72)
    return round(max(0,min(100,score)),1)

def analyst_upside(info):
    current = clean_num(info.get('currentPrice') or info.get('regularMarketPrice'))
    target = clean_num(info.get('targetMeanPrice'))
    if current is None or current <= 0 or target is None:
        return None
    return (target / current - 1) * 100


def analyst_conviction(info):
    upside = analyst_upside(info)
    count = clean_num(info.get('numberOfAnalystOpinions'))
    recommendation_mean = clean_num(info.get('recommendationMean'))
    if upside is None or count is None or count < 8:
        return None
    parts = [(curve_score(upside,[(-20,15),(-10,30),(0,45),(10,58),(20,72),(30,84),(40,92),(60,98)]),0.55),
             (curve_score(count,[(8,45),(12,55),(18,68),(25,78),(35,86),(50,92)]),0.20)]
    if recommendation_mean is not None:
        parts.append((curve_score(recommendation_mean,[(1.0,95),(1.5,86),(2.0,74),(2.5,60),(3.0,48),(4.0,25),(5.0,10)]),0.25))
    parts = [(a, w) for a, w in parts if a is not None]
    denom = sum(w for _, w in parts)
    return round(sum(a*w for a, w in parts)/denom, 1) if denom else None


def fetch_info_with_retry(ticker):
    t = yf.Ticker(ticker)
    info = {}
    for attempt in range(3):
        try:
            info = t.get_info() or {}
            if info:
                break
        except Exception:
            pass
        time.sleep(0.8 * (attempt + 1))
    try:
        hist = t.history(period='1y', auto_adjust=True)
    except Exception:
        hist = pd.DataFrame()
    try:
        fin = t.financials
    except Exception:
        fin = pd.DataFrame()
    try:
        bs = t.balance_sheet
    except Exception:
        bs = pd.DataFrame()
    try:
        insiders = t.insider_transactions
    except Exception:
        insiders = pd.DataFrame()
    metrics = {
        'eps_growth': pct(info.get('earningsGrowth')),
        'revenue_growth': pct(info.get('revenueGrowth')),
        'net_margin': pct(info.get('profitMargins')),
        'roic': calc_roic(fin, bs),
        'forward_pe': clean_num(info.get('forwardPE')),
        'analyst_upside': analyst_upside(info),
        'analyst_conviction': analyst_conviction(info),
        'institutional_ownership': pct(info.get('heldPercentInstitutions')),
        'insider_activity': insider_score(insiders),
        'momentum': momentum_12m(hist),
        'momentum_6m': momentum_period(hist,126),
        'momentum_3m': momentum_period(hist,63),
        'chart_health': chart_health(hist),
    }
    return info, metrics


def score_stock(m):
    raw={
        'EPS Growth': eps_growth_score(m.get('eps_growth')),
        'Revenue Growth': revenue_growth_score(m.get('revenue_growth')),
        'Net Margin': margin_score(m.get('net_margin')),
        'ROIC / Capital Efficiency': roic_score(m.get('roic')),
        'Forward P/E': valuation_score(m.get('forward_pe'),m.get('eps_growth')),
        'Analyst Conviction': clean_num(m.get('analyst_conviction')),
        'Institutional Ownership': institutional_score(m.get('institutional_ownership')),
        'Insider Activity': insider_activity_score(m.get('insider_activity')),
        '12M Momentum': momentum_score(m),
        'Chart Health': clean_num(m.get('chart_health')),
    }
    available_weight=sum(WEIGHTS[k] for k,v in raw.items() if v is not None)
    if available_weight==0: return None
    total=sum(v*WEIGHTS[k]/available_weight for k,v in raw.items() if v is not None)
    coverage=sum(v is not None for v in raw.values())
    total *= {10:1.00,9:.99,8:.97,7:.94,6:.90}.get(coverage,.86)
    return round(total,2)


def index_members(url, expected, ticker_col, name_col):
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 ConvictionAI-quarterly'}, timeout=30)
    r.raise_for_status()
    for table in pd.read_html(StringIO(r.text)):
        if set(expected).issubset({str(c) for c in table.columns}):
            out = table[[ticker_col, name_col]].copy()
            out.columns = ['ticker', 'company']
            out['ticker'] = out['ticker'].astype(str).str.strip().str.upper().str.replace('.', '-', regex=False)
            return out.drop_duplicates('ticker')
    raise RuntimeError('Constituent table not found')


def build_universe():
    sp = index_members('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', {'Symbol', 'Security'}, 'Symbol', 'Security')
    ndx = index_members('https://en.wikipedia.org/wiki/Nasdaq-100', {'Ticker', 'Company'}, 'Ticker', 'Company')
    return pd.concat([sp, ndx], ignore_index=True).drop_duplicates('ticker').reset_index(drop=True)


def quarter_id():
    dt = datetime.now(timezone.utc)
    q = (dt.month - 1) // 3 + 1
    return f'{dt.year}-Q{q}'


def score_one(row):
    ticker = row.ticker
    info, metrics = fetch_info_with_retry(ticker)
    scoring_keys = ['eps_growth','revenue_growth','net_margin','roic','forward_pe','analyst_conviction','institutional_ownership','insider_activity','momentum','chart_health']
    coverage = sum(metrics.get(k) is not None for k in scoring_keys)
    score = score_stock(metrics)
    if score is None or coverage < 7:
        return None
    return {
        'quarter': quarter_id(),
        'ticker': ticker,
        'company': info.get('longName') or info.get('shortName') or row.company,
        'score': round(score, 2),
        'coverage': coverage,
        'fetched_at': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
    }


def main():
    universe = build_universe()
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(score_one, row) for row in universe.itertuples(index=False)]
        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                if result:
                    rows.append(result)
            except Exception as e:
                print('ticker failed:', e)
            if i % 50 == 0:
                print(f'Completed {i}/{len(futures)}')

    current = pd.DataFrame(rows)
    if current.empty:
        raise SystemExit('No eligible scores were produced; refusing to overwrite snapshot file.')

    if OUT.exists():
        try:
            old = pd.read_csv(OUT)
        except Exception:
            old = pd.DataFrame(columns=current.columns)
    else:
        old = pd.DataFrame(columns=current.columns)

    q = quarter_id()
    if not old.empty and 'quarter' in old.columns:
        old = old[old['quarter'].astype(str) != q]
    out = pd.concat([old, current], ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f'Saved {len(current)} eligible tickers for {q} to {OUT}')


if __name__ == '__main__':
    main()
