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
    'Revenue Growth': 0.11,
    'Net Margin': 0.11,
    'ROIC / Capital Efficiency': 0.11,
    'Forward P/E': 0.10,
    'Analyst Upside': 0.11,
    'Institutional Ownership': 0.07,
    'Insider Activity': 0.05,
    '12M Momentum': 0.08,
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



def chart_health(history):
    if history is None or history.empty or 'Close' not in history:
        return None
    closes = history['Close'].dropna()
    if len(closes) < 200:
        return None
    price = float(closes.iloc[-1])
    sma50 = float(closes.tail(50).mean())
    sma200 = float(closes.tail(200).mean())
    high52 = float(closes.max())
    score = 0.0
    if price > sma50:
        score += 25.0
    if price > sma200:
        score += 25.0
    if sma50 > sma200:
        score += 20.0
    if len(closes) >= 63:
        mom3 = (price / float(closes.iloc[-63]) - 1) * 100
        score += normalize(mom3, -10, 15) * 0.15
    if high52 > 0:
        drawdown = (price / high52 - 1) * 100
        score += normalize(drawdown, -30, 0) * 0.15
    return round(max(0.0, min(100.0, score)), 1)

def analyst_upside(info):
    current = clean_num(info.get('currentPrice') or info.get('regularMarketPrice'))
    target = clean_num(info.get('targetMeanPrice'))
    if current is None or current <= 0 or target is None:
        return None
    return (target / current - 1) * 100


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
        'institutional_ownership': pct(info.get('heldPercentInstitutions')),
        'insider_activity': insider_score(insiders),
        'momentum': momentum_12m(hist),
        'chart_health': chart_health(hist),
    }
    return info, metrics


def score_stock(m):
    raw = {
        'EPS Growth': normalize(m.get('eps_growth'), -20, 50),
        'Revenue Growth': normalize(m.get('revenue_growth'), -10, 35),
        'Net Margin': normalize(m.get('net_margin'), -5, 35),
        'ROIC / Capital Efficiency': normalize(m.get('roic'), 0, 30),
        'Forward P/E': normalize(m.get('forward_pe'), 10, 45, reverse=True),
        'Analyst Upside': normalize(m.get('analyst_upside'), -10, 35),
        'Institutional Ownership': normalize(m.get('institutional_ownership'), 20, 90),
        'Insider Activity': normalize(m.get('insider_activity'), -5, 5),
        '12M Momentum': normalize(m.get('momentum'), -20, 40),
        'Chart Health': clean_num(m.get('chart_health')),
    }
    available_weight = sum(WEIGHTS[k] for k, v in raw.items() if v is not None)
    if available_weight == 0:
        return None
    return sum(v * WEIGHTS[k] / available_weight for k, v in raw.items() if v is not None)


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
    coverage = sum(v is not None for v in metrics.values())
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
