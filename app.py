import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import requests

import pandas as pd
import streamlit as st
import yfinance as yf

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from supabase import create_client
except ImportError:
    create_client = None

SNAPSHOT_FILE = Path(__file__).with_name("quarterly_scores.csv")
FALLBACK_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AVGO", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "LLY",
    "JPM", "V", "MA", "WMT", "COST", "NFLX", "ORCL", "AMD", "CRM", "PLTR",
    "UBER", "MU", "SPGI", "UNH", "XOM", "HD", "KO", "PEP", "ABBV", "GE"
]

st.set_page_config(page_title="Conviction AI", page_icon="📈", layout="wide")

# Mobile-first polish: tighter spacing and a calmer first screen.
st.markdown(
    """
    <style>
    .block-container {padding-top: 2.25rem; padding-bottom: 2.5rem;}
    h1 {margin-bottom: .35rem !important;}
    h3 {margin-top: .25rem !important; margin-bottom: .35rem !important;}
    div[data-testid="stAlert"] {border-radius: 12px;}
    .gem-badge {
        display: inline-block; padding: .22rem .55rem; border-radius: 999px;
        font-size: .78rem; font-weight: 700; margin: .15rem 0 .35rem 0;
    }
    .gem-quality {background: rgba(46, 160, 67, .14); color: #238636; border: 1px solid rgba(46, 160, 67, .25);}
    .gem-turnaround {background: rgba(210, 153, 34, .14); color: #9a6700; border: 1px solid rgba(210, 153, 34, .28);}
    @media (max-width: 640px) {
        .block-container {padding-top: 1.25rem; padding-left: 1rem; padding-right: 1rem;}
        h1 {font-size: 2.35rem !important; line-height: 1.05 !important;}
        h3 {font-size: 1.45rem !important;}
        .stButton > button {min-height: 3rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Conviction AI")
st.markdown("### Stock research made simple.")
st.caption("Quickly see growth, valuation, profitability, analyst sentiment, momentum, and chart health in one place.")
st.markdown("**10 factors. One simple Conviction Score.**")

WEIGHTS = {
    # Business quality carries most of the score. Market-opinion factors are confirmation, not the thesis.
    "EPS Growth": 0.16,
    "Revenue Growth": 0.13,
    "Net Margin": 0.10,
    "ROIC / Capital Efficiency": 0.13,
    "Forward P/E": 0.12,
    "Analyst Conviction": 0.08,
    "Institutional Ownership": 0.05,
    "Insider Activity": 0.04,
    "12M Momentum": 0.09,
    "Chart Health": 0.10,
}


def clean_num(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def pct(value):
    value = clean_num(value)
    return None if value is None else value * 100


def normalize(value, low, high, reverse=False):
    value = clean_num(value)
    if value is None:
        return None
    if high == low:
        return 50.0
    score = (value - low) / (high - low) * 100
    score = max(0.0, min(100.0, score))
    return 100.0 - score if reverse else score




def curve_score(value, points):
    """Piecewise-linear score. `points` is [(raw_value, score), ...]."""
    value = clean_num(value)
    if value is None:
        return None
    points = sorted(points)
    if value <= points[0][0]:
        return float(points[0][1])
    if value >= points[-1][0]:
        return float(points[-1][1])
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= value <= x1:
            if x1 == x0:
                return float(y1)
            t = (value - x0) / (x1 - x0)
            return float(y0 + t * (y1 - y0))
    return None


def eps_growth_score(value):
    # 100 should be exceptional. A routine 20-30% grower belongs in the 70s/80s, not at the ceiling.
    return curve_score(value, [
        (-30, 0), (-20, 10), (-10, 22), (0, 35), (5, 45), (10, 55),
        (15, 65), (20, 73), (30, 82), (40, 89), (60, 95), (100, 98), (150, 100)
    ])


def revenue_growth_score(value):
    return curve_score(value, [
        (-20, 0), (-10, 15), (0, 35), (5, 45), (10, 58), (15, 68),
        (20, 77), (30, 88), (40, 94), (60, 98), (80, 100)
    ])


def margin_score(value):
    # Sector-neutral compromise: good margins score well, but truly elite margins are needed for 95+.
    return curve_score(value, [
        (-10, 0), (0, 30), (5, 42), (10, 55), (15, 65), (20, 74),
        (25, 82), (30, 88), (40, 95), (50, 100)
    ])


def roic_score(value):
    return curve_score(value, [
        (-5, 0), (0, 20), (5, 35), (10, 50), (15, 65), (20, 77),
        (25, 86), (30, 92), (40, 97), (50, 100)
    ])


def valuation_score(forward_pe, eps_growth):
    """Growth-adjusted valuation. Uses a simple forward P/E + growth (PEG-like) blend."""
    pe = clean_num(forward_pe)
    growth = clean_num(eps_growth)
    if pe is None or pe <= 0:
        return None

    # Absolute P/E prevents a high-growth company from receiving a free pass at any valuation.
    pe_score = curve_score(pe, [(8, 95), (12, 92), (18, 84), (25, 72), (35, 55), (45, 38), (60, 20), (90, 5)])
    if growth is None or growth <= 0:
        return min(70.0, pe_score)

    peg = pe / max(growth, 1.0)  # growth is already in percentage points (e.g. 25 == 25%).
    peg_score = curve_score(peg, [(0.3, 98), (0.6, 94), (0.9, 86), (1.2, 76), (1.5, 65), (2.0, 50), (3.0, 30), (5.0, 10)])
    return round(0.65 * peg_score + 0.35 * pe_score, 1)


def institutional_score(value):
    # Ownership LEVEL is only a weak proxy for institutional conviction. Keep it near neutral and capped.
    value = clean_num(value)
    if value is None:
        return None
    return curve_score(value, [(0, 42), (20, 46), (40, 50), (60, 56), (75, 61), (90, 65), (100, 66)])


def insider_activity_score(value):
    # Open-market buying is more informative than routine selling. Neutral activity stays near 50.
    return curve_score(value, [(-5, 30), (-3, 38), (-1, 46), (0, 50), (1, 61), (2, 73), (3, 84), (4, 93), (5, 100)])


def momentum_score(m):
    """Trend persistence across 3/6/12 months, intentionally distinct from chart structure."""
    m3 = clean_num(m.get("momentum_3m"))
    m6 = clean_num(m.get("momentum_6m"))
    m12 = clean_num(m.get("momentum"))
    vals = []
    if m3 is not None: vals.append((curve_score(m3, [(-25, 5), (-10, 25), (0, 45), (5, 55), (10, 65), (20, 78), (35, 90), (55, 97), (80, 100)]), .25))
    if m6 is not None: vals.append((curve_score(m6, [(-35, 5), (-15, 25), (0, 45), (8, 57), (15, 68), (30, 82), (50, 93), (75, 98), (110, 100)]), .35))
    if m12 is not None: vals.append((curve_score(m12, [(-50, 0), (-20, 20), (0, 42), (10, 55), (20, 66), (35, 78), (55, 89), (80, 96), (120, 100)]), .40))
    if not vals:
        return None
    denom = sum(w for _, w in vals)
    score = sum(v*w for v,w in vals) / denom
    # Small consistency bonus/penalty: persistent positive trends beat a one-period spike.
    raw = [x for x in (m3,m6,m12) if x is not None]
    if len(raw) >= 2:
        positives = sum(x > 0 for x in raw)
        if positives == len(raw): score += 3
        elif positives <= 1: score -= 5
    return round(max(0.0, min(100.0, score)), 1)

def safe_row(df, names):
    if df is None or df.empty:
        return None
    for name in names:
        if name in df.index:
            vals = pd.to_numeric(df.loc[name], errors="coerce").dropna()
            if len(vals):
                return float(vals.iloc[0])
    return None


def calc_roic(financials, balance_sheet):
    """Approximate ROIC = NOPAT / invested capital using latest reported period."""
    ebit = safe_row(financials, ["EBIT", "Operating Income"])
    pretax = safe_row(financials, ["Pretax Income", "Income Before Tax"])
    tax = safe_row(financials, ["Tax Provision", "Income Tax Expense"])
    equity = safe_row(balance_sheet, ["Stockholders Equity", "Total Stockholder Equity"])
    debt = safe_row(balance_sheet, ["Total Debt"])
    cash = safe_row(balance_sheet, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"])

    if ebit is None or equity is None:
        return None
    tax_rate = 0.21
    if pretax not in (None, 0) and tax is not None:
        tax_rate = max(0.0, min(0.40, tax / pretax))
    invested_capital = equity + (debt or 0.0) - (cash or 0.0)
    if invested_capital <= 0:
        return None
    return (ebit * (1 - tax_rate) / invested_capital) * 100


def insider_score(insider_df):
    """Map recent reported insider buys/sells to -5..+5. Conservative when fields vary."""
    if insider_df is None or insider_df.empty:
        return None
    df = insider_df.copy()
    text_cols = [c for c in df.columns if any(k in str(c).lower() for k in ["transaction", "text", "type"])]
    if not text_cols:
        return None
    text = df[text_cols].astype(str).agg(" ".join, axis=1).str.lower()
    buys = text.str.contains("buy|purchase|acquisition").sum()
    sells = text.str.contains("sale|sell|disposition").sum()
    total = buys + sells
    if total == 0:
        return 0.0
    raw = (buys - sells) / total * 5
    return float(max(-5, min(5, raw)))


def momentum_12m(history):
    if history is None or history.empty or "Close" not in history:
        return None
    closes = history["Close"].dropna()
    if len(closes) < 2:
        return None
    return (float(closes.iloc[-1]) / float(closes.iloc[0]) - 1) * 100


def momentum_period(history, trading_days):
    if history is None or history.empty or "Close" not in history:
        return None
    closes = history["Close"].dropna()
    if len(closes) <= trading_days:
        return None
    return (float(closes.iloc[-1]) / float(closes.iloc[-trading_days]) - 1) * 100



def chart_health(history):
    """0-100 technical-health score. 90+ requires a genuinely strong, persistent trend."""
    if history is None or history.empty or "Close" not in history:
        return None
    closes = history["Close"].dropna()
    if len(closes) < 200:
        return None

    price = float(closes.iloc[-1])
    sma50 = float(closes.tail(50).mean())
    sma200 = float(closes.tail(200).mean())
    high52 = float(closes.max())
    sma50_20d_ago = float(closes.iloc[-70:-20].mean()) if len(closes) >= 70 else sma50

    score = 0.0
    if price > sma50: score += 15.0
    if price > sma200: score += 20.0
    if sma50 > sma200: score += 20.0

    mom3 = momentum_period(history, 63)
    if mom3 is not None:
        score += curve_score(mom3, [(-20, 0), (-5, 25), (0, 45), (8, 65), (15, 80), (25, 95), (40, 100)]) * 0.20

    if high52 > 0:
        drawdown = (price / high52 - 1) * 100
        score += curve_score(drawdown, [(-40, 0), (-25, 20), (-15, 45), (-10, 60), (-5, 80), (0, 100)]) * 0.15

    # A rising 50-day average helps distinguish a healthy trend from a recent bounce above the averages.
    if sma50_20d_ago > 0:
        slope = (sma50 / sma50_20d_ago - 1) * 100
        score += curve_score(slope, [(-8, 0), (-3, 20), (0, 45), (2, 65), (5, 85), (8, 100)]) * 0.10

    # Structural penalties keep weak charts from scoring well just because one sub-signal is hot.
    if price < sma200: score = min(score, 48.0)
    elif price < sma50: score = min(score, 64.0)
    if sma50 < sma200: score = min(score, 72.0)

    return round(max(0.0, min(100.0, score)), 1)

def analyst_upside(info):
    current = clean_num(info.get("currentPrice") or info.get("regularMarketPrice"))
    target = clean_num(info.get("targetMeanPrice"))
    if current is None or current <= 0 or target is None:
        return None
    return (target / current - 1) * 100


def analyst_conviction(info):
    """0-100 analyst signal. Requires meaningful coverage so a tiny analyst sample cannot dominate."""
    upside = analyst_upside(info)
    count = clean_num(info.get("numberOfAnalystOpinions"))
    recommendation_mean = clean_num(info.get("recommendationMean"))

    # Require a real analyst sample before this factor contributes to Conviction Score.
    if upside is None or count is None or count < 8:
        return None

    parts = []
    weights = []
    # Analyst opinion is confirmation, not the thesis. Strong scores require both meaningful upside and broad coverage.
    parts.append(curve_score(upside, [(-20, 15), (-10, 30), (0, 45), (10, 58), (20, 72), (30, 84), (40, 92), (60, 98)])); weights.append(0.55)
    parts.append(curve_score(count, [(8, 45), (12, 55), (18, 68), (25, 78), (35, 86), (50, 92)])); weights.append(0.20)
    if recommendation_mean is not None:
        # Yahoo convention is roughly 1=Strong Buy, 5=Sell.
        parts.append(curve_score(recommendation_mean, [(1.0, 95), (1.5, 86), (2.0, 74), (2.5, 60), (3.0, 48), (4.0, 25), (5.0, 10)])); weights.append(0.25)

    good = [(p, w) for p, w in zip(parts, weights) if p is not None]
    if not good:
        return None
    denom = sum(w for _, w in good)
    return round(sum(p*w for p, w in good) / denom, 1)


def _analysis_row(df, preferred_rows, column):
    """Read a numeric cell from a yfinance analyst DataFrame defensively."""
    if df is None or getattr(df, "empty", True):
        return None
    for row in preferred_rows:
        if row in df.index and column in df.columns:
            return clean_num(df.loc[row, column])
    return None


def estimate_revision_metrics(ticker_obj):
    """Return 90d EPS estimate revision %, 30d revision breadth, and a 0-100 revision score."""
    try:
        trend = ticker_obj.get_eps_trend()
    except Exception:
        trend = pd.DataFrame()
    try:
        revisions = ticker_obj.get_eps_revisions()
    except Exception:
        revisions = pd.DataFrame()

    current = _analysis_row(trend, ["+1y", "0y", "+1q", "0q"], "current")
    prior90 = _analysis_row(trend, ["+1y", "0y", "+1q", "0q"], "90daysAgo")
    if prior90 is None:
        prior90 = _analysis_row(trend, ["+1y", "0y", "+1q", "0q"], "60daysAgo")
    if prior90 is None:
        prior90 = _analysis_row(trend, ["+1y", "0y", "+1q", "0q"], "30daysAgo")

    revision_pct = None
    if current is not None and prior90 not in (None, 0):
        # Use change relative to the magnitude of the prior estimate so negative EPS estimates behave sensibly.
        revision_pct = (current - prior90) / abs(prior90) * 100

    up30 = _analysis_row(revisions, ["+1y", "0y", "+1q", "0q"], "upLast30days")
    down30 = _analysis_row(revisions, ["+1y", "0y", "+1q", "0q"], "downLast30days")
    breadth = None
    if up30 is not None or down30 is not None:
        breadth = (up30 or 0.0) - (down30 or 0.0)

    parts = []
    if revision_pct is not None:
        parts.append((curve_score(revision_pct, [(-20, 5), (-10, 18), (-5, 32), (0, 50), (3, 62), (6, 74), (10, 86), (15, 94), (25, 100)]), 0.72))
    if breadth is not None:
        parts.append((curve_score(breadth, [(-8, 10), (-4, 25), (-2, 38), (0, 50), (2, 64), (4, 78), (7, 92), (10, 100)]), 0.28))
    if not parts:
        return None, None, None
    denom = sum(w for _, w in parts)
    score = sum(v * w for v, w in parts) / denom
    return revision_pct, breadth, round(max(0.0, min(100.0, score)), 1)


def forward_growth_metrics(ticker_obj, fallback_eps_growth=None):
    """Get a forward EPS growth estimate and convert valuation-vs-growth into a 0-100 score."""
    forward_growth = None
    try:
        est = ticker_obj.get_earnings_estimate()
        raw = _analysis_row(est, ["+1y", "0y"], "growth")
        if raw is not None:
            forward_growth = raw * 100 if abs(raw) <= 3 else raw
    except Exception:
        pass

    if forward_growth is None:
        try:
            growth = ticker_obj.get_growth_estimates()
            raw = _analysis_row(growth, ["+5y", "+1y"], "stock")
            if raw is not None:
                forward_growth = raw * 100 if abs(raw) <= 3 else raw
        except Exception:
            pass

    if forward_growth is None:
        forward_growth = clean_num(fallback_eps_growth)
    return forward_growth


def value_vs_growth_score(forward_pe, forward_growth):
    pe = clean_num(forward_pe)
    growth = clean_num(forward_growth)
    if pe is None or pe <= 0 or growth is None or growth <= 0:
        return None
    # PEG-like comparison, intentionally capped so tiny denominators cannot create absurd scores.
    g = max(3.0, min(growth, 60.0))
    peg = pe / g
    score = curve_score(peg, [(0.35, 98), (0.6, 92), (0.85, 84), (1.0, 78), (1.25, 68), (1.5, 58), (2.0, 43), (2.75, 28), (4.0, 12)])
    if pe > 60:
        score = min(score, 55.0)
    elif pe > 45:
        score = min(score, 68.0)
    return round(score, 1)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_stock(symbol, cache_bust=None):
    t = yf.Ticker(symbol)
    errors = []

    info = {}
    last_info_error = None
    for attempt in range(3):
        try:
            info = t.get_info() or {}
            if info:
                break
        except Exception as e:
            last_info_error = e
        time.sleep(0.6 * (attempt + 1))
    if not info and last_info_error is not None:
        errors.append(f"Company data: {last_info_error}")

    try:
        hist = t.history(period="1y", auto_adjust=True)
    except Exception as e:
        hist = pd.DataFrame(); errors.append(f"Price history: {e}")

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

    revision_pct, revision_breadth, revision_score = estimate_revision_metrics(t)
    raw_eps_growth = pct(info.get("earningsGrowth"))
    forward_eps_growth = forward_growth_metrics(t, fallback_eps_growth=raw_eps_growth)
    value_growth = value_vs_growth_score(clean_num(info.get("forwardPE")), forward_eps_growth)

    metrics = {
        "eps_growth": raw_eps_growth,
        "revenue_growth": pct(info.get("revenueGrowth")),
        "net_margin": pct(info.get("profitMargins")),
        "roic": calc_roic(fin, bs),
        "forward_pe": clean_num(info.get("forwardPE")),
        "analyst_upside": analyst_upside(info),
        "analyst_conviction": analyst_conviction(info),
        "institutional_ownership": pct(info.get("heldPercentInstitutions")),
        "insider_activity": insider_score(insiders),
        "momentum": momentum_12m(hist),
        "momentum_6m": momentum_period(hist, 126),
        "momentum_3m": momentum_period(hist, 63),
        "chart_health": chart_health(hist),
        "estimate_revision_pct": revision_pct,
        "estimate_revision_breadth": revision_breadth,
        "estimate_revision_score": revision_score,
        "forward_eps_growth": forward_eps_growth,
        "value_vs_growth_score": value_growth,
    }

    return {
        "symbol": symbol,
        "company": info.get("longName") or info.get("shortName") or symbol,
        "sector": info.get("sector") or "—",
        "industry": info.get("industry") or "—",
        "price": clean_num(info.get("currentPrice") or info.get("regularMarketPrice")),
        "market_cap": clean_num(info.get("marketCap")),
        "recommendation": info.get("recommendationKey") or "—",
        "analyst_count": clean_num(info.get("numberOfAnalystOpinions")),
        "target_mean": clean_num(info.get("targetMeanPrice")),
        "metrics": metrics,
        "errors": errors,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


def score_stock(m):
    raw_scores = {
        "EPS Growth": eps_growth_score(m.get("eps_growth")),
        "Revenue Growth": revenue_growth_score(m.get("revenue_growth")),
        "Net Margin": margin_score(m.get("net_margin")),
        "ROIC / Capital Efficiency": roic_score(m.get("roic")),
        "Forward P/E": valuation_score(m.get("forward_pe"), m.get("eps_growth")),
        "Analyst Conviction": clean_num(m.get("analyst_conviction")),
        "Institutional Ownership": institutional_score(m.get("institutional_ownership")),
        "Insider Activity": insider_activity_score(m.get("insider_activity")),
        "12M Momentum": momentum_score(m),
        "Chart Health": clean_num(m.get("chart_health")),
    }
    available_weight = sum(WEIGHTS[k] for k, v in raw_scores.items() if v is not None)
    if available_weight == 0:
        return None, {}, raw_scores
    contributions = {
        k: (v * WEIGHTS[k] / available_weight if v is not None else None)
        for k, v in raw_scores.items()
    }
    total = sum(v for v in contributions.values() if v is not None)

    # Slight confidence haircut when important data is missing. This prevents a 7/10 stock from
    # reaching the same elite range as a fully observed company simply because weights were re-normalized.
    coverage = sum(v is not None for v in raw_scores.values())
    coverage_multiplier = {10: 1.00, 9: 0.99, 8: 0.97, 7: 0.94, 6: 0.90}.get(coverage, 0.86)
    total *= coverage_multiplier
    return round(total, 2), contributions, raw_scores




@st.cache_data(ttl=86400, show_spinner=False)
def fetch_index_universe(index_name):
    """Refresh index membership from public constituent tables; return a stable fallback if unavailable."""
    headers = {"User-Agent": "Mozilla/5.0 ConvictionAI/0.7.5"}
    if index_name == "S&P 500":
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        expected = {"Symbol", "Security"}
    elif index_name == "Nasdaq-100":
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        expected = {"Ticker", "Company"}
    else:
        raise ValueError(f"Unsupported index: {index_name}")

    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        for table in tables:
            cols = {str(c) for c in table.columns}
            if expected.issubset(cols):
                ticker_col = "Symbol" if "Symbol" in table.columns else "Ticker"
                name_col = "Security" if "Security" in table.columns else "Company"
                out = table[[ticker_col, name_col]].copy()
                out.columns = ["ticker", "company"]
                out["ticker"] = (
                    out["ticker"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
                )
                out = out[out["ticker"].str.match(r"^[A-Z0-9-]+$", na=False)]
                out["source_index"] = index_name
                return out.drop_duplicates("ticker").reset_index(drop=True)
    except Exception:
        pass

    fallback = pd.DataFrame({"ticker": FALLBACK_UNIVERSE, "company": FALLBACK_UNIVERSE})
    fallback["source_index"] = "Fallback"
    return fallback


@st.cache_data(ttl=86400, show_spinner=False)
def build_market_universe(selection):
    sp500 = fetch_index_universe("S&P 500")
    ndx = fetch_index_universe("Nasdaq-100")
    if selection == "S&P 500":
        return sp500
    if selection == "Nasdaq-100":
        return ndx
    combined = pd.concat([sp500, ndx], ignore_index=True)
    combined = combined.sort_values(["ticker", "source_index"]).drop_duplicates("ticker", keep="first")
    return combined.reset_index(drop=True)


def current_quarter_id(dt=None):
    dt = dt or datetime.now(timezone.utc)
    quarter = (dt.month - 1) // 3 + 1
    return f"{dt.year}-Q{quarter}"


SNAPSHOT_COLUMNS = ["quarter", "ticker", "company", "score", "coverage", "fetched_at"]


def _secret(name):
    # Streamlit Cloud secrets first; normal environment variables second.
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.environ.get(name)


@st.cache_resource(show_spinner=False)
def get_supabase_client():
    url = _secret("SUPABASE_URL")
    key = _secret("SUPABASE_KEY")
    if not url or not key or create_client is None:
        return None
    return create_client(url, key)


def storage_backend():
    return "Supabase/Postgres" if get_supabase_client() is not None else "Local CSV fallback"


def load_snapshots():
    client = get_supabase_client()
    if client is not None:
        try:
            response = (
                client.table("quarterly_scores")
                .select("quarter,ticker,company,score,coverage,fetched_at")
                .order("quarter", desc=False)
                .execute()
            )
            df = pd.DataFrame(response.data or [])
            for col in SNAPSHOT_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            return df[SNAPSHOT_COLUMNS]
        except Exception as exc:
            st.warning(f"Database read failed; using local fallback for this session. ({exc})")

    if not SNAPSHOT_FILE.exists():
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    try:
        df = pd.read_csv(SNAPSHOT_FILE)
        for col in SNAPSHOT_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[SNAPSHOT_COLUMNS]
    except Exception:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)


def save_quarter_snapshot(rows, quarter=None):
    quarter = quarter or current_quarter_id()
    new = pd.DataFrame(rows)
    if new.empty:
        return 0
    new["quarter"] = quarter
    new = new[SNAPSHOT_COLUMNS]

    payload = []
    for record in new.to_dict("records"):
        payload.append({
            "quarter": str(record["quarter"]),
            "ticker": str(record["ticker"]).upper(),
            "company": str(record["company"]),
            "score": float(record["score"]),
            "coverage": int(record["coverage"]),
            "fetched_at": str(record["fetched_at"]),
        })

    client = get_supabase_client()
    if client is not None:
        try:
            client.table("quarterly_scores").upsert(
                payload, on_conflict="quarter,ticker"
            ).execute()
            return len(payload)
        except Exception as exc:
            st.error(f"Database save failed: {exc}")
            return 0

    # Local-development fallback when Supabase credentials are not configured.
    old = load_snapshots()
    keys = set((r["quarter"], r["ticker"]) for r in payload)
    if not old.empty:
        keep = ~old.apply(lambda r: (str(r["quarter"]), str(r["ticker"]).upper()) in keys, axis=1)
        old = old[keep]
    out = pd.concat([old, pd.DataFrame(payload)], ignore_index=True)
    out.to_csv(SNAPSHOT_FILE, index=False)
    return len(payload)


def previous_quarter_id(qid):
    year, q = qid.split("-Q")
    year, q = int(year), int(q)
    return f"{year-1}-Q4" if q == 1 else f"{year}-Q{q-1}"


def _quarter_sort_key(qid):
    try:
        year, q = str(qid).split("-Q")
        return int(year), int(q)
    except Exception:
        return (0, 0)


def quarter_start_label(qid):
    try:
        year, q = str(qid).split("-Q")
        month = {"1": "Jan 1", "2": "Apr 1", "3": "Jul 1", "4": "Oct 1"}[q]
        return f"{month}, {year}"
    except Exception:
        return str(qid)


def quarter_movers(snapshot_df):
    if snapshot_df is None or snapshot_df.empty:
        return pd.DataFrame(), pd.DataFrame(), None, None
    work = snapshot_df.copy()
    work["coverage"] = pd.to_numeric(work["coverage"], errors="coerce")
    work = work[work["coverage"] >= 7]
    quarters = sorted(work["quarter"].dropna().astype(str).unique(), key=_quarter_sort_key)
    if len(quarters) < 2:
        latest = quarters[-1] if quarters else None
        return pd.DataFrame(), pd.DataFrame(), None, latest
    prior, latest = quarters[-2], quarters[-1]
    cur = work[work["quarter"] == latest].copy()
    prev = work[work["quarter"] == prior].copy()
    cur["score"] = pd.to_numeric(cur["score"], errors="coerce")
    prev["score"] = pd.to_numeric(prev["score"], errors="coerce")
    merged = cur.merge(prev[["ticker", "score"]], on="ticker", suffixes=("_current", "_prior"))
    merged["change"] = merged["score_current"] - merged["score_prior"]
    merged = merged.dropna(subset=["change"])
    winners = merged.sort_values("change", ascending=False).head(10)
    losers = merged.sort_values("change", ascending=True).head(10)
    return winners, losers, prior, latest


def scan_universe(tickers, workers=8, progress_callback=None):
    rows = []
    tickers = list(dict.fromkeys(tickers))

    def score_one(ticker):
        result = fetch_stock(ticker)
        score, _, _ = score_stock(result["metrics"])
        scoring_keys = ["eps_growth", "revenue_growth", "net_margin", "roic", "forward_pe", "analyst_conviction", "institutional_ownership", "insider_activity", "momentum", "chart_health"]
        coverage = sum(result["metrics"].get(k) is not None for k in scoring_keys)
        if score is None or coverage < 7:
            return None
        return {
            "ticker": ticker,
            "company": result["company"],
            "sector": result.get("sector") or "Other",
            "industry": result.get("industry") or "—",
            "score": round(score, 2),
            "coverage": coverage,
            "market_cap": result.get("market_cap"),
            "eps_growth": result["metrics"].get("eps_growth"),
            "revenue_growth": result["metrics"].get("revenue_growth"),
            "analyst_upside": result["metrics"].get("analyst_upside"),
            "analyst_conviction": result["metrics"].get("analyst_conviction"),
            "analyst_count": result.get("analyst_count"),
            "institutional_ownership": result["metrics"].get("institutional_ownership"),
            "insider_activity": result["metrics"].get("insider_activity"),
            "chart_health": result["metrics"].get("chart_health"),
            "momentum_3m": result["metrics"].get("momentum_3m"),
            "momentum_6m": result["metrics"].get("momentum_6m"),
            "one_year_return": result["metrics"].get("momentum"),
            "forward_pe": result["metrics"].get("forward_pe"),
            "estimate_revision_pct": result["metrics"].get("estimate_revision_pct"),
            "estimate_revision_breadth": result["metrics"].get("estimate_revision_breadth"),
            "estimate_revision_score": result["metrics"].get("estimate_revision_score"),
            "forward_eps_growth": result["metrics"].get("forward_eps_growth"),
            "value_vs_growth_score": result["metrics"].get("value_vs_growth_score"),
            "target_mean": result.get("target_mean"),
            "price": result.get("price"),
            "fetched_at": result["fetched_at"],
        }

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 12))) as pool:
        futures = {pool.submit(score_one, t): t for t in tickers}
        for future in as_completed(futures):
            try:
                row = future.result()
                if row:
                    rows.append(row)
            except Exception:
                pass
            done += 1
            if progress_callback:
                progress_callback(done, len(tickers))

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def add_sector_relative_hidden_features(df):
    """Add sector-relative percentiles so one sector's natural economics do not dominate Hidden Gems."""
    if df is None or df.empty:
        return df
    work = df.copy()
    if "sector" not in work.columns:
        work["sector"] = "Other"
    work["sector"] = work["sector"].fillna("Other").replace("—", "Other")

    for col, out in [
        ("score", "sector_quality_pct"),
        ("value_vs_growth_score", "sector_value_pct"),
        ("chart_health", "sector_chart_pct"),
    ]:
        vals = pd.to_numeric(work.get(col), errors="coerce")
        work[col] = vals
        work[out] = work.groupby("sector")[col].rank(pct=True, method="average") * 100
        # Tiny sector samples are not trustworthy percentiles; fall back toward neutral.
        counts = work.groupby("sector")[col].transform("count")
        work.loc[counts < 3, out] = 50.0
    return work


def hidden_gem_score(row):
    """Hidden Gem = quality first, then revisions/value/technical health and relative underfollowedness."""
    score = clean_num(row.get("score"))
    coverage = clean_num(row.get("coverage"))
    market_cap = clean_num(row.get("market_cap"))
    analysts = clean_num(row.get("analyst_count"))
    chart = clean_num(row.get("chart_health"))
    eps = clean_num(row.get("eps_growth"))
    revenue = clean_num(row.get("revenue_growth"))
    inst = clean_num(row.get("institutional_ownership"))
    insider = clean_num(row.get("insider_activity"))
    revision = clean_num(row.get("estimate_revision_score"))
    value_growth = clean_num(row.get("value_vs_growth_score"))
    sector_quality = clean_num(row.get("sector_quality_pct"))
    sector_value = clean_num(row.get("sector_value_pct"))

    # Quality floor only. Analyst price targets are intentionally NOT a gate.
    if score is None or score < 66 or coverage is None or coverage < 7:
        return None
    if market_cap is None or not (2e9 <= market_cap <= 300e9):
        return None
    if analysts is None or not (5 <= analysts <= 35):
        return None
    if chart is None or chart < 52:
        return None

    # Materially falling forward estimates are not a Hidden Gem. Missing revisions are allowed; genuinely falling revisions are not.
    revision_pct = clean_num(row.get("estimate_revision_pct"))
    if revision_pct is not None and revision_pct < -5:
        return None

    growth_values = [x for x in (eps, revenue) if x is not None]
    if not growth_values or max(growth_values) < 4:
        return None
    if len(growth_values) == 2 and eps < -8 and revenue < -8:
        return None
    if inst is not None and inst < 20:
        return None
    if insider is not None and insider < -5:
        return None

    # Distinguish normal quality compounders from improving turnarounds.
    fwd_growth = clean_num(row.get("forward_eps_growth"))
    quality_growth_ok = (fwd_growth is not None and fwd_growth > 0) or (revenue is not None and revenue >= 5)
    turnaround_ok = (
        fwd_growth is not None and fwd_growth <= 0
        and revision_pct is not None and revision_pct >= 3
        and chart >= 70
        and (clean_num(row.get("forward_pe")) is None or clean_num(row.get("forward_pe")) <= 35)
    )
    if not quality_growth_ok and not turnaround_ok:
        return None

    # 25% quality; 20% estimate revisions; 20% value vs growth; 15% chart; 10% institutional support; 10% underfollowed.
    absolute_quality = normalize(score, 66, 88)
    quality = absolute_quality
    if sector_quality is not None and absolute_quality is not None:
        quality = 0.60 * absolute_quality + 0.40 * sector_quality

    vg = value_growth
    if value_growth is not None and sector_value is not None:
        vg = 0.65 * value_growth + 0.35 * sector_value

    chart_part = normalize(chart, 52, 92)
    inst_part = institutional_score(inst) if inst is not None else None
    analyst_underfollowed = curve_score(analysts, [(5, 92), (8, 88), (12, 78), (18, 65), (25, 50), (35, 35)])
    cap_underfollowed = normalize(math.log10(market_cap), math.log10(2e9), math.log10(300e9), reverse=True)
    underfollowed = 0.60 * analyst_underfollowed + 0.40 * cap_underfollowed

    parts = [
        (quality, 0.25),
        (revision, 0.20),
        (vg, 0.20),
        (chart_part, 0.15),
        (inst_part, 0.10),
        (underfollowed, 0.10),
    ]
    usable = [(v, w) for v, w in parts if v is not None]
    # Require enough specific Hidden Gem evidence; otherwise the generic Conviction score is doing too much work.
    if len(usable) < 4:
        return None
    result = sum(v*w for v, w in usable) / sum(w for v, w in usable)
    return round(max(0.0, min(100.0, result)), 1)


def hidden_gem_type(row):
    """Classify the reason a qualifying name is interesting without pretending every setup is the same."""
    fwd = clean_num(row.get("forward_eps_growth"))
    rev = clean_num(row.get("revenue_growth"))
    revision = clean_num(row.get("estimate_revision_pct"))
    chart = clean_num(row.get("chart_health"))
    if fwd is not None and fwd <= 0 and revision is not None and revision >= 3 and (chart or 0) >= 70:
        return "Turnaround Gem"
    if (fwd is not None and fwd > 0) or (rev is not None and rev >= 5):
        return "Quality Gem"
    return "Discovery Candidate"


def hidden_gem_reasons(row):
    reasons = []
    score = clean_num(row.get("score"))
    revision_pct = clean_num(row.get("estimate_revision_pct"))
    revision_score = clean_num(row.get("estimate_revision_score"))
    vg = clean_num(row.get("value_vs_growth_score"))
    fwd_growth = clean_num(row.get("forward_eps_growth"))
    pe = clean_num(row.get("forward_pe"))
    analysts = clean_num(row.get("analyst_count"))
    chart = clean_num(row.get("chart_health"))
    sector = row.get("sector") or "its sector"
    sector_quality = clean_num(row.get("sector_quality_pct"))

    if score is not None:
        reasons.append(f"Conviction Score {score:.1f}/100")
    if sector_quality is not None and sector_quality >= 65:
        reasons.append(f"Quality ranks well versus other {sector} companies")
    if revision_pct is not None:
        direction = "rising" if revision_pct > 1 else "falling" if revision_pct < -1 else "stable"
        shown = ">+50%" if revision_pct > 50 else "<-50%" if revision_pct < -50 else f"{revision_pct:+.1f}%"
        reasons.append(f"EPS estimates are {direction} ({shown} vs ~90 days ago)")
    elif revision_score is not None:
        reasons.append(f"Estimate revision score {revision_score:.0f}/100")
    if vg is not None:
        label_vg = "Attractive" if vg >= 70 else "Reasonable" if vg >= 55 else "Mixed"
        reasons.append(f"Value vs growth: {label_vg} ({vg:.0f}/100)")
    elif pe is not None and fwd_growth is not None:
        reasons.append(f"Forward P/E {pe:.1f}x vs expected EPS growth {fwd_growth:.1f}%")
    if chart is not None:
        reasons.append(f"Chart health {chart:.0f}/100")
    if analysts is not None:
        reasons.append(f"Moderate coverage: {int(analysts)} analysts")
    return reasons[:5]


def pick_hidden_gem_from_df(df):
    if df is None or df.empty:
        return None, pd.DataFrame()
    work = add_sector_relative_hidden_features(df.copy())
    work["hidden_gem_score"] = work.apply(hidden_gem_score, axis=1)
    pool = work.dropna(subset=["hidden_gem_score"]).copy()
    pool = pool[pool["hidden_gem_score"] >= 65].copy()
    if pool.empty:
        return None, pool

    # Sector diversification: at most two qualifying names per sector in the discovery pool.
    pool = pool.sort_values(["hidden_gem_score", "score"], ascending=False)
    diversified = pool.groupby("sector", group_keys=False).head(2).copy()
    diversified = diversified.sort_values("hidden_gem_score", ascending=False)
    top_pool = diversified.head(min(30, len(diversified)))
    row = top_pool.sample(1).iloc[0].to_dict()
    return row, diversified


def lightweight_hidden_gem_scan(sample_size=120, workers=6, progress_callback=None):
    sp500 = build_market_universe("S&P 500")
    tickers = sp500["ticker"].dropna().astype(str).tolist()
    if not tickers:
        return pd.DataFrame()
    sample_size = min(sample_size, len(tickers))
    sample = random.sample(tickers, sample_size)
    return scan_universe(sample, workers=workers, progress_callback=progress_callback)


def label(score):
    if score is None: return "Insufficient Data"
    if score >= 85: return "Very High Conviction"
    if score >= 75: return "High Conviction"
    if score >= 65: return "Above Average"
    if score >= 50: return "Neutral"
    return "Low Conviction"


def fmt(v, suffix="%", digits=1):
    return "N/A" if v is None else f"{v:,.{digits}f}{suffix}"


def market_cap_fmt(v):
    if v is None: return "N/A"
    for n, s in [(1e12, "T"), (1e9, "B"), (1e6, "M")]:
        if abs(v) >= n: return f"${v/n:.2f}{s}"
    return f"${v:,.0f}"


ETF_UNIVERSE = [
    # Broad market / core
    ("VOO", "Vanguard S&P 500", "Broad Market", False), ("IVV", "iShares Core S&P 500", "Broad Market", False),
    ("SPY", "SPDR S&P 500", "Broad Market", False), ("SPLG", "SPDR Portfolio S&P 500", "Broad Market", False),
    ("VTI", "Vanguard Total Stock Market", "Broad Market", False), ("ITOT", "iShares Core S&P Total U.S.", "Broad Market", False),
    ("SCHB", "Schwab U.S. Broad Market", "Broad Market", False), ("RSP", "Invesco S&P 500 Equal Weight", "Broad Market", False),
    # Growth
    ("VOOG", "Vanguard S&P 500 Growth", "Growth", False), ("SCHG", "Schwab U.S. Large-Cap Growth", "Growth", False),
    ("VUG", "Vanguard Growth", "Growth", False), ("IWF", "iShares Russell 1000 Growth", "Growth", False),
    ("SPYG", "SPDR Portfolio S&P 500 Growth", "Growth", False), ("MGK", "Vanguard Mega Cap Growth", "Growth", False),
    ("QQQM", "Invesco Nasdaq-100", "Growth", False), ("QQQ", "Invesco QQQ", "Growth", False),
    # Technology / semis
    ("VGT", "Vanguard Information Technology", "Technology", False), ("XLK", "Technology Select Sector SPDR", "Technology", False),
    ("FTEC", "Fidelity MSCI Information Technology", "Technology", False), ("IYW", "iShares U.S. Technology", "Technology", False),
    ("IGV", "iShares Expanded Tech-Software", "Technology", False), ("CIBR", "First Trust Nasdaq Cybersecurity", "Technology", False),
    ("SMH", "VanEck Semiconductor", "Semiconductors", False), ("SOXX", "iShares Semiconductor", "Semiconductors", False),
    ("XSD", "SPDR S&P Semiconductor", "Semiconductors", False), ("PSI", "Invesco Semiconductors", "Semiconductors", False),
    # Momentum / quality / moat
    ("SPMO", "Invesco S&P 500 Momentum", "Momentum", False), ("MTUM", "iShares MSCI USA Momentum", "Momentum", False),
    ("PDP", "Invesco Dorsey Wright Momentum", "Momentum", False),
    ("MOAT", "VanEck Morningstar Wide Moat", "Quality", False), ("QUAL", "iShares MSCI USA Quality", "Quality", False),
    ("SPHQ", "Invesco S&P 500 Quality", "Quality", False), ("JQUA", "JPMorgan U.S. Quality Factor", "Quality", False),
    # Value / dividend
    ("VTV", "Vanguard Value", "Value", False), ("SCHV", "Schwab U.S. Large-Cap Value", "Value", False),
    ("IWD", "iShares Russell 1000 Value", "Value", False), ("SPYV", "SPDR Portfolio S&P 500 Value", "Value", False),
    ("SCHD", "Schwab U.S. Dividend Equity", "Dividend", False), ("VIG", "Vanguard Dividend Appreciation", "Dividend", False),
    ("DGRO", "iShares Core Dividend Growth", "Dividend", False), ("VYM", "Vanguard High Dividend Yield", "Dividend", False),
    ("DGRW", "WisdomTree U.S. Quality Dividend Growth", "Dividend", False),
    # Size
    ("IJH", "iShares Core S&P Mid-Cap", "Mid Cap", False), ("VO", "Vanguard Mid-Cap", "Mid Cap", False),
    ("MDY", "SPDR S&P MidCap 400", "Mid Cap", False), ("XMMO", "Invesco S&P MidCap Momentum", "Mid Cap", False),
    ("IJR", "iShares Core S&P Small-Cap", "Small Cap", False), ("VB", "Vanguard Small-Cap", "Small Cap", False),
    ("IWM", "iShares Russell 2000", "Small Cap", False), ("AVUV", "Avantis U.S. Small Cap Value", "Small Cap", False),
    # International
    ("VXUS", "Vanguard Total International Stock", "International", False), ("VEA", "Vanguard FTSE Developed Markets", "International", False),
    ("VWO", "Vanguard FTSE Emerging Markets", "International", False), ("IEFA", "iShares Core MSCI EAFE", "International", False),
    ("IEMG", "iShares Core MSCI Emerging Markets", "International", False), ("SCHF", "Schwab International Equity", "International", False),
    # Sectors
    ("XLE", "Energy Select Sector SPDR", "Sector", False), ("XLF", "Financial Select Sector SPDR", "Sector", False),
    ("XLV", "Health Care Select Sector SPDR", "Sector", False), ("XLI", "Industrial Select Sector SPDR", "Sector", False),
    ("XLY", "Consumer Discretionary Select Sector SPDR", "Sector", False), ("XLP", "Consumer Staples Select Sector SPDR", "Sector", False),
    ("XLU", "Utilities Select Sector SPDR", "Sector", False), ("XLRE", "Real Estate Select Sector SPDR", "Sector", False),
    ("XLB", "Materials Select Sector SPDR", "Sector", False), ("XLC", "Communication Services Select Sector SPDR", "Sector", False),
    # Themes / alternatives
    ("IBIT", "iShares Bitcoin Trust", "Alternative", False), ("ARKK", "ARK Innovation", "Thematic", False),
    ("BOTZ", "Global X Robotics & AI", "Thematic", False), ("AIQ", "Global X Artificial Intelligence & Technology", "Thematic", False),
    ("URA", "Global X Uranium", "Thematic", False), ("NLR", "VanEck Uranium and Nuclear", "Thematic", False),
    ("QTUM", "Defiance Quantum", "Thematic", False), ("ROBO", "ROBO Global Robotics & Automation", "Thematic", False),
    # Bonds / defensive
    ("BND", "Vanguard Total Bond Market", "Bonds", False), ("AGG", "iShares Core U.S. Aggregate Bond", "Bonds", False),
    ("SGOV", "iShares 0-3 Month Treasury Bond", "Bonds", False), ("TLT", "iShares 20+ Year Treasury Bond", "Bonds", False),
    # Leveraged — hidden by default
    # Additional broad / style funds to make category filters useful
    ("VT", "Vanguard Total World Stock", "Broad Market", False), ("ACWI", "iShares MSCI ACWI", "Broad Market", False),
    ("VV", "Vanguard Large-Cap", "Large Cap", False), ("SCHX", "Schwab U.S. Large-Cap", "Large Cap", False),
    ("MGC", "Vanguard Mega Cap", "Large Cap", False), ("VONE", "Vanguard Russell 1000", "Large Cap", False),
    ("VONG", "Vanguard Russell 1000 Growth", "Growth", False), ("IUSG", "iShares Core S&P U.S. Growth", "Growth", False),
    ("IWY", "iShares Russell Top 200 Growth", "Growth", False), ("QGRO", "American Century U.S. Quality Growth", "Growth", False),
    ("VBR", "Vanguard Small-Cap Value", "Small Cap", False), ("IJS", "iShares S&P Small-Cap 600 Value", "Small Cap", False),
    ("SCHA", "Schwab U.S. Small-Cap", "Small Cap", False), ("VXF", "Vanguard Extended Market", "Mid Cap", False),
    ("IUSV", "iShares Core S&P U.S. Value", "Value", False), ("VONV", "Vanguard Russell 1000 Value", "Value", False),
    ("USMV", "iShares MSCI USA Min Vol Factor", "Quality", False), ("SPLV", "Invesco S&P 500 Low Volatility", "Quality", False),
    ("OMFL", "Invesco Russell 1000 Dynamic Multifactor", "Quality", False), ("COWZ", "Pacer U.S. Cash Cows 100", "Quality", False),
    ("FDVV", "Fidelity High Dividend", "Dividend", False), ("HDV", "iShares Core High Dividend", "Dividend", False),
    ("NOBL", "ProShares S&P 500 Dividend Aristocrats", "Dividend", False), ("SDY", "SPDR S&P Dividend", "Dividend", False),
    ("IXUS", "iShares Core MSCI Total International", "International", False), ("EFA", "iShares MSCI EAFE", "International", False),
    ("SPDW", "SPDR Portfolio Developed World ex-US", "International", False), ("EMXC", "iShares MSCI Emerging Markets ex China", "International", False),
    ("EWJ", "iShares MSCI Japan", "International", False), ("INDA", "iShares MSCI India", "International", False),
    ("XHB", "SPDR S&P Homebuilders", "Sector", False), ("ITA", "iShares U.S. Aerospace & Defense", "Sector", False),
    ("PAVE", "Global X U.S. Infrastructure Development", "Sector", False), ("IHI", "iShares U.S. Medical Devices", "Sector", False),
    ("ICLN", "iShares Global Clean Energy", "Thematic", False), ("LIT", "Global X Lithium & Battery Tech", "Thematic", False),
    ("COPX", "Global X Copper Miners", "Thematic", False), ("GRID", "First Trust NASDAQ Clean Edge Smart Grid Infrastructure", "Thematic", False),
    ("HACK", "ETFMG Prime Cyber Security", "Technology", False), ("SKYY", "First Trust Cloud Computing", "Technology", False),
    ("VGSH", "Vanguard Short-Term Treasury", "Bonds", False), ("VGIT", "Vanguard Intermediate-Term Treasury", "Bonds", False),
    ("VCIT", "Vanguard Intermediate-Term Corporate Bond", "Bonds", False), ("TIP", "iShares TIPS Bond", "Bonds", False),
    ("TQQQ", "ProShares UltraPro QQQ", "Leveraged", True), ("SOXL", "Direxion Daily Semiconductor Bull 3X", "Leveraged", True),
    ("UPRO", "ProShares UltraPro S&P500", "Leveraged", True), ("SPXL", "Direxion Daily S&P 500 Bull 3X", "Leveraged", True),
]



# ETFs can belong to more than one beginner-friendly bucket.
ETF_EXTRA_TAGS = {
    "VOO": ["Core", "Large Cap", "US Equity"], "IVV": ["Core", "Large Cap", "US Equity"],
    "SPY": ["Core", "Large Cap", "US Equity"], "SPLG": ["Core", "Large Cap", "US Equity"],
    "VTI": ["Core", "US Equity", "Total Market"], "ITOT": ["Core", "US Equity", "Total Market"], "SCHB": ["Core", "US Equity", "Total Market"],
    "VOOG": ["Growth", "Large Cap", "US Equity"], "SCHG": ["Growth", "Large Cap", "US Equity"], "VUG": ["Growth", "Large Cap", "US Equity"],
    "IWF": ["Growth", "Large Cap", "US Equity"], "SPYG": ["Growth", "Large Cap", "US Equity"], "MGK": ["Growth", "Mega Cap", "US Equity"],
    "QQQM": ["Growth", "Nasdaq", "Technology-heavy", "US Equity"], "QQQ": ["Growth", "Nasdaq", "Technology-heavy", "US Equity"],
    "VGT": ["Technology", "Growth", "US Equity"], "XLK": ["Technology", "Growth", "US Equity"], "FTEC": ["Technology", "Growth", "US Equity"],
    "SMH": ["Semiconductors", "Technology", "Growth"], "SOXX": ["Semiconductors", "Technology", "Growth"], "XSD": ["Semiconductors", "Technology", "Growth"],
    "SPMO": ["Momentum", "Growth", "Large Cap"], "MTUM": ["Momentum", "Large Cap"],
    "MOAT": ["Quality", "Large Cap"], "QUAL": ["Quality", "Large Cap"], "SPHQ": ["Quality", "Large Cap"],
    "VTV": ["Value", "Large Cap"], "SCHV": ["Value", "Large Cap"], "IWD": ["Value", "Large Cap"],
    "SCHD": ["Dividend", "Income", "Quality"], "VIG": ["Dividend", "Income", "Quality"], "DGRO": ["Dividend", "Income", "Growth"], "VYM": ["Dividend", "Income"],
    "IJH": ["Mid Cap", "US Equity"], "VO": ["Mid Cap", "US Equity"], "XMMO": ["Mid Cap", "Momentum", "US Equity"],
    "IJR": ["Small Cap", "US Equity"], "VB": ["Small Cap", "US Equity"], "IWM": ["Small Cap", "US Equity"], "AVUV": ["Small Cap", "Value", "US Equity"],
    "VXUS": ["International", "Core", "Global"], "VEA": ["International", "Developed Markets"], "VWO": ["International", "Emerging Markets"],
    "IEFA": ["International", "Developed Markets"], "IEMG": ["International", "Emerging Markets"],
    "BND": ["Bonds", "Core", "Defensive"], "AGG": ["Bonds", "Core", "Defensive"], "SGOV": ["Bonds", "Cash-like", "Defensive"], "TLT": ["Bonds", "Long Duration"],
    "ARKK": ["Thematic", "Aggressive"], "BOTZ": ["Thematic", "AI", "Aggressive"], "AIQ": ["Thematic", "AI", "Aggressive"],
    "URA": ["Thematic", "Uranium", "Aggressive"], "NLR": ["Thematic", "Uranium", "Aggressive"], "QTUM": ["Thematic", "Quantum", "Aggressive"],
    "IBIT": ["Alternative", "Bitcoin", "Aggressive"],
}

def etf_tags(row):
    ticker, _name, category, _lev = row
    return sorted(set([category] + ETF_EXTRA_TAGS.get(ticker, [])))

def all_etf_filter_tags(include_leveraged=False):
    tags = set()
    for row in ETF_UNIVERSE:
        if row[3] and not include_leveraged:
            continue
        tags.update(etf_tags(row))
    preferred = ["All", "Core", "Growth", "Large Cap", "Technology", "Semiconductors", "Quality", "Momentum", "Dividend", "Value", "Small Cap", "Mid Cap", "International", "Bonds", "Thematic", "Aggressive"]
    rest = sorted(tags - set(preferred))
    return [x for x in preferred if x == "All" or x in tags] + rest


def _price_on_or_after(closes, date):
    if closes is None or closes.empty:
        return None, None
    idx = closes.index
    try:
        if getattr(idx, "tz", None) is not None:
            date = pd.Timestamp(date, tz=idx.tz)
        else:
            date = pd.Timestamp(date).tz_localize(None)
    except Exception:
        date = pd.Timestamp(date)
    subset = closes[closes.index >= date]
    if subset.empty:
        return None, None
    return float(subset.iloc[0]), subset.index[0]


def _trailing_return(closes, years=None, ytd=False):
    if closes is None or closes.empty or len(closes) < 2:
        return None
    end_price = float(closes.iloc[-1])
    end_date = closes.index[-1]
    if ytd:
        start_date = pd.Timestamp(year=end_date.year, month=1, day=1)
    else:
        start_date = pd.Timestamp(end_date) - pd.DateOffset(years=years)
    start_price, actual_start = _price_on_or_after(closes, start_date)
    if start_price is None or start_price <= 0:
        return None
    total = end_price / start_price - 1
    if ytd or years == 1:
        return total * 100
    days = max((pd.Timestamp(end_date).tz_localize(None) - pd.Timestamp(actual_start).tz_localize(None)).days, 1)
    return ((end_price / start_price) ** (365.25 / days) - 1) * 100


def _max_drawdown(closes, years=5):
    if closes is None or closes.empty:
        return None
    end_date = closes.index[-1]
    start_date = pd.Timestamp(end_date) - pd.DateOffset(years=years)
    subset = closes[closes.index >= start_date]
    if len(subset) < 30:
        return None
    dd = subset / subset.cummax() - 1
    return float(dd.min() * 100)


def _annualized_vol(closes, years=5):
    if closes is None or closes.empty:
        return None
    end_date = closes.index[-1]
    start_date = pd.Timestamp(end_date) - pd.DateOffset(years=years)
    subset = closes[closes.index >= start_date]
    returns = subset.pct_change().dropna()
    if len(returns) < 30:
        return None
    return float(returns.std() * (252 ** 0.5) * 100)


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_etf_metrics(ticker):
    t = yf.Ticker(ticker)
    try:
        hist = t.history(period="10y", auto_adjust=True)
    except Exception:
        hist = pd.DataFrame()
    closes = hist["Close"].dropna() if not hist.empty and "Close" in hist else pd.Series(dtype=float)
    try:
        info = t.get_info() or {}
    except Exception:
        info = {}

    return {
        "ticker": ticker,
        "ytd": _trailing_return(closes, ytd=True),
        "1Y": _trailing_return(closes, years=1),
        "3Y": _trailing_return(closes, years=3),
        "5Y": _trailing_return(closes, years=5),
        "10Y": _trailing_return(closes, years=10),
        "volatility_5y": _annualized_vol(closes, years=5),
        "max_drawdown_5y": _max_drawdown(closes, years=5),
        "expense_ratio": pct(info.get("annualReportExpenseRatio")),
        "aum": clean_num(info.get("totalAssets")),
        "avg_volume": clean_num(info.get("averageVolume")),
    }


def etf_all_around_score(row):
    """Beginner-oriented ETF score. Rewards durable returns and tradability; penalizes risk/cost."""
    pieces = {
        "5Y CAGR": (normalize(row.get("5Y"), 0, 25), 0.30),
        "10Y CAGR": (normalize(row.get("10Y"), 0, 22), 0.25),
        "Volatility": (normalize(row.get("volatility_5y"), 10, 40, reverse=True), 0.15),
        "Max Drawdown": (normalize(row.get("max_drawdown_5y"), -55, -10), 0.15),
        "Expense Ratio": (normalize(row.get("expense_ratio"), 0.03, 0.75, reverse=True), 0.08),
        "AUM": (normalize(math.log10(row.get("aum")) if clean_num(row.get("aum")) and row.get("aum") > 0 else None, 8, 11.5), 0.04),
        "Liquidity": (normalize(math.log10(row.get("avg_volume")) if clean_num(row.get("avg_volume")) and row.get("avg_volume") > 0 else None, 4, 7.5), 0.03),
    }
    available = [(score, weight) for score, weight in pieces.values() if score is not None]
    if not available:
        return None
    w = sum(weight for _, weight in available)
    return round(sum(score * weight for score, weight in available) / w, 1)


def scan_etfs(rows, workers=6):
    base = pd.DataFrame(rows, columns=["ticker", "name", "category", "leveraged"])
    metrics = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 10))) as pool:
        futures = {pool.submit(fetch_etf_metrics, t): t for t in base["ticker"]}
        for future in as_completed(futures):
            try:
                metrics.append(future.result())
            except Exception:
                pass
    if not metrics:
        return pd.DataFrame()
    out = base.merge(pd.DataFrame(metrics), on="ticker", how="left")
    out["all_around"] = out.apply(etf_all_around_score, axis=1)
    return out





@st.cache_resource(show_spinner=False)
def get_openai_client():
    """Create an optional OpenAI client from Streamlit Secrets / environment variables."""
    api_key = _secret("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


@st.cache_data(ttl=86400, show_spinner=False)
def two_sentence_ai_take(kind, facts):
    """Summarize only the supplied app facts. Returns None when AI is not configured."""
    client = get_openai_client()
    if client is None:
        return None
    model = _secret("OPENAI_MODEL") or "gpt-5-mini"
    try:
        response = client.responses.create(
            model=model,
            instructions=(
                "You write for a beginner investing education app. Use ONLY the facts supplied by the app. "
                "Write exactly two concise sentences in plain English. Do not give a buy/sell instruction, "
                "do not promise returns, and do not invent facts. Explain what stands out and the main tradeoff or risk."
            ),
            input=f"Analysis type: {kind}\nApp facts:\n{facts}",
            max_output_tokens=120,
        )
        text = (getattr(response, "output_text", "") or "").strip()
        return text or None
    except Exception:
        return None


def hidden_gem_fallback_take(gem):
    ticker = gem.get("ticker", "This company")
    score = clean_num(gem.get("score"))
    chart = clean_num(gem.get("chart_health"))
    rev = clean_num(gem.get("estimate_revision_pct"))
    vg = clean_num(gem.get("value_vs_growth_score"))
    pieces = []
    if score is not None: pieces.append(f"a {score:.0f}/100 Conviction Score")
    if rev is not None: pieces.append(f"EPS estimates {('rising' if rev > 1 else 'roughly stable' if rev >= -1 else 'falling')} by {rev:+.1f}% versus roughly 90 days ago")
    if vg is not None: pieces.append(f"a {vg:.0f}/100 value-vs-growth score")
    if chart is not None: pieces.append(f"{chart:.0f}/100 chart health")
    evidence = ", ".join(pieces[:3]) if pieces else "several quality signals"
    return f"{ticker} surfaced because it combines {evidence}. It is a discovery candidate rather than a recommendation, so the next step is reviewing the full 10-factor breakdown and the risks that could weaken the thesis."


def dca_fallback_take(goal, risk, daily_amount, model):
    core = sum(w for _t, w, role in model if "Core" in role or "Broad" in role or "Dividend core" in role)
    spicy = sum(w for _t, w, role in model if "Spicy" in role or "Theme" in role or "Momentum" in role or "Nasdaq" in role)
    first = f"This ${daily_amount:.0f}-per-trading-day {risk.lower()} example keeps about {core}% in core exposure and about {spicy}% in higher-octane satellites."
    second = f"It is built for the '{goal}' goal, but the more satellite exposure you choose, the more short-term swings you should expect."
    return first + " " + second


def _quarter_change_map(snapshot_df):
    """Return ticker -> last reported quarter-to-quarter Conviction Score change."""
    if snapshot_df is None or snapshot_df.empty:
        return {}
    work = snapshot_df.copy()
    work["coverage"] = pd.to_numeric(work["coverage"], errors="coerce")
    work["score"] = pd.to_numeric(work["score"], errors="coerce")
    work = work[work["coverage"] >= 7]
    quarters = sorted(work["quarter"].dropna().astype(str).unique(), key=_quarter_sort_key)
    if len(quarters) < 2:
        return {}
    prior, latest = quarters[-2], quarters[-1]
    a = work[work["quarter"] == latest][["ticker", "score"]].rename(columns={"score":"current"})
    b = work[work["quarter"] == prior][["ticker", "score"]].rename(columns={"score":"prior"})
    m = a.merge(b, on="ticker")
    m["change"] = m["current"] - m["prior"]
    return dict(zip(m["ticker"].astype(str), m["change"]))


def _cap(value, low, high):
    if value is None:
        return None
    return max(low, min(high, float(value)))


def _growth_trend_label(value):
    """Beginner-friendly display label; raw extreme growth is intentionally hidden."""
    v = clean_num(value)
    if v is None:
        return "Not enough data"
    if v < 0:
        return "Declining"
    if v < 5:
        return "Flat"
    if v < 15:
        return "Improving"
    if v < 30:
        return "Strong"
    if v < 60:
        return "Very strong"
    return "Exceptional (capped)"


@st.cache_data(ttl=3600, show_spinner=False)
def _benchmark_momentum():
    """S&P 500 momentum used so Emerging Leaders rewards market-relative acceleration."""
    try:
        hist = yf.Ticker("SPY").history(period="1y", auto_adjust=True)
        return {
            "m3": momentum_period(hist, 63),
            "m6": momentum_period(hist, 126),
            "m12": momentum_12m(hist),
        }
    except Exception:
        return {"m3": None, "m6": None, "m12": None}


def _relative_momentum_accel(row, benchmark):
    """Acceleration versus both the stock's own 12M pace and the S&P 500's recent pace."""
    m3 = clean_num(row.get("momentum_3m"))
    m6 = clean_num(row.get("momentum_6m"))
    m12 = clean_num(row.get("one_year_return"))
    b3, b6, b12 = (clean_num(benchmark.get(k)) for k in ("m3", "m6", "m12"))

    parts = []
    # Own-pacing acceleration.
    if m3 is not None and m12 is not None:
        own3 = m3 - m12 / 4.0
        if b3 is not None and b12 is not None:
            own3 -= (b3 - b12 / 4.0)
        parts.append(own3)
    if m6 is not None and m12 is not None:
        own6 = m6 - m12 / 2.0
        if b6 is not None and b12 is not None:
            own6 -= (b6 - b12 / 2.0)
        parts.append(own6)
    if not parts:
        return None
    # Gentle winsorization only for pathological prints; public display remains differentiated.
    return _cap(sum(parts) / len(parts), -30, 30)


def _momentum_accel_label(value):
    v = clean_num(value)
    if v is None:
        return "Not enough history"
    if v < 0:
        return "Lagging"
    if v < 3:
        return "Mild"
    if v < 7:
        return "Improving"
    if v < 12:
        return "Strong"
    if v < 18:
        return "Breakout"
    return "Exceptional"


def _why_emerging(row, qoq_change=None):
    """Pick the two strongest distinct reasons so rows do not all read the same."""
    accel = clean_num(row.get("Momentum Accel"))
    eps = clean_num(row.get("eps_growth"))
    rev = clean_num(row.get("revenue_growth"))
    chart = clean_num(row.get("chart_health"))
    revisions = clean_num(row.get("estimate_revision_pct"))
    pe = clean_num(row.get("forward_pe"))
    fwd = clean_num(row.get("forward_eps_growth"))

    candidates = []
    if qoq_change is not None and qoq_change >= 2:
        candidates.append((95 + min(qoq_change, 10), "QoQ score rising"))
    if revisions is not None and revisions >= 3:
        candidates.append((90 + min(revisions, 15) / 2, "EPS estimates rising"))
    if accel is not None and accel >= 12:
        candidates.append((88 + min(accel, 25) / 4, "Relative-strength breakout"))
    elif accel is not None and accel >= 5:
        candidates.append((80 + accel / 4, "Momentum improving vs S&P 500"))
    if rev is not None and rev >= 20:
        candidates.append((84 + min(rev, 60) / 8, "Revenue growth strong"))
    elif rev is not None and rev >= 10:
        candidates.append((74 + rev / 10, "Revenue trend improving"))
    if eps is not None and eps >= 25:
        candidates.append((82 + min(eps, 60) / 8, "EPS growth strong"))
    elif eps is not None and eps >= 12:
        candidates.append((73 + eps / 10, "EPS trend improving"))
    if chart is not None and chart >= 88:
        candidates.append((83 + chart / 20, "Very healthy chart"))
    elif chart is not None and chart >= 75:
        candidates.append((72 + chart / 25, "Healthy chart"))
    if pe is not None and fwd is not None and fwd > 0 and pe <= max(20, fwd * 1.25):
        candidates.append((76, "Valuation supports growth"))

    if not candidates:
        return "Multiple signals improving"

    candidates.sort(key=lambda x: x[0], reverse=True)
    picked = []
    for _strength, text in candidates:
        if text not in picked:
            picked.append(text)
        if len(picked) == 2:
            break
    return " + ".join(picked)


def emerging_leader_score(row, qoq_change=None):
    score = clean_num(row.get("score"))
    coverage = clean_num(row.get("coverage"))
    chart = clean_num(row.get("chart_health"))
    momentum_accel = clean_num(row.get("Momentum Accel"))
    eps = clean_num(row.get("eps_growth"))
    revenue = clean_num(row.get("revenue_growth"))
    inst = clean_num(row.get("institutional_ownership"))
    pe = clean_num(row.get("forward_pe"))

    # Keep this screen clearly below the elite absolute-score leaderboard.
    if score is None or coverage is None or coverage < 7 or score < 58 or score > 82:
        return None
    if chart is not None and chart < 52:
        return None
    if qoq_change is not None and qoq_change < -4:
        return None

    # Extreme base-effect growth is capped for scoring, and raw spikes are not shown in the table.
    eps_c = _cap(eps, -30, 60)
    rev_c = _cap(revenue, -20, 60)
    growth_vals = [x for x in (eps_c, rev_c) if x is not None]
    growth_quality = None if not growth_vals else sum(max(0, x) for x in growth_vals) / len(growth_vals)

    # Require at least two independent signs of improvement/strength.
    signals = 0
    if momentum_accel is not None and momentum_accel >= 2.5:
        signals += 1
    if qoq_change is not None and qoq_change >= 2:
        signals += 1
    if growth_quality is not None and growth_quality >= 10:
        signals += 1
    if chart is not None and chart >= 68:
        signals += 1
    if signals < 2:
        return None

    # Smooth curves avoid the old +25 ceiling pile-up and keep 90+ rare.
    accel_score = None if momentum_accel is None else curve_score(momentum_accel, [
        (-12, 15), (-7, 28), (-3, 40), (0, 48), (2, 54), (5, 62), (8, 70), (12, 79), (16, 85), (22, 90), (30, 94)
    ])
    qoq_score = None if qoq_change is None else curve_score(_cap(qoq_change, -8, 18), [
        (-5, 20), (0, 48), (2, 60), (5, 74), (8, 84), (12, 91), (18, 95)
    ])
    growth_score = None if growth_quality is None else curve_score(growth_quality, [
        (0, 35), (5, 45), (10, 56), (20, 69), (30, 78), (45, 87), (60, 92)
    ])
    chart_score = None if chart is None else curve_score(chart, [
        (52, 35), (60, 48), (68, 60), (75, 70), (82, 79), (90, 88), (100, 94)
    ])

    # Confirmation remains a small input only.
    inst_confirm = normalize(inst, 30, 90) if inst is not None else None
    val_confirm = None
    if pe is not None and pe > 0:
        val_confirm = curve_score(pe, [(10, 88), (18, 80), (25, 68), (35, 52), (50, 34), (80, 10)])
    confirms = [x for x in (inst_confirm, val_confirm) if x is not None]
    confirm_score = sum(confirms) / len(confirms) if confirms else None

    pieces = [
        (accel_score, 0.30),
        (qoq_score, 0.20),
        (growth_score, 0.20),
        (chart_score, 0.20),
        (confirm_score, 0.10),
    ]
    usable = [(v, w) for v, w in pieces if v is not None]
    if len(usable) < 3:
        return None

    raw = sum(v*w for v,w in usable) / sum(w for _,w in usable)
    calibrated = 52 + (raw - 50) * 0.72
    calibrated += min(3, max(0, signals - 2) * 1.5)
    return round(max(55, min(93, calibrated)), 1)


def build_emerging_leaders(leaderboard_df, snapshot_df):
    if leaderboard_df is None or leaderboard_df.empty:
        return pd.DataFrame()
    changes = _quarter_change_map(snapshot_df)
    work = leaderboard_df.copy()

    # Hard separation: current Top 20 by Conviction can never appear as Emerging Leaders.
    top20 = set(work.sort_values("score", ascending=False).head(20)["ticker"].astype(str))
    work = work[~work["ticker"].astype(str).isin(top20)].copy()
    work["QoQ Change"] = work["ticker"].map(changes)

    benchmark = _benchmark_momentum()
    work["Momentum Accel"] = work.apply(lambda r: _relative_momentum_accel(r, benchmark), axis=1)
    work["Emerging Score"] = work.apply(lambda r: emerging_leader_score(r, r.get("QoQ Change")), axis=1)
    work = work.dropna(subset=["Emerging Score"]).copy()
    work["EPS Trend"] = work["eps_growth"].map(_growth_trend_label)
    work["Revenue Trend"] = work["revenue_growth"].map(_growth_trend_label)
    work["Why Emerging?"] = work.apply(lambda r: _why_emerging(r, r.get("QoQ Change")), axis=1)
    work = work.sort_values(["Emerging Score", "score"], ascending=False)
    return work.reset_index(drop=True)



def _value_vs_growth_display(row):
    vg = clean_num(row.get("value_vs_growth_score"))
    if vg is not None:
        return f"{vg:.0f}/100"
    fwd = clean_num(row.get("forward_eps_growth"))
    pe = clean_num(row.get("forward_pe"))
    if fwd is not None and fwd <= 0:
        return "Forward EPS declining"
    if pe is None:
        return "P/E data unavailable"
    if fwd is None:
        return "Growth estimate unavailable"
    return "Not enough data"


def _chart_display(value):
    v = clean_num(value)
    return "Not enough history" if v is None else f"{v:.0f}/100"


def _trend_cell_style(value):
    """Subtle beginner-friendly cues for tables; presentation only, never changes scores."""
    text = str(value).lower()
    if any(k in text for k in ["rising", "strong", "breakout", "exceptional", "healthy", "attractive", "improving", "quality gem"]):
        return "background-color: rgba(46,160,67,.10); color: #1f7a35; font-weight: 600;"
    if any(k in text for k in ["turnaround", "reasonable", "mixed", "flat", "mild", "stable"]):
        return "background-color: rgba(210,153,34,.10); color: #8a6100; font-weight: 600;"
    if any(k in text for k in ["declining", "falling", "lagging", "weak"]):
        return "background-color: rgba(218,54,51,.09); color: #b42318; font-weight: 600;"
    if any(k in text for k in ["not enough", "unavailable", "no prior", "forward eps declining"]):
        return "background-color: rgba(110,118,129,.08); color: #6e7781;"
    return ""


def _styled_table(df, cue_columns):
    cue_columns = [c for c in cue_columns if c in df.columns]
    if not cue_columns:
        return df
    return df.style.map(_trend_cell_style, subset=cue_columns)

# ------------------------------
# Clean beginner-facing UI
# ------------------------------

st.caption("Start with a ticker or browse the market. Stocks need at least **7/10 factors** before they can appear in rankings.")

main_stocks, main_etfs, main_dca = st.tabs(["📈 Stocks", "🧺 ETFs", "🧱 DCA Builder"])


def render_stock_result(symbol, force=False):
    with st.spinner(f"Pulling available data for {symbol}…"):
        result = fetch_stock(symbol, cache_bust=(str(time.time()) if force else None))

    m = result["metrics"]
    score, contributions, raw_scores = score_stock(m)
    scoring_keys = ["eps_growth", "revenue_growth", "net_margin", "roic", "forward_pe", "analyst_conviction", "institutional_ownership", "insider_activity", "momentum", "chart_health"]
    available = sum(m.get(k) is not None for k in scoring_keys)

    st.divider()
    st.subheader(f"{result['company']} ({symbol})")
    st.caption(f"{result['sector']} • {result['industry']} • Updated {result['fetched_at']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Conviction Score", "N/A" if score is None else f"{score:.1f}/100")
    c2.metric("Rating", label(score))
    c3.metric("Price", "N/A" if result["price"] is None else f"${result['price']:,.2f}")
    c4.metric("Data Coverage", f"{available}/10")

    if available < 7:
        st.warning(
            f"Only {available}/10 factors came back from the free data feed. This stock is **not eligible for leaderboard ranking**. "
            "Try **Force Refresh** once."
        )

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Market Cap", market_cap_fmt(result["market_cap"]))
    c6.metric("Mean Analyst Target", "N/A" if result["target_mean"] is None else f"${result['target_mean']:,.2f}")
    c7.metric("Target Upside", fmt(m["analyst_upside"]))
    c8.metric("Analysts", "N/A" if result["analyst_count"] is None else f"{int(result['analyst_count'])}")

    metric_rows = [
        ("EPS Growth", m["eps_growth"], "%"),
        ("Revenue Growth", m["revenue_growth"], "%"),
        ("Net Margin", m["net_margin"], "%"),
        ("ROIC / Capital Efficiency", m["roic"], "%"),
        ("Forward P/E", m["forward_pe"], "x"),
        ("Analyst Conviction", m["analyst_conviction"], "/100"),
        ("Institutional Ownership", m["institutional_ownership"], "%"),
        ("Insider Activity", m["insider_activity"], "/5"),
        ("12M Momentum", m["momentum"], "%"),
        ("Chart Health", m["chart_health"], "/100"),
    ]

    table = pd.DataFrame([
        {
            "Factor": name,
            "Live Value": "N/A" if value is None else (f"{value:.1f}x" if unit == "x" else f"{value:.1f}{unit}"),
            "Factor Score": "N/A" if raw_scores.get(name) is None else f"{raw_scores[name]:.0f}/100",
            "Weight": f"{WEIGHTS[name]:.0%}",
        }
        for name, value, unit in metric_rows
    ])
    st.markdown("### 10-factor breakdown")
    st.dataframe(table, use_container_width=True, hide_index=True)

    chart_df = pd.DataFrame(
        {"Factor": [k for k, v in raw_scores.items() if v is not None],
         "Score": [v for v in raw_scores.values() if v is not None]}
    ).set_index("Factor")
    if not chart_df.empty:
        st.bar_chart(chart_df)

    strengths, risks = [], []
    if m["eps_growth"] is not None and m["eps_growth"] >= 20: strengths.append("Strong EPS growth")
    if m["revenue_growth"] is not None and m["revenue_growth"] >= 15: strengths.append("Healthy revenue growth")
    if m["net_margin"] is not None and m["net_margin"] >= 20: strengths.append("High profitability")
    if m["roic"] is not None and m["roic"] >= 15: strengths.append("Strong capital efficiency")
    if m["analyst_upside"] is not None and (result["analyst_count"] or 0) >= 8 and m["analyst_upside"] >= 15: strengths.append("Strong analyst target upside with broad coverage")
    if m["institutional_ownership"] is not None and m["institutional_ownership"] >= 65: strengths.append("High institutional ownership")
    if m["momentum"] is not None and m["momentum"] >= 15: strengths.append("Strong 12-month momentum")
    if m["chart_health"] is not None and m["chart_health"] >= 75: strengths.append("Healthy chart and trend")

    if m["forward_pe"] is not None and m["forward_pe"] > 40: risks.append("Elevated forward valuation")
    if m["eps_growth"] is not None and m["eps_growth"] < 5: risks.append("Weak/negative EPS growth")
    if m["net_margin"] is not None and m["net_margin"] < 5: risks.append("Thin profitability")
    if m["analyst_upside"] is not None and (result["analyst_count"] or 0) >= 8 and m["analyst_upside"] < 0: risks.append("Mean analyst target below current price")
    if m["insider_activity"] is not None and m["insider_activity"] < -2: risks.append("Recent reported insider activity skews negative")
    if m["momentum"] is not None and m["momentum"] < -10: risks.append("Negative 12-month momentum")
    if m["chart_health"] is not None and m["chart_health"] < 40: risks.append("Weak chart health / trend")

    s1, s2 = st.columns(2)
    with s1:
        st.markdown("### Strengths")
        st.write("\n".join(f"• {x}" for x in strengths) if strengths else "• No standout strength threshold triggered")
    with s2:
        st.markdown("### Watch-outs")
        st.write("\n".join(f"• {x}" for x in risks) if risks else "• No major risk threshold triggered")

    with st.expander("How the score works"):
        st.write(
            "Conviction AI is a transparent weighted research score, not a prediction. Business quality carries more weight than market opinion. Missing metrics are excluded, remaining weights are re-normalized, and incomplete coverage receives a small confidence haircut. "
            "Analyst Conviction requires at least 8 analysts and combines target upside, analyst count, and consensus rating. "
            "Chart Health uses the 50-day and 200-day moving averages, trend relationship, 3-month momentum, and distance from the 52-week high."
        )
        weight_df = pd.DataFrame({"Factor": list(WEIGHTS.keys()), "Weight": [f"{v:.0%}" for v in WEIGHTS.values()]})
        st.dataframe(weight_df, hide_index=True, use_container_width=True)


with main_stocks:
    stock_search_tab, market_tab, emerging_tab, hidden_tab = st.tabs(["🔎 Search a Stock", "🏆 Market Leaders", "🚀 Emerging Leaders", "💎 Hidden Gems"])

    with stock_search_tab:
        st.markdown("### Search any stock")
        st.caption("Enter a ticker and get the key numbers without bouncing between multiple sites.")
        symbol = st.text_input("Ticker", value="AVGO", placeholder="AVGO, GOOGL, META…", key="stock_search_ticker").upper().strip()
        run = st.button("Analyze", type="primary", use_container_width=True, key="stock_analyze")

        force = False
        with st.expander("Having trouble loading a ticker?"):
            st.caption("Use this only if the first result is missing several data points.")
            force = st.button("Retry live data", use_container_width=True, key="stock_force")

        if (run or force) and symbol:
            render_stock_result(symbol, force=force)
        elif run:
            st.warning("Enter a ticker first.")
        else:
            st.caption("Examples: AVGO · GOOGL · META · SPGI · VST")

    with market_tab:
        st.markdown("### Market Leaders")
        st.caption("Three simple views. No stock appears unless at least **7 of 10 factors** are available.")

        universe_choice = st.radio(
            "Market universe",
            ["S&P 500 + Nasdaq-100", "S&P 500", "Nasdaq-100"],
            horizontal=True,
            key="market_universe_choice",
        )
        universe_df = build_market_universe(universe_choice)
        leaderboard_universe = universe_df["ticker"].tolist()

        scan = st.button(
            f"Refresh market scan ({len(leaderboard_universe):,} stocks)",
            type="primary",
            use_container_width=True,
            key="market_scan_button",
        )

        if "leaderboard_df" not in st.session_state:
            st.session_state.leaderboard_df = pd.DataFrame()

        if scan:
            progress = st.progress(0.0, text=f"Scoring 0 / {len(leaderboard_universe)} stocks…")
            def update_progress(done, total):
                progress.progress(done / max(total, 1), text=f"Scoring {done:,} / {total:,} stocks…")
            st.session_state.leaderboard_df = scan_universe(
                leaderboard_universe, workers=4, progress_callback=update_progress
            )
            progress.empty()

        leaderboard_df = st.session_state.leaderboard_df
        snapshots = load_snapshots()
        winners, losers, prior_q, latest_q = quarter_movers(snapshots)

        top_tab, improve_tab = st.tabs(["Top Stocks", "Biggest Improvers"])

        with top_tab:
            if leaderboard_df.empty:
                st.info("Click **Refresh market scan** to build the list.")
            else:
                top10 = leaderboard_df.head(10).copy()
                top10.insert(0, "Rank", range(1, len(top10) + 1))
                top10["Score"] = top10["score"].map(lambda x: f"{x:.1f}")
                top10["Coverage"] = top10["coverage"].map(lambda x: f"{int(x)}/10")
                top10["EPS Trend"] = top10["eps_growth"].map(_growth_trend_label)
                top10["Value vs Growth"] = top10["value_vs_growth_score"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.0f}/100")
                top10["Chart Health"] = top10["chart_health"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.0f}/100")
                st.dataframe(
                    top10[["Rank", "ticker", "company", "Score", "Coverage", "EPS Trend", "Value vs Growth", "Chart Health"]],
                    use_container_width=True,
                    hide_index=True,
                )

        with improve_tab:
            st.caption("Updates on **Jan 1, Apr 1, Jul 1, and Oct 1** using Conviction Score—not price movement.")
            if winners.empty:
                if latest_q:
                    st.info(f"First snapshot saved for **{quarter_start_label(latest_q)}**. This list appears after the next quarterly snapshot.")
                else:
                    st.info("No quarterly snapshots yet. The scheduled job creates them automatically on Jan 1, Apr 1, Jul 1, and Oct 1.")
            else:
                st.caption(f"Comparing **{quarter_start_label(prior_q)} → {quarter_start_label(latest_q)}**")
                show = winners.copy()
                show.insert(0, "Rank", range(1, len(show) + 1))
                show["Current Score"] = show["score_current"].map(lambda x: f"{x:.1f}")
                show["Prior Score"] = show["score_prior"].map(lambda x: f"{x:.1f}")
                show["Change"] = show["change"].map(lambda x: f"+{x:.1f}" if x >= 0 else f"{x:.1f}")
                st.dataframe(show[["Rank", "ticker", "company", "Current Score", "Prior Score", "Change"]], use_container_width=True, hide_index=True)

            with st.expander("Show biggest fallers"):
                if losers.empty:
                    st.caption("Fallers will appear once two quarterly snapshots exist.")
                else:
                    fall = losers.copy()
                    fall.insert(0, "Rank", range(1, len(fall) + 1))
                    fall["Current Score"] = fall["score_current"].map(lambda x: f"{x:.1f}")
                    fall["Prior Score"] = fall["score_prior"].map(lambda x: f"{x:.1f}")
                    fall["Change"] = fall["change"].map(lambda x: f"{x:.1f}")
                    st.dataframe(fall[["Rank", "ticker", "company", "Current Score", "Prior Score", "Change"]], use_container_width=True, hide_index=True)

        with st.expander("Wall Street price gaps (secondary view)"):
            st.caption("Largest gaps between current price and the mean analyst target. Requires **8+ analysts**. Use this as supporting evidence, not the main thesis.")
            if leaderboard_df.empty:
                st.info("Click **Refresh market scan** first.")
            else:
                opp = leaderboard_df.copy()
                opp["analyst_count"] = pd.to_numeric(opp.get("analyst_count"), errors="coerce")
                opp["analyst_upside"] = pd.to_numeric(opp.get("analyst_upside"), errors="coerce")
                opp = opp[(opp["analyst_count"] >= 8) & opp["analyst_upside"].notna()].sort_values("analyst_upside", ascending=False).head(10)
                if opp.empty:
                    st.info("No stocks in this scan had enough analyst coverage plus a usable mean target.")
                else:
                    show = opp.copy()
                    show.insert(0, "Rank", range(1, len(show) + 1))
                    show["Price"] = show["price"].map(lambda x: "N/A" if pd.isna(x) else f"${x:,.2f}")
                    show["Mean Target"] = show["target_mean"].map(lambda x: "N/A" if pd.isna(x) else f"${x:,.2f}")
                    show["Upside"] = show["analyst_upside"].map(lambda x: f"{x:+.1f}%")
                    show["Analysts"] = show["analyst_count"].map(lambda x: f"{int(x)}")
                    st.dataframe(show[["Rank", "ticker", "company", "Price", "Mean Target", "Upside", "Analysts"]], use_container_width=True, hide_index=True)


    with emerging_tab:
        st.markdown("### 🚀 Emerging Leaders")
        st.caption("Companies whose setup is **accelerating**, not simply the highest-scoring companies. Current Top 20 Conviction names are excluded, and at least two improvement signals are required.")
        leaderboard_df = st.session_state.get("leaderboard_df", pd.DataFrame())
        snapshots = load_snapshots()
        emerging = build_emerging_leaders(leaderboard_df, snapshots)
        if leaderboard_df.empty:
            st.info("Go to **Market Leaders** and run **Refresh market scan** first. Emerging Leaders uses that same full-market scan—no second wait.")
        elif emerging.empty:
            st.info("No clear acceleration candidates surfaced in this scan. That is okay—this screen is intentionally selective rather than duplicating Top Stocks.")
        else:
            top = emerging.head(15).copy()
            top.insert(0, "Rank", range(1, len(top)+1))
            top["Emerging"] = top["Emerging Score"].map(lambda x: f"{x:.1f}")
            top["Conviction"] = top["score"].map(lambda x: f"{x:.1f}")
            top["Momentum Accel"] = top["Momentum Accel"].map(_momentum_accel_label)
            top["QoQ"] = top["QoQ Change"].map(lambda x: "No prior quarter" if pd.isna(x) else f"{x:+.1f}")
            top["Chart"] = top["chart_health"].map(_chart_display)
            emerging_view = top[["Rank","ticker","company","Emerging","Conviction","Momentum Accel","QoQ","EPS Trend","Revenue Trend","Chart","Why Emerging?"]]
            st.dataframe(
                _styled_table(emerging_view, ["Momentum Accel", "QoQ", "EPS Trend", "Revenue Trend", "Why Emerging?"]),
                use_container_width=True, hide_index=True
            )
            st.caption("Growth and momentum are shown as normalized labels instead of noisy capped numbers. Momentum acceleration is measured relative to the S&P 500, and each name must show at least two independent improvement signals.")

            with st.expander("How Emerging Leaders is different"):
                st.markdown(
                    "**Top Stocks** = strongest absolute Conviction Scores today.  \n"
                    "**Emerging Leaders** = improving setups outside the current Top 20.  \n"
                    "**Hidden Gems** = quality companies that are relatively less followed."
                )


    with hidden_tab:
        st.markdown("### 💎 Hidden Gems")
        st.caption("Discover an established S&P 500 company that looks unusually attractive versus its sector while receiving less attention than the obvious mega-cap names.")

        with st.expander("What counts as a Hidden Gem?"):
            st.markdown(
                "A company needs a solid quality floor, at least **7/10 data factors**, moderate analyst coverage, and no obvious red flags. **Quality Gems** need positive forward EPS growth or solid revenue growth; **Turnaround Gems** can have declining forward EPS only when estimate revisions are improving, the chart is healthy, and valuation is reasonable. "
                "Hidden Gem scoring then emphasizes **sector-relative quality, EPS estimate revisions, value vs growth, chart health, institutional support, and underfollowedness**. "
                "Only names with a **65+ Hidden Gem Score** enter the randomizer, and the pool is capped at **two names per sector** so one industry cannot dominate."
            )

        c1, c2 = st.columns([2, 1])
        with c1:
            st.caption("For a quick discovery, Conviction AI checks a larger random slice of the S&P 500, applies a quality floor, then ranks the survivors by quality + how underfollowed they are.")
        with c2:
            sample_size = st.selectbox("Discovery depth", [80, 120, 160], index=1, key="hidden_sample_size")

        find_gem = st.button("💎 Find a Hidden Gem", type="primary", use_container_width=True, key="hidden_find")

        if find_gem:
            progress = st.progress(0.0, text=f"Checking 0 / {sample_size} stocks…")
            def update_hidden_progress(done, total):
                progress.progress(done / max(total, 1), text=f"Checking {done} / {total} stocks…")
            scan_df = lightweight_hidden_gem_scan(sample_size=sample_size, workers=6, progress_callback=update_hidden_progress)
            progress.empty()
            gem, pool = pick_hidden_gem_from_df(scan_df)
            st.session_state.hidden_gem_scan = scan_df
            st.session_state.hidden_gem_pool = pool
            st.session_state.hidden_gem = gem

        gem = st.session_state.get("hidden_gem")
        pool = st.session_state.get("hidden_gem_pool", pd.DataFrame())

        if gem:
            st.divider()
            st.markdown(f"## {gem['ticker']} — {gem['company']}")
            gem_type = hidden_gem_type(gem)
            badge_class = "gem-quality" if gem_type == "Quality Gem" else "gem-turnaround" if gem_type == "Turnaround Gem" else ""
            badge_icon = "●" if gem_type == "Quality Gem" else "▲" if gem_type == "Turnaround Gem" else "•"
            st.markdown(f'<span class="gem-badge {badge_class}">{badge_icon} {gem_type}</span>', unsafe_allow_html=True)
            if gem_type == "Turnaround Gem":
                st.caption("Forward EPS is still declining, but expectations are improving and other signals are strong. Discovery only — not a recommendation.")
            else:
                st.caption("Positive forward growth with sector-relative quality. Randomly selected from companies that passed the Hidden Gem screen. Discovery only — not a recommendation.")
            a, b, c, d = st.columns(4)
            a.metric("Conviction", f"{gem['score']:.1f}/100")
            b.metric("Hidden Gem Score", f"{gem['hidden_gem_score']:.1f}/100")
            rev = clean_num(gem.get("estimate_revision_pct"))
            if rev is None:
                rev_display = "Revision data unavailable"
            elif rev > 50:
                rev_display = "Rising >50%"
            elif rev < -50:
                rev_display = "Falling >50%"
            elif rev > 1:
                rev_display = f"Rising {rev:+.1f}%"
            elif rev < -1:
                rev_display = f"Falling {rev:+.1f}%"
            else:
                rev_display = "Stable"
            c.metric("Estimate Revisions", rev_display)
            vg = clean_num(gem.get("value_vs_growth_score"))
            
            if vg is None:
                fwd = clean_num(gem.get("forward_eps_growth"))
                vg_display = "Forward EPS declining" if fwd is not None and fwd <= 0 else "Insufficient data"
            else:
                vg_display = f"{vg:.0f}/100"
            d.metric("Value vs Growth", vg_display)
            status_parts = []
            if rev is not None:
                status_parts.append("🟢 Estimates rising" if rev > 1 else "🔴 Estimates falling" if rev < -1 else "🟡 Estimates stable")
            if vg is not None:
                status_parts.append("🟢 Attractive value/growth" if vg >= 70 else "🟡 Reasonable value/growth" if vg >= 55 else "🔴 Weak value/growth")
            elif clean_num(gem.get("forward_eps_growth")) is not None and clean_num(gem.get("forward_eps_growth")) <= 0:
                status_parts.append("🟡 Forward EPS declining")
            if status_parts:
                st.caption("  •  ".join(status_parts))

            st.markdown("### Why it surfaced")
            st.write("\n".join(f"• {x}" for x in hidden_gem_reasons(gem)))

            facts = "\n".join([
                f"Ticker: {gem.get('ticker')}",
                f"Company: {gem.get('company')}",
                f"Gem type: {hidden_gem_type(gem)}",
                f"Conviction Score: {clean_num(gem.get('score'))}",
                f"Hidden Gem Score: {clean_num(gem.get('hidden_gem_score'))}",
                f"EPS growth: {clean_num(gem.get('eps_growth'))}%",
                f"Revenue growth: {clean_num(gem.get('revenue_growth'))}%",
                f"Chart health: {clean_num(gem.get('chart_health'))}/100",
                f"12M return: {clean_num(gem.get('one_year_return'))}%",
                f"Sector: {gem.get('sector')}",
                f"EPS estimate revision vs roughly 90 days ago: {clean_num(gem.get('estimate_revision_pct'))}%",
                f"Estimate revision score: {clean_num(gem.get('estimate_revision_score'))}/100",
                f"Forward EPS growth estimate: {clean_num(gem.get('forward_eps_growth'))}%",
                f"Value vs growth score: {clean_num(gem.get('value_vs_growth_score'))}/100",
                f"Analyst count: {clean_num(gem.get('analyst_count'))}",
            ])
            ai_take = two_sentence_ai_take("Hidden Gem", facts)
            st.markdown("### ✨ Quick take")
            st.write(ai_take or hidden_gem_fallback_take(gem))
            if ai_take is None:
                st.caption("Plain-English fallback shown. Add an OpenAI API key to Streamlit Secrets to turn this into an AI-generated two-sentence summary.")

            if isinstance(pool, pd.DataFrame) and not pool.empty:
                st.caption(f"This discovery scan found **{len(pool)} qualifying Hidden Gems** among {len(st.session_state.get('hidden_gem_scan', []))} randomly checked S&P 500 stocks.")

            r1, r2 = st.columns(2)
            with r1:
                if st.button("🎲 Show another from this pool", use_container_width=True, key="hidden_again"):
                    next_gem, _ = pick_hidden_gem_from_df(pool.drop(columns=["hidden_gem_score"], errors="ignore"))
                    if next_gem:
                        st.session_state.hidden_gem = next_gem
                        st.rerun()
            with r2:
                if st.button("🔎 Open full 10-factor analysis", use_container_width=True, key="hidden_open"):
                    st.session_state.hidden_open_ticker = gem["ticker"]

            if st.session_state.get("hidden_open_ticker") == gem["ticker"]:
                render_stock_result(gem["ticker"], force=False)
        elif find_gem:
            st.info("No stock in this random sample cleared the Hidden Gem quality floor. Tap **Find a Hidden Gem** again for a fresh sample.")
        else:
            st.info("Tap **Find a Hidden Gem** and Conviction AI will look for a quality company you may not already be watching.")


with main_etfs:
    st.markdown("### ETF research made simple")
    st.caption("Find, compare, and rank ETFs without needing to understand every fund metric first. Leveraged ETFs are hidden by default.")

    finder_tab, compare_tab, top_etf_tab, all_around_tab = st.tabs(["🔎 ETF Finder", "⚖️ Compare ETFs", "🏁 Top ETFs", "⭐ Best All-Around"])

    with finder_tab:
        etf_lookup = st.text_input("ETF ticker", placeholder="VOO, VOOG, QQQM, SPMO, MOAT…", key="etf_lookup_clean").strip().upper()
        if etf_lookup:
            known = {r[0]: r for r in ETF_UNIVERSE}
            if etf_lookup in known:
                row = known[etf_lookup]
                with st.spinner(f"Loading {etf_lookup}…"):
                    m = fetch_etf_metrics(etf_lookup)
                st.subheader(f"{row[1]} ({etf_lookup})")
                st.caption(" • ".join(etf_tags(row)))
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("YTD", "N/A" if m.get("ytd") is None else f"{m['ytd']:+.1f}%")
                c2.metric("1Y", "N/A" if m.get("1Y") is None else f"{m['1Y']:+.1f}%")
                c3.metric("3Y CAGR", "N/A" if m.get("3Y") is None else f"{m['3Y']:.1f}%")
                c4.metric("5Y CAGR", "N/A" if m.get("5Y") is None else f"{m['5Y']:.1f}%")
                c5.metric("10Y CAGR", "N/A" if m.get("10Y") is None else f"{m['10Y']:.1f}%")
                e1, e2, e3 = st.columns(3)
                e1.metric("Expense Ratio", "N/A" if m.get("expense_ratio") is None else f"{m['expense_ratio']:.2f}%")
                e2.metric("5Y Volatility", "N/A" if m.get("volatility_5y") is None else f"{m['volatility_5y']:.1f}%")
                e3.metric("5Y Max Drawdown", "N/A" if m.get("max_drawdown_5y") is None else f"{m['max_drawdown_5y']:.1f}%")
            else:
                st.warning("That ETF is not in the curated universe yet.")
        else:
            st.info("Try **VOO**, **VOOG**, **QQQM**, **SCHG**, **SPMO**, **MOAT**, **VGT**, or **SMH**.")

    with compare_tab:
        st.caption("Pick 2–5 ETFs and compare the numbers side by side.")
        etf_options = [r[0] for r in ETF_UNIVERSE if not r[3]]
        compare_tickers = st.multiselect("ETFs to compare", etf_options, default=["VOO", "VOOG", "QQQM"], max_selections=5, key="etf_compare_tickers")
        if st.button("Compare selected ETFs", type="primary", use_container_width=True, key="etf_compare_btn"):
            if len(compare_tickers) < 2:
                st.warning("Pick at least two ETFs.")
            else:
                lookup = {r[0]: r for r in ETF_UNIVERSE}
                rows = []
                with st.spinner("Building comparison…"):
                    for t in compare_tickers:
                        m = fetch_etf_metrics(t)
                        rows.append({
                            "ETF": t, "Name": lookup[t][1],
                            "YTD": m.get("ytd"), "1Y": m.get("1Y"), "3Y CAGR": m.get("3Y"), "5Y CAGR": m.get("5Y"), "10Y CAGR": m.get("10Y"),
                            "Expense Ratio": m.get("expense_ratio"), "5Y Volatility": m.get("volatility_5y"), "5Y Max Drawdown": m.get("max_drawdown_5y"),
                        })
                df = pd.DataFrame(rows)
                display = df.copy()
                for col in ["YTD", "1Y", "3Y CAGR", "5Y CAGR", "10Y CAGR", "Expense Ratio", "5Y Volatility", "5Y Max Drawdown"]:
                    display[col] = display[col].map(lambda x: "N/A" if pd.isna(x) else f"{x:.1f}%")
                st.dataframe(display, use_container_width=True, hide_index=True)
                numeric = df.set_index("ETF")
                callouts=[]
                if numeric["5Y CAGR"].notna().any(): callouts.append(f"**Best 5Y growth:** {numeric['5Y CAGR'].idxmax()}")
                if numeric["Expense Ratio"].notna().any(): callouts.append(f"**Lowest cost:** {numeric['Expense Ratio'].idxmin()}")
                if numeric["5Y Volatility"].notna().any(): callouts.append(f"**Lowest volatility:** {numeric['5Y Volatility'].idxmin()}")
                if numeric["5Y Max Drawdown"].notna().any(): callouts.append(f"**Shallowest 5Y drawdown:** {numeric['5Y Max Drawdown'].idxmax()}")
                if callouts:
                    st.markdown(" · ".join(callouts))
                st.caption("A higher return is not automatically better; cost, volatility, diversification, and drawdowns matter too.")

    with top_etf_tab:
        period = st.radio("Performance period", ["YTD", "1Y", "3Y CAGR", "5Y CAGR", "10Y CAGR"], horizontal=True, index=3, key="etf_period_clean")
        c1, c2 = st.columns([2, 1])
        with c2:
            include_leveraged = st.toggle("Include leveraged", value=False, key="etf_leveraged_clean")
        with c1:
            category = st.selectbox("ETF type", all_etf_filter_tags(include_leveraged), key="etf_category_clean")

        eligible = [r for r in ETF_UNIVERSE if (include_leveraged or not r[3]) and (category == "All" or category in etf_tags(r))]
        st.caption(f"{len(eligible)} ETFs match this filter. Funds can appear in more than one category.")
        if st.button(f"Refresh ETF rankings ({len(eligible)} funds)", type="primary", use_container_width=True, key="etf_refresh_clean"):
            with st.spinner(f"Comparing {len(eligible)} ETFs…"):
                st.session_state.etf_df = scan_etfs(eligible)

        if "etf_df" not in st.session_state:
            st.session_state.etf_df = pd.DataFrame()
        etf_df = st.session_state.etf_df

        if etf_df.empty:
            st.info("Choose a period and click **Refresh ETF rankings**.")
        else:
            period_key = {"YTD":"ytd", "1Y":"1Y", "3Y CAGR":"3Y", "5Y CAGR":"5Y", "10Y CAGR":"10Y"}[period]
            ranked = etf_df.dropna(subset=[period_key]).sort_values(period_key, ascending=False).head(10).copy()
            if ranked.empty:
                st.info("Not enough history was available for this selection.")
            else:
                ranked.insert(0, "Rank", range(1, len(ranked)+1))
                ranked["Return"] = ranked[period_key].map(lambda x: f"{x:+.1f}%")
                ranked["Expense"] = ranked["expense_ratio"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2f}%")
                st.dataframe(ranked[["Rank","ticker","name","category","Return","Expense"]], use_container_width=True, hide_index=True)
                st.caption("YTD and 1Y are total returns. 3Y, 5Y and 10Y are annualized CAGR.")

    with all_around_tab:
        st.caption("Balances long-term returns with volatility, drawdown, expenses, fund size, and liquidity.")
        include_leveraged_all = st.toggle("Include leveraged funds", value=False, key="etf_leveraged_all")
        all_rows = [r for r in ETF_UNIVERSE if include_leveraged_all or not r[3]]
        if st.button(f"Build all-around ranking ({len(all_rows)} funds)", use_container_width=True, key="etf_all_refresh"):
            with st.spinner(f"Comparing {len(all_rows)} ETFs…"):
                st.session_state.etf_all_df = scan_etfs(all_rows)

        if "etf_all_df" not in st.session_state:
            st.session_state.etf_all_df = pd.DataFrame()
        all_df = st.session_state.etf_all_df

        if all_df.empty:
            st.info("Click **Build all-around ranking**.")
        else:
            ranked = all_df.dropna(subset=["all_around"]).sort_values("all_around", ascending=False).head(10).copy()
            ranked.insert(0, "Rank", range(1, len(ranked)+1))
            ranked["ETF Score"] = ranked["all_around"].map(lambda x: f"{x:.1f}")
            ranked["5Y CAGR"] = ranked["5Y"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.1f}%")
            ranked["10Y CAGR"] = ranked["10Y"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.1f}%")
            ranked["Expense"] = ranked["expense_ratio"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2f}%")
            st.dataframe(ranked[["Rank","ticker","name","category","ETF Score","5Y CAGR","10Y CAGR","Expense"]], use_container_width=True, hide_index=True)


# ------------------------------
# DCA Builder
# ------------------------------
DCA_MODELS = {
    ("Keep it simple", "Conservative"): [("VOO", 55, "Core U.S. stocks"), ("VXUS", 20, "International diversification"), ("BND", 25, "Bonds / stability")],
    ("Keep it simple", "Balanced"): [("VOO", 70, "Core U.S. stocks"), ("VXUS", 20, "International diversification"), ("BND", 10, "Bonds / stability")],
    ("Keep it simple", "Growth"): [("VOO", 75, "Core U.S. stocks"), ("VOOG", 15, "Growth tilt"), ("VXUS", 10, "International diversification")],
    ("Keep it simple", "Aggressive"): [("VOO", 65, "Core U.S. stocks"), ("VOOG", 20, "Growth tilt"), ("QQQM", 15, "Nasdaq growth")],
    ("Long-term growth", "Conservative"): [("VOO", 65, "Core compounder"), ("VOOG", 15, "Growth tilt"), ("VXUS", 10, "Diversification"), ("BND", 10, "Stability")],
    ("Long-term growth", "Balanced"): [("VOO", 60, "Core compounder"), ("VOOG", 20, "Growth tilt"), ("QQQM", 10, "Nasdaq growth"), ("AVUV", 10, "Small-cap diversification")],
    ("Long-term growth", "Growth"): [("VOO", 50, "Core compounder"), ("VOOG", 25, "Growth tilt"), ("QQQM", 15, "Nasdaq growth"), ("SPMO", 10, "Momentum satellite")],
    ("Long-term growth", "Aggressive"): [("VOO", 45, "Core compounder"), ("VOOG", 25, "Growth tilt"), ("QQQM", 20, "Nasdaq growth"), ("SMH", 10, "Spicy satellite")],
    ("Core + a little spice", "Conservative"): [("VOO", 75, "Core compounder"), ("VXUS", 15, "Diversification"), ("MOAT", 10, "Quality satellite")],
    ("Core + a little spice", "Balanced"): [("VOO", 70, "Core compounder"), ("VOOG", 15, "Growth tilt"), ("MOAT", 10, "Quality satellite"), ("SMH", 5, "Spicy satellite")],
    ("Core + a little spice", "Growth"): [("VOO", 65, "Core compounder"), ("VOOG", 15, "Growth tilt"), ("SPMO", 10, "Momentum satellite"), ("SMH", 10, "Spicy satellite")],
    ("Core + a little spice", "Aggressive"): [("VOO", 55, "Core compounder"), ("VOOG", 15, "Growth tilt"), ("QQQM", 10, "Nasdaq growth"), ("SMH", 10, "Spicy satellite"), ("URA", 5, "Theme satellite"), ("QTUM", 5, "Theme satellite")],
    ("Income & stability", "Conservative"): [("SCHD", 40, "Dividend core"), ("VOO", 30, "Broad U.S. stocks"), ("BND", 20, "Bonds"), ("SGOV", 10, "Short Treasury")],
    ("Income & stability", "Balanced"): [("SCHD", 40, "Dividend core"), ("VOO", 40, "Broad U.S. stocks"), ("BND", 20, "Bonds")],
    ("Income & stability", "Growth"): [("VOO", 50, "Broad U.S. stocks"), ("SCHD", 30, "Dividend quality"), ("VIG", 20, "Dividend growth")],
    ("Income & stability", "Aggressive"): [("VOO", 50, "Broad U.S. stocks"), ("SCHD", 25, "Dividend quality"), ("VIG", 15, "Dividend growth"), ("QQQM", 10, "Growth satellite")],
}


@st.cache_data(ttl=21600, show_spinner=False)
def historical_cagr_5y(ticker):
    """Return trailing CAGR using up to 5 years of adjusted price history.

    For newer securities, annualize the longest available history once at least
    ~1 year is available. Returns (cagr_pct, years_used).
    """
    ticker = (ticker or "").upper().strip()
    if not ticker:
        return None, None
    try:
        hist = yf.Ticker(ticker).history(period="5y", auto_adjust=True)
    except Exception:
        return None, None
    if hist is None or hist.empty or "Close" not in hist:
        return None, None
    closes = hist["Close"].dropna()
    if len(closes) < 2:
        return None, None
    start_price = clean_num(closes.iloc[0])
    end_price = clean_num(closes.iloc[-1])
    if not start_price or start_price <= 0 or end_price is None:
        return None, None
    start_dt = pd.Timestamp(closes.index[0]).tz_localize(None)
    end_dt = pd.Timestamp(closes.index[-1]).tz_localize(None)
    years_used = max((end_dt - start_dt).days / 365.25, 0)
    if years_used < 0.9:
        return None, years_used
    cagr = ((end_price / start_price) ** (1 / years_used) - 1) * 100
    return float(cagr), float(years_used)


def project_dca(starting_balance, daily_amount, annual_rate_pct, years):
    """Project a trading-day DCA using 252 market days per year, converted to monthly-equivalent contributions and monthly compounding."""
    months = int(years * 12)
    monthly_contribution = daily_amount * 252 / 12
    monthly_rate = (1 + annual_rate_pct / 100) ** (1 / 12) - 1 if annual_rate_pct > -100 else -1
    if abs(monthly_rate) < 1e-12:
        return starting_balance + monthly_contribution * months
    growth = (1 + monthly_rate) ** months
    return starting_balance * growth + monthly_contribution * ((growth - 1) / monthly_rate)


def custom_dca_fallback_take(daily_amount, blended_cagr, rows, horizon_value):
    tickers = ", ".join(r["Ticker"] for r in rows if r["Ticker"])
    first = f"This model puts ${daily_amount:.2f} per trading day across {tickers or 'your selected investments'} and uses a dollar-weighted {blended_cagr:.1f}% expected annual return."
    second = f"At those assumptions, the 20-year projection is about ${horizon_value:,.0f}; historical CAGR is only a starting reference and future returns can be materially lower or higher."
    return first + " " + second


with main_dca:
    st.markdown("### 🧱 DCA Builder")
    st.caption("Use a guided model or build your own up to 5-investment DCA plan and see how the assumptions compound over time.")

    guided_tab, custom_tab = st.tabs(["🧭 Guided DCA", "🛠️ Build Your Own"])

    with guided_tab:
        c1, c2, c3 = st.columns(3)
        with c1:
            daily_amount = st.number_input("Amount per trading day", min_value=1.0, max_value=1000.0, value=10.0, step=1.0, key="dca_daily")
        with c2:
            dca_goal = st.selectbox("What are you trying to build?", ["Keep it simple", "Long-term growth", "Core + a little spice", "Income & stability"], key="dca_goal")
        with c3:
            dca_risk = st.selectbox("How aggressive?", ["Conservative", "Balanced", "Growth", "Aggressive"], index=2, key="dca_risk")

        model = DCA_MODELS[(dca_goal, dca_risk)]
        rows=[]
        for ticker, weight, role in model:
            rows.append({"ETF": ticker, "Role": role, "Weight": f"{weight}%", "Per Day": f"${daily_amount*weight/100:.2f}", "Per Week": f"${daily_amount*5*weight/100:.2f}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        annual = daily_amount * 252
        st.caption("Assumes 252 trading days per year. $10 per trading day = about $2,520 per year.")
        st.metric("Approx. yearly contributions", f"${annual:,.0f}")
        core_weight = sum(w for _t,w,role in model if "Core" in role or "Broad" in role or "Dividend core" in role)
        spicy_weight = sum(w for _t,w,role in model if "Spicy" in role or "Theme" in role or "Momentum" in role or "Nasdaq" in role)
        st.caption(f"This example is about **{core_weight}% core** and **{spicy_weight}% higher-octane satellite** exposure, with the rest used for diversification/stability.")

        dca_facts = "\n".join([
            f"Trading-day contribution: ${daily_amount:.2f}",
            f"Goal: {dca_goal}",
            f"Risk setting: {dca_risk}",
            f"Approximate annual contribution: ${annual:.0f}",
            "Allocation: " + "; ".join(f"{t} {w}% ({role})" for t,w,role in model),
            f"Core weight: {core_weight}%",
            f"Higher-octane satellite weight: {spicy_weight}%",
        ])
        dca_ai = two_sentence_ai_take("DCA model", dca_facts)
        st.markdown("### ✨ Quick take")
        st.write(dca_ai or dca_fallback_take(dca_goal, dca_risk, daily_amount, model))
        if dca_ai is None:
            st.caption("Plain-English fallback shown. Add an OpenAI API key to Streamlit Secrets to turn this into an AI-generated two-sentence summary.")

    with custom_tab:
        st.markdown("### Build your own DCA model")
        st.caption("Add up to 5 stocks or ETFs. Type a ticker and Conviction AI will pre-fill its trailing historical CAGR (up to 5 years) as a starting assumption. You can edit it.")

        a1, a2 = st.columns(2)
        with a1:
            starting_balance = st.number_input("Starting balance", min_value=0.0, max_value=10000000.0, value=0.0, step=100.0, key="custom_dca_start")
        with a2:
            investment_count = st.selectbox("Number of investments", [1,2,3,4,5], index=2, key="custom_dca_count")

        defaults = [
            ("VOO", 6.0),
            ("VOOG", 3.0),
            ("SMH", 1.0),
            ("", 0.0),
            ("", 0.0),
        ]
        custom_rows=[]
        st.markdown("#### Your investments")
        for i in range(investment_count):
            c1, c2, c3 = st.columns([1.05, 1, 1.15])
            with c1:
                ticker = st.text_input(f"Ticker {i+1}", value=defaults[i][0], key=f"custom_ticker_{i}").upper().strip()
            hist_cagr, hist_years = historical_cagr_5y(ticker) if ticker else (None, None)
            with c2:
                dollars = st.number_input(
                    f"$ per trading day {i+1}",
                    min_value=0.0,
                    max_value=5000.0,
                    value=defaults[i][1],
                    step=1.0,
                    key=f"custom_dollars_{i}",
                )
            with c3:
                default_cagr = round(hist_cagr, 1) if hist_cagr is not None else 8.0
                cagr_key = f"custom_cagr_{i}_{ticker or 'blank'}"
                cagr = st.number_input(
                    f"Expected CAGR % {i+1}",
                    min_value=-20.0,
                    max_value=50.0,
                    value=float(max(-20.0, min(50.0, default_cagr))),
                    step=0.5,
                    key=cagr_key,
                    help="Pre-filled from trailing adjusted-price history when available. Edit this assumption if you want a more conservative or aggressive projection.",
                )
            hist_label = None
            if hist_cagr is not None and hist_years is not None:
                hist_label = f"{hist_cagr:.1f}% over {hist_years:.1f} years"
                st.caption(f"**{ticker} historical CAGR:** {hist_label}")
            elif ticker:
                st.caption(f"**{ticker}:** not enough price history to calculate a reliable historical CAGR.")
            custom_rows.append({"Ticker": ticker, "Daily Dollars": dollars, "Expected CAGR": cagr, "Historical CAGR": hist_cagr, "History Years": hist_years})

        active_rows = [r for r in custom_rows if r["Ticker"] and r["Daily Dollars"] > 0]
        total_daily = sum(r["Daily Dollars"] for r in active_rows)

        if not active_rows or total_daily <= 0:
            st.warning("Add at least one ticker with a dollar amount above $0 per trading day to calculate the model.")
        else:
            blended_cagr = sum(r["Daily Dollars"] * r["Expected CAGR"] for r in active_rows) / total_daily
            annual_contribution = total_daily * 252
            st.caption("Projection assumes 252 trading days per year. Dollar amounts automatically determine each investment's portfolio weight.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total per trading day", f"${total_daily:,.2f}")
            c2.metric("Dollar-weighted expected CAGR", f"{blended_cagr:.2f}%")
            c3.metric("Approx. yearly contributions", f"${annual_contribution:,.0f}")

            allocation_rows=[]
            for r in active_rows:
                weight = r["Daily Dollars"] / total_daily * 100
                hist_text = "N/A" if r["Historical CAGR"] is None else f"{r['Historical CAGR']:.1f}%"
                allocation_rows.append({
                    "Ticker": r["Ticker"],
                    "Per Day": f"${r['Daily Dollars']:.2f}",
                    "Auto Weight": f"{weight:.1f}%",
                    "Historical CAGR": hist_text,
                    "Expected CAGR": f"{r['Expected CAGR']:.1f}%",
                    "Per Year": f"${r['Daily Dollars']*252:,.0f}",
                })
            st.dataframe(pd.DataFrame(allocation_rows), use_container_width=True, hide_index=True)

            horizons = [5, 10, 15, 20, 25, 30]
            projection_rows=[]
            for years in horizons:
                portfolio_value = 0.0
                for r in active_rows:
                    share = r["Daily Dollars"] / total_daily
                    allocated_start = starting_balance * share
                    portfolio_value += project_dca(allocated_start, r["Daily Dollars"], r["Expected CAGR"], years)
                contributed = starting_balance + annual_contribution * years
                projection_rows.append({
                    "Years": years,
                    "Projected Value": portfolio_value,
                    "Total Contributed": contributed,
                    "Estimated Growth": portfolio_value - contributed,
                })
            proj = pd.DataFrame(projection_rows)
            display_proj = proj.copy()
            for col in ["Projected Value", "Total Contributed", "Estimated Growth"]:
                display_proj[col] = display_proj[col].map(lambda x: f"${x:,.0f}")
            st.markdown("#### What it could grow to")
            st.dataframe(display_proj, use_container_width=True, hide_index=True)

            chart_data = proj.set_index("Years")[["Projected Value", "Total Contributed"]]
            st.line_chart(chart_data)

            twenty_year = float(proj.loc[proj["Years"] == 20, "Projected Value"].iloc[0])
            custom_facts = "\n".join([
                f"Total trading-day contribution: ${total_daily:.2f}",
                f"Starting balance: ${starting_balance:.0f}",
                f"Dollar-weighted expected CAGR assumption: {blended_cagr:.2f}%",
                "Holdings: " + "; ".join(
                    f"{r['Ticker']} ${r['Daily Dollars']:.2f}/day at {r['Expected CAGR']:.1f}% expected CAGR"
                    + (f" (historical {r['Historical CAGR']:.1f}% over {r['History Years']:.1f}y)" if r['Historical CAGR'] is not None and r['History Years'] is not None else "")
                    for r in active_rows
                ),
                f"20-year projected value: ${twenty_year:.0f}",
            ])
            custom_ai = two_sentence_ai_take("custom DCA projection", custom_facts)
            st.markdown("### ✨ Quick take")
            st.write(custom_ai or custom_dca_fallback_take(total_daily, blended_cagr, active_rows, twenty_year))

            st.caption("Historical CAGR is based on adjusted price history and is not a forecast. Projection assumes steady monthly-equivalent contributions and constant annual returns; it ignores taxes, fees, inflation, and changing market returns.")

    with st.expander("What does DCA mean?"):
        st.write("Dollar-cost averaging means investing a fixed dollar amount on a regular schedule instead of trying to guess the perfect day to buy. It can make a long-term plan easier to stick with, but it does not prevent losses.")
    with st.expander("How do expected CAGR projections work?"):
        st.write("CAGR is the annual return assumption used for the projection. A 10% expected CAGR does not mean the investment will earn 10% every year; real markets are uneven, and future returns can be much lower or higher.")
    st.info("These are educational models, not personalized investment recommendations. A real allocation should also consider time horizon, emergency savings, taxes, diversification, and ability to tolerate losses.")

st.divider()
st.caption("For research and educational purposes only. Not investment advice.")
with st.expander("About Conviction AI"):
    st.write(
        "Conviction AI is designed to make stock, ETF, and DCA research easier for beginner investors by summarizing public market data into simple rankings, comparisons, and educational model allocations."
    )
with st.expander("Data & disclaimer"):
    st.write(
        "Live market data is sourced from Yahoo Finance through yfinance and may be delayed, incomplete, or unavailable. "
        "Conviction AI is a research tool, not investment advice. Always verify material figures before making an investment decision."
    )
