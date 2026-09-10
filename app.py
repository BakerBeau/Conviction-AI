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
    "EPS Growth": 0.16,
    "Revenue Growth": 0.11,
    "Net Margin": 0.11,
    "ROIC / Capital Efficiency": 0.11,
    "Forward P/E": 0.10,
    "Analyst Conviction": 0.11,
    "Institutional Ownership": 0.07,
    "Insider Activity": 0.05,
    "12M Momentum": 0.08,
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



def chart_health(history):
    """0-100 technical health score using trend, moving averages, and proximity to the 52-week high."""
    if history is None or history.empty or "Close" not in history:
        return None
    closes = history["Close"].dropna()
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

    # 3-month trend contributes up to 15 points.
    if len(closes) >= 63:
        mom3 = (price / float(closes.iloc[-63]) - 1) * 100
        score += normalize(mom3, -10, 15) * 0.15

    # Staying close to the 52-week high contributes up to 15 points.
    if high52 > 0:
        drawdown = (price / high52 - 1) * 100
        score += normalize(drawdown, -30, 0) * 0.15

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
    parts.append(normalize(upside, -10, 40)); weights.append(0.60)
    parts.append(normalize(count, 8, 35)); weights.append(0.20)
    if recommendation_mean is not None:
        # Yahoo convention is roughly 1=Strong Buy, 5=Sell.
        parts.append(normalize(recommendation_mean, 1.0, 4.0, reverse=True)); weights.append(0.20)

    good = [(p, w) for p, w in zip(parts, weights) if p is not None]
    if not good:
        return None
    denom = sum(w for _, w in good)
    return round(sum(p*w for p, w in good) / denom, 1)


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

    metrics = {
        "eps_growth": pct(info.get("earningsGrowth")),
        "revenue_growth": pct(info.get("revenueGrowth")),
        "net_margin": pct(info.get("profitMargins")),
        "roic": calc_roic(fin, bs),
        "forward_pe": clean_num(info.get("forwardPE")),
        "analyst_upside": analyst_upside(info),
        "analyst_conviction": analyst_conviction(info),
        "institutional_ownership": pct(info.get("heldPercentInstitutions")),
        "insider_activity": insider_score(insiders),
        "momentum": momentum_12m(hist),
        "chart_health": chart_health(hist),
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
        "EPS Growth": normalize(m.get("eps_growth"), -20, 50),
        "Revenue Growth": normalize(m.get("revenue_growth"), -10, 35),
        "Net Margin": normalize(m.get("net_margin"), -5, 35),
        "ROIC / Capital Efficiency": normalize(m.get("roic"), 0, 30),
        "Forward P/E": normalize(m.get("forward_pe"), 10, 45, reverse=True),
        "Analyst Conviction": clean_num(m.get("analyst_conviction")),
        "Institutional Ownership": normalize(m.get("institutional_ownership"), 20, 90),
        "Insider Activity": normalize(m.get("insider_activity"), -5, 5),
        "12M Momentum": normalize(m.get("momentum"), -20, 40),
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
    return total, contributions, raw_scores




@st.cache_data(ttl=86400, show_spinner=False)
def fetch_index_universe(index_name):
    """Refresh index membership from public constituent tables; return a stable fallback if unavailable."""
    headers = {"User-Agent": "Mozilla/5.0 ConvictionAI/0.6.4"}
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
            "one_year_return": result["metrics"].get("momentum"),
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


def hidden_gem_score(row):
    """Score established, relatively underfollowed S&P 500 companies without requiring a perfect checklist."""
    score = clean_num(row.get("score"))
    coverage = clean_num(row.get("coverage"))
    market_cap = clean_num(row.get("market_cap"))
    analysts = clean_num(row.get("analyst_count"))
    chart = clean_num(row.get("chart_health"))
    eps = clean_num(row.get("eps_growth"))
    revenue = clean_num(row.get("revenue_growth"))
    inst = clean_num(row.get("institutional_ownership"))
    upside = clean_num(row.get("analyst_upside"))
    insider = clean_num(row.get("insider_activity"))

    # Hard floor: enough real data and enough overall quality to be worth discovering.
    if score is None or score < 67 or coverage is None or coverage < 7:
        return None
    if market_cap is None or not (2e9 <= market_cap <= 250e9):
        return None
    if analysts is None or not (6 <= analysts <= 30):
        return None
    if chart is None or chart < 55:
        return None

    # Growth should be healthy, but we no longer require both growth lines to be perfect.
    growth_values = [x for x in (eps, revenue) if x is not None]
    if not growth_values or max(growth_values) < 5:
        return None
    if len(growth_values) == 2 and eps < -5 and revenue < -5:
        return None

    # Avoid clear red flags while allowing neutral / missing signals.
    if inst is not None and inst < 25:
        return None
    if upside is not None and upside < 0:
        return None
    if insider is not None and insider < -5:
        return None

    # Rank rather than over-filter: quality + underfollowed + technical health + upside/growth.
    analyst_underfollowed = normalize(analysts, 6, 30, reverse=True)
    cap_underfollowed = normalize(math.log10(market_cap), math.log10(2e9), math.log10(250e9), reverse=True)
    quality = normalize(score, 67, 90)
    chart_part = normalize(chart, 55, 90)
    growth_best = max(growth_values) if growth_values else None
    growth_part = normalize(growth_best, 5, 30) if growth_best is not None else None
    upside_part = normalize(upside, 0, 30) if upside is not None else None

    parts = [
        (quality, 0.30),
        (analyst_underfollowed, 0.20),
        (cap_underfollowed, 0.18),
        (chart_part, 0.14),
        (growth_part, 0.10),
        (upside_part, 0.08),
    ]
    usable = [(v, w) for v, w in parts if v is not None]
    if not usable:
        return None
    return round(sum(v*w for v, w in usable) / sum(w for v, w in usable), 1)


def hidden_gem_reasons(row):
    reasons = []
    score = clean_num(row.get("score"))
    eps = clean_num(row.get("eps_growth"))
    revenue = clean_num(row.get("revenue_growth"))
    upside = clean_num(row.get("analyst_upside"))
    analysts = clean_num(row.get("analyst_count"))
    chart = clean_num(row.get("chart_health"))
    inst = clean_num(row.get("institutional_ownership"))
    market_cap = clean_num(row.get("market_cap"))
    if score is not None: reasons.append(f"Conviction Score {score:.1f}/100")
    if eps is not None and eps >= 10: reasons.append(f"EPS growth {eps:+.1f}%")
    elif revenue is not None and revenue >= 10: reasons.append(f"Revenue growth {revenue:+.1f}%")
    if chart is not None: reasons.append(f"Chart health {chart:.0f}/100")
    if upside is not None: reasons.append(f"Analyst target upside {upside:+.1f}%")
    if analysts is not None: reasons.append(f"Only {int(analysts)} analysts covering it")
    if inst is not None and inst >= 35: reasons.append(f"Institutional ownership {inst:.0f}%")
    if market_cap is not None: reasons.append(f"Market cap {market_cap_fmt(market_cap)}")
    return reasons[:5]


def pick_hidden_gem_from_df(df):
    if df is None or df.empty:
        return None, pd.DataFrame()
    work = df.copy()
    work["hidden_gem_score"] = work.apply(hidden_gem_score, axis=1)
    pool = work.dropna(subset=["hidden_gem_score"]).copy()
    if pool.empty:
        return None, pool
    # Keep randomness, but bias the draw toward stronger underfollowed candidates.
    top_pool = pool.sort_values("hidden_gem_score", ascending=False).head(min(25, len(pool)))
    row = top_pool.sample(1).iloc[0].to_dict()
    return row, pool.sort_values("hidden_gem_score", ascending=False)


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
            "Conviction AI is a transparent weighted research score, not a prediction. Missing metrics are excluded and the remaining weights are re-normalized. "
            "Analyst Conviction requires at least 8 analysts and combines target upside, analyst count, and consensus rating. "
            "Chart Health uses the 50-day and 200-day moving averages, trend relationship, 3-month momentum, and distance from the 52-week high."
        )
        weight_df = pd.DataFrame({"Factor": list(WEIGHTS.keys()), "Weight": [f"{v:.0%}" for v in WEIGHTS.values()]})
        st.dataframe(weight_df, hide_index=True, use_container_width=True)


with main_stocks:
    stock_search_tab, market_tab, hidden_tab = st.tabs(["🔎 Search a Stock", "🏆 Market Leaders", "💎 Hidden Gems"])

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
                top10["1Y Return"] = top10["one_year_return"].map(lambda x: "N/A" if pd.isna(x) else f"{x:+.1f}%")
                top10["Chart Health"] = top10["chart_health"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.0f}/100")
                st.dataframe(
                    top10[["Rank", "ticker", "company", "Score", "Coverage", "1Y Return", "Chart Health"]],
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



    with hidden_tab:
        st.markdown("### 💎 Hidden Gems")
        st.caption("Discover an established S&P 500 company with strong fundamentals that may be getting less attention than the mega-cap names.")

        with st.expander("What counts as a Hidden Gem?"):
            st.markdown(
                "A company must have about a **67+ Conviction Score**, at least **7/10 data factors**, roughly **$2B–$250B market cap**, "
                "**6–30 covering analysts**, **55+ chart health**, at least one healthy growth signal, and no major negative institutional/insider signal. "
                "The final pick is randomized from the qualifying pool. Low trading volume by itself is **not** considered a positive signal."
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
            st.caption("Randomly selected from companies that passed the Hidden Gem screen. Discovery only — not a recommendation.")
            a, b, c, d = st.columns(4)
            a.metric("Conviction", f"{gem['score']:.1f}/100")
            b.metric("Hidden Gem Score", f"{gem['hidden_gem_score']:.1f}/100")
            c.metric("Analyst Upside", "N/A" if pd.isna(gem.get("analyst_upside")) else f"{gem['analyst_upside']:+.1f}%")
            d.metric("Chart Health", "N/A" if pd.isna(gem.get("chart_health")) else f"{gem['chart_health']:.0f}/100")

            st.markdown("### Why it surfaced")
            st.write("\n".join(f"• {x}" for x in hidden_gem_reasons(gem)))

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

with main_dca:
    st.markdown("### 🧱 Build a simple DCA plan")
    st.caption("Choose what you are trying to build and how aggressive you want to be. Conviction AI turns the daily amount into an easy educational example allocation.")
    c1, c2, c3 = st.columns(3)
    with c1:
        daily_amount = st.number_input("Amount per day", min_value=1.0, max_value=1000.0, value=10.0, step=1.0, key="dca_daily")
    with c2:
        dca_goal = st.selectbox("What are you trying to build?", ["Keep it simple", "Long-term growth", "Core + a little spice", "Income & stability"], key="dca_goal")
    with c3:
        dca_risk = st.selectbox("How aggressive?", ["Conservative", "Balanced", "Growth", "Aggressive"], index=2, key="dca_risk")

    model = DCA_MODELS[(dca_goal, dca_risk)]
    rows=[]
    for ticker, weight, role in model:
        rows.append({"ETF": ticker, "Role": role, "Weight": f"{weight}%", "Per Day": f"${daily_amount*weight/100:.2f}", "Per Week": f"${daily_amount*7*weight/100:.2f}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    annual = daily_amount * 365
    st.metric("Approx. yearly contributions", f"${annual:,.0f}")
    core_weight = sum(w for _t,w,role in model if "Core" in role or "Broad" in role or "Dividend core" in role)
    spicy_weight = sum(w for _t,w,role in model if "Spicy" in role or "Theme" in role or "Momentum" in role or "Nasdaq" in role)
    st.caption(f"This example is about **{core_weight}% core** and **{spicy_weight}% higher-octane satellite** exposure, with the rest used for diversification/stability.")
    with st.expander("What does DCA mean?"):
        st.write("Dollar-cost averaging means investing a fixed dollar amount on a regular schedule instead of trying to guess the perfect day to buy. It can make a long-term plan easier to stick with, but it does not prevent losses.")
    with st.expander("Why core + satellite?"):
        st.write("The core is meant to do most of the long-term compounding. Smaller satellite positions can add growth, momentum, sectors, or themes without letting the entire plan depend on one concentrated bet.")
    st.info("These are educational model allocations, not personalized investment recommendations. A real allocation should also consider time horizon, emergency savings, taxes, and ability to tolerate losses.")

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
