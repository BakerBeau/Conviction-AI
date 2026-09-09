import math
import os
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

st.title("📈 Conviction AI")
st.caption("Type a ticker. Conviction AI pulls available market/fundamental data and converts it into a transparent 0–100 research score.")
st.info(
    "Research tool only — not investment advice. Live data is sourced from Yahoo Finance through yfinance and may be delayed, incomplete, or unavailable. "
    "Always verify material figures before making an investment decision."
)

WEIGHTS = {
    "EPS Growth": 0.18,
    "Revenue Growth": 0.12,
    "Net Margin": 0.12,
    "ROIC / Capital Efficiency": 0.12,
    "Forward P/E": 0.10,
    "Analyst Upside": 0.12,
    "Institutional Ownership": 0.08,
    "Insider Activity": 0.06,
    "12M Momentum": 0.10,
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


def analyst_upside(info):
    current = clean_num(info.get("currentPrice") or info.get("regularMarketPrice"))
    target = clean_num(info.get("targetMeanPrice"))
    if current is None or current <= 0 or target is None:
        return None
    return (target / current - 1) * 100


@st.cache_data(ttl=900, show_spinner=False)
def fetch_stock(symbol):
    t = yf.Ticker(symbol)
    errors = []

    try:
        info = t.info or {}
    except Exception as e:
        info, errors = {}, [f"Company data: {e}"]

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
        "institutional_ownership": pct(info.get("heldPercentInstitutions")),
        "insider_activity": insider_score(insiders),
        "momentum": momentum_12m(hist),
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
        "Analyst Upside": normalize(m.get("analyst_upside"), -10, 35),
        "Institutional Ownership": normalize(m.get("institutional_ownership"), 20, 90),
        "Insider Activity": normalize(m.get("insider_activity"), -5, 5),
        "12M Momentum": normalize(m.get("momentum"), -20, 40),
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
    headers = {"User-Agent": "Mozilla/5.0 ConvictionAI/0.5"}
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


def quarter_movers(snapshot_df, quarter=None):
    quarter = quarter or current_quarter_id()
    prior = previous_quarter_id(quarter)
    cur = snapshot_df[snapshot_df["quarter"] == quarter].copy()
    prev = snapshot_df[snapshot_df["quarter"] == prior].copy()
    if cur.empty or prev.empty:
        return pd.DataFrame(), pd.DataFrame(), prior
    cur["score"] = pd.to_numeric(cur["score"], errors="coerce")
    prev["score"] = pd.to_numeric(prev["score"], errors="coerce")
    merged = cur.merge(prev[["ticker", "score"]], on="ticker", suffixes=("_current", "_prior"))
    merged["change"] = merged["score_current"] - merged["score_prior"]
    merged = merged.dropna(subset=["change"])
    winners = merged.sort_values("change", ascending=False).head(10)
    losers = merged.sort_values("change", ascending=True).head(10)
    return winners, losers, prior


def scan_universe(tickers, workers=8, progress_callback=None):
    rows = []
    tickers = list(dict.fromkeys(tickers))

    def score_one(ticker):
        result = fetch_stock(ticker)
        score, _, _ = score_stock(result["metrics"])
        coverage = sum(v is not None for v in result["metrics"].values())
        if score is None:
            return None
        return {
            "ticker": ticker,
            "company": result["company"],
            "score": round(score, 2),
            "coverage": coverage,
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


with st.sidebar:
    st.header("Conviction Engine")
    st.caption(f"Quarterly storage: {storage_backend()}")
    st.write("Weights")
    for k, v in WEIGHTS.items():
        st.caption(f"{k}: {v:.0%}")
    st.divider()
    st.caption("Missing metrics are excluded and remaining weights are re-normalized, rather than scored as zero.")

st.markdown("## Market Leaderboards")
st.caption("Rank a chosen stock universe by the same Conviction AI score, then track score changes from one saved quarter to the next.")

with st.expander("Leaderboard universe", expanded=True):
    universe_choice = st.selectbox(
        "Automatic universe",
        ["S&P 500 + Nasdaq-100", "S&P 500", "Nasdaq-100"],
        index=0,
        help="Membership is refreshed from public constituent tables and cached for 24 hours.",
    )
    universe_df = build_market_universe(universe_choice)
    leaderboard_universe = universe_df["ticker"].tolist()
    u1, u2, u3 = st.columns(3)
    u1.metric("Universe size", f"{len(leaderboard_universe):,}")
    u2.metric("S&P 500 members loaded", f"{len(fetch_index_universe('S&P 500')):,}")
    u3.metric("Nasdaq-100 members loaded", f"{len(fetch_index_universe('Nasdaq-100')):,}")
    st.caption("The combined universe is deduplicated by ticker. Class-share tickers are normalized for Yahoo Finance (for example, BRK.B → BRK-B).")

q_now = current_quarter_id()
scan_col, snap_col = st.columns([1, 1])
with scan_col:
    scan = st.button("Refresh Top 10", use_container_width=True)
with snap_col:
    save_snapshot = st.button(f"Save {q_now} Snapshot", use_container_width=True)

if "leaderboard_df" not in st.session_state:
    st.session_state.leaderboard_df = pd.DataFrame()

if scan or save_snapshot:
    progress = st.progress(0.0, text=f"Scoring 0 / {len(leaderboard_universe)} tickers…")

    def update_progress(done, total):
        progress.progress(done / max(total, 1), text=f"Scoring {done:,} / {total:,} tickers…")

    st.session_state.leaderboard_df = scan_universe(
        leaderboard_universe, workers=8, progress_callback=update_progress
    )
    progress.empty()

if save_snapshot:
    if st.session_state.leaderboard_df.empty:
        st.warning("No scores were available to save.")
    else:
        saved = save_quarter_snapshot(st.session_state.leaderboard_df.to_dict("records"), q_now)
        st.success(f"Saved {saved} ticker scores for {q_now}.")

leaderboard_df = st.session_state.leaderboard_df
snapshots = load_snapshots()
winners, losers, prior_q = quarter_movers(snapshots, q_now)

tab_top, tab_up, tab_down = st.tabs(["🏆 Current Top 10", "🚀 Top 10 Movers", "📉 Top 10 Losers"])
with tab_top:
    if leaderboard_df.empty:
        st.info("Click **Refresh Top 10** to score the current universe.")
    else:
        top10 = leaderboard_df.head(10).copy()
        top10.insert(0, "Rank", range(1, len(top10) + 1))
        top10["Score"] = top10["score"].map(lambda x: f"{x:.1f}")
        top10["Coverage"] = top10["coverage"].map(lambda x: f"{int(x)}/9")
        st.dataframe(top10[["Rank", "ticker", "company", "Score", "Coverage"]], use_container_width=True, hide_index=True)
        st.bar_chart(top10.set_index("ticker")[["score"]])

with tab_up:
    if winners.empty:
        st.info(f"QoQ movers need saved snapshots for both **{prior_q}** and **{q_now}**. Save one snapshot each quarter and this list will populate automatically.")
    else:
        show = winners.copy()
        show.insert(0, "Rank", range(1, len(show) + 1))
        show["Current"] = show["score_current"].map(lambda x: f"{x:.1f}")
        show["Prior"] = show["score_prior"].map(lambda x: f"{x:.1f}")
        show["QoQ Change"] = show["change"].map(lambda x: f"+{x:.1f}" if x >= 0 else f"{x:.1f}")
        st.dataframe(show[["Rank", "ticker", "company", "Current", "Prior", "QoQ Change"]], use_container_width=True, hide_index=True)

with tab_down:
    if losers.empty:
        st.info(f"QoQ losers need saved snapshots for both **{prior_q}** and **{q_now}**. Save one snapshot each quarter and this list will populate automatically.")
    else:
        show = losers.copy()
        show.insert(0, "Rank", range(1, len(show) + 1))
        show["Current"] = show["score_current"].map(lambda x: f"{x:.1f}")
        show["Prior"] = show["score_prior"].map(lambda x: f"{x:.1f}")
        show["QoQ Change"] = show["change"].map(lambda x: f"{x:.1f}")
        st.dataframe(show[["Rank", "ticker", "company", "Current", "Prior", "QoQ Change"]], use_container_width=True, hide_index=True)

st.divider()
st.markdown("## Single-stock analysis")

left, right = st.columns([3, 1])
with left:
    symbol = st.text_input("Ticker", value="AVGO", placeholder="AVGO, GOOGL, META…").upper().strip()
with right:
    st.write("")
    st.write("")
    run = st.button("Analyze Live", type="primary", use_container_width=True)

if run and symbol:
    with st.spinner(f"Pulling available data for {symbol}…"):
        result = fetch_stock(symbol)

    m = result["metrics"]
    score, contributions, raw_scores = score_stock(m)
    available = sum(v is not None for v in m.values())

    st.divider()
    st.subheader(f"{result['company']} ({symbol})")
    st.caption(f"{result['sector']} • {result['industry']} • Data fetched {result['fetched_at']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Conviction Score", "N/A" if score is None else f"{score:.1f}/100")
    c2.metric("Rating", label(score))
    c3.metric("Price", "N/A" if result["price"] is None else f"${result['price']:,.2f}")
    c4.metric("Data Coverage", f"{available}/9")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Market Cap", market_cap_fmt(result["market_cap"]))
    c6.metric("Analyst Mean Target", "N/A" if result["target_mean"] is None else f"${result['target_mean']:,.2f}")
    c7.metric("Analyst Upside", fmt(m["analyst_upside"]))
    c8.metric("Consensus", str(result["recommendation"]).replace("_", " ").title())

    metric_rows = [
        ("EPS Growth", m["eps_growth"], "%"),
        ("Revenue Growth", m["revenue_growth"], "%"),
        ("Net Margin", m["net_margin"], "%"),
        ("ROIC / Capital Efficiency", m["roic"], "%"),
        ("Forward P/E", m["forward_pe"], "x"),
        ("Analyst Upside", m["analyst_upside"], "%"),
        ("Institutional Ownership", m["institutional_ownership"], "%"),
        ("Insider Activity", m["insider_activity"], "/5"),
        ("12M Momentum", m["momentum"], "%"),
    ]

    table = pd.DataFrame([
        {
            "Factor": name,
            "Live Value": "N/A" if value is None else (f"{value:.1f}{unit}" if unit != "x" else f"{value:.1f}x"),
            "Factor Score": "N/A" if raw_scores.get(name) is None else f"{raw_scores[name]:.0f}/100",
            "Weight": f"{WEIGHTS[name]:.0%}",
        }
        for name, value, unit in metric_rows
    ])
    st.markdown("### Live Research Breakdown")
    st.dataframe(table, use_container_width=True, hide_index=True)

    chart_df = pd.DataFrame(
        {"Factor": [k for k, v in raw_scores.items() if v is not None],
         "Score": [v for v in raw_scores.values() if v is not None]}
    ).set_index("Factor")
    if not chart_df.empty:
        st.bar_chart(chart_df)

    strengths, risks = [], []
    if m["eps_growth"] is not None and m["eps_growth"] >= 20: strengths.append("Strong EPS growth")
    if m["revenue_growth"] is not None and m["revenue_growth"] >= 15: strengths.append("Healthy top-line growth")
    if m["net_margin"] is not None and m["net_margin"] >= 20: strengths.append("High profitability")
    if m["roic"] is not None and m["roic"] >= 15: strengths.append("Strong capital efficiency")
    if m["analyst_upside"] is not None and m["analyst_upside"] >= 15: strengths.append("Positive analyst implied upside")
    if m["institutional_ownership"] is not None and m["institutional_ownership"] >= 65: strengths.append("High institutional ownership")
    if m["momentum"] is not None and m["momentum"] >= 15: strengths.append("Strong 12-month momentum")

    if m["forward_pe"] is not None and m["forward_pe"] > 40: risks.append("Elevated forward valuation")
    if m["eps_growth"] is not None and m["eps_growth"] < 5: risks.append("Weak/negative EPS growth")
    if m["net_margin"] is not None and m["net_margin"] < 5: risks.append("Thin profitability")
    if m["analyst_upside"] is not None and m["analyst_upside"] < 0: risks.append("Mean analyst target below current price")
    if m["insider_activity"] is not None and m["insider_activity"] < -2: risks.append("Recent reported insider activity skews negative")
    if m["momentum"] is not None and m["momentum"] < -10: risks.append("Negative 12-month momentum")

    s1, s2 = st.columns(2)
    with s1:
        st.markdown("### Strengths")
        st.write("\n".join(f"• {x}" for x in strengths) if strengths else "• No standout strength threshold triggered")
    with s2:
        st.markdown("### Risks")
        st.write("\n".join(f"• {x}" for x in risks) if risks else "• No major risk threshold triggered")

    st.markdown("### Shareable Summary")
    if score is not None:
        summary = f"{symbol} scores {score:.1f}/100 on Conviction AI ({label(score)}) using {available}/9 available live factors."
        if strengths:
            summary += " Strengths: " + ", ".join(strengths[:3]) + "."
        if risks:
            summary += " Key risk: " + risks[0] + "."
        st.code(summary)

    with st.expander("Methodology & data caveats"):
        st.write(
            "The score is a transparent weighted model, not a prediction model. ROIC is an approximation calculated from the latest statements when the needed rows are available. "
            "Institutional ownership is a current ownership percentage, not hedge-fund flow. Insider activity is a rough signal from reported transactions. Analyst upside uses the mean analyst target versus current price. "
            "Yahoo/yfinance fields can be delayed, missing, or defined differently by issuer."
        )

    if result["errors"]:
        with st.expander("Data warnings"):
            for err in result["errors"]:
                st.warning(err)

elif run:
    st.warning("Enter a ticker first.")
else:
    st.markdown("### Try it")
    st.write("Enter **AVGO**, **GOOGL**, **META**, **AMZN**, or another U.S.-listed ticker and click **Analyze Live**.")

st.divider()
st.caption("Version 0.5 — automatic S&P 500 / Nasdaq-100 universe + parallel scoring + persistent Supabase/Postgres quarterly history + Top 10 and QoQ score movers. Next: premium market data, accounts, alerts, and Stripe.")
