# Conviction AI v0.6.2

Beginner-friendly stock and ETF research dashboard built with Streamlit.

## Cleaned-up layout

The app now uses two simple top-level sections:

- **Stocks**
  - Search a Stock
  - Market Leaders
    - Top Stocks
    - Biggest Improvers
    - Analyst Opportunities
    - Biggest Fallers tucked inside an expander
- **ETFs**
  - ETF Finder
  - Top ETFs
  - Best All-Around

The developer-style sidebar and extra leaderboard clutter were removed from the main experience.

## Stocks

- 10-factor Conviction Score
- Minimum 7/10 factor coverage for leaderboard eligibility
- Top Stocks shows score, coverage, 1-year return and analyst upside
- Analyst Opportunities requires at least 8 analysts
- Quarterly score changes update on Jan 1, Apr 1, Jul 1 and Oct 1
- Biggest Fallers remain available but no longer take up a primary tab

## ETFs

- Expanded curated universe including VOO, VOOG, QQQM, SCHG, SPMO, MOAT, VGT, SMH, XMMO, AVUV and many more
- ETF Finder for direct ticker lookup
- Top ETFs with YTD, 1Y, 3Y CAGR, 5Y CAGR and 10Y CAGR toggles
- Best All-Around score balances returns, volatility, drawdown, expenses, AUM and liquidity
- Leveraged ETFs hidden by default

## Deploy on Streamlit Community Cloud

1. Unzip this package.
2. In your GitHub `Conviction-AI` repository, use **Add file → Upload files**.
3. Upload/replace the files from this package.
4. Click **Commit changes**.
5. Your existing Streamlit app should redeploy automatically from `app.py`.

## Quarterly snapshots

The included GitHub Actions workflow runs `quarterly_snapshot.py` at the start of each quarter and commits the updated `quarterly_scores.csv` back to the repository.

## Data note

This MVP uses Yahoo Finance through `yfinance`. Free public data can be delayed, incomplete or temporarily unavailable. Under-covered stocks are excluded from rankings rather than being treated as zero.

Conviction AI is a research tool, not personalized investment advice.
