# Conviction AI v0.6

Beginner-friendly stock and ETF research dashboard built with Streamlit.

## What's new in v0.6

### Stocks
- 10-factor Conviction Score
- Analyst Conviction replaces simple analyst upside
- Analyst Conviction requires at least 8 covering analysts and blends:
  - mean target upside vs current price
  - analyst coverage count
  - consensus recommendation strength
- Analyst Opportunities leaderboard ranks the biggest current-price vs mean-target gaps, with a minimum 8-analyst filter
- Chart Health remains one of the 10 stock factors
- Stocks need at least 7/10 available factors to be leaderboard-eligible
- Quarterly risers/fallers compare Jan 1 / Apr 1 / Jul 1 / Oct 1 snapshots

### ETFs
- Beginner-friendly ETF Leaderboard
- Toggle between YTD, 1Y, 3Y CAGR, 5Y CAGR, and 10Y CAGR
- Filter by ETF category
- Leveraged ETFs excluded by default, with an optional toggle to include them
- Best All-Around ETF score blends long-term return, volatility, max drawdown, expense ratio, fund size, and liquidity when those fields are available

## Deploy on Streamlit Community Cloud

1. Upload all files in this folder to your GitHub repository.
2. In Streamlit Community Cloud, deploy `app.py` from the repository.
3. Streamlit will install the packages in `requirements.txt`.
4. Existing Streamlit deployments normally redeploy automatically after the GitHub commit.

## Quarterly snapshots

The included GitHub Actions workflow is designed to run the stock snapshot process at the start of each quarter. The generated `quarterly_scores.csv` powers the Quarterly Risers and Quarterly Fallers tabs.

## Data note

This MVP uses Yahoo Finance through `yfinance`. Free public data can be delayed, incomplete, or temporarily unavailable. The app excludes under-covered stocks from ranking rather than treating missing fields as zero. Before charging users, a licensed production market-data provider would be preferable.

## Important

Conviction AI is a research tool, not personalized investment advice.
