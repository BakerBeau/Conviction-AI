# Conviction AI v0.6.6

Beginner-friendly stock, ETF, and DCA research app built with Streamlit.

## What is new in v0.6.6

- Analyst data is de-emphasized in Market Leaders. It remains one of the 10 stock factors, while the price-target-gap table is now a secondary expandable view.
- Top Stocks now highlights Chart Health instead of analyst upside.
- ETF Compare lets users compare 2–5 ETFs side by side across YTD, 1Y, 3Y/5Y/10Y CAGR, expense ratio, volatility, and drawdown.
- ETF filters use multiple tags, so a fund can appear in Growth, Large Cap, Technology-heavy, Core, etc. instead of being trapped in one category.
- Expanded curated ETF universe so category filters return healthier lists.
- New DCA Builder: choose a daily dollar amount, goal, and aggressiveness to see an educational ETF model allocation.
- DCA presets emphasize a core + satellite structure and clearly label examples as educational, not personalized investment recommendations.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Upload the contents of this folder to your GitHub repository. Streamlit Community Cloud should redeploy automatically after the commit.

## Data note

Market data is sourced through `yfinance` and may be delayed, incomplete, or unavailable. Conviction AI is for research and education, not investment advice.
