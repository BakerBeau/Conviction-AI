# Conviction AI v0.6.5

Beginner-friendly stock and ETF research in one Streamlit app.

## New in v0.6.5 — Hidden Gems

The Stocks section now includes a final **Hidden Gems** tab.

A Hidden Gem must first pass a quality screen:
- Conviction Score of 70+
- At least 7/10 scoring factors available
- S&P 500 company
- Approx. $2B–$150B market cap
- 8–20 covering analysts
- Chart Health of 60+
- Positive EPS/revenue growth, with at least one at 10%+
- No clearly weak institutional/insider signal
- At least 5% analyst target upside when that data is available

The app then calculates a separate **Hidden Gem Score** that rewards quality plus being relatively underfollowed. Low trading volume by itself is not treated as a positive signal.

The discovery button checks a random slice of the S&P 500, builds a qualifying pool, and randomly surfaces one candidate. Users can roll another candidate from the same pool or open the full 10-factor analysis.

## Main sections

### Stocks
- Search a Stock
- Market Leaders
  - Top Stocks
  - Biggest Improvers
  - Analyst Opportunities
- Hidden Gems

### ETFs
- ETF Finder
- Top ETFs
- Best All-Around

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Upload these files to the existing GitHub repository and commit the changes. Streamlit Community Cloud should redeploy automatically.

## Disclaimer

For research and educational purposes only. Not investment advice. Live market data may be delayed, incomplete, or unavailable.


## Hidden Gems v0.6.5
The Hidden Gems screen now uses a broader quality floor and a ranking model instead of a narrow perfect-checklist filter. Default discovery depth is 120 randomly selected S&P 500 companies, with 80/120/160 options.
