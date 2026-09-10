# Conviction AI v0.7.1

Beginner-friendly stock, ETF, and DCA research app built with Streamlit.

## What is new in v0.6.7

- New **Emerging Leaders** stock-discovery tab between Market Leaders and Hidden Gems.
- Emerging Leaders ranks stocks that may not be Top 10 yet but show a useful mix of Conviction Score, chart health, momentum, growth, institutional support, and quarter-to-quarter score improvement when history exists.
- New optional **2-sentence AI Quick Take** for Hidden Gems.
- New optional **2-sentence AI Quick Take** for DCA model allocations.
- AI summaries are constrained to the facts already calculated by Conviction AI and are instructed not to invent facts or issue buy/sell commands.
- If no OpenAI API key is configured, the app automatically shows a plain-English local fallback summary instead, so nothing breaks.

## Turn on the AI summaries

Do **not** put your API key in GitHub.

In Streamlit Community Cloud, open your app settings and add this to **Secrets**:

```toml
OPENAI_API_KEY = "your_api_key_here"
```

Optional model override:

```toml
OPENAI_MODEL = "gpt-5-mini"
```

The app uses OpenAI's Responses API. AI output is cached for 24 hours for the same set of facts to reduce repeat API calls.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

For local AI summaries, set `OPENAI_API_KEY` as an environment variable. `OPENAI_MODEL` is optional.

## Deploy

Upload the contents of this folder to your GitHub repository. Streamlit Community Cloud should redeploy automatically after the commit.

## Data note

Market data is sourced through `yfinance` and may be delayed, incomplete, or unavailable. Conviction AI is for research and education, not investment advice.


## DCA convention
Daily DCA inputs are treated as trading-day contributions using 252 trading days per year. For example, $10 per trading day is modeled as approximately $2,520 per year.


## v0.7.0 custom DCA updates
- Build Your Own DCA now uses **dollar amounts per trading day** instead of manual percentage weights.
- Portfolio weights are calculated automatically from those dollar amounts.
- Entering a ticker pre-fills **Expected CAGR** from its trailing adjusted-price historical CAGR (up to 5 years) when enough history exists.
- Users can override the historical CAGR assumption.
- Multi-holding projections compound each holding using its own expected CAGR and sum the results.


## v0.7.1 scoring calibration
- EPS growth is much harder to max out; 20-30% growth now scores as strong rather than perfect.
- Revenue growth, margins, and ROIC use tougher piecewise curves.
- Valuation is growth-adjusted using forward P/E plus a PEG-like growth relationship.
- Analyst Conviction has less weight and requires stronger evidence for elite scores.
- Institutional ownership is treated as a weak confirmation signal and is capped because ownership level alone is not a bullish catalyst.
- Insider activity rewards buying more than it punishes routine selling.
- Momentum blends 3-, 6-, and 12-month persistence so it overlaps less with Chart Health.
- Chart Health is stricter and includes moving-average structure, 3-month trend, distance from the 52-week high, and the direction of the 50-day average.
- Incomplete factor coverage receives a small confidence haircut after re-normalization.
- Goal: make 90+ scores rare and make the 65-85 range more informative.
