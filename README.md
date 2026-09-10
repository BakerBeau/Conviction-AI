# Conviction AI v0.7.7

Beginner-friendly stock, ETF, and DCA research app built with Streamlit.

## v0.7.7 beginner-discovery polish

- **Quality Gem** and **Turnaround Gem** now use visually distinct badges and clearer explanatory copy.
- **Why Emerging?** now selects the strongest two signal-specific reasons (estimate revisions, relative-strength breakout, revenue/EPS strength, chart health, QoQ score change, or supportive valuation) instead of repeating the same generic phrase.
- Generic `N/A` labels on the main discovery screens are replaced with plain-English reasons such as **Forward EPS declining**, **Growth estimate unavailable**, **P/E data unavailable**, **No prior quarter**, and **Not enough history**.
- Added subtle green/amber/red/gray table cues for positive, mixed, negative, and unavailable trend labels.
- Hidden Gem cards also show a compact status line for estimate revisions and value-vs-growth condition.

## What is new in v0.7.7

- Rebuilt **Hidden Gems** to reduce sector bias and stop financial/asset-management names from dominating discovery.
- Hidden Gem quality is now partly **sector-relative**, so a company is compared with other companies in its own sector rather than only against universal thresholds.
- Added **EPS Estimate Revisions** using yfinance EPS trend/revision data. The Hidden Gem card now shows whether estimates are rising, stable, or falling versus roughly 90 days ago.
- Added **Value vs Growth**, a forward-P/E-versus-forward-EPS-growth score. This replaces raw analyst target upside on the Hidden Gem card.
- Hidden Gem scoring now emphasizes: 25% quality, 20% estimate revisions, 20% value vs growth, 15% chart health, 10% institutional support, and 10% underfollowedness. Missing data is reweighted, but at least four Hidden Gem evidence groups must be available.
- Hidden Gems now require a **65+ Hidden Gem Score** after the quality screen.
- The discovery pool is limited to **two stocks per sector** before random selection, improving sector diversity.
- Analyst price-target upside is no longer a Hidden Gem gate or headline metric.
- Estimate-revision display is capped into beginner-friendly labels when extreme base effects would otherwise create distracting percentages.

## Data note

Estimate revisions use yfinance's EPS trend and EPS revision datasets when available. Forward growth uses yfinance earnings estimates or growth estimates when available, with a fallback to reported EPS growth. Free market-data feeds can be incomplete or delayed, so missing signals are handled conservatively.

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


## v0.7.3 scoring calibration
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


## v0.7.3 Emerging Leaders recalibration
- Excludes the current Top 20 Conviction stocks so Emerging Leaders cannot duplicate Market Leaders.
- Focuses on acceleration rather than absolute score.
- Uses recent momentum pace vs 12-month pace, QoQ Conviction change when available, current growth, chart health, and light valuation/institutional confirmation.
- Keeps candidates mostly in the 58-82 Conviction band with at least 7/10 factor coverage.


## v0.7.3 Emerging Leaders recalibration
- Caps EPS/revenue growth inputs to reduce one-time/base-effect distortions.
- Requires at least two independent improvement signals.
- Separates EPS Growth and Revenue Growth in the table.
- Recalibrates Emerging Scores so 90+ is rare.
- Renames Recent Pace to Momentum Accel for clearer interpretation.


## v0.7.6 Emerging Leaders display cleanup
- Raw extreme EPS/revenue spikes are hidden from the leaderboard and replaced with beginner-friendly trend labels.
- Momentum acceleration is now relative to the S&P 500 and uses a smooth scoring curve instead of piling up at a hard +25 cap.
- Added a concise **Why Emerging?** explanation for every candidate.
- Preserves the v0.7.4 Hidden Gems sector-relative/revisions/value-vs-growth rebuild.


## v0.7.6 discovery cleanup
- Market Leaders no longer displays the unreliable 1Y return column; it shows EPS Trend, Value vs Growth, and Chart Health instead.
- Emerging Leaders shows qualitative momentum-acceleration labels instead of capped +20-point values.
- Hidden Gems now require a 65+ Hidden Gem Score and reject materially falling EPS estimate revisions (< -5%).
- Hidden Gems are labeled **Quality Gem** or **Turnaround Gem** so positive-growth compounders are not mixed with improving-but-still-declining EPS stories.
- Value vs Growth now explains when it is not applicable because forward EPS is declining.
