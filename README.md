# Conviction AI v0.5.2

A beginner-friendly Streamlit stock research app that turns ten commonly used research signals into one transparent 0–100 Conviction Score.

## What changed in v0.5.2

- Added **Chart Health** as the 10th factor.
- Chart Health uses price vs. 50-day SMA, price vs. 200-day SMA, 50/200-day trend, 3-month momentum, and distance from the 52-week high.
- Leaderboards now require **at least 7/10 factors**.
- Added beginner-facing homepage copy: **Stock research made simple** and **10 factors. One simple Conviction Score.**
- Quarterly movers/fallers remain fixed to snapshots taken Jan 1, Apr 1, Jul 1, and Oct 1.

## Files you need in GitHub

- `app.py`
- `quarterly_snapshot.py`
- `quarterly_scores.csv`
- `requirements.txt`
- `.github/workflows/quarterly-snapshot.yml`

The other Supabase files from older versions are not required for this no-database version.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data caveat

The MVP uses Yahoo Finance through `yfinance`. Some fields can be missing or rate-limited. Stocks with fewer than 7 of the 10 factors are excluded from leaderboards rather than receiving a misleading low score.
