# Conviction AI v0.4

A Streamlit stock-research MVP that scores companies on a transparent 0–100 Conviction Score, ranks a configurable stock universe, and tracks quarter-over-quarter score movers using persistent Supabase/Postgres storage.

## Features

- Live ticker analysis using public Yahoo Finance data through `yfinance`
- 9-factor weighted Conviction Score
- Current Top 10 leaderboard
- Top 10 QoQ score improvers
- Top 10 QoQ score decliners
- Persistent quarterly snapshots in Supabase/Postgres
- Automatic local CSV fallback when Supabase is not configured

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Without Supabase credentials the app still works and stores quarterly snapshots in `quarterly_scores.csv`.

## Set up Supabase

1. Create a Supabase project.
2. Open the SQL editor and run `supabase_schema.sql`.
3. Add the project URL and a server-side Supabase key to your Streamlit secrets. Copy `.streamlit_secrets.toml.example` as a template.
4. Never commit real keys to GitHub.

Local `.streamlit/secrets.toml` example:

```toml
SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
SUPABASE_KEY = "YOUR_SERVER_SIDE_KEY"
```

On Streamlit Community Cloud, put those same entries in the app's **Secrets** settings instead of committing the file.

## Database behavior

`quarterly_scores` uses `(quarter, ticker)` as its primary key. Saving a quarter uses an upsert, so refreshing the same quarter updates each ticker instead of duplicating it. The next quarter creates a new row and enables QoQ score-change rankings.

## Security note

The included schema enables Row Level Security and intentionally does not create public anonymous policies. Keep the database key server-side in Streamlit Secrets. Before adding accounts or exposing direct browser/database access, add proper user-scoped RLS policies.

## Deploy

Push these files to GitHub, create a Streamlit Community Cloud app with `app.py` as the entrypoint, and add the two Supabase secrets in the deployment settings.

## Data caveat

`yfinance` is appropriate for prototyping, but a commercial paid product should move to a licensed/reliable market-data provider before launch.

## v0.5 automatic market universe

The leaderboard no longer requires a manually maintained ticker list. Choose one of:

- S&P 500
- Nasdaq-100
- S&P 500 + Nasdaq-100 (deduplicated)

Index membership is refreshed from public constituent tables and cached for 24 hours. Ticker symbols are normalized for Yahoo Finance (for example, `BRK.B` becomes `BRK-B`). The scoring pass uses a bounded thread pool to make broad-universe refreshes substantially faster than the earlier serial scan.

### Important production note

A 500+ stock scan can still encounter Yahoo/yfinance throttling because the MVP pulls several fields per company. The app skips failed names and shows data coverage. For a paid production product, replace the public-data layer with a licensed bulk fundamentals/estimates provider and run scheduled universe scoring server-side rather than on-demand in the Streamlit request.
