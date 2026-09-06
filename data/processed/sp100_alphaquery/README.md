# S&P 100 AlphaQuery Short-Window Dataset

Generated from AlphaQuery chart JSON discovered through BrowserOS page inspection, plus Yahoo adjusted-close daily returns for graph construction.

Important limitation: unauthenticated AlphaQuery currently exposes only the free recent chart window, not the 2021-2026 history used in the Dow 30 manuscript. The local files here cover 2026-03-10 to 2026-06-09.

## Files

- `merged_iv_data_filled.csv`: AlphaQuery 30-Day Implied Volatility Mean (`iv-mean`), shape (64, 102).
- `merged_rv_data_filled.csv`: AlphaQuery 30-Day Historical Volatility close-to-close (`historical-volatility`), shape (64, 102).
- `daily_returns.csv`: Yahoo adjusted-close daily returns, shape (64, 102).
- `raw_alphaquery_json/`: per-ticker raw chart JSON cache for resume/rebuild.
- `coverage_summary.csv`: non-null counts by ticker.
- `tickers_model_usable.txt`: 100 tickers with non-empty IV, RV, and return data.
- `tickers_complete_coverage.txt`: 96 tickers with full IV/RV coverage and full returns coverage in this window.

## Coverage Notes

Strict complete-coverage tickers: 96.
Model-usable tickers after excluding all-empty series: 100.
Known issue: `BNY` has no IV values and only partial historical-volatility values in this AlphaQuery short window.

For a publication-scale run matching the Dow 30 manuscript, log into AlphaQuery/VolVue subscription access or provide exported historical CSV/API access, then rerun `scripts/data/fetch_alphaquery_option_stats.py` with the same ticker list.
