# S&P 100 VolVue / AlphaQuery Download Check

Checked on 2026-06-10 using BrowserOS MCP against the logged-in browser session.

## Result

The current VolVue login is active but not subscribed for full S&P 100 historical
exports. In browser page state, `localStorage.user` was:

```json
{"status":1}
```

With this session:

- `AAPL` 30-day implied volatility mean returned 1,256 rows, from 2021-06-09 to
  2026-06-09.
- `AAPL` 30-day close-to-close historical volatility returned 1,256 rows, from
  2021-06-09 to 2026-06-09.
- `META` 30-day implied volatility mean returned 65 rows, from 2026-03-09 to
  2026-06-09.
- `META` 30-day close-to-close historical volatility returned 65 rows, from
  2026-03-09 to 2026-06-09.

This matches VolVue's page behavior: Dow 30 tickers can show five years under
the free exception, while non-Dow S&P 100 tickers have disabled `6M`, `1Y`,
`2Y`, and `5Y` chart buttons.

## Existing Downloaded Data

The project already contains the complete currently accessible S&P 100 short
window data:

- `../sp100_alphaquery/merged_iv_data_filled.csv`
- `../sp100_alphaquery/merged_rv_data_filled.csv`
- `../sp100_alphaquery/raw_alphaquery_json/`

Those files cover 101 S&P 100 ticker columns from 2026-03-10 to 2026-06-09
using AlphaQuery's 30-day implied volatility mean and 30-day close-to-close
historical volatility.

## Blocker For The Requested Five-Year S&P 100 Dataset

The requested "every S&P 100 stock, five years, realized volatility and implied
volatility" dataset requires a VolVue paid plan with broader history access.
The current logged-in status only unlocks five-year chart data for the Dow 30
exception set, not the full S&P 100 universe.
