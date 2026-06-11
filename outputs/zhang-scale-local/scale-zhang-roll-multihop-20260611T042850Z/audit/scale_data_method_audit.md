# S&P Scale Experiment Data and Method Audit

## Executive Finding

The S&P 100 and S&P 500 RV, IV, and return panels are paired on the same daily date range and ticker namespace. The current preliminary scale run should not be used as a final GLASSO scale conclusion, however, because the S&P 500 graph construction fell back from GLASSO to an absolute-correlation graph and the model implementation used a static split rather than Zhang's rolling-window design.

## Data Pairing

| Universe | RV shape | IV shape | Returns shape | Common dates | Common tickers | Return fetch failures |
|---|---:|---:|---:|---:|---:|---:|
| sp100 | [1256, 101] | [1256, 101] | [1256, 101] | 1256 | 101 | 0 |
| sp500 | [1256, 503] | [1256, 503] | [1256, 503] | 1256 | 503 | 0 |

The panels use the date range shared by realized volatility, implied volatility, and returns. Non-positive RV/IV values were already removed from the processed panels. Remaining missing values are concentrated in newly added or reorganized tickers and are handled by coverage filtering, not by filling pre-listing history.

## Old Run Status

| Universe | Selected tickers | Old graph method | Edges | Density | Fallback |
|---|---:|---|---:|---:|---|
| sp100 | 99 | glasso | 1083 | 0.22325293753865183 |  |
| sp500 | 449 | glasso_fallback_corr | 6928 | 0.06888323258033725 | FloatingPointError: Non SPD result: the system is too ill-conditioned for this solver. The system is too ill-conditioned for this solver; used absolute correlation graph |

## Method Gap Against Zhang's Original Code

Zhang's public implementation recomputes the graph for each forecast block. For a forecast origin \(t\), it uses up to 1000 prior return observations, reserves the latest 22 observations as validation for neural models, and forecasts the next 22 trading days. The preliminary scale script instead used one chronological split and one static graph. It also used a custom concatenated message-passing neural architecture rather than Zhang's residual \(H_1 + \mathrm{GCN}\) architecture.

Therefore, the old S&P 500 result is best labeled as a preliminary static-graph screen. The next run should be a rolling Zhang-style experiment that logs graph success per block and treats IV as an explicit extension.
