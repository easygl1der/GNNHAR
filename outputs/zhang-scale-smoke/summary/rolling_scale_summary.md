# Zhang-style Rolling Scale Summary

This summary reads only corrected rolling outputs from `gnnhar_iv_zhang_scale_pipeline.py`. Interpret scale gains only after checking `graph_audit.csv`; if a large universe has many fallback blocks, it is not a clean GLASSO scale test.

## Universe Summary

| universe | path | n_assets | n_dates | n_test_dates | n_blocks | fallback_blocks | best_qlike_model | best_test_qlike | best_mse_model | best_test_mse | best_gnn_qlike_model | best_gnn_iv_qlike_model | har_qlike | ghar_qlike | har_iv_qlike | ghar_iv_qlike |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sp100-smoke | outputs/zhang-scale-smoke/sp100 | 12 | 1234 | 22 | 1 | 0 | GHAR+IV | 0.0040152785368263 | GHAR | 4.971018314361572 | GNNHAR1L | GNNHAR1L-IV | 0.0040166075341403 | 0.0040197568014264 | 0.0040174201130867 | 0.0040152785368263 |
| sp500-smoke | outputs/zhang-scale-smoke/sp500 | 25 | 1234 | 22 | 1 | 0 | GHAR+IV | 0.0076115694828331 | GHAR+IV | 34.259124755859375 | GNNHAR1L |  | 0.0078843533992767 | 0.0078932298347353 | 0.0076213777065277 | 0.0076115694828331 |

## Graph Audit

| universe | n_assets | n_blocks | fallback_blocks | median_edges | median_density | methods |
| --- | --- | --- | --- | --- | --- | --- |
| sp100-smoke | 12 | 1 | 0 | 44.0 | 0.6666666666666666 | glasso_cv:1 |
| sp500-smoke | 25 | 1 | 0 | 152.0 | 0.5066666666666667 | glasso_cv:1 |

## Gains

| universe | n_assets | comparison | base_model | improved_model | mse_gain | qlike_gain |
| --- | --- | --- | --- | --- | --- | --- |
| sp100-smoke | 12 | Graph gain without IV: GHAR vs HAR | HAR | GHAR | 0.003295869937379181 | -0.0007840614895362474 |
| sp100-smoke | 12 | Graph gain with IV: GHAR+IV vs HAR+IV | HAR+IV | GHAR+IV | 0.005075437744265665 | 0.0005330725192078667 |
| sp100-smoke | 12 | IV gain in HAR: HAR+IV vs HAR | HAR | HAR+IV | -0.025217281896358967 | -0.00020230479067051554 |
| sp100-smoke | 12 | IV gain in GHAR: GHAR+IV vs GHAR | GHAR | GHAR+IV | -0.023386805213362116 | 0.001114063566858281 |
| sp100-smoke | 12 | Best non-IV GNN vs GHAR: GNNHAR1L | GHAR | GNNHAR1L | -14.789295347396797 | -12.001774485361045 |
| sp100-smoke | 12 | Best IV GNN vs GHAR+IV: GNNHAR1L-IV | GHAR+IV | GNNHAR1L-IV | -17.288031967642436 | -13.224612695841307 |
| sp100-smoke | 12 | IV gain in best GNN: GNNHAR1L-IV vs GNNHAR1L | GNNHAR1L | GNNHAR1L-IV | -0.18534299328887505 | -0.09283279671418154 |
| sp500-smoke | 25 | Graph gain without IV: GHAR vs HAR | HAR | GHAR | -0.0001563902397729855 | -0.001125829222649255 |
| sp500-smoke | 25 | Graph gain with IV: GHAR+IV vs HAR+IV | HAR+IV | GHAR+IV | 0.0005000380064498211 | 0.0012869357840905593 |
| sp500-smoke | 25 | IV gain in HAR: HAR+IV vs HAR | HAR | HAR+IV | 0.02483283826036753 | 0.03335412296119611 |
| sp500-smoke | 25 | IV gain in GHAR: GHAR+IV vs GHAR | GHAR | GHAR+IV | 0.02547286543611893 | 0.03568378950055562 |
| sp500-smoke | 25 | Best non-IV GNN vs GHAR: GNNHAR1L | GHAR | GNNHAR1L | -5.895079642921746 | -10.188752810965559 |
