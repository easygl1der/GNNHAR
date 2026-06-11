# Zhang-style Rolling Scale Summary

This summary reads only corrected rolling outputs from `gnnhar_iv_zhang_scale_pipeline.py`. Interpret scale gains only after checking `graph_audit.csv`; if a large universe has many fallback blocks, it is not a clean GLASSO scale test.

## Universe Summary

| universe | path | n_assets | n_dates | n_test_dates | n_blocks | fallback_blocks | best_qlike_model | best_test_qlike | best_mse_model | best_test_mse | best_gnn_qlike_model | best_gnn_iv_qlike_model | har_qlike | ghar_qlike | har_iv_qlike | ghar_iv_qlike |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sp100 | outputs/zhang-scale-local/scale-zhang-roll-multihop-20260611T042850Z/full/sp100 | 99 | 1073 | 44 | 2 | 0 | GHAR+IV | 0.0027252461295574 | HAR+IV | 6.107729434967041 | GNNHAR3L | GNNHAR2L-IV | 0.0027713687159121 | 0.0027774714399129 | 0.0027261369396001 | 0.0027252461295574 |
| sp500 | outputs/zhang-scale-local/scale-zhang-roll-multihop-20260611T042850Z/full/sp500 | 449 | 1223 | 44 | 2 | 0 | GHAR+IV | 0.0054479339160025 | HAR+IV | 16.45408058166504 | GNNHAR3L | GNNHAR2L-IV | 0.0057699433527886 | 0.0057499096728861 | 0.0054527665488421 | 0.0054479339160025 |

## Graph Audit

| universe | n_assets | n_blocks | fallback_blocks | median_edges | median_density | methods |
| --- | --- | --- | --- | --- | --- | --- |
| sp100 | 99 | 2 | 0 | 881.5 | 0.1817151102865388 | glasso_cv:2 |
| sp500 | 449 | 2 | 0 | 2558.5 | 0.0254384743875278 | glasso_cv:2 |

## Gains

| universe | n_assets | comparison | base_model | improved_model | mse_gain | qlike_gain |
| --- | --- | --- | --- | --- | --- | --- |
| sp100 | 99 | Graph gain without IV: GHAR vs HAR | HAR | GHAR | -0.001338659690196975 | -0.00220206137341461 |
| sp100 | 99 | Graph gain with IV: GHAR+IV vs HAR+IV | HAR+IV | GHAR+IV | -0.0002469388907049286 | 0.0003267664326614961 |
| sp100 | 99 | IV gain in HAR: HAR+IV vs HAR | HAR | HAR+IV | 0.03311865511426659 | 0.01632109652255842 |
| sp100 | 99 | IV gain in GHAR: GHAR+IV vs GHAR | GHAR | GHAR+IV | 0.03417280843655934 | 0.01880318537393766 |
| sp100 | 99 | Best non-IV GNN vs GHAR: GNNHAR3L | GHAR | GNNHAR3L | -4.753958764485994 | -7.645739943394142 |
| sp100 | 99 | Best IV GNN vs GHAR+IV: GNNHAR2L-IV | GHAR+IV | GNNHAR2L-IV | -4.778176709334998 | -7.712142144922666 |
| sp100 | 99 | IV gain in best GNN: GNNHAR2L-IV vs GNNHAR3L | GNNHAR3L | GNNHAR2L-IV | 0.030107720274419658 | 0.011267262589939708 |
| sp500 | 449 | Graph gain without IV: GHAR vs HAR | HAR | GHAR | -0.0009031660090095261 | 0.003472075664801344 |
| sp500 | 449 | Graph gain with IV: GHAR+IV vs HAR+IV | HAR+IV | GHAR+IV | -0.0010906864833353236 | 0.000886271729462984 |
| sp500 | 449 | IV gain in HAR: HAR+IV vs HAR | HAR | HAR+IV | 0.040359473198028684 | 0.054970523028308294 |
| sp500 | 449 | IV gain in GHAR: GHAR+IV vs GHAR | GHAR | GHAR+IV | 0.04017968333135691 | 0.05251834795033006 |
| sp500 | 449 | Best non-IV GNN vs GHAR: GNNHAR3L | GHAR | GNNHAR3L | -5.118077395467484 | -9.783771765046835 |
| sp500 | 449 | Best IV GNN vs GHAR+IV: GNNHAR2L-IV | GHAR+IV | GNNHAR2L-IV | -5.330323713774967 | -10.343172684223735 |
| sp500 | 449 | IV gain in best GNN: GNNHAR2L-IV vs GNNHAR3L | GNNHAR3L | GNNHAR2L-IV | 0.006881914231449016 | 0.0033683734694363032 |
