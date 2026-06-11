# Zhang-style Rolling Scale Experiment: sp100-smoke

## Design

This run uses Zhang's rolling forecast structure rather than the earlier static split. For each forecast origin, the graph is recomputed from pre-origin returns over a window of at most 1000 observations; neural models use the last 22 pre-origin observations as validation and forecast the next 22 trading days. Linear HAR/GHAR baselines use `LinearRegression` and the same non-positive forecast replacement rule used in Zhang's public `GHAR.py`.

The IV models are an extension: IV HAR lags are appended to the node features, and GHAR+IV also includes graph-aggregated IV lags. These rows should be interpreted as the user's research extension, not as part of Zhang's original baseline.

## Data

- Assets selected: 12
- Valid aligned dates after HAR lags: 2021-07-12 to 2026-06-09
- Source common date range: 2021-06-09 to 2026-06-09
- Coverage threshold: 0.950
- Rolling blocks: 1
- Forecasted dates: 22

## Graph Audit

- Requested graph method: glasso_cv
- Blocks with graph fallback: 0 / 1
- Median graph density: 0.666667
- Median graph edges: 44.0

## Result

The best model by rolling test QLIKE is `GHAR+IV` with QLIKE 0.00401528 and MSE 5.08727.

The complete ranking is in `tables/model_losses.csv`. The graph log is in `tables/graph_blocks.csv`, which is the first place to check before interpreting S&P 500 scale effects.

## IV Decomposition

IV decomposition is unavailable because the corresponding fake-IV controls were not run.

## Interpretation Rule

If S&P 500 still has graph fallback blocks, do not describe its graph result as a clean GLASSO scale result. If the graph succeeds but gains shrink, the likely explanations to test next are graph density/neighbor caps, sector heterogeneity, noisy IV coverage, and GNN hyperparameter capacity rather than data-pairing failure.
