# GNNHAR-IV Completion Audit

## Current State

- Repository branch: `2026-06-01`
- Full-run notebook commit used in Colab: `8dbd9d4`
- Full-run Colab URL:
  `https://colab.research.google.com/github/easygl1der/GNNHAR/blob/8dbd9d4/notebooks/gnnhar_iv_colab_full_run.ipynb`
- Drive output path:
  `/content/drive/MyDrive/GNNHAR-colab-runs/outputs`
- Local synced Drive path inspected:
  `/Users/yitwah/Library/CloudStorage/GoogleDrive-easyglider458@gmail.com/My Drive/GNNHAR-colab-runs/outputs`

## Execution Evidence

The full-run notebook was executed in Colab after cloning:

```bash
git clone --depth 1 --branch 2026-06-01 https://github.com/easygl1der/GNNHAR.git /content/GNNHAR
```

The pipeline was run with:

```bash
python /content/GNNHAR/scripts/analysis/gnnhar_iv_pipeline.py \
  --data-dir /content/GNNHAR/experiments/dow30/data \
  --output-dir /content/drive/MyDrive/GNNHAR-colab-runs/outputs \
  --epochs 250
```

`run_metadata.json` in Google Drive confirms:

- `epochs`: `250`
- `fast`: `false`
- `mcs_bootstrap`: `300`
- `output_dir`: `/content/drive/MyDrive/GNNHAR-colab-runs/outputs`
- `n_dates`: `1234`
- `split_sizes`: train `863`, validation `185`, test `186`

## Required Outputs Verified

Tables in Drive:

- `tables/model_losses.csv`
- `tables/model_losses.tex`
- `tables/loss_ratios.csv`
- `tables/loss_ratios.tex`
- `tables/mcs_results.csv`
- `tables/mcs_results.tex`
- `tables/dm_tests.csv`
- `tables/dm_tests.tex`
- `tables/iv_decomposition.csv`
- `tables/iv_decomposition.tex`
- `tables/regime_results.csv`
- `tables/regime_results.tex`

Figures in Drive:

- `figures/glasso_adjacency_heatmap.png`
- `figures/forecast_error_boxplot.png`
- `figures/forecast_ratio_boxplot.png`
- `figures/model_comparison_bar.png`
- `figures/iv_decomposition.png`

Report in Drive:

- `report/gnnhar_iv_report.md`

Reproducible notebooks:

- `notebooks/gnnhar_iv_colab_pipeline.ipynb`
- `notebooks/gnnhar_iv_colab_full_run.ipynb`

Reusable script:

- `scripts/analysis/gnnhar_iv_pipeline.py`

## Methodological Coverage

The generated `model_losses.csv` includes:

- HAR
- GHAR
- GNNHAR1L
- GNNHAR2L
- GNNHAR3L
- HAR+IV
- GHAR+IV
- GNNHAR1L-IV
- GNNHAR2L-IV
- GNNHAR3L-IV
- HAR+fakeIV
- GHAR+fakeIV
- GNNHAR1L-IV+fakeIV
- QLIKE-trained GNNHAR variants
- Random-adjacency GHAR robustness check

The pipeline implements:

- HAR daily, weekly, and monthly lag features.
- GLASSO adjacency from training-window returns.
- Linear HAR/GHAR MSE estimation.
- GNNHAR one-, two-, and three-hop graph neural models.
- MSE and QLIKE forecast losses.
- QLIKE-trained GNN variants for estimation-criterion comparison.
- MCS at the 5 percent level.
- DM tests for the required pairwise comparisons.
- IV versus fake-IV decomposition into total improvement, genuine information gain, and parameter expansion gain.
- Calm versus volatile regime split using market-average realized volatility.
- Forecast error and forecast ratio boxplots.

## Key Full-Run Result

The full run ranked `GHAR+IV` best by test QLIKE:

- `GHAR+IV`: test MSE `1.0272092819`, test QLIKE `0.0007849337`
- `HAR+IV`: test MSE `1.0464029312`, test QLIKE `0.0007928116`
- `GHAR`: test MSE `1.0848363638`, test QLIKE `0.0008110855`
- `HAR`: test MSE `1.1270402670`, test QLIKE `0.0008219654`

The IV decomposition table reports positive genuine IV information gains for HAR and GHAR, while the GNNHAR1L-IV comparison indicates instability and fake-IV/parameter-expansion effects. This is discussed in the generated report.

## Remaining Caveats

- The GNNHAR neural variants are empirically unstable on this Dow 30 sample; the report treats this as a result and limitation rather than hiding it.
- The full run uses the Dow 30 data available in the repository, not Zhang et al.'s larger S&P 100 universe.
- Google Colab runtime stochasticity remains a limitation even with fixed seeds.
