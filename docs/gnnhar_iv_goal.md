# GNNHAR-IV Research Goal

## Objective

Build a reproducible, publication-style empirical analysis pipeline for extending Zhang et al. GNNHAR to GNNHAR-IV. The final analysis should follow Zhang et al.'s forecasting and statistical evaluation framework, while adding implied volatility (IV) as an exogenous information channel.

## Inputs

- Current repository branch: `2026-06-01`
- Colab code notebook: `https://colab.research.google.com/drive/1Lylnb5FcNL846MTrMKEmnoUU1SabHni7`
- Zhang et al. GNNHAR paper: `references/papers/zhang-forecasting-realized-volatility-gnn-ijf-2025.pdf`
- Colab GitHub + Drive workflow notebook: `notebooks/gnnhar_colab_github_drive_workflow.ipynb`
- Colab repo path: `/content/GNNHAR`
- Drive data path: `/content/drive/MyDrive/GNNHAR-colab-runs/data`
- Drive output path: `/content/drive/MyDrive/GNNHAR-colab-runs/outputs`

## Research Framing

The original Zhang et al. paper models realized volatility (RV) spillovers using HAR, GHAR, and GNNHAR. This project extends that framework by adding implied volatility (IV) to construct HAR-IV, GHAR-IV, and GNNHAR-IV models. The key question is whether IV adds genuine predictive information beyond the parameter expansion effect from adding more regressors/features.

## Required Methodological Alignment With Zhang et al.

Read the paper and reproduce the main evaluation logic:

- HAR, GHAR, GNNHAR1L, GNNHAR2L, GNNHAR3L
- GLASSO adjacency construction
- MSE and QLIKE training criteria
- distinction between estimation criterion and forecast loss
- out-of-sample MSE and QLIKE
- model loss ratios relative to HAR
- MCS test at 5 percent level
- DM tests for key pairwise comparisons
- calm versus volatile regime split
- forecast error and forecast ratio plots
- multi-hop and nonlinearity discussion
- robustness checks where feasible

## Required Model Comparisons

Compare at least these models:

- HAR
- GHAR
- GNNHAR1L, GNNHAR2L, GNNHAR3L
- HAR+IV
- GHAR+IV
- GNNHAR1L-IV, GNNHAR2L-IV, GNNHAR3L-IV
- HAR+fakeIV
- GHAR+fakeIV
- GNNHAR-IV+fakeIV

Use fake/pseudo IV to separate genuine IV information from parameter expansion.

## Required IV Contribution Analysis

For each relevant model family, report:

- no-IV baseline loss
- real-IV loss
- fake-IV loss
- total IV improvement
- genuine information gain
- parameter expansion gain

Explain whether GNNHAR-IV improves because IV contains real information, because the model gains more parameters, or both.

## Required Statistical Evaluation

Produce:

- Test MSE
- Test QLIKE
- loss ratios relative to HAR
- loss ratios relative to HAR+IV where relevant
- MCS test results
- DM tests for:
  - HAR versus GHAR
  - GHAR versus GNNHAR1L
  - GHAR+IV versus GNNHAR-IV
  - real IV versus fake IV
  - GNNHAR1L-IV versus GNNHAR2L-IV
- regime-stratified results using market-average RV or another defensible volatility proxy

## Required Outputs

Write all outputs to `/content/drive/MyDrive/GNNHAR-colab-runs/outputs`.

Produce:

- cleaned runnable Colab notebook
- reusable scripts/modules if appropriate
- result tables as CSV and LaTeX
- figures as PNG or PDF
- structured report draft in Markdown or LaTeX

Minimum tables:

- model loss table
- relative loss ratio table
- MCS result table
- real IV versus fake IV decomposition table
- regime-stratified result table

Minimum figures:

- GLASSO adjacency heatmap
- forecast error boxplots
- forecast ratio boxplots
- model comparison bar chart
- IV information decomposition chart
- optional DM-test significance plot

## Report Requirements

The report should be more than a notebook summary. It should read like an empirical finance forecasting analysis:

- explain Zhang et al.'s original GNNHAR setup
- explain how IV extends the original framework
- distinguish training loss from forecast loss
- include statistical significance, not only raw losses
- discuss nonlinear spillover, multi-hop behavior, and IV contribution
- discuss limitations:
  - Dow 30 is smaller than the original S&P 100 setting
  - IV data availability and quality
  - possible overparameterization
  - GNN instability under QLIKE
  - Colab runtime stochasticity

## Execution Requirements

Run on Colab, not local machine resources.

Start by cloning the branch:

```bash
git clone --depth 1 --branch 2026-06-01 https://github.com/easygl1der/GNNHAR.git /content/GNNHAR
```

Use Google Drive for persistent data and outputs.

## Success Criteria

The final result should not be just "models were run." It should show that GNNHAR-IV is a clear extension of Zhang et al., evaluated with Zhang-style forecasting losses, statistical tests, robustness checks, and publication-style tables/figures.
