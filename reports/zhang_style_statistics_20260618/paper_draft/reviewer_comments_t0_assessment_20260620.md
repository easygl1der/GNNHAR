# Reviewer Comments and T0 Assessment

Paper reviewed: `main.pdf` / `main.tex` in `reports/zhang_style_statistics_20260618/paper_draft`  
Review date: 2026-06-20  
Perspective: financial econometrics / forecasting reviewer, with attention to whether the draft can plausibly become a top-tier financial forecasting paper.

## Overall Verdict

The draft is now a coherent working paper with a defensible empirical narrative:

> Option-implied volatility is the most stable added information channel across Dow30, S&P100, and S&P500, while larger graph universes do not mechanically improve GNNHAR gains.

However, in its current form it is **not yet T0-ready**. It is closer to a promising empirical project paper or early journal draft. The key reason is not writing quality alone. The main issue is that several central claims still depend on incomplete robustness, a non-common-calendar S&P500 comparison, an unresolved S&P500 GNN implementation failure, and a target proxy that differs materially from high-frequency realized volatility.

The strongest current contribution is the IV result. The weakest current contribution is the broad GNN/large-graph claim, because the S&P500 GNN evidence may be partly an implementation artifact.

## What Is Already Strong

1. **The paper now has a clear empirical message.**  
   The draft no longer overclaims that GNNHAR dominates. It correctly says that IV-augmented models are the stable winners and that larger graphs do not automatically improve forecasting.

2. **Dow30 MCS is now in the correct 234-date evidence path.**  
   The old 232-date Dow30 run has been separated into `.trash`, and the current Dow30 aligned run uses 234 dates from 2025-07-07 to 2026-06-09. This fixes an earlier evidence-contamination problem.

3. **The best-loss pattern is plausible and interpretable.**  
   Current saved results show:

   | Universe | Best MSE model | MSE gain vs HAR_M | Best QLIKE model | QLIKE gain vs HAR_M |
   |---|---:|---:|---:|---:|
   | Dow30 | HAR_M_IV | 4.1% | GNNHAR2L_Q_IV | 7.7% |
   | S&P100 | HAR_M_IV | 3.5% | HAR_Q_IV | 6.7% |
   | S&P500 | HAR_M_IV | 3.4% | GHAR_M_IV | 2.7% |

   This supports a conservative but interesting claim: IV matters more robustly than graph depth.

4. **The draft is transparent about limitations.**  
   It explicitly says the current RV target is a 30-day close-to-close historical-volatility proxy, not the intraday RV used by Zhang-style benchmark papers. This is important and should remain visible.

5. **The appendix is moving in the right direction.**  
   The paper now has data audit, MCS, DM/depth, FVU, smoothing, S&P500 diagnostics, and reference-map appendices. That is much better than a short appendix with only placeholder claims.

## Major Concerns

### 1. The Target Variable Is Not Strong Enough for the Current Framing

The paper repeatedly uses “realized volatility” language, but the target is a 30-day close-to-close historical-volatility proxy converted to daily variance scale. This is not equivalent to high-frequency realized volatility.

For a top-tier financial forecasting paper, this matters because QLIKE and HAR literature usually expects a carefully justified volatility proxy. The paper cites realized-volatility work, but the empirical target is closer to a smoothed historical volatility measure. The current wording is honest in the notation section, but the abstract/introduction still risks sounding like a standard RV forecasting paper.

**Required fix:** make the title, abstract, and data section more precise. Possible framing:

> We forecast a daily variance-scale 30-day historical-volatility proxy, and use Zhang-style HAR/GHAR/GNNHAR architecture as the forecasting framework.

If the paper wants to compete as a true realized-volatility forecasting paper, it should add a high-frequency RV target or a much stronger explanation for why the current proxy is acceptable.

### 2. The S&P500 Calendar Mismatch Weakens RQ3

RQ3 asks whether graph gains increase as the universe expands from Dow30 to S&P100 to S&P500. But Dow30 and S&P100 use 234 test dates, while S&P500 currently uses 223 test dates.

The paper correctly labels this as a near-calendar comparison. Still, for a T0 submission, that caveat is too central. The main cross-universe conclusion should not depend on a partially aligned design.

**Required fix:** rerun S&P500 on the Dow30/S&P100 calendar, or move RQ3 from a primary research question to a clearly exploratory result.

### 3. The S&P500 GNN Failure Cannot Yet Be Treated as a Model Finding

The current draft says S&P500 GNNHAR strongly over-predicts low-volatility observations, possibly because a ReLU constraint is applied on standardized target scale. That is a plausible diagnostic, but it means the S&P500 GNN result may reflect a coding/design choice rather than an economic failure of nonlinear graph aggregation.

For a serious forecasting paper, a model-family conclusion cannot rest on a suspected output-layer artifact.

**Required fix:** run corrected-output GNNHAR:

- hidden-layer ReLU retained;
- final layer unconstrained; or
- log-volatility target with exponential inverse transform;
- same train/validation/test split;
- same graph and feature construction;
- saved predictions, hidden states, and diagnostics.

Only after that can the paper claim that GNNHAR fails or does not scale in S&P500.

### 4. Robustness Section Is Mostly a To-Do List

The robustness section is admirably honest, but many entries are marked “not yet run”:

- alternative validation split;
- alternative graph construction;
- IV interaction effects;
- weekly and monthly forecast horizons;
- corrected-output GNNHAR;
- exact hidden-state MAD.

For a T0 paper, this is not acceptable as final evidence. A reviewer will read this as “the authors know what they still need to do.”

**Required fix:** convert the robustness section into a completed robustness matrix. At minimum:

| Robustness item | Dow30 | S&P100 | S&P500 | Required for submission |
|---|---:|---:|---:|---:|
| Common calendar | done | done | missing | yes |
| Corrected-output GNN | missing | optional | missing | yes |
| Alternative validation window | missing | missing | optional | yes |
| IV interaction | missing | missing | missing | yes |
| Alternative graph | missing | missing | optional | yes |
| Horizon \(h=5,22\) | missing | missing | missing | yes |

### 5. Benchmark Set Is Too Narrow

The paper currently compares HAR, GHAR, GNNHAR, and IV variants. That is internally consistent, but top-tier forecasting papers usually expect stronger external baselines.

Potential missing benchmarks:

- HAR-RV and HAR-IV variants with standard OLS/rolling regression details;
- HEAVY or HARQ-style volatility models;
- GARCH/EGARCH/GJR-GARCH or realized-GARCH baselines;
- tree boosting or random forest baselines using the same lagged RV/IV features;
- simple IV-only and RV+IV linear models;
- equal-weight or persistence benchmarks.

The current result that HAR_M_IV wins MSE in all three universes is actually a warning: the paper may be more about the power of IV features than graph neural networks. Stronger non-graph baselines are needed to prove that the graph machinery adds something beyond a good IV regression.

### 6. The IV Mechanism Is Not Yet Identified

The paper says IV contains forward-looking volatility information. That is economically plausible, but the current evidence mainly shows that adding IV improves loss.

Alternative explanations remain:

- IV is just a smoothed volatility proxy;
- IV embeds moneyness/liquidity/option-market microstructure effects;
- the IV feature absorbs scale differences across stocks;
- the IV gain is concentrated in a few event stocks or high-volatility dates;
- IV improves QLIKE but not necessarily economically meaningful risk decisions.

**Required mechanism tests:**

- IV-only benchmark;
- lagged-IV placebo, e.g. \(IV_{t-22}\);
- lead-IV falsification check if available only for diagnostic purposes, not for forecasting;
- \(IV \times RV\) interaction;
- event-date split, such as earnings or macro-announcement windows;
- cross-sectional IV coverage/liquidity controls;
- per-sector IV contribution.

### 7. Statistical Inference Needs More Panel-Aware Treatment

MCS and DM are useful, but the draft should explain exactly what the loss sequence is:

- averaged across tickers per date?
- pooled over stock-date observations?
- block bootstrap over dates only?
- any cross-sectional dependence correction?

Given the panel structure, a reviewer may object if inference treats thousands of stock-date observations as independent. The current MCS reports \(n_{\text{dates}}=234\) or 223, which suggests date-level aggregation. That is good, but the paper should state this explicitly and justify the block size.

**Required fix:** add a short inference-design paragraph:

> Losses are first averaged across stocks within each out-of-sample date; MCS and DM are then run on the date-level loss-differential series, with block bootstrap over dates.

If that is not exactly what the code does, the code and paper should be aligned.

### 8. Data Construction and Survivorship Bias Need More Detail

The S&P500 run uses 449 model tickers. The draft mentions that this is the formal model output, but a top-tier reviewer will ask:

- Which S&P500 constituents were used?
- Are constituents fixed as of a date, or dynamically selected?
- What tickers were dropped and why?
- Does survivorship bias enter because only current constituents are used?
- How is option IV missingness handled?
- Are delisted/merged firms excluded?

**Required fix:** add a constituent and missingness audit table:

| Universe | Raw tickers | Model tickers | Main drop reason | Date range | IV coverage rule |
|---|---:|---:|---|---|---|

The appendix should also include a ticker-drop file or table, not just a high-level statement.

### 9. The Paper Needs Stronger Economic Evaluation

Forecasting papers at top venues usually need more than statistical loss. The current paper has MSE, QLIKE, MCS, DM, FVU, and diagnostics, but it does not yet show economic value.

Possible additions:

- volatility-targeting portfolio backtest;
- minimum-variance portfolio allocation;
- VaR or expected shortfall backtest;
- option hedging or variance-risk management application;
- turnover and transaction-cost sensitivity.

If the paper remains purely statistical, it should target financial econometrics / forecasting methodology rather than claiming broad investment usefulness.

### 10. The Current Novelty Claim Needs Sharpening

The draft’s real novelty is not “GNNHAR plus IV across larger universes” by itself. A skeptical reviewer may say that adding IV to HAR is known, and graph neural volatility models are known.

A stronger novelty statement is:

> We show that when a Zhang-style graph-HAR/GNNHAR framework is extended with option-implied volatility and scaled from Dow30 to S&P500, the robust gain comes from IV rather than graph depth; larger graph universes introduce estimation and optimization fragility rather than monotone improvement.

This is more distinctive and better supported by current evidence.

## Section-Level Comments

### Abstract

The abstract is honest but still too defensive. It says “current evidence” and “draft” too much for a paper abstract. For the working draft this is fine; for submission it should be rewritten after robustness is complete.

Suggested change after completing robustness:

> We find that IV-augmented models consistently enter the superior model set across all universes, whereas deeper graph neural HAR variants do not deliver monotone gains as the graph expands.

### Introduction

The RQs are clear. RQ3 is interesting, but it currently overcommits because S&P500 is not calendar-aligned and the GNN implementation issue remains unresolved.

Recommendation: keep RQ1 and RQ2 as primary; make RQ3 conditional/exploratory until S&P500 is rerun.

### Methodology

The model equations are useful, but the draft should add:

- exact rolling estimation window;
- validation window;
- graph re-estimation frequency;
- training objective for MSE-trained versus QLIKE-trained models;
- optimizer and early stopping details for GNNHAR;
- whether forecasts are clipped or constrained positive.

The output positivity issue is central because QLIKE requires positive forecasts.

### Empirical Results

The main tables are useful. However, T0 readers need clearer separation between:

- descriptive mean-loss ranking;
- MCS membership;
- pairwise DM evidence;
- diagnostic failure cases.

Recommendation: add a compact “main finding table” in the main text and move more implementation details to appendix.

### Statistical Analysis

The SATS/EchoStar discussion is valuable. It shows the model is stress-tested by real event observations. But it should not become too anecdotal. Add a table showing how much of total S&P500 SSE/QLIKE is explained by the top 1, 5, and 10 stock-date observations.

### Robustness

This is the weakest section for submission because it is mostly prospective. It is fine for an internal draft, but not for a final manuscript.

### Conclusion

The conclusion answers the three RQs well, but it should avoid final-sounding claims on RQ3 until S&P500 is fully aligned and corrected-output GNNHAR is run.

### Appendix

The appendix is better than before, but still reads partly like engineering notes. It should become more table-driven and audit-like:

- full model list and naming convention;
- raw-to-model ticker audit;
- data-source table;
- missingness table;
- reproducibility command table;
- MCS settings and code path;
- all robustness status in one matrix;
- Colab/AutoDL reproduction path.

## Decision Recommendation

If this were submitted now to a top finance/forecasting venue, my likely recommendation would be:

**Reject / major revision**, mainly because the core empirical design is not yet fully aligned and several robustness checks are still planned rather than completed.

If the required reruns and robustness checks are completed, the paper could become:

**A credible financial forecasting paper with a strong empirical message**, especially if it is positioned as evidence that option-implied information dominates graph-depth complexity in large-universe volatility forecasting.

To become T0-level, it needs:

1. a fully common-calendar design;
2. corrected-output GNNHAR;
3. stronger non-graph and econometric baselines;
4. completed robustness matrix;
5. clearer IV mechanism evidence;
6. panel-aware statistical inference explanation;
7. economic-value evaluation;
8. a cleaner appendix and reproducibility package.

## Priority Action List

### Must Fix Before Any Serious Submission

1. Rerun S&P500 on the Dow30/S&P100 calendar.
2. Rerun corrected-output GNNHAR for S&P500.
3. Recompute Dow30 regime diagnostics on the 234-date aligned run.
4. Add IV-only, RV-only, additive IV, and IV-interaction baselines.
5. Add at least one alternative validation split.
6. Add at least one alternative graph construction.
7. Make the inference design explicit: date-level loss aggregation, block bootstrap, MCS/DM settings.
8. Add a ticker/drop/missingness audit table.

### Strongly Recommended for T0 Ambition

1. Add HEAVY/GARCH/realized-GARCH or strong ML baselines.
2. Add economic evaluation such as volatility targeting or VaR backtesting.
3. Add weekly and monthly horizons.
4. Save hidden states and reproduce exact MAD/over-smoothing diagnostics.
5. Add sector-level and event-date heterogeneity.
6. Turn the appendix into a clean reproducibility appendix rather than a project log.

### Writing and Framing Improvements

1. Avoid overusing “current evidence” in the final submission version.
2. Replace “realized volatility” with “30-day historical-volatility proxy” wherever ambiguity could hurt.
3. State clearly that the paper is not a numerical replication of Zhang et al.
4. Reframe the contribution around IV dominance and graph-scaling fragility.
5. Keep the title, abstract, and conclusion aligned with the actual strongest evidence.

## Bottom Line

The paper has a real idea and the current results are not trivial. The best supported claim is:

> Across three equity universes, IV-augmented volatility forecasting models are consistently competitive, while graph neural depth does not deliver monotone gains as the universe expands.

That is publishable if backed by complete robustness and cleaner identification. It is not yet enough for T0 because the current S&P500 result mixes an economically interesting scaling question with an unresolved implementation diagnostic.
