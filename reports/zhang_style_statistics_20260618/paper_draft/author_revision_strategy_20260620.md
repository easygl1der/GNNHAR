# Author-Side Revision Strategy

This note converts the reviewer-style concerns into manuscript strategy. The goal is not to chase every possible T0 objection immediately, but to keep the paper's center of gravity clear:

> The paper is mainly about whether Zhang-style GNNHAR depth and graph scaling remain useful once option-implied volatility is added across equity universes. The current evidence says IV is robust, while graph neural depth is fragile, especially in the large-universe setting.

## 1. Data Target: Do Not Apologize Too Much, Reframe Precisely

The paper should not pretend to forecast high-frequency realized volatility. It forecasts a daily variance-scale 30-day close-to-close historical-volatility proxy. That must be explicit.

But this should not be framed only as a weakness. The stronger author-side framing is:

- The 30-day close-to-close HV target is smoother and more persistent than intraday RV.
- That makes HAR and IV baselines strong and harder to beat.
- If GNNHAR is unstable even on this smoother proxy, then moving to noisier intraday RV is not automatically a solution.
- The result is therefore relevant to the stability of graph neural volatility forecasting, not only to the choice of target.

Suggested text:

> Our target is not high-frequency intraday realized volatility. We forecast a daily variance-scale 30-day close-to-close historical-volatility proxy. This smoother target is deliberately useful for evaluating model stability: it reduces microstructure noise and extreme intraday irregularity, yet the large-universe GNNHAR rows still display substantial instability. Thus the evidence should be read as a Zhang-style proxy-volatility extension, not as a direct numerical replication of intraday RV forecasting.

## 2. Why 30-Day HV Can Be Defensible

The 30-day proxy has mechanical overlap with monthly HAR features and IV30. A reviewer can attack this. The response should be honest:

- yes, the overlap exists;
- yes, it weakens causal claims;
- but this is not a causal IV paper;
- the purpose is out-of-sample predictive comparison under a realistic option-volatility horizon;
- IV30 and HV30 are naturally matched maturity/horizon objects.

Suggested text:

> The 30-day horizon also aligns the historical-volatility target with the maturity of the option-implied volatility feature. This alignment is not intended to identify a causal effect of IV. Instead, it creates a stringent forecasting benchmark: if IV improves forecasts even after HAR lags capture strong 30-day persistence, the evidence supports IV as a robust forward-looking state variable in this proxy-volatility setting.

## 3. Mention 10-Day HV as Future Robustness, Not Main Evidence

You have downloaded 10-day VolVue data, and conceptually 10-day HV may be a sharper target than 30-day HV. But the current paper-ready results are still 30-day. So the draft should say:

- 10-day HV is a natural next robustness target;
- it is less smoothed than 30-day HV;
- it may better expose short-run forecast differences;
- it does not replace the current evidence until rerun.

Suggested text:

> A natural next step is to repeat the pipeline on 10-day HV/IV panels. A 10-day target would reduce the mechanical overlap with the monthly HAR component and provide a less smoothed proxy than HV30, while still avoiding some microstructure noise in intraday realized volatility.

## 4. SATS / Extreme Event Framing

The SATS example should support the paper's central message:

- extreme stock-date events create irregular losses;
- high-frequency RV would likely amplify such irregularity;
- the 30-day target smooths the event but does not remove the stress;
- GNNHAR instability under the smoother target is therefore meaningful.

Suggested text:

> The SATS event is not treated as an observation to delete. It is a stress case. A roughly 70% close-to-close return would likely produce an even more irregular high-frequency realized-volatility observation. The fact that the large-universe GNNHAR rows already struggle under the smoother 30-day target is consistent with the paper's broader caution about nonlinear graph depth under irregular volatility dynamics.

## 5. S&P500 Calendar: Use “Nested Near-Calendar,” Not “Different Calendar”

I checked the actual arrays:

- Dow30 aligned: 234 dates, 2025-07-07 to 2026-06-09.
- S&P100: 234 dates, 2025-07-07 to 2026-06-09.
- S&P500: 223 dates, 2025-07-14 to 2026-06-01.
- Every S&P500 test date is inside the Dow30/S&P100 calendar.
- Missing S&P500 dates are only the first five and last six boundary dates.

So the correct wording is not “incompatible calendar.” It is:

> S&P500 is a 223-date subset of the 234-date Dow30/S&P100 calendar.

This is still a limitation, but a smaller one.

Suggested text:

> The S&P500 run is not fully identical in length, but it is nested within the Dow30/S&P100 calendar. The comparison is therefore a near-calendar scale comparison rather than a fully disjoint sample comparison.

## 6. Robustness: Do Not Let It Dominate the Paper

The current robustness section is too apologetic if read as a submission draft. It should be rewritten around three categories:

1. Completed evidence: main losses, MCS, DM, FVU, S&P500 diagnostics.
2. Low-cost near-term additions: date-subset check, IV placebo using existing arrays if available, recompute Dow30 regime on 234-date arrays, ticker/sector heterogeneity summaries.
3. Computational future work: full S&P500 rerun, corrected-output GNNHAR, exact hidden-state MAD, 10-day target rerun.

This keeps the paper focused and does not make the manuscript read like an unfinished checklist.

## 7. Economic Contribution: Put GNNHAR Fragility Beside IV Value

Do not frame the contribution only as “IV is useful.” That is known.

A stronger contribution is:

> In a Zhang-style graph-HAR/GNNHAR framework, IV is the stable source of out-of-sample improvement, while graph neural depth is not robustly rewarded by QLIKE, especially when scaling to hundreds of equities.

This makes the machine-learning contribution clear: more complex graph neural architectures are not automatically better for financial forecasting, even when evaluated in a large panel.

Suggested contribution bullets:

- We extend Zhang-style HAR/GHAR/GNNHAR with option-implied volatility.
- We evaluate whether graph-based gains scale from Dow30 to S&P100 and S&P500.
- We show that IV-augmented models repeatedly enter the best QLIKE set.
- We show that deeper GNNHAR does not produce monotone gains and can fail badly in the large-universe setting.
- We document diagnostic evidence linking this failure to low-volatility over-prediction and output-scale instability.

## 8. Benchmark Criticism: Respond by Scope, Not by Overbuilding

The paper does not need to become a universal benchmark zoo. Zhang et al. were published with a focused HAR/GHAR/GNNHAR comparison, and your paper is an extension of that line.

The response should be:

- The benchmark family is deliberately Zhang-style.
- HAR is the canonical volatility benchmark.
- GHAR and GNNHAR are the models under test.
- IV variants are the paper's extension.
- Additional ML baselines are useful future work, but not required for the core Zhang-style question.

Suggested text:

> The model set is intentionally focused on the Zhang-style HAR, GHAR, and GNNHAR family rather than a general-purpose machine-learning horse race. This scope allows us to isolate whether graph aggregation, graph neural depth, and IV augmentation improve the same heterogeneous-autoregressive backbone.

## 9. MCS / DM: Keep It, But Clarify the Loss Sequence

MCS and DM are already strong enough for the current scope if clearly explained. The paper should state whether losses are averaged by date before MCS/DM. If the code uses date-level loss series, say so explicitly.

Suggested text:

> MCS and DM tests are applied to date-level loss series obtained by aggregating losses across stocks within each out-of-sample date. This keeps inference at the time-series level and avoids treating stock-date observations as independent.

If the code does not do this, the diagnostic code should be aligned with this statement.

## 10. Appendix: Keep Reproducibility, Move the Economic Evidence Forward

The appendix can keep repo paths and artifacts. That is useful. But the main text should not feel like a local experiment log.

Recommended split:

- Main text: economic question, model comparison, main tables, MCS interpretation, GNNHAR fragility.
- Appendix A: data and reproducibility map.
- Appendix B: MCS/DM details.
- Appendix C: S&P500 diagnostic failure case.
- Appendix D: robustness status matrix.

The current engineering details should remain, but they should be framed as reproducibility evidence rather than part of the main argument.

## Recommended Main-Text Positioning

The paper should converge to this thesis:

> We do not claim that GNNHAR universally dominates HAR. Instead, we show that when the Zhang-style graph-HAR/GNNHAR framework is extended with option-implied volatility and scaled across equity universes, the stable predictive information comes from IV, while nonlinear graph depth is fragile under QLIKE evaluation. This fragility appears even under a smooth 30-day volatility proxy, suggesting that graph neural volatility forecasting requires careful target choice, output scaling, and robustness diagnostics before being treated as a superior financial forecasting tool.

This thesis is closer to your actual evidence and makes the paper stronger than a generic “IV improves forecasts” paper.
