# Implied Volatility in Volatility Forecasting

This folder collects papers on using option-implied volatility (IV) or volatility-index information to improve traditional volatility forecasting models. The search was run with Academic MCP on 2026-06-19. Journal quartile notes were checked separately because Academic MCP returns paper metadata but not journal quartiles.

For the Chinese working summary, see `summary.zh.md`.

## Files

- `summary.zh.md`: Chinese synthesis of how the papers add IV to traditional models.
- `manifest.json`: machine-readable paper metadata, local file paths, and download status.
- `references.bib`: BibTeX entries.
- `yfanti-karanasos-2021-aim-heavy-jors.pdf`: open repository PDF for Yfanti and Karanasos, Journal of the Operational Research Society.
- `becker-clements-mcclelland-2009-vix-jump-jbf.pdf`: open QUT eprints PDF for Becker, Clements, and McClelland, Journal of Banking and Finance.
- `kambouroudis-mcmillan-tsakou-2019-wp-har-iv.pdf`: open working-paper version of Kambouroudis, McMillan, and Tsakou's later Journal of Futures Markets article.
- `yuan-zhou-zhang-cui-2019-garch-ito-option-data-arxiv.pdf`: arXiv version of Yuan, Zhou, Zhang, and Cui's later Canadian Journal of Statistics article.
- `jeon-taylor-2011-caviar-iv-jof.pdf`: open pre-publication PDF for Jeon and Taylor's Journal of Forecasting article.
- `yao-izzeldin-2018-model-free-iv-jfm.pdf`: open PDF for Yao and Izzeldin's Journal of Futures Markets article.
- `text/`: fallback local text extraction for PDFs that MinerU could not parse during this run.

Publisher pages were blocked or closed in several cases. No fake local PDF was created for papers where only metadata or a landing page was available.

MinerU parsing was attempted for the saved PDFs, but the MinerU service returned `503 Service Unavailable`; the summaries therefore use Academic MCP metadata plus local PDF text extraction where available.

## Core Q1 Candidates

These are the strongest matches to "Q1 journal + traditional volatility model + option-implied volatility".

1. Yfanti and Karanasos, "Financial volatility modeling with option-implied information and important macro-factors", Journal of the Operational Research Society, DOI `10.1080/01605682.2021.1966327`.
   - Method: extends the HEAVY framework into an AIM-HEAVY system with asymmetry, IV, realized variance, and macro uncertainty.
   - Local PDF: `yfanti-karanasos-2021-aim-heavy-jors.pdf`.
   - Quartile note: Resurchify/SJR snapshot checked on 2026-06-19 lists best quartile Q1 for 2024.

2. Becker, Clements, and McClelland, "The jump component of S&P 500 volatility and the VIX index", Journal of Banking and Finance, DOI `10.1016/j.jbankfin.2008.10.015`.
   - Method: compares VIX with model-based forecasts and decomposes realized volatility into continuous and jump components.
   - Local PDF: `becker-clements-mcclelland-2009-vix-jump-jbf.pdf`.
   - Quartile note: Resurchify/SJR snapshot checked on 2026-06-19 lists Journal of Banking and Finance as Q1 in Economics and Econometrics and Finance.

3. Seo and Kim, "The information content of option-implied information for volatility forecasting with investor sentiment", Journal of Banking and Finance, DOI `10.1016/j.jbankfin.2014.09.010`.
   - Method: studies whether option-implied volatility information helps more or less depending on investor sentiment; included as metadata-only because no open PDF was found.
   - Local PDF: not saved.
   - Quartile note: same Journal of Banking and Finance Q1 evidence as above.

4. Qiao, Teng, Li, and Liu, "Improving volatility forecasting based on Chinese volatility index information: Evidence from CSI 300 index and futures markets", North American Journal of Economics and Finance, DOI `10.1016/j.najef.2019.04.003`.
   - Method: adds Chinese implied volatility index iVX to HAR and time-varying coefficient HAR models using CSI 300 index and futures realized volatility.
   - Local PDF: not saved; no direct open PDF was resolved.
   - Quartile note: Resurchify/SJR snapshot checked on 2026-06-19 lists best quartile Q1 for 2024.

## Methodologically Close Papers

These papers are highly useful for model design but should not all be presented as Q1 evidence without checking the target ranking system and year.

- Kambouroudis, McMillan, and Tsakou, Journal of Futures Markets, DOI `10.1002/fut.22241`: direct HAR-IV template. Journal of Futures Markets is Q2 in the checked 2024 SJR snapshot.
- Yuan, Zhou, Zhang, and Cui, Canadian Journal of Statistics, DOI `10.1002/cjs.11746`: GARCH-Ito-OI and GARCH-Ito-IV models that combine low-frequency, high-frequency, and option-implied information. Canadian Journal of Statistics is Q2 in the checked 2024 snapshot.
- Wu, Wang, and Wang, Applied Economics Letters, DOI `10.1080/13504851.2020.1785617`: realized EGARCH-MIDAS extended with IV. Applied Economics Letters is Q3 in the checked 2024 snapshot.
- Yao and Izzeldin, Journal of Futures Markets, DOI `10.1002/fut.21881`: robustness over model-free IV constructions rather than a single vendor IV series.
- Jeon and Taylor, Journal of Forecasting, DOI `10.1002/for.1251`: CAViaR plus IV for VaR / quantile forecasting; Journal of Forecasting is Q1 in the checked 2024 snapshot, but the task target here is VaR rather than realized-volatility point forecasting.
- Donaldson and Kamstra, Journal of Financial Research, DOI `10.1111/j.1475-6803.2005.00137.x`: ARCH versus IV forecasts conditional on trading volume; included for mechanism.

## Practical Modeling Takeaways

1. The cleanest direct baseline is HAR-IV:

\[
RV_{t+h}=\alpha+\beta_d RV_t+\beta_w RV_{t:t-4}+\beta_m RV_{t:t-21}+\gamma IV_t+\varepsilon_{t+h}.
\]

2. HEAVY/GARCH-type extensions treat IV as a forward-looking state or exogenous information channel:

\[
h_t=f(h_{t-1}, r_{t-1}^2, RM_{t-1}, IV_{t-1}, U_{t-1}, A_{t-1}),
\]

where \(RM\) is a realized measure, \(U\) is uncertainty, and \(A\) captures asymmetry.

3. A useful graph-model extension is:

\[
\widehat{RV}_{i,t+h}
= f\left(H^{RV}_{i,t}, H^{IV}_{i,t}, W H^{RV}_{t}, W H^{IV}_{t}\right),
\]

where \(W\) is a return-network or volatility-spillover graph operator.

4. The literature suggests three robustness checks for GNNHAR-IV:
   - Compare IV construction choices: raw IV, VIX-style model-free IV, corridor IV, call/put-side measures.
   - Split by market regime: jump days, high-volatility states, high/low sentiment, high/low volume.
   - Compare "IV as regressor" against "IV as noisy measurement of latent future variance."
