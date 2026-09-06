# Dow30 Aligned Full-Model Run

This directory is reserved for the corrected 234-date Dow30 full-model run used by the paper draft.

Expected test calendar:

- Start: 2025-07-07
- End: 2026-06-09
- Test dates: 234
- Model tickers: 30

Expected layout:

```text
arrays/
  truth.npy
  tickers.npy
  test_dates.npy
predictions/
  pred_HAR_M.npy
  pred_HAR_M_IV.npy
  pred_HAR_Q.npy
  pred_HAR_Q_IV.npy
  pred_GHAR_M.npy
  pred_GHAR_M_IV.npy
  pred_GHAR_Q.npy
  pred_GHAR_Q_IV.npy
  pred_GNNHAR1L_M.npy
  ...
tables/
  loss_table.csv
  dm_tests.csv
  fvu.csv
diagnostics/
  mcs_mse.csv
  mcs_qlike.csv
  mcs_summary.csv
```

After exporting aligned arrays and prediction files, recompute diagnostics with:

```bash
python3 scripts/paper_ready/recompute_diagnostics.py \
  --run-dir outputs/paper_ready_20260617/universes/dow30/aligned_full_model_20260619 \
  --bootstrap 10000 \
  --block-size 2 \
  --algorithm SQ \
  --seed 0
```

The older 232-date Dow30 full-model run is archived under `../.trash/legacy_232date_full_model_20260614/` and must not be used for current paper MCS inference.
