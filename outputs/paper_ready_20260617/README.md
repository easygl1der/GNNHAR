# GNNHAR Paper-Ready Results

This folder is the structured analysis entrypoint for the Zhang-style paper draft.
It separates current paper evidence from archived runs whose calendars no longer match the manuscript.

## Current Status

| Universe | Main evidence status | Model tickers | Test dates | Test window | Notes |
|---|---:|---:|---:|---|---|
| Dow30 | current aligned full-model evidence | 30 | 234 | 2025-07-07 to 2026-06-09 | Imported from `20260619T071426Z_aligned_full_model`; 234-date MCS has been recomputed. |
| S&P100 | current full-model evidence | 91 | 234 | 2025-07-07 to 2026-06-09 | Current saved diagnostics and MCS are retained. |
| S&P500 | current AutoDL full-model evidence | 449 | 223 | 2025-07-14 to 2026-06-01 | Current saved diagnostics and MCS are retained; calendar is near-aligned, not identical. |

The older Dow30 232-date full-model run has been moved to:

```text
universes/dow30/.trash/legacy_232date_full_model_20260614/
```

Do not use that folder for current paper MCS tables. It is kept only for provenance.

## Layout

- `universes/dow30/aligned_full_model_20260619/`: corrected 234-date Dow30 full-model run and recomputed diagnostics.
- `universes/dow30/supplements/wide_multihop_ghar_20260618/`: 234-date GHAR2H/GHAR3H supplement only; it is not a full HAR/GHAR/GNNHAR model family.
- `universes/dow30/.trash/`: archived incompatible Dow30 artifacts.
- `universes/sp100/`: current S&P100 full-model paper-ready artifacts.
- `universes/sp500/`: current S&P500 AutoDL full-model paper-ready artifacts.

For each current full-model run, use this structure:

```text
RUN_DIR/
  arrays/
    truth.npy
    tickers.npy
    test_dates.npy
  predictions/
    pred_HAR_M.npy
    pred_HAR_M_IV.npy
    pred_GHAR_M.npy
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

## Recompute Diagnostics and MCS

To recompute paper diagnostics after changing aligned truth or prediction arrays, run:

```bash
python scripts/paper_ready/recompute_diagnostics.py \
  --run-dir outputs/paper_ready_20260617/universes/dow30/aligned_full_model_20260619 \
  --bootstrap 10000 \
  --block-size 2 \
  --algorithm SQ \
  --seed 0
```

This writes per-date MSE/QLIKE losses, `mcs_mse.csv`, `mcs_qlike.csv`, `mcs_summary.csv`, loss summaries, and integrity checks under `RUN_DIR/diagnostics/`.

## Colab Reproduction

Use the notebook:

```text
notebooks/paper_ready_reproduction_colab.ipynb
```

The notebook is organized as:

1. Mount Google Drive and clone the repository branch.
2. Verify data paths and GPU.
3. Run a smoke test.
4. Run Dow30/S&P100 full-model Colab jobs.
5. Run or import the S&P500 AutoDL output.
6. Export arrays into this paper-ready layout.
7. Recompute diagnostics and MCS.

The notebook is intended to be run top-to-bottom after the data folders are present. Long full-model cells are separated from smoke-test cells so that a reviewer can verify the code path before launching expensive training.

## Paper Draft

The manuscript lives at:

```text
reports/zhang_style_statistics_20260618/paper_draft/main.tex
```

The Dow30 MCS source is `universes/dow30/aligned_full_model_20260619/diagnostics/mcs_summary.csv`, which reports 234 test dates.
