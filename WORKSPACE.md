# Cloud workspace notes

This file orients a Cursor Cloud checkout so implied-volatility / GNNHAR paper work can continue without re-deriving the tree. It does not replace `outputs/paper_ready_20260617/README.md` and does not restate manuscript numbers.

## Branch

- Working branch for this paper-ready tree: `2026-06-01`
- Verified against `origin/2026-06-01` at `6ea605b` (`Sync paper-ready GNNHAR-IV essentials from local Volatility tree`)
- Re-check with `git fetch origin 2026-06-01 && git log -1 --oneline origin/2026-06-01` after later pulls

## What was synced

Commit `6ea605b` brought the paper-ready GNNHAR-IV essentials from the local Volatility tree into this repo:

- Dow30 aligned full-model evidence under `outputs/paper_ready_20260617/universes/dow30/aligned_full_model_20260619/`
- S&P100 and S&P500 paper-ready universe trees under `outputs/paper_ready_20260617/universes/sp100/` and `.../sp500/`
- Reproduction script `scripts/paper_ready/recompute_diagnostics.py` and Colab notebook `notebooks/paper_ready_reproduction_colab.ipynb`
- Processed IV inputs (for example `data/processed/sp100_alphaquery/`)
- Manuscript draft assets under `reports/zhang_style_statistics_20260618/`
- `.gitignore` rules for bulk raw data, smoke outputs, and deliverables

## How to reproduce (notebook + diagnostics)

Canonical instructions live in `outputs/paper_ready_20260617/README.md`. Short path:

1. Open `notebooks/paper_ready_reproduction_colab.ipynb` and run top-to-bottom after the data folders are present. It clones/checks out `2026-06-01`, verifies paths and GPU, runs a smoke test, launches Dow30 / S&P100 full-model Colab jobs, imports or runs S&P500 AutoDL output, exports arrays into the paper-ready layout, then recomputes diagnostics/MCS. Keep smoke-test cells separate from expensive full-model cells.
2. After aligned `arrays/` and `predictions/` exist, recompute diagnostics only (no training):

```bash
python scripts/paper_ready/recompute_diagnostics.py \
  --run-dir outputs/paper_ready_20260617/universes/dow30/aligned_full_model_20260619 \
  --bootstrap 10000 \
  --block-size 2 \
  --algorithm SQ \
  --seed 0
```

Point `--run-dir` at `universes/sp100` or `universes/sp500` the same way. The script writes MCS/loss/integrity files under `RUN_DIR/diagnostics/`.

Do not use `universes/dow30/.trash/legacy_232date_full_model_20260614/` for current paper MCS tables.

## Paths verified in this checkout

Present on disk and tracked as expected:

- `outputs/paper_ready_20260617/universes/dow30/aligned_full_model_20260619/` (arrays, predictions, tables, diagnostics)
- `outputs/paper_ready_20260617/universes/sp100/`
- `outputs/paper_ready_20260617/universes/sp500/`
- `scripts/paper_ready/recompute_diagnostics.py`
- `notebooks/paper_ready_reproduction_colab.ipynb`

## Intentionally not in git

From `.gitignore` (absent here, as intended):

- `data/raw/`
- `data/google_drive_upload/`, `data/packages/`
- `deliverables/`
- smoke outputs: `outputs/**/*smoke*/`, `outputs/scale-experiment-smoke/`, `outputs/zhang-scale-smoke/`
- other bulk local runs: `outputs/autodl/`, `outputs/gnnhar_iv_fixed_*/`, `outputs/gnnhar_iv_relation_*/`, `outputs/gnnhar_iv_tuned_*/`, `outputs/**/.trash/`

Bring those from the local Volatility tree, Google Drive, or AutoDL when a full retrain or raw-data rebuild is needed.

## Next useful commands

```bash
# confirm paper branch
git fetch origin 2026-06-01
git checkout 2026-06-01
git log -1 --oneline

# Python deps used by the paper-ready notebook
python3 -m pip install -r requirements-scale.txt

# recompute MCS/diagnostics on an existing exported run
python scripts/paper_ready/recompute_diagnostics.py \
  --run-dir outputs/paper_ready_20260617/universes/dow30/aligned_full_model_20260619 \
  --bootstrap 10000 --block-size 2 --algorithm SQ --seed 0

# manuscript
# reports/zhang_style_statistics_20260618/paper_draft/main.tex
```

For training, use the Colab notebook (or `autodl/` for S&P500). For interpretation of current evidence, start from `outputs/paper_ready_20260617/README.md`, not the older root-README Dow30 table.
