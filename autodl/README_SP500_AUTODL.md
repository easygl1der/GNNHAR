# SP500 AutoDL Run Package

This package runs the S&P 500 Zhang-style GNNHAR experiment on an AutoDL GPU
server and writes outputs in the same layout used by the Colab/Drive workflow.

The archive already contains the SP500 data under `data/scale_experiment/sp500`.
The current panel has 1256 common dates from `2021-06-09` to `2026-06-09`.
With the Zhang-style driver threshold of `0.99`, the full run selects 449
well-covered nodes from 503 common tickers.

## What It Runs

- Universe: `sp500`
- Forecast horizon: daily, `H1`
- Rolling protocol: 1000-day lookback, 22-day validation, 22-day forecast blocks
- Models:
  - HAR, GHAR, GHAR2H, GHAR3H
  - HAR+IV, GHAR+IV, GHAR2H+IV, GHAR3H+IV
  - GNNHAR1L to GNNHAR5L
  - GNNHAR1L-IV to GNNHAR5L-IV
- Training losses: MSE and QLIKE
- Post-run diagnostics:
  - MCS with `alpha=0.05`, `B=10000`, `w=2`, `SQ`
  - 90% market-state split
  - MAD / smoothing diagnostics
  - DM, FVU, integrity checks

## Recommended AutoDL Workflow

Choose an AutoDL image that already has Python, CUDA, PyTorch, NumPy, pandas,
SciPy, scikit-learn, and matplotlib. The entrypoint can install missing Python
packages, but for a long GPU job it is better to start from a PyTorch/CUDA
image.

Upload the generated archive to `/root/autodl-tmp`, then unpack it:

```bash
cd /root/autodl-tmp
tar -xzf gnnhar_sp500_autodl_*.tar.gz
cd gnnhar_sp500_autodl
```

Run a short smoke test first:

```bash
MAX_BLOCKS=1 MAX_TICKERS=80 EPOCHS=40 MCS_BOOTSTRAP=20 bash autodl/run_sp500_full.sh
```

If the smoke test finishes and creates a result archive, run the full SP500 job:

```bash
bash autodl/run_sp500_full.sh
```

To enforce a specific GPU type:

```bash
REQUIRE_GPU_NAME=A100 bash autodl/run_sp500_full.sh
```

or:

```bash
REQUIRE_GPU_NAME=H100 bash autodl/run_sp500_full.sh
```

For local debugging only, when there is no NVIDIA GPU:

```bash
SKIP_PIP_INSTALL=1 SKIP_GPU_CHECK=1 MAX_BLOCKS=1 MAX_TICKERS=20 EPOCHS=2 MCS_BOOTSTRAP=2 bash autodl/run_sp500_full.sh
```

## Main Environment Variables

- `EPOCHS`: default `5000`
- `HIDDEN_GRID`: default `9`
- `LR_GRID`: default `0.001`
- `MCS_BOOTSTRAP`: default `10000`
- `MAX_BLOCKS`: default `0`, meaning all blocks
- `MAX_TICKERS`: default `0`, meaning all selected tickers
- `REQUIRE_GPU_NAME`: optional substring such as `A100` or `H100`
- `SKIP_PIP_INSTALL`: local/debug switch; use `1` only when dependencies are already installed
- `SKIP_GPU_CHECK`: local/debug switch; do not use for the formal AutoDL run

## Outputs

The final Colab-style result folder is:

```text
autodl_runs/<RUN_ID>/gnnhar_colab_runs/sp500/<RUN_ID>/
```

Important files:

- `truth.npy`
- `test_dates.npy`
- `tickers.npy`
- `pred_*.npy`
- `loss_table.csv`
- `dm_tests.csv`
- `dm_depth_tests.csv`
- `fvu.csv`
- `post_run_diagnostics/`
- `run_manifest.json`

The script also creates:

```text
autodl_runs/<RUN_ID>/sp500_results_<RUN_ID>.tar.gz
```

Download that archive back to the local machine. Inside it, the folder
`gnnhar_colab_runs/sp500/<RUN_ID>` can be copied into the Google Drive output
tree used by the Colab notebook.

## Practical Notes

- The full job runs MSE and QLIKE separately, then merges the forecasts into one
  Colab-style result folder.
- The post-run diagnostic stage is run after the merge, so MCS, market-state
  split, and MAD tables are stored next to `loss_table.csv`.
- If package installation is slow on AutoDL, use an AutoDL PyTorch/CUDA image or
  set a PyPI mirror with `PIP_INDEX_URL` before running the script.
- If the server is interrupted, keep the `autodl_runs/<RUN_ID>` directory. Logs
  are in `autodl_runs/<RUN_ID>/logs`.
