#!/usr/bin/env bash
set -Eeuo pipefail

# AutoDL entrypoint for the SP500 Zhang-style GNNHAR run.
#
# Usage on AutoDL:
#   cd /root/autodl-tmp/gnnhar_sp500_autodl
#   bash autodl/run_sp500_full.sh
#
# Useful overrides:
#   EPOCHS=300 MCS_BOOTSTRAP=10000 REQUIRE_GPU_NAME=A100 bash autodl/run_sp500_full.sh
#   MAX_BLOCKS=1 MAX_TICKERS=80 EPOCHS=40 MCS_BOOTSTRAP=20 bash autodl/run_sp500_full.sh
#   SKIP_PIP_INSTALL=1 SKIP_GPU_CHECK=1 MAX_BLOCKS=1 MAX_TICKERS=20 EPOCHS=2 bash autodl/run_sp500_full.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
WORK_ROOT="${WORK_ROOT:-${ROOT_DIR}/autodl_runs/${RUN_ID}}"
SOURCE_OUT_ROOT="${SOURCE_OUT_ROOT:-${WORK_ROOT}/zhang_scale_source}"
COLAB_OUT_ROOT="${COLAB_OUT_ROOT:-${WORK_ROOT}/gnnhar_colab_runs}"
LOG_DIR="${LOG_DIR:-${WORK_ROOT}/logs}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/scale_experiment/sp500}"
RETURNS_FILE="${RETURNS_FILE:-${DATA_DIR}/daily_returns.csv}"

MODELS="${MODELS:-HAR,GHAR,GHAR2H,GHAR3H,HAR+IV,GHAR+IV,GHAR2H+IV,GHAR3H+IV,GNNHAR1L,GNNHAR2L,GNNHAR3L,GNNHAR4L,GNNHAR5L,GNNHAR1L-IV,GNNHAR2L-IV,GNNHAR3L-IV,GNNHAR4L-IV,GNNHAR5L-IV}"
HIDDEN_GRID="${HIDDEN_GRID:-9}"
LR_GRID="${LR_GRID:-0.001}"
EPOCHS="${EPOCHS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_NN="${NUM_NN:-1}"
MCS_BOOTSTRAP="${MCS_BOOTSTRAP:-10000}"
LOOKBACK="${LOOKBACK:-1000}"
# Avoid the generic WINDOW environment variable because terminal multiplexers
# such as screen may set WINDOW=0.
FORECAST_WINDOW="${FORECAST_WINDOW:-22}"
VALID_LEN="${VALID_LEN:-22}"
BLOCK_STRIDE="${BLOCK_STRIDE:-22}"
MAX_BLOCKS="${MAX_BLOCKS:-0}"
MAX_TICKERS="${MAX_TICKERS:-0}"
MAX_NEIGHBORS="${MAX_NEIGHBORS:-0}"
GRAPH_METHOD="${GRAPH_METHOD:-glasso_cv}"
REQUIRE_GPU_NAME="${REQUIRE_GPU_NAME:-}"
SKIP_PIP_INSTALL="${SKIP_PIP_INSTALL:-0}"
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-0}"

mkdir -p "$WORK_ROOT" "$SOURCE_OUT_ROOT" "$COLAB_OUT_ROOT" "$LOG_DIR"

require_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer; got '$value'" >&2
    exit 2
  fi
}

require_nonnegative_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "$name must be a non-negative integer; got '$value'" >&2
    exit 2
  fi
}

require_positive_int LOOKBACK "$LOOKBACK"
require_positive_int FORECAST_WINDOW "$FORECAST_WINDOW"
require_positive_int VALID_LEN "$VALID_LEN"
require_positive_int BLOCK_STRIDE "$BLOCK_STRIDE"
require_positive_int EPOCHS "$EPOCHS"
require_positive_int BATCH_SIZE "$BATCH_SIZE"
require_positive_int NUM_NN "$NUM_NN"
require_positive_int MCS_BOOTSTRAP "$MCS_BOOTSTRAP"
require_nonnegative_int MAX_BLOCKS "$MAX_BLOCKS"
require_nonnegative_int MAX_TICKERS "$MAX_TICKERS"
require_nonnegative_int MAX_NEIGHBORS "$MAX_NEIGHBORS"

echo "ROOT_DIR=$ROOT_DIR"
echo "WORK_ROOT=$WORK_ROOT"
echo "DATA_DIR=$DATA_DIR"
echo "RUN_ID=$RUN_ID"
echo "MODELS=$MODELS"
echo "EPOCHS=$EPOCHS HIDDEN_GRID=$HIDDEN_GRID LR_GRID=$LR_GRID BATCH_SIZE=$BATCH_SIZE NUM_NN=$NUM_NN"
echo "LOOKBACK=$LOOKBACK FORECAST_WINDOW=$FORECAST_WINDOW VALID_LEN=$VALID_LEN BLOCK_STRIDE=$BLOCK_STRIDE"
echo "MAX_BLOCKS=$MAX_BLOCKS MAX_TICKERS=$MAX_TICKERS MAX_NEIGHBORS=$MAX_NEIGHBORS"
echo "MCS_BOOTSTRAP=$MCS_BOOTSTRAP GRAPH_METHOD=$GRAPH_METHOD"

if [[ "$SKIP_PIP_INSTALL" == "1" ]]; then
  echo "SKIP_PIP_INSTALL=1; skipping dependency installation."
else
python - <<'PY'
import importlib.util
import subprocess
import sys

required = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "torch": "torch",
}
missing = [pkg for module, pkg in required.items() if importlib.util.find_spec(module) is None]
if missing:
    print("Installing missing packages:", missing)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
else:
    print("Required Python packages are already available.")
PY
fi

if [[ "$SKIP_GPU_CHECK" == "1" ]]; then
  echo "SKIP_GPU_CHECK=1; skipping nvidia-smi GPU check."
else
python - <<PY
import json
import os
import subprocess
import sys

required = os.environ.get("REQUIRE_GPU_NAME", "")
try:
    res = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
    )
except FileNotFoundError:
    raise SystemExit("nvidia-smi not found. Start an AutoDL GPU instance before running this job.")
print(res.stdout.strip())
names = [line.strip() for line in res.stdout.splitlines() if line.strip()]
if not names:
    raise SystemExit("No GPU detected by nvidia-smi.")
if required and not any(required.lower() in name.lower() for name in names):
    raise SystemExit(f"Required GPU containing {required!r}, got {names!r}")
PY
fi

python - <<PY
from pathlib import Path
import pandas as pd

data_dir = Path("$DATA_DIR")
required = ["merged_rv_data_filled.csv", "merged_iv_data_filled.csv", "daily_returns.csv"]
missing = [name for name in required if not (data_dir / name).exists()]
if missing:
    raise SystemExit(f"Missing SP500 data files under {data_dir}: {missing}")
for name in required:
    path = data_dir / name
    header = pd.read_csv(path, nrows=0)
    n_rows = sum(1 for _ in path.open()) - 1
    print(f"{name}: rows={n_rows}, columns={len(header.columns)}")
PY

COMMON_ARGS=(
  --universes sp500
  --horizons 1
  --models "$MODELS"
  --hidden-grid "$HIDDEN_GRID"
  --lr-grid "$LR_GRID"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-nn "$NUM_NN"
  --mcs-bootstrap "$MCS_BOOTSTRAP"
  --lookback "$LOOKBACK"
  --window "$FORECAST_WINDOW"
  --valid-len "$VALID_LEN"
  --block-stride "$BLOCK_STRIDE"
  --graph-method "$GRAPH_METHOD"
  --output-root "$SOURCE_OUT_ROOT"
  --rerun-existing
)

if [[ "$MAX_BLOCKS" != "0" ]]; then
  COMMON_ARGS+=(--max-blocks "$MAX_BLOCKS")
fi
if [[ "$MAX_TICKERS" != "0" ]]; then
  COMMON_ARGS+=(--max-tickers "$MAX_TICKERS")
fi
if [[ "$MAX_NEIGHBORS" != "0" ]]; then
  COMMON_ARGS+=(--max-neighbors "$MAX_NEIGHBORS")
fi
if [[ -n "$REQUIRE_GPU_NAME" ]]; then
  COMMON_ARGS+=(--require-gpu-name "$REQUIRE_GPU_NAME")
fi
if [[ "$SKIP_GPU_CHECK" == "1" ]]; then
  COMMON_ARGS+=(--allow-missing-gpu)
fi

echo "Starting SP500 MSE stage..."
python scripts/analysis/run_zhang_scale_colab_full.py \
  "${COMMON_ARGS[@]}" \
  --losses MSE \
  2>&1 | tee "$LOG_DIR/sp500_mse.log"

echo "Starting SP500 QLIKE stage..."
python scripts/analysis/run_zhang_scale_colab_full.py \
  "${COMMON_ARGS[@]}" \
  --losses QLIKE \
  2>&1 | tee "$LOG_DIR/sp500_qlike.log"

echo "Converting source outputs to Colab-style layout..."
python scripts/analysis/convert_zhang_scale_to_colab_outputs.py \
  --source-root "$SOURCE_OUT_ROOT" \
  --dest-root "$COLAB_OUT_ROOT" \
  --universe sp500 \
  --horizon 1 \
  --run-id "$RUN_ID" \
  2>&1 | tee "$LOG_DIR/convert_colab_layout.log"

echo "Running post-run diagnostics on Colab-style output..."
python scripts/post_run_diagnostics.py \
  --output-root "$COLAB_OUT_ROOT" \
  --universes sp500 \
  --bootstrap "$MCS_BOOTSTRAP" \
  --block-size 2 \
  2>&1 | tee "$LOG_DIR/post_run_diagnostics.log"

cat > "$WORK_ROOT/RUN_SUMMARY.txt" <<EOF
SP500 AutoDL run completed.

Run ID: $RUN_ID
Source pipeline output: $SOURCE_OUT_ROOT
Colab-style output: $COLAB_OUT_ROOT/sp500/$RUN_ID
Logs: $LOG_DIR
Result archive: $WORK_ROOT/sp500_results_${RUN_ID}.tar.gz

Main files:
- $COLAB_OUT_ROOT/sp500/$RUN_ID/truth.npy
- $COLAB_OUT_ROOT/sp500/$RUN_ID/pred_*.npy
- $COLAB_OUT_ROOT/sp500/$RUN_ID/loss_table.csv
- $COLAB_OUT_ROOT/sp500/$RUN_ID/dm_tests.csv
- $COLAB_OUT_ROOT/sp500/$RUN_ID/fvu.csv
- $COLAB_OUT_ROOT/sp500/$RUN_ID/post_run_diagnostics/
EOF

tar -C "$WORK_ROOT" -czf "$WORK_ROOT/sp500_results_${RUN_ID}.tar.gz" \
  "$(basename "$SOURCE_OUT_ROOT")" \
  "$(basename "$COLAB_OUT_ROOT")" \
  logs \
  RUN_SUMMARY.txt

echo "Done."
cat "$WORK_ROOT/RUN_SUMMARY.txt"
