#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BUILD_ROOT="${BUILD_ROOT:-${ROOT_DIR}/tmp/autodl_package_${STAMP}}"
PACKAGE_DIR="${BUILD_ROOT}/gnnhar_sp500_autodl"
PACKAGE_PATH="${PACKAGE_PATH:-${ROOT_DIR}/tmp/gnnhar_sp500_autodl_${STAMP}.tar.gz}"

rm -rf "$BUILD_ROOT"
mkdir -p "$PACKAGE_DIR"

mkdir -p "$PACKAGE_DIR/scripts/analysis" "$PACKAGE_DIR/scripts/data" "$PACKAGE_DIR/src" "$PACKAGE_DIR/autodl"

rsync -a \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  src/ "$PACKAGE_DIR/src/"

rsync -a \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  scripts/analysis/ "$PACKAGE_DIR/scripts/analysis/"

cp scripts/post_run_diagnostics.py "$PACKAGE_DIR/scripts/post_run_diagnostics.py"

mkdir -p "$PACKAGE_DIR/data/scale_experiment/sp500"
rsync -a data/scale_experiment/sp500/ "$PACKAGE_DIR/data/scale_experiment/sp500/"

cp autodl/run_sp500_full.sh "$PACKAGE_DIR/autodl/run_sp500_full.sh"
cp autodl/README_SP500_AUTODL.md "$PACKAGE_DIR/autodl/README_SP500_AUTODL.md"

cat > "$PACKAGE_DIR/requirements.txt" <<'EOF'
numpy
pandas
scipy
scikit-learn
matplotlib
torch
EOF

cat > "$PACKAGE_DIR/PACKAGE_MANIFEST.txt" <<EOF
Created UTC: ${STAMP}
Purpose: AutoDL SP500 GNNHAR run package
Entrypoint: bash autodl/run_sp500_full.sh
Data path: data/scale_experiment/sp500
EOF

mkdir -p "$(dirname "$PACKAGE_PATH")"
tar -C "$BUILD_ROOT" -czf "$PACKAGE_PATH" gnnhar_sp500_autodl

echo "$PACKAGE_PATH"
du -sh "$PACKAGE_PATH"
