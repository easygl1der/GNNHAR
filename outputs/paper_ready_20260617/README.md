# GNNHAR Paper-Ready Results

Created UTC: 2026-06-18T08:23:24.381590+00:00

This folder is the structured analysis entrypoint for Colab and report generation.
It is a copy of selected run outputs; the original Colab and AutoDL result folders are unchanged.

## Layout

- `universes/<universe>/arrays/`: `truth.npy`, `tickers.npy`, and `test_dates.npy`.
- `universes/<universe>/predictions/`: one `pred_*.npy` file per model.
- `universes/<universe>/tables/`: main loss, DM, FVU, and multi-hop tables.
- `universes/<universe>/diagnostics/`: MCS, market-regime, data-integrity, and smoothing diagnostics.
- `universes/<universe>/graphs/`: exported GLASSO adjacency matrices when available.
- `universes/<universe>/zhang_source/`: loss-specific Zhang-style source tables when available.
- `universes/<universe>/supplements/`: date-aligned supplemental runs that should not be silently merged into arrays with a different test-date index.
- `manifest.json`: machine-readable inventory.
- `checksums_sha256.txt`: checksum audit for every copied file.

## Universe Summary

| Universe | Truth shape | Predictions | Tables | Diagnostics | Graph matrices | Source |
|---|---:|---:|---:|---:|---:|---|
| dow30 | [232, 30] | 20 | 3 | 12 | 0 | Colab-style Dow30 full run written to Google Drive. |
| sp100 | [234, 91] | 36 | 5 | 12 | 0 | Colab-style SP100 full run written to Google Drive. |
| sp500 | [223, 449] | 36 | 4 | 12 | 22 | AutoDL A100 SP500 full run, converted to Colab-style outputs. |

## Colab Loading

Use this folder as the stable entrypoint after mounting Google Drive in Colab:

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

RESULT_ROOT = Path('/content/drive/MyDrive/GNNHAR_Research/results/paper_ready_20260617')

def load_universe(universe):
    root = RESULT_ROOT / 'universes' / universe
    truth = np.load(root / 'arrays' / 'truth.npy')
    tickers = np.load(root / 'arrays' / 'tickers.npy', allow_pickle=True)
    test_dates = np.load(root / 'arrays' / 'test_dates.npy', allow_pickle=True)
    predictions = {p.stem.removeprefix('pred_'): np.load(p) for p in sorted((root / 'predictions').glob('pred_*.npy'))}
    tables = {p.stem: pd.read_csv(p) for p in sorted((root / 'tables').glob('*.csv'))}
    diagnostics = {p.stem: pd.read_csv(p) for p in sorted((root / 'diagnostics').glob('*.csv')) if p.stat().st_size > 0}
    manifest = json.loads((root / 'universe_manifest.json').read_text())
    return truth, tickers, test_dates, predictions, tables, diagnostics, manifest

truth, tickers, test_dates, predictions, tables, diagnostics, manifest = load_universe('sp500')
print(truth.shape, len(predictions), sorted(tables))
```

For SP500 GLASSO matrices:

```python
graph_files = sorted((RESULT_ROOT / 'universes' / 'sp500' / 'graphs').rglob('*.npz'))
graph = np.load(graph_files[0], allow_pickle=True)
W = graph['adjacency']
info = json.loads(str(graph['info_json']))
print(len(graph_files), W.shape, info.keys())
```

The model names are encoded in prediction filenames. For example, `pred_GNNHAR5L_QLIKE_IV.npy` is the SP500 prediction array for the 5-layer IV GNNHAR trained under QLIKE.

## Notes

- `hidden_state_mad.csv` may be empty for runs that did not save hidden representations; use `mad_smoothing_diagnostics.csv` and `oversmoothing_depth_summary.csv` as prediction-level smoothing diagnostics.
- SP500 graph matrices are normalized adjacency matrices used by the models, not raw precision matrices.
