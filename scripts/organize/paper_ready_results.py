#!/usr/bin/env python3
"""Build a Drive-ready results tree for the GNNHAR universes.

The source outputs came from several runs: Dow30 and SP100 from Colab-backed
Drive folders, and SP500 from the AutoDL A100 run.  This script creates a
single, structured copy without moving or modifying the original outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVE_ROOT = Path(
    "/Users/yitwah/Library/CloudStorage/GoogleDrive-easyglider458@gmail.com/My Drive"
)


SOURCES = {
    "dow30": {
        "colab": DRIVE_ROOT
        / "GNNHAR_Research (1)"
        / "outputs"
        / "gnnhar_colab_runs"
        / "dow30"
        / "20260614T032943Z",
        "source_note": "Colab-style Dow30 full run written to Google Drive.",
        "supplements": {
            "wide_multihop_ghar_20260618": DRIVE_ROOT
            / "GNNHAR_Research"
            / "outputs"
            / "gnnhar_colab_runs"
            / "dow30"
            / "20260618T075711Z_wide_multihop_ghar",
        },
    },
    "sp100": {
        "colab": DRIVE_ROOT
        / "GNNHAR_Research (1)"
        / "outputs"
        / "gnnhar_colab_runs"
        / "sp100"
        / "20260614T181453Z",
        "source_note": "Colab-style SP100 full run written to Google Drive.",
    },
    "sp500": {
        "colab": REPO_ROOT
        / "outputs"
        / "autodl"
        / "sp500_parallel_a100_20260617T0706Z"
        / "extracted"
        / "gnnhar_colab_runs"
        / "sp500"
        / "sp500_parallel_a100_20260617T0706Z",
        "zhang_source": REPO_ROOT
        / "outputs"
        / "autodl"
        / "sp500_parallel_a100_20260617T0706Z"
        / "extracted"
        / "zhang_scale_source",
        "archive": REPO_ROOT
        / "outputs"
        / "autodl"
        / "sp500_parallel_a100_20260617T0706Z"
        / "sp500_results_sp500_parallel_a100_20260617T0706Z.tar.gz",
        "log": REPO_ROOT
        / "outputs"
        / "autodl"
        / "sp500_parallel_a100_20260617T0706Z"
        / "sp500_parallel_a100_20260617T0706Z.log",
        "source_note": "AutoDL A100 SP500 full run, converted to Colab-style outputs.",
    },
}


CORE_TABLES = [
    "loss_table.csv",
    "dm_tests.csv",
    "dm_depth_tests.csv",
    "multi_hop_dm_tests.csv",
    "fvu.csv",
    "incremental_fvu.csv",
]

METADATA_FILES = [
    "run_manifest.json",
    "run_config.json",
    "rolling_blocks.csv",
    "model_training_log.csv",
    "multi_hop_dm_summary.json",
    "posthoc_ghar_multihop_summary.json",
]

SUPPLEMENT_FILES = [
    "truth.npy",
    "tickers.npy",
    "test_dates.npy",
    "loss_table.csv",
    "dm_tests.csv",
    "fvu.csv",
    "run_manifest.json",
    "rolling_blocks.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=REPO_ROOT / "outputs" / "paper_ready_20260617",
        help="Destination directory for the structured results tree.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def copy_file(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_dir_contents(src: Path, dst: Path) -> list[str]:
    copied: list[str] = []
    if not src.exists() or not src.is_dir():
        return copied
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        copy_file(path, dst / rel)
        copied.append(str((dst / rel).relative_to(dst.parents[2])))
    return copied


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def npy_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    arr = np.load(path, allow_pickle=True)
    return {
        "shape": [int(x) for x in arr.shape],
        "dtype": str(arr.dtype),
        "finite_share": float(np.isfinite(arr).mean()) if np.issubdtype(arr.dtype, np.number) else None,
    }


def csv_summary(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {"exists": path.exists(), "empty": path.exists()}
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return {"exists": True, "empty": True}
    return {
        "exists": True,
        "empty": False,
        "rows": int(len(df)),
        "columns": list(map(str, df.columns)),
    }


def safe_rel(path: Path, base: Path) -> str:
    return str(path.relative_to(base))


def build_universe(universe: str, cfg: dict, dest_root: Path) -> dict:
    src = Path(cfg["colab"])
    if not src.exists():
        raise FileNotFoundError(f"{universe}: missing source {src}")

    dst = dest_root / "universes" / universe
    dst.mkdir(parents=True, exist_ok=True)

    files: list[str] = []
    source_files: dict[str, str] = {}

    # Core arrays for Colab loading.
    for name in ["truth.npy", "tickers.npy", "test_dates.npy"]:
        out = dst / "arrays" / name
        if copy_file(src / name, out):
            files.append(safe_rel(out, dest_root))
            source_files[safe_rel(out, dest_root)] = str(src / name)

    pred_dir = dst / "predictions"
    for pred in sorted(src.glob("pred_*.npy")):
        out = pred_dir / pred.name
        copy_file(pred, out)
        files.append(safe_rel(out, dest_root))
        source_files[safe_rel(out, dest_root)] = str(pred)

    # Main paper-facing tables.
    table_dir = dst / "tables"
    for name in CORE_TABLES:
        out = table_dir / name
        if copy_file(src / name, out):
            files.append(safe_rel(out, dest_root))
            source_files[safe_rel(out, dest_root)] = str(src / name)

    # Diagnostics: MCS, market regimes, data checks, and smoothing proxies.
    diag_src = src / "post_run_diagnostics"
    if diag_src.exists():
        for diag in sorted(diag_src.iterdir()):
            if diag.is_file():
                out = dst / "diagnostics" / diag.name
                copy_file(diag, out)
                files.append(safe_rel(out, dest_root))
                source_files[safe_rel(out, dest_root)] = str(diag)

    # Metadata and run logs.
    meta_dir = dst / "metadata"
    for name in METADATA_FILES:
        out = meta_dir / name
        if copy_file(src / name, out):
            files.append(safe_rel(out, dest_root))
            source_files[safe_rel(out, dest_root)] = str(src / name)

    # Preserve full SP500 Zhang-source tables and graph cache, which are unique
    # to the AutoDL run.  For Dow30/SP100 the available Drive source is already
    # the Colab-style layout.
    zhang_source = cfg.get("zhang_source")
    if zhang_source:
        zhang_source = Path(zhang_source)
        for loss in ["MSE", "QLIKE"]:
            source_tables = zhang_source / "full" / universe / loss / "H1" / "tables"
            for table in sorted(source_tables.glob("*")):
                if table.is_file():
                    out = dst / "zhang_source" / loss.lower() / "tables" / table.name
                    copy_file(table, out)
                    files.append(safe_rel(out, dest_root))
                    source_files[safe_rel(out, dest_root)] = str(table)
            source_pred = zhang_source / "full" / universe / loss / "H1" / "predictions_test.npz"
            out_pred = dst / "zhang_source" / loss.lower() / "predictions_test.npz"
            if copy_file(source_pred, out_pred):
                files.append(safe_rel(out_pred, dest_root))
                source_files[safe_rel(out_pred, dest_root)] = str(source_pred)
            for extra_name in ["run_metadata.json", "ticker_coverage.csv"]:
                extra_src = zhang_source / "full" / universe / loss / "H1" / extra_name
                extra_out = dst / "zhang_source" / loss.lower() / extra_name
                if copy_file(extra_src, extra_out):
                    files.append(safe_rel(extra_out, dest_root))
                    source_files[safe_rel(extra_out, dest_root)] = str(extra_src)

        graph_cache = zhang_source / "graph_cache"
        if graph_cache.exists():
            for graph in sorted(graph_cache.rglob("*.npz")):
                out = dst / "graphs" / graph.relative_to(graph_cache)
                copy_file(graph, out)
                files.append(safe_rel(out, dest_root))
                source_files[safe_rel(out, dest_root)] = str(graph)

    for optional in ["archive", "log"]:
        if optional in cfg:
            path = Path(cfg[optional])
            out = dst / "raw_run_artifacts" / path.name
            if copy_file(path, out):
                files.append(safe_rel(out, dest_root))
                source_files[safe_rel(out, dest_root)] = str(path)

    supplement_manifests: dict[str, dict] = {}
    for supplement_name, supplement_src_raw in cfg.get("supplements", {}).items():
        supplement_src = Path(supplement_src_raw)
        if not supplement_src.exists():
            raise FileNotFoundError(f"{universe}: missing supplement {supplement_name}: {supplement_src}")
        supplement_dst = dst / "supplements" / supplement_name
        supplement_files: list[str] = []
        for name in SUPPLEMENT_FILES:
            out = supplement_dst / name
            if copy_file(supplement_src / name, out):
                rel = safe_rel(out, dest_root)
                files.append(rel)
                supplement_files.append(rel)
                source_files[rel] = str(supplement_src / name)
        pred_dst = supplement_dst / "predictions"
        for pred in sorted(supplement_src.glob("pred_*.npy")):
            out = pred_dst / pred.name
            copy_file(pred, out)
            rel = safe_rel(out, dest_root)
            files.append(rel)
            supplement_files.append(rel)
            source_files[rel] = str(pred)
        for manifest_src in sorted(supplement_src.glob("*manifest*.json")):
            out = supplement_dst / manifest_src.name
            copy_file(manifest_src, out)
            rel = safe_rel(out, dest_root)
            files.append(rel)
            supplement_files.append(rel)
            source_files[rel] = str(manifest_src)
        supplement_truth = supplement_dst / "truth.npy"
        supplement_dates = supplement_dst / "test_dates.npy"
        supplement_preds = sorted((supplement_dst / "predictions").glob("pred_*.npy"))
        date_summary = {}
        if supplement_dates.exists():
            dates = pd.to_datetime(np.load(supplement_dates, allow_pickle=True)).normalize()
            date_summary = {
                "n_test_dates": int(len(dates)),
                "start": str(dates[0].date()) if len(dates) else None,
                "end": str(dates[-1].date()) if len(dates) else None,
            }
        supplement_manifests[supplement_name] = {
            "source_root": str(supplement_src),
            "truth": npy_summary(supplement_truth),
            "prediction_file_count": len(supplement_preds),
            "date_summary": date_summary,
            "files": sorted(supplement_files),
        }

    run_manifest = read_json(src / "run_manifest.json")
    run_config = read_json(src / "run_config.json")
    truth = dst / "arrays" / "truth.npy"
    preds = sorted((dst / "predictions").glob("pred_*.npy"))

    diag_files = sorted((dst / "diagnostics").glob("*.csv")) if (dst / "diagnostics").exists() else []
    table_files = sorted((dst / "tables").glob("*.csv")) if (dst / "tables").exists() else []
    graph_files = sorted((dst / "graphs").rglob("*.npz")) if (dst / "graphs").exists() else []

    universe_manifest = {
        "universe": universe,
        "source_note": cfg["source_note"],
        "source_root": str(src),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "truth": npy_summary(truth),
        "prediction_file_count": len(preds),
        "table_file_count": len(table_files),
        "diagnostic_file_count": len(diag_files),
        "graph_matrix_file_count": len(graph_files),
        "run_manifest_summary": {
            "universe": run_manifest.get("universe"),
            "files_count": len(run_manifest.get("files", [])) if isinstance(run_manifest.get("files"), list) else None,
            "diagnostics": run_manifest.get("diagnostics", {}),
        },
        "run_config_summary": {
            "run_id": run_config.get("run_id"),
            "universe": run_config.get("universe"),
            "n_tickers": run_config.get("n_tickers"),
            "n_test_dates": run_config.get("n_test_dates"),
            "notes": run_config.get("notes"),
        },
        "tables": {p.name: csv_summary(p) for p in table_files},
        "diagnostics": {p.name: csv_summary(p) for p in diag_files},
        "supplements": supplement_manifests,
        "files": sorted(files),
        "source_files": source_files,
    }
    manifest_path = dst / "universe_manifest.json"
    manifest_path.write_text(json.dumps(universe_manifest, indent=2, default=str), encoding="utf-8")
    files.append(safe_rel(manifest_path, dest_root))
    return universe_manifest


def write_readme(dest: Path, manifests: dict[str, dict]) -> None:
    lines = [
        "# GNNHAR Paper-Ready Results",
        "",
        f"Created UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This folder is the structured analysis entrypoint for Colab and report generation.",
        "It is a copy of selected run outputs; the original Colab and AutoDL result folders are unchanged.",
        "",
        "## Layout",
        "",
        "- `universes/<universe>/arrays/`: `truth.npy`, `tickers.npy`, and `test_dates.npy`.",
        "- `universes/<universe>/predictions/`: one `pred_*.npy` file per model.",
        "- `universes/<universe>/tables/`: main loss, DM, FVU, and multi-hop tables.",
        "- `universes/<universe>/diagnostics/`: MCS, market-regime, data-integrity, and smoothing diagnostics.",
        "- `universes/<universe>/graphs/`: exported GLASSO adjacency matrices when available.",
        "- `universes/<universe>/zhang_source/`: loss-specific Zhang-style source tables when available.",
        "- `universes/<universe>/supplements/`: date-aligned supplemental runs that should not be silently merged into arrays with a different test-date index.",
        "- `manifest.json`: machine-readable inventory.",
        "- `checksums_sha256.txt`: checksum audit for every copied file.",
        "",
        "## Universe Summary",
        "",
        "| Universe | Truth shape | Predictions | Tables | Diagnostics | Graph matrices | Source |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for universe, manifest in manifests.items():
        truth_shape = manifest.get("truth", {}).get("shape")
        lines.append(
            "| {u} | {shape} | {preds} | {tables} | {diags} | {graphs} | {source} |".format(
                u=universe,
                shape=str(truth_shape),
                preds=manifest.get("prediction_file_count"),
                tables=manifest.get("table_file_count"),
                diags=manifest.get("diagnostic_file_count"),
                graphs=manifest.get("graph_matrix_file_count"),
                source=manifest.get("source_note"),
            )
        )
    lines += [
        "",
        "## Colab Loading",
        "",
        "Use this folder as the stable entrypoint after mounting Google Drive in Colab:",
        "",
        "```python",
        "from pathlib import Path",
        "import json",
        "import numpy as np",
        "import pandas as pd",
        "",
        "RESULT_ROOT = Path('/content/drive/MyDrive/GNNHAR_Research/results/paper_ready_20260617')",
        "",
        "def load_universe(universe):",
        "    root = RESULT_ROOT / 'universes' / universe",
        "    truth = np.load(root / 'arrays' / 'truth.npy')",
        "    tickers = np.load(root / 'arrays' / 'tickers.npy', allow_pickle=True)",
        "    test_dates = np.load(root / 'arrays' / 'test_dates.npy', allow_pickle=True)",
        "    predictions = {p.stem.removeprefix('pred_'): np.load(p) for p in sorted((root / 'predictions').glob('pred_*.npy'))}",
        "    tables = {p.stem: pd.read_csv(p) for p in sorted((root / 'tables').glob('*.csv'))}",
        "    diagnostics = {p.stem: pd.read_csv(p) for p in sorted((root / 'diagnostics').glob('*.csv')) if p.stat().st_size > 0}",
        "    manifest = json.loads((root / 'universe_manifest.json').read_text())",
        "    return truth, tickers, test_dates, predictions, tables, diagnostics, manifest",
        "",
        "truth, tickers, test_dates, predictions, tables, diagnostics, manifest = load_universe('sp500')",
        "print(truth.shape, len(predictions), sorted(tables))",
        "```",
        "",
        "For SP500 GLASSO matrices:",
        "",
        "```python",
        "graph_files = sorted((RESULT_ROOT / 'universes' / 'sp500' / 'graphs').rglob('*.npz'))",
        "graph = np.load(graph_files[0], allow_pickle=True)",
        "W = graph['adjacency']",
        "info = json.loads(str(graph['info_json']))",
        "print(len(graph_files), W.shape, info.keys())",
        "```",
        "",
        "The model names are encoded in prediction filenames. For example, `pred_GNNHAR5L_QLIKE_IV.npy` is the SP500 prediction array for the 5-layer IV GNNHAR trained under QLIKE.",
        "",
        "## Notes",
        "",
        "- `hidden_state_mad.csv` may be empty for runs that did not save hidden representations; use `mad_smoothing_diagnostics.csv` and `oversmoothing_depth_summary.csv` as prediction-level smoothing diagnostics.",
        "- SP500 graph matrices are normalized adjacency matrices used by the models, not raw precision matrices.",
    ]
    (dest / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_checksums(dest: Path) -> None:
    rows = []
    for path in sorted(dest.rglob("*")):
        if path.is_file() and path.name != "checksums_sha256.txt":
            rows.append(f"{sha256(path)}  {path.relative_to(dest)}")
    (dest / "checksums_sha256.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dest = args.dest
    if dest.exists() and args.overwrite:
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    manifests = {universe: build_universe(universe, cfg, dest) for universe, cfg in SOURCES.items()}
    root_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "destination": str(dest),
        "universes": manifests,
    }
    (dest / "manifest.json").write_text(json.dumps(root_manifest, indent=2, default=str), encoding="utf-8")
    write_readme(dest, manifests)
    write_checksums(dest)
    print(json.dumps({"destination": str(dest), "universes": list(manifests)}, indent=2))


if __name__ == "__main__":
    main()
