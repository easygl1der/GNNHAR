#!/usr/bin/env python3
"""Run SP500 Zhang-style workers in parallel and merge model outputs.

Each worker runs the existing rolling pipeline for a subset of models.  The
workers share a graph cache so every rolling block estimates the GLASSO graph
once and the other workers wait for the cached adjacency.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GROUPS = {
    "linear": "HAR,GHAR,GHAR2H,GHAR3H,HAR+IV,GHAR+IV,GHAR2H+IV,GHAR3H+IV",
    "gnn_low": "GNNHAR1L,GNNHAR2L",
    "gnn_high": "GNNHAR3L,GNNHAR4L,GNNHAR5L",
    "gnn_iv": "GNNHAR1L-IV,GNNHAR2L-IV,GNNHAR3L-IV,GNNHAR4L-IV,GNNHAR5L-IV",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--losses", default="MSE,QLIKE")
    parser.add_argument("--groups", default="linear,gnn_low,gnn_high,gnn_iv")
    parser.add_argument("--data-dir", default="data/scale_experiment/sp500")
    parser.add_argument("--returns-file", default="data/scale_experiment/sp500/daily_returns.csv")
    parser.add_argument("--coverage-threshold", type=float, default=0.99)
    parser.add_argument("--hidden-grid", default="9")
    parser.add_argument("--lr-grid", default="0.001")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-nn", type=int, default=1)
    parser.add_argument("--mcs-bootstrap", type=int, default=10000)
    parser.add_argument("--lookback", type=int, default=1000)
    parser.add_argument("--window", type=int, default=22)
    parser.add_argument("--valid-len", type=int, default=22)
    parser.add_argument("--block-stride", type=int, default=22)
    parser.add_argument("--max-blocks", type=int, default=0)
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument("--max-neighbors", type=int, default=0)
    parser.add_argument("--graph-method", choices=["glasso_cv", "glasso", "corr"], default="glasso_cv")
    parser.add_argument("--graph-cache-dir", default="")
    parser.add_argument("--require-gpu-name", default="")
    parser.add_argument("--allow-missing-gpu", action="store_true")
    parser.add_argument("--skip-figures", action="store_true", default=True)
    parser.add_argument("--write-figures", action="store_true")
    return parser.parse_args()


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def check_gpu(require_name: str, allow_missing: bool) -> dict:
    payload = {"required": require_name, "available": False, "raw": ""}
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        if allow_missing:
            payload["raw"] = "nvidia-smi not found"
            return payload
        raise SystemExit("nvidia-smi not found")
    payload["raw"] = (res.stdout + res.stderr).strip()
    names = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    payload["available"] = bool(names)
    payload["names"] = names
    if not names:
        if allow_missing:
            return payload
        raise SystemExit("No GPU detected by nvidia-smi")
    if require_name and not any(require_name.lower() in name.lower() for name in names):
        raise SystemExit(f"Required GPU containing {require_name!r}, got {names!r}")
    return payload


def group_models(group: str) -> str:
    if group not in GROUPS:
        raise ValueError(f"Unknown group {group!r}; expected one of {sorted(GROUPS)}")
    return GROUPS[group]


def pipeline_cmd(args: argparse.Namespace, loss: str, group: str, output_dir: Path, graph_cache_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/analysis/gnnhar_iv_zhang_scale_pipeline.py",
        "--universe-name",
        "sp500",
        "--data-dir",
        args.data_dir,
        "--returns-file",
        args.returns_file,
        "--output-dir",
        str(output_dir),
        "--coverage-threshold",
        str(args.coverage_threshold),
        "--models",
        group_models(group),
        "--hidden-grid",
        args.hidden_grid,
        "--lr-grid",
        args.lr_grid,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-nn",
        str(args.num_nn),
        "--mcs-bootstrap",
        str(args.mcs_bootstrap),
        "--lookback",
        str(args.lookback),
        "--window",
        str(args.window),
        "--valid-len",
        str(args.valid_len),
        "--block-stride",
        str(args.block_stride),
        "--graph-method",
        args.graph_method,
        "--graph-cache-dir",
        str(graph_cache_dir),
        "--worker-output-only",
        "--loss",
        loss,
        "--horizon",
        "1",
    ]
    if args.max_blocks > 0:
        cmd.extend(["--max-blocks", str(args.max_blocks)])
    if args.max_tickers > 0:
        cmd.extend(["--max-tickers", str(args.max_tickers)])
    if args.max_neighbors > 0:
        cmd.extend(["--max-neighbors", str(args.max_neighbors)])
    if args.skip_figures and not args.write_figures:
        cmd.append("--skip-figures")
    return cmd


def start_worker(cmd: list[str], log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    proc._gnnhar_log_file = log  # type: ignore[attr-defined]
    return proc


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def wait_workers(workers: dict[str, subprocess.Popen], progress_path: Path) -> None:
    failures = []
    for name, proc in workers.items():
        rc = int(proc.wait())
        log = getattr(proc, "_gnnhar_log_file", None)
        if log is not None:
            log.close()
        event = {"event": "worker_end", "worker": name, "returncode": rc, "time": time.time()}
        append_jsonl(progress_path, event)
        if rc != 0:
            failures.append((name, rc))
    if failures:
        raise SystemExit(f"Workers failed: {failures}")


def load_npz(path: Path) -> dict[str, object]:
    import numpy as np

    payload = np.load(path, allow_pickle=True)
    return {key: payload[key] for key in payload.files}


def merge_loss_outputs(worker_dirs: dict[str, Path], merged_dir: Path) -> None:
    import numpy as np
    import pandas as pd
    from scripts.analysis.gnnhar_iv_pipeline import build_iv_decomposition, build_mcs_table, build_regime_table, evaluate_runs, save_tables
    from scripts.analysis.gnnhar_iv_scale_pipeline import build_dm_table
    from scripts.analysis.gnnhar_iv_zhang_scale_pipeline import EPS, ModelRun, model_adjacency, model_family, model_iv_channel

    merged_dir.mkdir(parents=True, exist_ok=True)
    (merged_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (merged_dir / "tables").mkdir(parents=True, exist_ok=True)
    tables: dict[str, list[pd.DataFrame]] = {}
    arrays: dict[str, object] = {}
    base_truth = base_dates = base_tickers = None
    metadata = None
    source_dirs = {}

    for group, run_dir in worker_dirs.items():
        source_dirs[group] = str(run_dir)
        payload = load_npz(run_dir / "predictions_test.npz")
        truth = payload["truth"]
        dates = payload["dates"]
        tickers = payload["tickers"]
        if base_truth is None:
            base_truth, base_dates, base_tickers = truth, dates, tickers
            arrays["truth"] = truth
            arrays["dates"] = dates
            arrays["tickers"] = tickers
        else:
            if truth.shape != base_truth.shape or not np.array_equal(dates, base_dates) or not np.array_equal(tickers, base_tickers):
                raise ValueError(f"Panel mismatch in worker {group}")
            if not np.allclose(truth, base_truth, equal_nan=True):
                raise ValueError(f"Truth mismatch in worker {group}")
        for key, value in payload.items():
            if key.startswith("pred_"):
                arrays[key] = value
        for table in (run_dir / "tables").glob("*.csv"):
            try:
                frame = pd.read_csv(table)
            except pd.errors.EmptyDataError:
                continue
            tables.setdefault(table.stem, []).append(frame)
        if metadata is None and (run_dir / "run_metadata.json").exists():
            metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        for pred_csv in (run_dir / "predictions").glob("*.csv"):
            if pred_csv.name == "truth.csv":
                continue
            shutil.copy2(pred_csv, merged_dir / "predictions" / pred_csv.name)

    if base_truth is None:
        raise ValueError("No worker predictions found")
    np.savez_compressed(merged_dir / "predictions_test.npz", **arrays)
    shutil.copy2(next(iter(worker_dirs.values())) / "predictions" / "truth.csv", merged_dir / "predictions" / "truth.csv")

    # The source npz key transform is not reversible for every name. Re-read the
    # model names from worker metadata and map them through the same key rule.
    pred_arrays = {}
    for group, run_dir in worker_dirs.items():
        payload = load_npz(run_dir / "predictions_test.npz")
        if (run_dir / "run_metadata.json").exists():
            meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
            for model in meta.get("args", {}).get("models", "").split(","):
                model = model.strip()
                if not model:
                    continue
                key = "pred_" + model.replace("+", "plus").replace("-", "_")
                if key in payload:
                    pred_arrays[model] = payload[key]

    for name, frames in tables.items():
        if name in {"model_losses", "loss_ratios", "dm_tests", "mcs_results", "iv_decomposition", "regime_results"}:
            continue
        merged = frames[0].copy() if name == "graph_blocks" else pd.concat(frames, ignore_index=True)
        merged.to_csv(merged_dir / "tables" / f"{name}.csv", index=False)
        try:
            merged.to_latex(merged_dir / "tables" / f"{name}.tex", index=False, float_format="%.6g")
        except Exception:
            pass

    if metadata is None:
        metadata = {}
    test_idx = np.arange(base_truth.shape[0])
    eval_panel = type(
        "EvalPanel",
        (),
        {
            "dates": pd.to_datetime(base_dates),
            "tickers": [str(ticker) for ticker in base_tickers],
            "target": np.asarray(base_truth),
            "split": {"test": test_idx},
        },
    )()
    estimation = "QLIKE" if "QLIKE" in str(merged_dir) else "MSE"
    runs = [
        ModelRun(
            name=name,
            family=model_family(name),
            iv_channel=model_iv_channel(name),
            adjacency=model_adjacency(name),
            estimation=estimation,
            prediction=np.clip(pred, EPS, None),
        )
        for name, pred in sorted(pred_arrays.items())
    ]
    loss_table, losses = evaluate_runs(runs, eval_panel)
    ratio_table = loss_table[
        [
            "model",
            "estimation",
            "mse_ratio_vs_har",
            "qlike_ratio_vs_har",
            "mse_ratio_vs_har_iv",
            "qlike_ratio_vs_har_iv",
        ]
    ].copy()
    try:
        dm_table = build_dm_table(losses["qlike"])
    except Exception as exc:
        dm_table = pd.DataFrame([{"note": f"DM table unavailable: {type(exc).__name__}: {exc}"}])
    mcs_table = build_mcs_table(losses["qlike"], int(metadata.get("args", {}).get("mcs_bootstrap", 80)))
    iv_decomposition = build_iv_decomposition(loss_table)
    regime_table = build_regime_table(runs, eval_panel)
    save_tables(
        {
            "model_losses": loss_table,
            "loss_ratios": ratio_table,
            "dm_tests": dm_table,
            "mcs_results": mcs_table,
            "iv_decomposition": iv_decomposition,
            "regime_results": regime_table,
        },
        merged_dir,
    )
    metadata["parallel_workers"] = {
        "source_dirs": source_dirs,
        "merged": True,
        "merged_dir": str(merged_dir),
        "models": sorted(pred_arrays),
    }
    (merged_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    coverage = next(iter(worker_dirs.values())) / "ticker_coverage.csv"
    if coverage.exists():
        shutil.copy2(coverage, merged_dir / "ticker_coverage.csv")
    report_src = next(iter(worker_dirs.values())) / "report"
    if report_src.exists():
        shutil.copytree(report_src, merged_dir / "report", dirs_exist_ok=True)


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    progress_path = out_root / "parallel_progress.jsonl"
    append_jsonl(progress_path, {"event": "parallel_start", "run_id": args.run_id, "gpu": check_gpu(args.require_gpu_name, args.allow_missing_gpu), "time": time.time()})

    groups = parse_csv(args.groups)
    losses = [loss.upper() for loss in parse_csv(args.losses)]
    for loss in losses:
        graph_cache_dir = Path(args.graph_cache_dir) if args.graph_cache_dir else out_root / "graph_cache" / loss
        worker_dirs = {group: out_root / "workers" / loss / group for group in groups}
        workers = {}
        for group, output_dir in worker_dirs.items():
            cmd = pipeline_cmd(args, loss, group, output_dir, graph_cache_dir)
            append_jsonl(progress_path, {"event": "worker_start", "loss": loss, "group": group, "output_dir": str(output_dir), "time": time.time()})
            workers[f"{loss}:{group}"] = start_worker(cmd, out_root / "logs" / f"{loss.lower()}_{group}.log")
        wait_workers(workers, progress_path)
        merged_dir = out_root / "full" / "sp500" / loss / "H1"
        merge_loss_outputs(worker_dirs, merged_dir)
        append_jsonl(progress_path, {"event": "loss_merged", "loss": loss, "merged_dir": str(merged_dir), "time": time.time()})

    append_jsonl(progress_path, {"event": "parallel_end", "run_id": args.run_id, "time": time.time()})
    print(json.dumps({"output_root": str(out_root), "losses": losses, "groups": groups}, indent=2))


if __name__ == "__main__":
    main()
