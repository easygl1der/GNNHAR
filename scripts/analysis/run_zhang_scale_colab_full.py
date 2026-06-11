#!/usr/bin/env python3
"""Colab driver for the full Zhang-style rolling scale experiment.

The core pipeline is intentionally one-universe/one-loss.  This wrapper runs the
complete Dow30/SP100/SP500 by MSE/QLIKE matrix, records progress after each
sub-run, and builds the combined summary when possible.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_ROOT = (
    Path("/content/GNNHAR-colab-runs") / time.strftime("scale-zhang-full-%Y%m%dT%H%M%SZ", time.gmtime())
    if Path("/content").exists()
    else REPO_ROOT / "outputs" / "zhang-scale-full" / time.strftime("scale-zhang-full-%Y%m%dT%H%M%SZ", time.gmtime())
)

DEFAULT_MODELS = ",".join(
    [
        "HAR",
        "GHAR",
        "GHAR2H",
        "GHAR3H",
        "HAR+IV",
        "GHAR+IV",
        "GHAR2H+IV",
        "GHAR3H+IV",
        "GNNHAR1L",
        "GNNHAR2L",
        "GNNHAR3L",
        "GNNHAR4L",
        "GNNHAR5L",
        "GNNHAR1L-IV",
        "GNNHAR2L-IV",
        "GNNHAR3L-IV",
        "GNNHAR4L-IV",
        "GNNHAR5L-IV",
    ]
)


@dataclass(frozen=True)
class UniverseConfig:
    key: str
    label: str
    data_dir: str
    returns_file: str
    coverage_threshold: float


UNIVERSES = {
    "dow30": UniverseConfig(
        key="dow30",
        label="Dow30",
        data_dir="experiments/dow30/data",
        returns_file="experiments/dow30/data/dow30_daily_returns_2021_2026.csv",
        coverage_threshold=0.99,
    ),
    "sp100": UniverseConfig(
        key="sp100",
        label="SP100",
        data_dir="data/scale_experiment/sp100",
        returns_file="data/scale_experiment/sp100/daily_returns.csv",
        coverage_threshold=0.95,
    ),
    "sp500": UniverseConfig(
        key="sp500",
        label="SP500",
        data_dir="data/scale_experiment/sp500",
        returns_file="data/scale_experiment/sp500/daily_returns.csv",
        coverage_threshold=0.99,
    ),
}


def parse_csv(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--universes", default="dow30,sp100,sp500", help="Comma-separated subset of dow30,sp100,sp500")
    parser.add_argument("--losses", default="MSE,QLIKE", help="Comma-separated subset of MSE,QLIKE")
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--hidden-grid", default="9,16")
    parser.add_argument("--lr-grid", default="0.001,0.0003")
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-nn", type=int, default=1)
    parser.add_argument("--mcs-bootstrap", type=int, default=60)
    parser.add_argument("--lookback", type=int, default=1000)
    parser.add_argument("--window", type=int, default=22)
    parser.add_argument("--valid-len", type=int, default=22)
    parser.add_argument("--block-stride", type=int, default=22)
    parser.add_argument("--max-blocks", type=int, default=0)
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument("--fill-limit", type=int, default=5)
    parser.add_argument("--max-neighbors", type=int, default=0)
    parser.add_argument("--graph-method", choices=["glasso_cv", "glasso", "corr"], default="glasso_cv")
    parser.add_argument("--skip-figures", action="store_true", default=True)
    parser.add_argument("--write-figures", action="store_true", help="Generate per-run figures instead of skipping them")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-gpu-name", default="", help="Fail if nvidia-smi does not contain this string")
    parser.add_argument("--allow-missing-gpu", action="store_true")
    return parser.parse_args()


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def run_streaming(cmd: Sequence[str], cwd: Path, log_path: Path, dry_run: bool) -> int:
    print("+ " + " ".join(cmd), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return 0
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return int(proc.wait())


def check_gpu(require_name: str, allow_missing: bool) -> dict:
    payload = {"required": require_name, "available": False, "name": "", "raw": ""}
    if not require_name and allow_missing:
        return payload
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        if allow_missing:
            return payload
        raise SystemExit("nvidia-smi is not available; cannot verify the requested Colab GPU")
    payload["raw"] = (res.stdout + res.stderr).strip()
    names = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    payload["available"] = bool(names)
    payload["name"] = names[0] if names else ""
    if require_name and not any(require_name.lower() in name.lower() for name in names):
        raise SystemExit(f"Required GPU containing {require_name!r}, but nvidia-smi reported {names!r}")
    if not names and not allow_missing:
        raise SystemExit("No GPU was reported by nvidia-smi")
    return payload


def complete_run(path: Path) -> bool:
    return (path / "tables" / "model_losses.csv").exists() and (path / "run_metadata.json").exists()


def selected_universes(keys: Iterable[str]) -> List[UniverseConfig]:
    configs: List[UniverseConfig] = []
    for key in keys:
        normalized = key.lower()
        if normalized not in UNIVERSES:
            raise ValueError(f"Unknown universe {key!r}; expected one of {sorted(UNIVERSES)}")
        configs.append(UNIVERSES[normalized])
    return configs


def pipeline_command(args: argparse.Namespace, universe: UniverseConfig, loss: str, output_dir: Path) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/analysis/gnnhar_iv_zhang_scale_pipeline.py",
        "--universe-name",
        universe.key,
        "--data-dir",
        universe.data_dir,
        "--returns-file",
        universe.returns_file,
        "--output-dir",
        str(output_dir),
        "--coverage-threshold",
        str(universe.coverage_threshold),
        "--models",
        args.models,
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
        "--fill-limit",
        str(args.fill_limit),
        "--graph-method",
        args.graph_method,
        "--loss",
        loss,
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


def summarize_existing(out_root: Path, configs: List[UniverseConfig], losses: List[str], dry_run: bool) -> None:
    run_dirs: List[Path] = []
    labels: List[str] = []
    for universe in configs:
        for loss in losses:
            path = out_root / "full" / universe.key / loss
            if complete_run(path):
                run_dirs.append(path)
                labels.append(f"{universe.label}-{loss}")
    if not run_dirs:
        print("No complete run directories found for summary.", flush=True)
        return
    summary_dir = out_root / "summary"
    cmd = [
        sys.executable,
        "scripts/analysis/summarize_zhang_scale_experiment.py",
        "--run-dirs",
        *[str(path) for path in run_dirs],
        "--labels",
        *labels,
        "--output-dir",
        str(summary_dir),
    ]
    rc = run_streaming(cmd, REPO_ROOT, out_root / "logs" / "summary.log", dry_run)
    if rc != 0:
        raise SystemExit(f"Summary failed with exit code {rc}")


def main() -> None:
    args = parse_args()
    if args.write_figures:
        args.skip_figures = False
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    progress_path = out_root / "driver_progress.jsonl"
    configs = selected_universes(parse_csv(args.universes))
    losses = [loss.upper() for loss in parse_csv(args.losses)]
    unknown_losses = sorted(set(losses).difference({"MSE", "QLIKE"}))
    if unknown_losses:
        raise ValueError(f"Unknown losses: {unknown_losses}")

    gpu = check_gpu(args.require_gpu_name, args.allow_missing_gpu)
    append_jsonl(
        progress_path,
        {
            "event": "driver_start",
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "output_root": str(out_root),
            "gpu": gpu,
            "universes": [cfg.key for cfg in configs],
            "losses": losses,
            "models": parse_csv(args.models),
        },
    )
    print(json.dumps({"output_root": str(out_root), "gpu": gpu}, indent=2), flush=True)

    for universe in configs:
        for loss in losses:
            output_dir = out_root / "full" / universe.key / loss
            log_path = out_root / "logs" / f"{universe.key}_{loss}.log"
            if complete_run(output_dir) and not args.rerun_existing:
                append_jsonl(progress_path, {"event": "skip_existing", "universe": universe.key, "loss": loss})
                print(f"Skipping existing complete run: {output_dir}", flush=True)
                continue
            append_jsonl(
                progress_path,
                {
                    "event": "run_start",
                    "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "universe": universe.key,
                    "loss": loss,
                    "output_dir": str(output_dir),
                },
            )
            started = time.time()
            rc = run_streaming(pipeline_command(args, universe, loss, output_dir), REPO_ROOT, log_path, args.dry_run)
            elapsed = time.time() - started
            append_jsonl(
                progress_path,
                {
                    "event": "run_end" if rc == 0 else "run_error",
                    "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "universe": universe.key,
                    "loss": loss,
                    "returncode": rc,
                    "elapsed_sec": round(elapsed, 2),
                    "output_dir": str(output_dir),
                    "log_path": str(log_path),
                },
            )
            if rc != 0:
                raise SystemExit(f"{universe.key} {loss} failed with exit code {rc}; see {log_path}")

    summarize_existing(out_root, configs, losses, args.dry_run)
    append_jsonl(
        progress_path,
        {
            "event": "driver_end",
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "output_root": str(out_root),
        },
    )
    print(f"Driver finished. Output root: {out_root}", flush=True)


if __name__ == "__main__":
    main()
