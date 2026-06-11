#!/usr/bin/env python3
"""Validate GNNHAR-IV scale experiment output directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_MODELS = {
    "HAR",
    "HAR+IV",
    "GHAR",
    "GHAR+IV",
    "GHAR2H",
    "GHAR2H+IV",
    "GHAR3H",
    "GHAR3H+IV",
    "GNNHAR1L",
    "GNNHAR1L-IV",
    "GNNHAR2L",
    "GNNHAR2L-IV",
    "GNNHAR3L",
    "GNNHAR3L-IV",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="outputs/colab-scale-runs/<run_id>")
    parser.add_argument("--universes", default="sp100,sp500")
    parser.add_argument("--require-compiled-report", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def require_finite(table: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    for column in columns:
        if column not in table:
            raise ValueError(f"{path}: missing column {column}")
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: non-finite values in {column}")


def validate_universe(run_dir: Path, universe: str) -> dict:
    root = run_dir / universe
    metadata_path = root / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    losses_path = root / "tables" / "model_losses.csv"
    losses = read_csv(losses_path)
    require_finite(losses, ["test_mse", "test_qlike", "mse_ratio_vs_har", "qlike_ratio_vs_har"], losses_path)
    models = set(losses["model"].astype(str))
    missing = sorted(REQUIRED_MODELS - models)
    if missing:
        raise ValueError(f"{losses_path}: missing required models {missing}")
    coverage_path = root / "ticker_coverage.csv"
    coverage = read_csv(coverage_path)
    if coverage.empty:
        raise ValueError(f"{coverage_path}: empty coverage table")
    report_path = root / "report" / "scale_experiment_report.md"
    if not report_path.exists() or report_path.stat().st_size == 0:
        raise ValueError(f"{report_path}: missing or empty report")
    return {
        "universe": universe,
        "n_tickers": metadata["n_tickers"],
        "n_dates": metadata["n_dates"],
        "best_model": str(losses.sort_values("test_qlike").iloc[0]["model"]),
        "best_qlike": float(losses["test_qlike"].min()),
        "model_count": int(len(losses)),
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    universes = [item.strip() for item in args.universes.split(",") if item.strip()]
    rows = [validate_universe(run_dir, universe) for universe in universes]
    summary_path = run_dir / "summary" / "scale_summary.csv"
    gains_path = run_dir / "summary" / "scale_gains.csv"
    read_csv(summary_path)
    read_csv(gains_path)
    if args.require_compiled_report:
        pdf_path = run_dir / "summary" / "report" / "scale_experiment_report.pdf"
        if not pdf_path.exists() or pdf_path.stat().st_size == 0:
            raise ValueError(f"{pdf_path}: missing compiled report")
    print(json.dumps({"run_dir": str(run_dir), "universes": rows}, indent=2))


if __name__ == "__main__":
    main()
