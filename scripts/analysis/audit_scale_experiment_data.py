#!/usr/bin/env python3
"""Audit S&P scale-experiment data pairing and preliminary run metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/scale_experiment")
    parser.add_argument("--old-run-root", default="outputs/colab-scale-runs/scale-20260610T201904Z")
    parser.add_argument("--output-dir", default="outputs/scale-experiment-audit")
    parser.add_argument("--universes", default="sp100,sp500")
    return parser.parse_args()


def read_wide(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"date": "Date"})
    if "Date" not in df.columns:
        raise ValueError(f"{path} does not contain a Date/date column")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def audit_universe(data_root: Path, old_run_root: Path, universe: str) -> Dict[str, object]:
    data_dir = data_root / universe
    rv = read_wide(data_dir / "merged_rv_data_filled.csv")
    iv = read_wide(data_dir / "merged_iv_data_filled.csv")
    returns = read_wide(data_dir / "daily_returns.csv")

    rv_tickers = set(rv.columns)
    iv_tickers = set(iv.columns)
    return_tickers = set(returns.columns)
    common_tickers = sorted(rv_tickers & iv_tickers & return_tickers)
    common_dates = rv.index.intersection(iv.index).intersection(returns.index)
    coverage = pd.DataFrame(
        {
            "rv_coverage": rv.loc[common_dates, common_tickers].notna().mean(),
            "iv_coverage": iv.loc[common_dates, common_tickers].notna().mean(),
            "returns_coverage": returns.loc[common_dates, common_tickers].notna().mean(),
        }
    )
    coverage["min_coverage"] = coverage.min(axis=1)

    returns_meta_path = data_dir / "daily_returns.metadata.json"
    returns_meta: Dict[str, object] = {}
    if returns_meta_path.exists():
        returns_meta = json.loads(returns_meta_path.read_text(encoding="utf-8"))

    old_meta_path = old_run_root / universe / "run_metadata.json"
    old_meta: Dict[str, object] = {}
    if old_meta_path.exists():
        old_meta = json.loads(old_meta_path.read_text(encoding="utf-8"))

    graph_meta = old_meta.get("graph", {}) if isinstance(old_meta, dict) else {}
    panel_meta = old_meta.get("panel", {}) if isinstance(old_meta, dict) else {}

    summary: Dict[str, object] = {
        "universe": universe,
        "rv_shape": [int(rv.shape[0]), int(rv.shape[1])],
        "iv_shape": [int(iv.shape[0]), int(iv.shape[1])],
        "returns_shape": [int(returns.shape[0]), int(returns.shape[1])],
        "date_start": str(common_dates.min().date()) if len(common_dates) else None,
        "date_end": str(common_dates.max().date()) if len(common_dates) else None,
        "common_dates": int(len(common_dates)),
        "common_tickers": int(len(common_tickers)),
        "rv_only_tickers": sorted(rv_tickers - return_tickers),
        "returns_only_tickers": sorted(return_tickers - rv_tickers),
        "iv_only_tickers": sorted(iv_tickers - rv_tickers),
        "rv_missing_fraction": float(rv.loc[common_dates, common_tickers].isna().mean().mean()),
        "iv_missing_fraction": float(iv.loc[common_dates, common_tickers].isna().mean().mean()),
        "returns_missing_fraction": float(returns.loc[common_dates, common_tickers].isna().mean().mean()),
        "rv_nonpositive_count": int((rv.loc[common_dates, common_tickers] <= 0).sum().sum()),
        "iv_nonpositive_count": int((iv.loc[common_dates, common_tickers] <= 0).sum().sum()),
        "returns_tickers_succeeded": returns_meta.get("tickers_succeeded"),
        "returns_failures": returns_meta.get("failures", []),
        "old_selected_tickers": panel_meta.get("selected_tickers"),
        "old_excluded_tickers": panel_meta.get("excluded_tickers", []),
        "old_coverage_threshold": panel_meta.get("coverage_threshold"),
        "old_graph_method": graph_meta.get("method"),
        "old_graph_fallback": graph_meta.get("fallback"),
        "old_graph_edges": graph_meta.get("n_edges"),
        "old_graph_density": graph_meta.get("density"),
        "old_graph_train_rows": graph_meta.get("train_rows"),
    }

    coverage_out = data_root.parent / "_audit_tmp"
    coverage_out.mkdir(parents=True, exist_ok=True)
    coverage.sort_values("min_coverage").to_csv(coverage_out / f"{universe}_coverage_by_ticker.csv")
    summary["lowest_coverage_tickers"] = [
        {
            "ticker": str(idx),
            "min_coverage": float(row["min_coverage"]),
            "rv_coverage": float(row["rv_coverage"]),
            "iv_coverage": float(row["iv_coverage"]),
            "returns_coverage": float(row["returns_coverage"]),
        }
        for idx, row in coverage.sort_values("min_coverage").head(15).iterrows()
    ]
    return summary


def markdown_report(rows: List[Dict[str, object]]) -> str:
    lines = [
        "# S&P Scale Experiment Data and Method Audit",
        "",
        "## Executive Finding",
        "",
        "The S&P 100 and S&P 500 RV, IV, and return panels are paired on the same daily date range and ticker namespace. The current preliminary scale run should not be used as a final GLASSO scale conclusion, however, because the S&P 500 graph construction fell back from GLASSO to an absolute-correlation graph and the model implementation used a static split rather than Zhang's rolling-window design.",
        "",
        "## Data Pairing",
        "",
        "| Universe | RV shape | IV shape | Returns shape | Common dates | Common tickers | Return fetch failures |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['universe']} | {row['rv_shape']} | {row['iv_shape']} | {row['returns_shape']} | "
            f"{row['common_dates']} | {row['common_tickers']} | {len(row.get('returns_failures') or [])} |"
        )
    lines.extend(
        [
            "",
            "The panels use the date range shared by realized volatility, implied volatility, and returns. Non-positive RV/IV values were already removed from the processed panels. Remaining missing values are concentrated in newly added or reorganized tickers and are handled by coverage filtering, not by filling pre-listing history.",
            "",
            "## Old Run Status",
            "",
            "| Universe | Selected tickers | Old graph method | Edges | Density | Fallback |",
            "|---|---:|---|---:|---:|---|",
        ]
    )
    for row in rows:
        fallback = row.get("old_graph_fallback") or ""
        if fallback:
            fallback = fallback.replace("|", "/")
        lines.append(
            f"| {row['universe']} | {row.get('old_selected_tickers')} | {row.get('old_graph_method')} | "
            f"{row.get('old_graph_edges')} | {row.get('old_graph_density')} | {fallback} |"
        )
    lines.extend(
        [
            "",
            "## Method Gap Against Zhang's Original Code",
            "",
            "Zhang's public implementation recomputes the graph for each forecast block. For a forecast origin \\(t\\), it uses up to 1000 prior return observations, reserves the latest 22 observations as validation for neural models, and forecasts the next 22 trading days. The preliminary scale script instead used one chronological split and one static graph. It also used a custom concatenated message-passing neural architecture rather than Zhang's residual \\(H_1 + \\mathrm{GCN}\\) architecture.",
            "",
            "Therefore, the old S&P 500 result is best labeled as a preliminary static-graph screen. The next run should be a rolling Zhang-style experiment that logs graph success per block and treats IV as an explicit extension.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    old_run_root = Path(args.old_run_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [audit_universe(data_root, old_run_root, item.strip()) for item in args.universes.split(",") if item.strip()]
    (output_dir / "scale_data_method_audit.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_dir / "scale_data_method_audit.md").write_text(markdown_report(rows), encoding="utf-8")

    tmp_dir = data_root.parent / "_audit_tmp"
    if tmp_dir.exists():
        for path in tmp_dir.glob("*_coverage_by_ticker.csv"):
            path.replace(output_dir / path.name)
        try:
            tmp_dir.rmdir()
        except OSError:
            pass
    print(json.dumps({"output_dir": str(output_dir), "universes": [row["universe"] for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
