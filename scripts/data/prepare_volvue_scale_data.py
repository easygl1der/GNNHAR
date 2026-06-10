#!/usr/bin/env python3
"""Prepare VolVue S&P 100 / S&P 500 wide panels for scale experiments."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict

import pandas as pd


DATASETS: Dict[str, Dict[str, str]] = {
    "sp100": {
        "rv": "sp100_historical_volatility_close_to_close_30d_wide_clean.csv",
        "iv": "sp100_implied_volatility_mean_30d_wide_clean.csv",
    },
    "sp500": {
        "rv": "sp500_historical_volatility_close_to_close_30d_wide_clean.csv",
        "iv": "sp500_implied_volatility_mean_30d_wide_clean.csv",
    },
}


def read_wide(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df = df.rename(columns={"date": "Date"})
    if "Date" not in df.columns:
        raise ValueError(f"{path} must contain a date or Date column")
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    ordered = ["Date"] + sorted([col for col in df.columns if col != "Date"])
    return df[ordered]


def write_dataset(name: str, source_root: Path, output_root: Path) -> dict:
    spec = DATASETS[name]
    rv = read_wide(source_root / spec["rv"])
    iv = read_wide(source_root / spec["iv"])
    rv_tickers = set(rv.columns) - {"Date"}
    iv_tickers = set(iv.columns) - {"Date"}
    tickers = sorted(rv_tickers & iv_tickers)
    if not tickers:
        raise ValueError(f"{name}: no common RV/IV tickers")

    out_dir = output_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rv_out = rv[["Date", *tickers]]
    iv_out = iv[["Date", *tickers]]
    rv_out.to_csv(out_dir / "merged_rv_data_filled.csv", index=False)
    iv_out.to_csv(out_dir / "merged_iv_data_filled.csv", index=False)
    (out_dir / "tickers.txt").write_text("\n".join(tickers) + "\n", encoding="utf-8")

    summary = {
        "name": name,
        "source_root": str(source_root),
        "n_dates_rv": int(len(rv_out)),
        "n_dates_iv": int(len(iv_out)),
        "n_tickers": int(len(tickers)),
        "date_start": str(max(rv_out["Date"].min(), iv_out["Date"].min())),
        "date_end": str(min(rv_out["Date"].max(), iv_out["Date"].max())),
        "rv_missing_fraction": float(rv_out.drop(columns=["Date"]).isna().mean().mean()),
        "iv_missing_fraction": float(iv_out.drop(columns=["Date"]).isna().mean().mean()),
        "tickers": tickers,
    }
    (out_dir / "volvue_panel_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", default="data/scale_experiment")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = [write_dataset(name, source_root, output_root) for name in DATASETS]
    audit = source_root / "processing_audit.json"
    if audit.exists():
        shutil.copy2(audit, output_root / "volvue_processing_audit.json")
    (output_root / "scale_data_manifest.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
