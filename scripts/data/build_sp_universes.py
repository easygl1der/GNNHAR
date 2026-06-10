#!/usr/bin/env python3
"""Build current S&P 100 and S&P 500 ticker lists for scale experiments."""

from __future__ import annotations

import argparse
import io
import urllib.request
from pathlib import Path

import pandas as pd


SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
SP100_URL = "https://en.wikipedia.org/wiki/S%26P_100"


def read_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def normalize_symbol(symbol: str) -> str:
    # AlphaQuery uses the index-style dotted class-share symbols, e.g. BRK.B.
    return str(symbol).strip()


def yahoo_symbol(symbol: str) -> str:
    # Yahoo Finance uses dash for class shares, e.g. BRK.B -> BRK-B.
    return normalize_symbol(symbol).replace(".", "-")


def write_universe(df: pd.DataFrame, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["Symbol"] = df["Symbol"].map(normalize_symbol)
    df["YahooSymbol"] = df["Symbol"].map(yahoo_symbol)
    df = df.drop_duplicates("Symbol").sort_values("Symbol").reset_index(drop=True)
    df.to_csv(output_dir / "constituents.csv", index=False)
    (output_dir / "tickers.txt").write_text(",".join(df["Symbol"].tolist()) + "\n", encoding="utf-8")
    print(f"{name}: {len(df)} tickers -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="data/processed")
    args = parser.parse_args()

    output_root = Path(args.output_root)

    sp500 = pd.read_csv(io.BytesIO(read_url(SP500_URL)))
    write_universe(sp500, output_root / "sp500", "sp500")

    sp100_tables = pd.read_html(io.BytesIO(read_url(SP100_URL)))
    sp100 = next(table for table in sp100_tables if "Symbol" in table.columns and "Name" in table.columns)
    write_universe(sp100, output_root / "sp100", "sp100")


if __name__ == "__main__":
    main()
