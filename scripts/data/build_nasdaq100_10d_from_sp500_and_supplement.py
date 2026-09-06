#!/usr/bin/env python3
"""Build Nasdaq-100 10-day VolVue panels from S&P 500 data plus supplements."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


FIELDS = [
    "implied_volatility_mean_10d",
    "historical_volatility_close_to_close_10d",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default="tmp/nasdaq100_tickers.txt")
    parser.add_argument("--sp500-root", default="data/raw/volvue_sp500_10d_5y_20260619")
    parser.add_argument("--supplement-root", default="data/raw/volvue_nasdaq100_missing_10d_5y_20260619")
    parser.add_argument("--out", default="data/raw/volvue_nasdaq100_10d_5y_20260619")
    return parser.parse_args()


def read_wide(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        tickers = header[1:]
        data = {ticker: {} for ticker in tickers}
        dates = []
        for row in reader:
            date = row[0]
            dates.append(date)
            for ticker, value in zip(tickers, row[1:]):
                data[ticker][date] = value
    return dates, data


def write_wide(path: Path, dates: list[str], tickers: list[str], data_by_ticker: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Date", *tickers])
        for date in dates:
            writer.writerow([date, *[data_by_ticker[ticker].get(date, "") for ticker in tickers]])


def missing_fraction(path: Path) -> float:
    total = 0
    missing = 0
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            for cell in row[1:]:
                total += 1
                if cell == "":
                    missing += 1
    return missing / total if total else 0.0


def main() -> None:
    args = parse_args()
    tickers = [line.strip() for line in Path(args.tickers).read_text().splitlines() if line.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    metadata = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "tickersFile": args.tickers,
        "tickerCount": len(tickers),
        "sp500Root": args.sp500_root,
        "supplementRoot": args.supplement_root,
        "fields": {},
    }

    for field in FIELDS:
        sp_dates, sp_data = read_wide(Path(args.sp500_root) / "wide" / f"{field}.csv")
        sup_dates, sup_data = read_wide(Path(args.supplement_root) / "wide" / f"{field}.csv")
        dates = sorted(set(sp_dates) | set(sup_dates))
        combined = {}
        source = {}
        missing = []
        for ticker in tickers:
            if ticker in sp_data:
                combined[ticker] = sp_data[ticker]
                source[ticker] = "sp500"
            elif ticker in sup_data:
                combined[ticker] = sup_data[ticker]
                source[ticker] = "supplement"
            else:
                combined[ticker] = {}
                source[ticker] = "missing"
                missing.append(ticker)
        output = out / "wide" / f"{field}.csv"
        write_wide(output, dates, tickers, combined)
        metadata["fields"][field] = {
            "file": str(output),
            "nDates": len(dates),
            "dateStart": dates[0] if dates else None,
            "dateEnd": dates[-1] if dates else None,
            "nTickers": len(tickers),
            "sp500Covered": sum(1 for ticker in tickers if source[ticker] == "sp500"),
            "supplementCovered": sum(1 for ticker in tickers if source[ticker] == "supplement"),
            "missingTickers": missing,
            "missingFraction": missing_fraction(output),
        }

    (out / "nasdaq100_tickers.txt").write_text("\n".join(tickers) + "\n", encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
