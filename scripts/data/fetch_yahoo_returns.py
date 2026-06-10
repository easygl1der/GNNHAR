#!/usr/bin/env python3
"""Fetch adjusted-close daily returns from Yahoo chart API."""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd


def unix_date(date_text: str) -> int:
    dt = datetime.fromisoformat(date_text).replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def read_symbols(constituents_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(constituents_csv)
    if "YahooSymbol" not in df.columns:
        df["YahooSymbol"] = df["Symbol"].astype(str).str.replace(".", "-", regex=False)
    return df


def fetch_adj_close(yahoo_symbol: str, start: str, end: str, timeout: int) -> pd.Series:
    # Include one extra prior calendar week so the first requested return can be
    # computed from the previous trading close.
    start_dt = datetime.fromisoformat(start).date() - timedelta(days=7)
    end_dt = datetime.fromisoformat(end).date() + timedelta(days=1)
    params = urllib.parse.urlencode(
        {
            "period1": unix_date(start_dt.isoformat()),
            "period2": unix_date(end_dt.isoformat()),
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(yahoo_symbol)}?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    result = payload["chart"]["result"][0]
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators", {})
    adjclose = indicators.get("adjclose", [{}])[0].get("adjclose") or []
    dates = [datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat() for ts in timestamps]
    return pd.Series(pd.to_numeric(adjclose, errors="coerce"), index=pd.Index(dates, name="Date"), name=yahoo_symbol)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constituents-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--failures-ok", action="store_true")
    args = parser.parse_args()

    constituents = read_symbols(Path(args.constituents_csv))
    closes: Dict[str, pd.Series] = {}
    failures: List[str] = []
    for pos, row in enumerate(constituents.itertuples(index=False), start=1):
        symbol = str(row.Symbol)
        yahoo_symbol = str(row.YahooSymbol)
        try:
            close = fetch_adj_close(yahoo_symbol, args.start, args.end, args.timeout)
            returns = close.sort_index().pct_change()
            returns = returns[(returns.index >= args.start) & (returns.index <= args.end)]
            returns.name = symbol
            closes[symbol] = returns
            print(f"{pos}/{len(constituents)} {symbol}/{yahoo_symbol}: {returns.notna().sum()} returns", flush=True)
        except Exception as exc:
            msg = f"{symbol}/{yahoo_symbol}: {type(exc).__name__}: {exc}"
            failures.append(msg)
            print(msg, flush=True)
            if not args.failures_ok:
                raise
        time.sleep(args.sleep)

    panel = pd.concat(closes.values(), axis=1).sort_index().copy()
    panel.index.name = "Date"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.reset_index().to_csv(output, index=False)
    metadata = {
        "constituents_csv": str(Path(args.constituents_csv).resolve()),
        "output": str(output.resolve()),
        "start": args.start,
        "end": args.end,
        "tickers_succeeded": len(closes),
        "failures": failures,
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
