#!/usr/bin/env python3
"""Fetch AlphaQuery option-statistic chart JSON into project CSV panels.

This uses the same public chart endpoint loaded by the AlphaQuery web page. In
an unauthenticated/free session the endpoint currently returns only the recent
short window shown on the web chart. Longer history requires AlphaQuery/VolVue
subscription access and should not be bypassed.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List

import pandas as pd


ALPHAQUERY_CHART_URL = "https://www.alphaquery.com/data/option-statistic-chart"

INDICATOR_TO_OUTPUT = {
    "iv-mean": "merged_iv_data_filled.csv",
    "historical-volatility": "merged_rv_data_filled.csv",
    "parkinson-historical-volatility": "merged_parkinson_hv_data_filled.csv",
    "iv-call": "merged_iv_call_data_filled.csv",
    "iv-put": "merged_iv_put_data_filled.csv",
    "put-call-iv-ratio": "merged_put_call_iv_ratio_data_filled.csv",
    "iv-mean-skew": "merged_iv_mean_skew_data_filled.csv",
    "put-call-ratio-volume": "merged_put_call_ratio_volume_data_filled.csv",
    "put-call-ratio-oi": "merged_put_call_ratio_oi_data_filled.csv",
}


def read_tickers(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8").replace("\n", ",")
    return [part.strip() for part in text.split(",") if part.strip()]


def fetch_chart(ticker: str, indicator: str, per_type: str, timeout: int) -> pd.Series:
    params = urllib.parse.urlencode(
        {
            "ticker": ticker,
            "perType": per_type,
            "identifier": indicator,
        }
    )
    url = f"{ALPHAQUERY_CHART_URL}?{params}"
    referer = (
        "https://www.alphaquery.com/stock/"
        f"{ticker}/volatility-option-statistics/{per_type.lower()}/{indicator}"
    )
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/javascript,*/*;q=0.01",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": referer,
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected payload for {ticker} {indicator}: {type(payload)}")
    if not payload:
        return pd.Series(dtype=float, name=ticker)
    dates = [pd.to_datetime(row["x"]).date().isoformat() for row in payload]
    values = pd.to_numeric([row.get("value") for row in payload], errors="coerce")
    return pd.Series(values, index=pd.Index(dates, name="Date"), name=ticker, dtype=float)


def load_cached_series(path: Path, ticker: str) -> pd.Series:
    df = pd.read_json(path)
    if df.empty:
        return pd.Series(dtype=float, name=ticker)
    date_col = "Date" if "Date" in df.columns else "index"
    value_cols = [col for col in df.columns if col != date_col]
    if not value_cols:
        return pd.Series(dtype=float, name=ticker)
    values = pd.to_numeric(df[value_cols[0]], errors="coerce")
    dates = pd.to_datetime(df[date_col]).dt.date.astype(str)
    return pd.Series(values.to_numpy(dtype=float), index=pd.Index(dates, name="Date"), name=ticker)


def write_wide_panel(series_by_ticker: Dict[str, pd.Series], path: Path) -> None:
    if not series_by_ticker:
        pd.DataFrame(columns=["Date"]).to_csv(path, index=False)
        return
    panel = pd.concat(series_by_ticker.values(), axis=1).sort_index().copy()
    panel.index.name = "Date"
    panel.reset_index().to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-type", default="30-Day")
    parser.add_argument(
        "--indicators",
        nargs="+",
        default=["iv-mean", "historical-volatility"],
        choices=sorted(INDICATOR_TO_OUTPUT),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.35)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--failures-ok", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="ignore cached raw JSON and refetch everything")
    args = parser.parse_args()

    tickers = read_tickers(Path(args.tickers_file))
    if args.limit:
        tickers = tickers[: args.limit]

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw_alphaquery_json"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "tickers_file": str(Path(args.tickers_file).resolve()),
        "n_tickers_requested": len(tickers),
        "per_type": args.per_type,
        "indicators": args.indicators,
        "note": "Unauthenticated AlphaQuery chart endpoint currently returns the free recent chart window only.",
        "failures": [],
        "counts": {},
    }

    for indicator in args.indicators:
        series_by_ticker: Dict[str, pd.Series] = {}
        for pos, ticker in enumerate(tickers, start=1):
            raw_path = raw_dir / f"{ticker}_{indicator}.json"
            source = "failed"
            try:
                if raw_path.exists() and not args.no_resume:
                    series = load_cached_series(raw_path, ticker)
                    source = "cached"
                else:
                    series = fetch_chart(ticker, indicator, args.per_type, args.timeout)
                    raw_path.write_text(
                        series.reset_index().to_json(orient="records"),
                        encoding="utf-8",
                    )
                    source = "fetched"
                series_by_ticker[ticker] = series
                print(f"[{indicator}] {pos}/{len(tickers)} {ticker}: {len(series)} rows ({source})", flush=True)
            except Exception as exc:
                message = f"{ticker} {indicator}: {type(exc).__name__}: {exc}"
                metadata["failures"].append(message)
                print(f"[{indicator}] {pos}/{len(tickers)} {message}", flush=True)
                if not args.failures_ok:
                    raise
            if source == "fetched":
                time.sleep(args.sleep)

        write_wide_panel(series_by_ticker, output_dir / INDICATOR_TO_OUTPUT[indicator])
        metadata["counts"][indicator] = {
            "tickers_succeeded": len(series_by_ticker),
            "min_rows": int(min((len(s) for s in series_by_ticker.values()), default=0)),
            "max_rows": int(max((len(s) for s in series_by_ticker.values()), default=0)),
        }

    (output_dir / "alphaquery_fetch_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
