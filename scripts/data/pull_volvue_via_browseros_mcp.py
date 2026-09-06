#!/usr/bin/env python3
"""Pull authenticated VolVue ticker chart data through BrowserOS MCP."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FIELDS = {
    "implied_volatility_mean_10d": "implied-volatility-mean",
    "historical_volatility_close_to_close_10d": "historical-volatility-close-to-close",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default="data/scale_experiment/sp500/tickers.txt")
    parser.add_argument("--out", default="data/raw/volvue_sp500_10d_5y_20260619")
    parser.add_argument("--page", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--start-date", default="2021-06-09")
    parser.add_argument("--end-date", default="2026-06-18")
    return parser.parse_args()


def browseros_call(page: int, expression: str, timeout: int = 120) -> dict[str, Any]:
    payload = {
        "jsonrpc": "2.0",
        "id": int(time.time() * 1000) % 1_000_000_000,
        "method": "tools/call",
        "params": {
            "name": "evaluate_script",
            "arguments": {"page": page, "expression": expression},
        },
    }
    raw = subprocess.check_output(
        [
            "/usr/bin/curl",
            "-sS",
            "--max-time",
            str(timeout),
            "-X",
            "POST",
            "http://127.0.0.1:9000/mcp",
            "-H",
            "Content-Type: application/json",
            "-H",
            "Accept: application/json, text/event-stream",
            "--data",
            json.dumps(payload),
        ],
        text=True,
    )
    response = json.loads(raw)
    if response.get("result", {}).get("isError"):
        text = response["result"]["content"][0].get("text", "")
        raise RuntimeError(text)
    text = response["result"]["content"][0]["text"]
    return json.loads(text)


def fetch_batch(page: int, jobs: list[dict[str, str]]) -> dict[str, Any]:
    expression = f"""
(async () => {{
  const jobs = {json.dumps(jobs)};
  const out = [];
  const failures = [];
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  async function getJsonWithRetry(job, attempts = 3) {{
    const url = '/service/ticker-data-chart?ticker=' + encodeURIComponent(job.ticker)
      + '&timeFrame=10-day&urlName=' + encodeURIComponent(job.urlName);
    let last;
    for (let i = 0; i < attempts; i++) {{
      try {{
        const response = await fetch(url, {{credentials: 'same-origin'}});
        if (!response.ok) throw new Error('HTTP ' + response.status + ' ' + response.statusText);
        const json = await response.json();
        if (json.error) throw new Error(json.error);
        return json;
      }} catch (error) {{
        last = error;
        await sleep(500 * (i + 1));
      }}
    }}
    throw last;
  }}
  for (const job of jobs) {{
    try {{
      const json = await getJsonWithRetry(job);
      out.push({{
        ticker: job.ticker,
        fieldName: job.fieldName,
        urlName: job.urlName,
        columnNames: json.columnNames,
        columnTypes: json.columnTypes,
        field: json.field,
        data: json.data,
        fetchedAt: new Date().toISOString()
      }});
    }} catch (error) {{
      failures.push({{
        ticker: job.ticker,
        fieldName: job.fieldName,
        urlName: job.urlName,
        error: String(error),
        fetchedAt: new Date().toISOString()
      }});
    }}
    await sleep(60);
  }}
  return JSON.stringify({{records: out, failures}});
}})()
"""
    return browseros_call(page, expression)


def date_from_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date().isoformat()


def write_wide(records: list[dict[str, Any]], out_dir: Path, start_date: str, end_date: str) -> dict[str, Any]:
    dates: set[str] = set()
    by_ticker: dict[str, dict[str, float | None]] = {}
    for record in records:
        series: dict[str, float | None] = {}
        for row in record.get("data", []):
            date = date_from_ms(row[0])
            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
                continue
            dates.add(date)
            series[date] = row[1]
        by_ticker[record["ticker"]] = series

    ordered_dates = sorted(dates)
    tickers = sorted(by_ticker)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{records[0]['fieldName']}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Date", *tickers])
        for date in ordered_dates:
            writer.writerow([date, *[by_ticker[ticker].get(date) for ticker in tickers]])
    return {
        "file": str(path),
        "nDates": len(ordered_dates),
        "dateStart": ordered_dates[0] if ordered_dates else None,
        "dateEnd": ordered_dates[-1] if ordered_dates else None,
        "nTickers": len(tickers),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    raw_dir = out / "raw_json"
    raw_dir.mkdir(parents=True, exist_ok=True)

    tickers = [line.strip() for line in Path(args.tickers).read_text().splitlines() if line.strip()]
    jobs = [
        {"ticker": ticker, "fieldName": field_name, "urlName": url_name}
        for ticker in tickers
        for field_name, url_name in FIELDS.items()
    ]
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    remaining_jobs: list[dict[str, str]] = []

    for job in jobs:
        cache_file = raw_dir / f"{job['ticker']}_{job['fieldName']}.json"
        if cache_file.exists():
            records.append(json.loads(cache_file.read_text(encoding="utf-8")))
        else:
            remaining_jobs.append(job)

    if records:
        print(f"resuming with {len(records)} cached records; {len(remaining_jobs)} jobs remain", flush=True)

    for start in range(0, len(remaining_jobs), args.batch_size):
        batch = remaining_jobs[start : start + args.batch_size]
        result = fetch_batch(args.page, batch)
        for record in result["records"]:
            records.append(record)
            (raw_dir / f"{record['ticker']}_{record['fieldName']}.json").write_text(
                json.dumps(record, indent=2),
                encoding="utf-8",
            )
        failures.extend(result["failures"])
        done = len(records) + len(failures)
        print(f"done {done}/{len(jobs)} records={len(records)} failures={len(failures)}", flush=True)

    fields: dict[str, Any] = {}
    for field_name in FIELDS:
        field_records = [record for record in records if record["fieldName"] == field_name]
        fields[field_name] = write_wide(field_records, out / "wide", args.start_date, args.end_date)
        fields[field_name]["successCount"] = len(field_records)
        fields[field_name]["failureCount"] = len([f for f in failures if f["fieldName"] == field_name])

    metadata = {
        "source": "BrowserOS MCP page-context fetch of https://volvue.com/service/ticker-data-chart",
        "requestedAt": datetime.now(timezone.utc).isoformat(),
        "tickersFile": args.tickers,
        "tickerCount": len(tickers),
        "jobCount": len(jobs),
        "successCount": len(records),
        "failureCount": len(failures),
        "failures": failures,
        "startDate": args.start_date,
        "endDate": args.end_date,
        "fields": fields,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
