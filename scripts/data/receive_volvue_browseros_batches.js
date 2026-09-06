#!/usr/bin/env node
/**
 * Receive authenticated VolVue fetch batches from a BrowserOS page.
 *
 * This avoids copying BrowserOS cookies into shell commands. The browser page
 * fetches VolVue with its own session and POSTs decoded JSON batches here.
 */

const fs = require("fs/promises");
const http = require("http");
const path = require("path");

function parseArgs(argv) {
  const args = {
    out: "data/raw/volvue_sp500_10d_5y_browseros",
    expected: 1006,
    port: 17880,
    startDate: "2021-06-09",
    endDate: "2026-06-18",
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--out") args.out = argv[++i];
    else if (arg === "--expected") args.expected = Number(argv[++i]);
    else if (arg === "--port") args.port = Number(argv[++i]);
    else if (arg === "--start-date") args.startDate = argv[++i];
    else if (arg === "--end-date") args.endDate = argv[++i];
    else throw new Error(`Unknown argument: ${arg}`);
  }
  return args;
}

function dateFromMs(ms) {
  return new Date(ms).toISOString().slice(0, 10);
}

function csvEscape(value) {
  if (value == null || Number.isNaN(value)) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

async function writeJson(file, value) {
  await fs.mkdir(path.dirname(file), { recursive: true });
  await fs.writeFile(file, JSON.stringify(value, null, 2), "utf8");
}

async function writeWide(fieldName, records, outDir, startDate, endDate) {
  const dateSet = new Set();
  const byTicker = {};
  for (const record of records) {
    const series = {};
    for (const row of record.data || []) {
      const date = dateFromMs(row[0]);
      if (startDate && date < startDate) continue;
      if (endDate && date > endDate) continue;
      dateSet.add(date);
      series[date] = row[1];
    }
    byTicker[record.ticker] = series;
  }
  const dates = Array.from(dateSet).sort();
  const tickers = Object.keys(byTicker).sort();
  const lines = [["Date", ...tickers].map(csvEscape).join(",")];
  for (const date of dates) {
    lines.push([date, ...tickers.map((ticker) => byTicker[ticker][date])].map(csvEscape).join(","));
  }
  const file = path.join(outDir, "wide", `${fieldName}.csv`);
  await fs.mkdir(path.dirname(file), { recursive: true });
  await fs.writeFile(file, `${lines.join("\n")}\n`, "utf8");
  return { file, nDates: dates.length, dateStart: dates[0] || null, dateEnd: dates[dates.length - 1] || null, nTickers: tickers.length };
}

async function main() {
  const args = parseArgs(process.argv);
  const rawDir = path.join(args.out, "raw_json");
  await fs.mkdir(rawDir, { recursive: true });
  const records = [];
  const failures = [];
  let finalized = false;

  async function finalize(reason) {
    if (finalized) return;
    finalized = true;
    const byField = {};
    for (const record of records) {
      if (!byField[record.fieldName]) byField[record.fieldName] = [];
      byField[record.fieldName].push(record);
    }
    const fields = {};
    for (const [fieldName, fieldRecords] of Object.entries(byField)) {
      fields[fieldName] = await writeWide(fieldName, fieldRecords, args.out, args.startDate, args.endDate);
      fields[fieldName].successCount = fieldRecords.length;
    }
    const metadata = {
      source: "BrowserOS page-context fetch of https://volvue.com/service/ticker-data-chart",
      requestedAt: new Date().toISOString(),
      reason,
      expected: args.expected,
      received: records.length,
      failureCount: failures.length,
      failures,
      startDate: args.startDate,
      endDate: args.endDate,
      fields,
    };
    await writeJson(path.join(args.out, "metadata.json"), metadata);
    console.log(JSON.stringify(metadata, null, 2));
    server.close(() => process.exit(failures.length ? 2 : 0));
  }

  const server = http.createServer((req, res) => {
    if (req.method === "GET" && req.url === "/status") {
      res.writeHead(200, { "content-type": "application/json", "access-control-allow-origin": "*" });
      res.end(JSON.stringify({ received: records.length, failures: failures.length, expected: args.expected }));
      return;
    }
    if (req.method !== "POST" || !["/record", "/failure", "/done"].includes(req.url)) {
      res.writeHead(404, { "access-control-allow-origin": "*" });
      res.end();
      return;
    }
    let body = "";
    req.setEncoding("utf8");
    req.on("data", (chunk) => { body += chunk; });
    req.on("end", async () => {
      try {
        const payload = body ? JSON.parse(body) : {};
        if (req.url === "/record") {
          records.push(payload);
          await writeJson(path.join(rawDir, `${payload.ticker}_${payload.fieldName}.json`), payload);
          if (records.length % 25 === 0 || records.length === args.expected) {
            console.log(`received ${records.length}/${args.expected}, failures ${failures.length}`);
          }
        } else if (req.url === "/failure") {
          failures.push(payload);
          console.log(`failure ${payload.ticker} ${payload.fieldName}: ${payload.error}`);
        } else if (req.url === "/done") {
          res.writeHead(200, { "content-type": "application/json", "access-control-allow-origin": "*" });
          res.end(JSON.stringify({ ok: true }));
          await finalize("browser_done");
          return;
        }
        res.writeHead(200, { "content-type": "application/json", "access-control-allow-origin": "*" });
        res.end(JSON.stringify({ ok: true }));
      } catch (error) {
        res.writeHead(500, { "content-type": "application/json", "access-control-allow-origin": "*" });
        res.end(JSON.stringify({ error: error.message }));
      }
    });
  });

  server.listen(args.port, "127.0.0.1", () => {
    console.log(`listening http://127.0.0.1:${args.port}, expected ${args.expected}`);
  });

  process.on("SIGINT", () => finalize("sigint"));
  process.on("SIGTERM", () => finalize("sigterm"));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
