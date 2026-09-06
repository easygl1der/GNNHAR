#!/usr/bin/env node
/**
 * Download VolVue 10-day IV/HV chart data for an S&P 500 ticker list.
 *
 * The VolVue chart endpoint returns AES-GCM encrypted data plus a response key.
 * This script mirrors the frontend decoder from the public app bundle and writes
 * both raw responses and wide Date x ticker panels.
 */

const crypto = require("crypto").webcrypto;
const fs = require("fs/promises");
const path = require("path");

const FIELDS = {
  implied_volatility_mean_10d: "implied-volatility-mean",
  historical_volatility_close_to_close_10d: "historical-volatility-close-to-close",
};

const KEY_PERMUTATION = [
  37, 22, 10, 57, 28, 33, 3, 5, 52, 54, 38, 8, 43, 29, 26, 47,
  59, 25, 14, 36, 12, 60, 40, 21, 20, 45, 31, 34, 16, 44, 0, 7,
  15, 6, 41, 55, 23, 1, 24, 50, 53, 9, 13, 49, 2, 35, 30, 32,
  63, 51, 46, 39, 4, 11, 19, 61, 58, 56, 42, 62, 18, 27, 17, 48,
];

function aesName() {
  const chars = [
    String.fromCharCode("F".charCodeAt(0) - 3),
    String.fromCharCode("H".charCodeAt(0) - 27),
    String.fromCharCode("I".charCodeAt(0) + 10),
    String.fromCharCode("Z".charCodeAt(0) - 19),
    String.fromCharCode("V".charCodeAt(0) - 21),
    String.fromCharCode("B".charCodeAt(0) + 11),
    String.fromCharCode("T".charCodeAt(0) - 15),
  ];
  return [chars[4], chars[6], chars[2], chars[1], chars[3], chars[0], chars[5]].join("");
}

async function decodePayload(data, keyString) {
  if (!keyString) return data;
  const encrypted = Uint8Array.from(Buffer.from(data, "base64"));
  const keyBytes = Uint8Array.from(Buffer.from(keyString, "base64"));
  const rawKey = new Uint8Array(32);
  for (let i = 0; i < rawKey.length; i += 1) rawKey[i] = keyBytes[KEY_PERMUTATION[i]];
  const key = await crypto.subtle.importKey("raw", rawKey, { name: aesName() }, false, ["decrypt"]);
  const iv = encrypted.slice(0, 12);
  const cipherText = encrypted.slice(12);
  const decrypted = await crypto.subtle.decrypt({ name: aesName(), iv }, key, cipherText);
  return JSON.parse(new TextDecoder().decode(decrypted));
}

function parseArgs(argv) {
  const args = {
    tickers: "data/scale_experiment/sp500/tickers.txt",
    out: null,
    concurrency: 8,
    startDate: null,
    endDate: null,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--tickers") args.tickers = argv[++i];
    else if (arg === "--out") args.out = argv[++i];
    else if (arg === "--concurrency") args.concurrency = Number(argv[++i]);
    else if (arg === "--start-date") args.startDate = argv[++i];
    else if (arg === "--end-date") args.endDate = argv[++i];
    else throw new Error(`Unknown argument: ${arg}`);
  }
  if (!args.out) {
    const stamp = new Date().toISOString().slice(0, 10).replace(/-/g, "");
    args.out = `data/raw/volvue_sp500_10d_5y_${stamp}`;
  }
  return args;
}

async function readTickers(file) {
  const text = await fs.readFile(file, "utf8");
  return text.split(/\r?\n/).map((x) => x.trim()).filter(Boolean);
}

function dateFromMs(ms) {
  return new Date(ms).toISOString().slice(0, 10);
}

function csvEscape(value) {
  if (value == null || Number.isNaN(value)) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

async function fetchField(ticker, urlName) {
  const url = new URL("https://volvue.com/service/ticker-data-chart");
  url.searchParams.set("ticker", ticker);
  url.searchParams.set("timeFrame", "10-day");
  url.searchParams.set("urlName", urlName);
  const response = await fetch(url, {
    headers: {
      accept: "application/json,text/plain,*/*",
      "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 Chrome/148 Safari/537.36",
      referer: `https://volvue.com/ticker/${ticker}/10-day/${urlName}`,
    },
  });
  if (!response.ok) throw new Error(`HTTP ${response.status} ${response.statusText}`);
  const json = await response.json();
  if (json.error) throw new Error(json.error);
  const rows = await decodePayload(json.data, json.k);
  return { ...json, data: rows };
}

async function withRetries(fn, attempts = 3) {
  let last;
  for (let i = 0; i < attempts; i += 1) {
    try {
      return await fn();
    } catch (error) {
      last = error;
      await new Promise((resolve) => setTimeout(resolve, 750 * (i + 1)));
    }
  }
  throw last;
}

async function mapLimit(items, limit, worker) {
  const results = new Array(items.length);
  let index = 0;
  async function run() {
    while (index < items.length) {
      const current = index;
      index += 1;
      results[current] = await worker(items[current], current);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, run));
  return results;
}

async function writeWide(fieldName, tickerRows, outDir, startDate, endDate) {
  const dateSet = new Set();
  const byTicker = {};
  for (const [ticker, rows] of Object.entries(tickerRows)) {
    const series = {};
    for (const row of rows) {
      const date = dateFromMs(row[0]);
      if (startDate && date < startDate) continue;
      if (endDate && date > endDate) continue;
      dateSet.add(date);
      series[date] = row[1];
    }
    byTicker[ticker] = series;
  }
  const dates = Array.from(dateSet).sort();
  const tickers = Object.keys(tickerRows).sort();
  const lines = [["Date", ...tickers].map(csvEscape).join(",")];
  for (const date of dates) {
    lines.push([date, ...tickers.map((ticker) => byTicker[ticker][date])].map(csvEscape).join(","));
  }
  const file = path.join(outDir, `${fieldName}.csv`);
  await fs.writeFile(file, `${lines.join("\n")}\n`, "utf8");
  return { file, nDates: dates.length, dateStart: dates[0] || null, dateEnd: dates[dates.length - 1] || null };
}

async function main() {
  const args = parseArgs(process.argv);
  const tickers = await readTickers(args.tickers);
  const rawDir = path.join(args.out, "raw_json");
  const wideDir = path.join(args.out, "wide");
  await fs.mkdir(rawDir, { recursive: true });
  await fs.mkdir(wideDir, { recursive: true });

  const metadata = {
    source: "https://volvue.com/service/ticker-data-chart",
    timeFrame: "10-day",
    requestedAt: new Date().toISOString(),
    tickersFile: args.tickers,
    tickerCount: tickers.length,
    startDate: args.startDate,
    endDate: args.endDate,
    fields: {},
  };

  for (const [fieldName, urlName] of Object.entries(FIELDS)) {
    console.log(`Downloading ${fieldName} (${urlName}) for ${tickers.length} tickers`);
    const tickerRows = {};
    const failures = [];
    await mapLimit(tickers, args.concurrency, async (ticker, i) => {
      try {
        const payload = await withRetries(() => fetchField(ticker, urlName));
        tickerRows[ticker] = payload.data;
        const rawPath = path.join(rawDir, `${ticker}_${fieldName}.json`);
        await fs.writeFile(rawPath, JSON.stringify(payload, null, 2), "utf8");
        if ((i + 1) % 25 === 0 || i + 1 === tickers.length) {
          console.log(`  ${fieldName}: ${i + 1}/${tickers.length}`);
        }
      } catch (error) {
        failures.push({ ticker, error: error.message });
        console.log(`  FAILED ${ticker} ${fieldName}: ${error.message}`);
      }
    });
    const wide = await writeWide(fieldName, tickerRows, wideDir, args.startDate, args.endDate);
    metadata.fields[fieldName] = {
      urlName,
      successCount: Object.keys(tickerRows).length,
      failureCount: failures.length,
      failures,
      wide,
    };
  }

  await fs.writeFile(path.join(args.out, "metadata.json"), JSON.stringify(metadata, null, 2), "utf8");
  console.log(JSON.stringify(metadata, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
