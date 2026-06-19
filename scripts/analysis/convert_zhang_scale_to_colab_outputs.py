#!/usr/bin/env python3
"""Convert Zhang-scale pipeline outputs into the Colab-style run layout.

The AutoDL runner uses ``gnnhar_iv_zhang_scale_pipeline.py`` because it is a
stable script entrypoint for long remote jobs.  That pipeline writes one folder
per training loss.  The Colab notebook and downstream report code expect a
single run folder with arrays named ``truth.npy`` and ``pred_<model>.npy``.

This converter merges the MSE-trained and QLIKE-trained folders into that
layout while preserving the original pipeline outputs.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


MODEL_RENAMES = {
    "HAR": "HAR",
    "GHAR": "GHAR",
    "GHAR2H": "GHAR2H",
    "GHAR3H": "GHAR3H",
    "GNNHAR1L": "GNNHAR1L",
    "GNNHAR2L": "GNNHAR2L",
    "GNNHAR3L": "GNNHAR3L",
    "GNNHAR4L": "GNNHAR4L",
    "GNNHAR5L": "GNNHAR5L",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="AutoDL output root containing full/<universe>/<loss>/H<horizon>")
    parser.add_argument("--dest-root", type=Path, required=True, help="Colab-style output root; universe/run_id will be created under it")
    parser.add_argument("--universe", default="sp500")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--run-id", default="", help="Optional run id; defaults to UTC timestamp")
    parser.add_argument("--baseline", default="HAR_M")
    return parser.parse_args()


def npz_key_for_model(model: str) -> str:
    return "pred_" + model.replace("+", "plus").replace("-", "_")


def colab_name(model: str, loss: str) -> str:
    loss_suffix = "M" if loss.upper() == "MSE" else "Q"
    uses_iv = "+IV" in model or model.endswith("-IV")
    base = model.replace("+IV", "").replace("-IV", "")
    base = MODEL_RENAMES.get(base, base)
    return f"{base}_{loss_suffix}_IV" if uses_iv else f"{base}_{loss_suffix}"


def load_loss_run(run_dir: Path, loss: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], pd.DataFrame]:
    npz_path = run_dir / "predictions_test.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    payload = np.load(npz_path, allow_pickle=True)
    truth = payload["truth"]
    dates = payload["dates"].astype(str)
    tickers = payload["tickers"].astype(str)
    losses = pd.read_csv(run_dir / "tables" / "model_losses.csv")

    preds: dict[str, np.ndarray] = {}
    for model in losses["model"].astype(str):
        if "fakeIV" in model:
            continue
        key = npz_key_for_model(model)
        if key not in payload:
            continue
        name = colab_name(model, loss)
        pred = payload[key]
        if pred.shape != truth.shape:
            raise ValueError(f"{run_dir}: {model} prediction shape {pred.shape} != truth shape {truth.shape}")
        preds[name] = pred.astype(np.float32)
    return truth.astype(np.float32), dates, tickers, preds, losses


def assert_same_panel(lhs: tuple[np.ndarray, np.ndarray, np.ndarray], rhs: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    truth_a, dates_a, tickers_a = lhs
    truth_b, dates_b, tickers_b = rhs
    if truth_a.shape != truth_b.shape:
        raise ValueError(f"truth shapes differ: {truth_a.shape} vs {truth_b.shape}")
    if not np.array_equal(dates_a, dates_b):
        raise ValueError("test dates differ between MSE and QLIKE source runs")
    if not np.array_equal(tickers_a, tickers_b):
        raise ValueError("tickers differ between MSE and QLIKE source runs")
    if not np.allclose(truth_a, truth_b, equal_nan=True):
        raise ValueError("truth arrays differ between MSE and QLIKE source runs")


def mse_loss(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.nanmean((pred - truth) ** 2))


def qlike_loss(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = np.clip(pred, 1e-10, None)
    truth = np.clip(truth, 1e-10, None)
    ratio = truth / pred
    return float(np.nanmean(ratio - np.log(np.clip(ratio, 1e-10, None)) - 1))


def qlike_array(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    pred = np.clip(pred, 1e-10, None)
    truth = np.clip(truth, 1e-10, None)
    ratio = truth / pred
    return ratio - np.log(np.clip(ratio, 1e-10, None)) - 1


def loss_table(truth: np.ndarray, preds: dict[str, np.ndarray], baseline: str) -> pd.DataFrame:
    if baseline not in preds:
        raise KeyError(f"baseline {baseline!r} not found in predictions")
    base_mse = mse_loss(preds[baseline], truth)
    base_qlike = qlike_loss(preds[baseline], truth)
    rows = []
    for model, pred in preds.items():
        model_mse = mse_loss(pred, truth)
        model_qlike = qlike_loss(pred, truth)
        rows.append(
            {
                "model": model,
                "mse": model_mse,
                "qlike": model_qlike,
                "mse_ratio": model_mse / base_mse,
                "qlike_ratio": model_qlike / base_qlike,
            }
        )
    return pd.DataFrame(rows).sort_values("qlike_ratio").reset_index(drop=True)


def dm_test(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[np.ndarray, np.ndarray]:
    d = e1 - e2
    d_bar = d.mean(axis=0)
    n_obs = d.shape[0]
    gamma0 = np.mean(d**2, axis=0) - d_bar**2
    gamma1 = np.mean(d[1:] * d[:-1], axis=0) if h > 1 else 0
    var_d = (gamma0 + 2 * gamma1) / n_obs
    dm_stat = d_bar / np.sqrt(np.maximum(var_d, 1e-20))
    corr = np.sqrt((n_obs + 1 - 2 * h + h * (h - 1) / n_obs) / n_obs)
    dm_corr = dm_stat * corr
    p_vals = 2 * (1 - stats.t.cdf(np.abs(dm_corr), df=n_obs - 1))
    return dm_stat, p_vals


def dm_tables(truth: np.ndarray, preds: dict[str, np.ndarray], baseline: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_sq = (preds[baseline] - truth) ** 2
    base_ql = qlike_array(preds[baseline], truth)
    rows = []
    for model, pred in preds.items():
        if model == baseline:
            continue
        sq = (pred - truth) ** 2
        ql = qlike_array(pred, truth)
        dm_mse, p_mse = dm_test(base_sq, sq)
        dm_ql, p_ql = dm_test(base_ql, ql)
        rows.append(
            {
                "model": model,
                "DM_MSE_avg": float(np.mean(dm_mse)),
                "p_MSE_avg": float(np.mean(p_mse)),
                "DM_QL_avg": float(np.mean(dm_ql)),
                "p_QL_avg": float(np.mean(p_ql)),
            }
        )

    baseline_rows = []
    for base, cand in depth_comparisons(preds):
        base_ql = qlike_array(preds[base], truth)
        cand_ql = qlike_array(preds[cand], truth)
        dm_ql, p_ql = dm_test(base_ql, cand_ql)
        baseline_rows.append(
            {
                "base_model": base,
                "candidate_model": cand,
                "DM_QL_avg_positive_favors_candidate": float(np.mean(dm_ql)),
                "p_QL_avg": float(np.mean(p_ql)),
                "mean_qlike_base": qlike_loss(preds[base], truth),
                "mean_qlike_candidate": qlike_loss(preds[cand], truth),
            }
        )
    return pd.DataFrame(rows).sort_values("p_QL_avg").reset_index(drop=True), pd.DataFrame(baseline_rows)


def depth_comparisons(preds: dict[str, np.ndarray]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for suffix in ["M", "Q", "M_IV", "Q_IV"]:
        for base, cand in [(f"GNNHAR1L_{suffix}", f"GNNHAR2L_{suffix}"), (f"GNNHAR2L_{suffix}", f"GNNHAR3L_{suffix}"), (f"GNNHAR3L_{suffix}", f"GNNHAR4L_{suffix}"), (f"GNNHAR4L_{suffix}", f"GNNHAR5L_{suffix}")]:
            if base in preds and cand in preds:
                pairs.append((base, cand))
        for base, cand in [(f"GHAR_{suffix}", f"GHAR2H_{suffix}"), (f"GHAR2H_{suffix}", f"GHAR3H_{suffix}")]:
            if base in preds and cand in preds:
                pairs.append((base, cand))
    return pairs


def fvu_table(truth: np.ndarray, preds: dict[str, np.ndarray], baseline: str) -> pd.DataFrame:
    base_pred = preds[baseline]
    rows = []
    for model, pred in preds.items():
        if model == baseline:
            continue
        diff = (base_pred - pred) ** 2
        denom = (pred - pred.mean(axis=1, keepdims=True)) ** 2
        fvu = diff.sum(axis=1) / np.maximum(denom.sum(axis=1), 1e-20)
        rows.append({"model": model, "FVU_mean": float(np.mean(fvu))})
    return pd.DataFrame(rows).sort_values("FVU_mean").reset_index(drop=True)


def copy_source_tables(source_dirs: dict[str, Path], dest: Path) -> list[str]:
    files = []
    for loss, src in source_dirs.items():
        table_dir = src / "tables"
        if not table_dir.exists():
            continue
        out = dest / "source_tables" / loss.lower()
        out.mkdir(parents=True, exist_ok=True)
        for table in table_dir.glob("*.csv"):
            target = out / table.name
            target.write_bytes(table.read_bytes())
            files.append(str(target.relative_to(dest)))
    return files


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    source_dirs = {
        "MSE": args.source_root / "full" / args.universe / "MSE" / f"H{args.horizon}",
        "QLIKE": args.source_root / "full" / args.universe / "QLIKE" / f"H{args.horizon}",
    }
    loaded = {loss: load_loss_run(path, loss) for loss, path in source_dirs.items()}
    assert_same_panel(loaded["MSE"][:3], loaded["QLIKE"][:3])

    truth, dates, tickers = loaded["MSE"][:3]
    preds: dict[str, np.ndarray] = {}
    for loss in ["MSE", "QLIKE"]:
        preds.update(loaded[loss][3])

    out_dir = args.dest_root / args.universe / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "truth.npy", truth)
    np.save(out_dir / "test_dates.npy", dates)
    np.save(out_dir / "tickers.npy", tickers)
    for model, pred in sorted(preds.items()):
        np.save(out_dir / f"pred_{model}.npy", pred.astype(np.float32))

    loss_df = loss_table(truth, preds, args.baseline)
    loss_df.to_csv(out_dir / "loss_table.csv", index=False)
    dm_df, depth_dm_df = dm_tables(truth, preds, args.baseline)
    dm_df.to_csv(out_dir / "dm_tests.csv", index=False)
    depth_dm_df.to_csv(out_dir / "dm_depth_tests.csv", index=False)
    fvu_df = fvu_table(truth, preds, args.baseline)
    fvu_df.to_csv(out_dir / "fvu.csv", index=False)
    copied = copy_source_tables(source_dirs, out_dir)

    model_configs = []
    for model in sorted(preds):
        depth = re.search(r"GNNHAR(\d+)L", model)
        model_configs.append(
            {
                "name": model,
                "use_iv": model.endswith("_IV"),
                "is_linear": model.startswith("HAR") or model.startswith("GHAR"),
                "n_layers": int(depth.group(1)) if depth else 0,
                "loss_fn": "QL" if "_Q" in model else "MSE",
            }
        )
    config = {
        "universe": args.universe,
        "horizon": args.horizon,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.source_root),
        "source_dirs": {key: str(value) for key, value in source_dirs.items()},
        "output_dir": str(out_dir),
        "models": model_configs,
        "baseline": args.baseline,
        "conversion": "zhang-scale outputs merged into Colab-style single-run layout",
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    manifest = {
        **config,
        "n_models": len(preds),
        "n_test_dates": int(truth.shape[0]),
        "truth_shape": tuple(int(x) for x in truth.shape),
        "files": sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file()),
        "source_tables": copied,
        "created_unix": time.time(),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "n_models": len(preds), "truth_shape": truth.shape}, indent=2))


if __name__ == "__main__":
    main()
