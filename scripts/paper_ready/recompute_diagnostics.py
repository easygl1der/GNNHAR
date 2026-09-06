#!/usr/bin/env python3
"""Recompute paper-ready diagnostics from a structured run directory.

Expected run layout:

  RUN_DIR/
    arrays/truth.npy
    arrays/test_dates.npy
    arrays/tickers.npy
    predictions/pred_HAR_M.npy
    predictions/pred_*.npy

The script writes:

  RUN_DIR/diagnostics/per_date_mse_losses.csv
  RUN_DIR/diagnostics/per_date_qlike_losses.csv
  RUN_DIR/diagnostics/mcs_mse.csv
  RUN_DIR/diagnostics/mcs_qlike.csv
  RUN_DIR/diagnostics/mcs_summary.csv
  RUN_DIR/diagnostics/diagnostic_loss_summary.csv
  RUN_DIR/diagnostics/data_integrity_checks.csv
  RUN_DIR/diagnostics/recompute_manifest.json

It intentionally does not train models. Use it after a Colab or AutoDL run has
exported aligned truth and prediction arrays into the paper-ready layout.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from gnnhar.mcs import ModelConfidenceSet
except ImportError:  # pragma: no cover
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from gnnhar.mcs import ModelConfidenceSet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline", default="HAR_M")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--algorithm", choices=["SQ", "R"], default="SQ")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def infer_model_metadata(model_name: str) -> dict:
    depth_match = re.search(r"GNNHAR(\d+)L", model_name)
    if model_name.startswith("HAR"):
        model_class = "HAR"
    elif model_name.startswith("GHAR"):
        model_class = "GHAR"
    elif model_name.startswith("GNNHAR"):
        model_class = "GNNHAR"
    else:
        model_class = "OTHER"
    return {
        "model_class": model_class,
        "depth": int(depth_match.group(1)) if depth_match else np.nan,
        "uses_iv": model_name.endswith("_IV"),
        "training_loss": "QLIKE-trained" if "_Q" in model_name else "MSE-trained" if "_M" in model_name else "unknown",
    }


def load_structured_run(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    arrays = run_dir / "arrays"
    predictions = run_dir / "predictions"
    truth = np.load(arrays / "truth.npy")
    test_dates = np.load(arrays / "test_dates.npy", allow_pickle=True)
    tickers = np.load(arrays / "tickers.npy", allow_pickle=True)
    preds: dict[str, np.ndarray] = {}
    for path in sorted(predictions.glob("pred_*.npy")):
        model = path.stem.replace("pred_", "")
        pred = np.load(path)
        if pred.shape != truth.shape:
            raise ValueError(f"{path} has shape {pred.shape}, expected {truth.shape}")
        preds[model] = pred
    if not preds:
        raise FileNotFoundError(f"No prediction files found under {predictions}")
    return truth, np.asarray(test_dates), np.asarray(tickers), preds


def per_date_loss_frames(truth: np.ndarray, preds: dict[str, np.ndarray], eps: float = 1e-10) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth_clip = np.clip(truth, eps, None)
    mse: dict[str, np.ndarray] = {}
    qlike: dict[str, np.ndarray] = {}
    for model, pred in preds.items():
        pred_clip = np.clip(pred, eps, None)
        mse[model] = np.nanmean((pred_clip - truth_clip) ** 2, axis=1)
        ratio = truth_clip / pred_clip
        qlike[model] = np.nanmean(ratio - np.log(np.clip(ratio, eps, None)) - 1, axis=1)
    return pd.DataFrame(mse), pd.DataFrame(qlike)


def run_mcs_table(
    loss_df: pd.DataFrame,
    metric: str,
    alpha: float,
    bootstrap: int,
    block_size: int,
    algorithm: str,
    seed: int,
) -> pd.DataFrame:
    clean = loss_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if clean.shape[0] < 5 or clean.shape[1] < 2:
        raise ValueError(f"Need at least 5 dates and 2 models for MCS; got {clean.shape}")
    np.random.seed(seed)
    names = np.array(clean.columns.astype(str).tolist(), dtype=object)
    result = ModelConfidenceSet(clean.values, alpha=alpha, B=bootstrap, w=block_size, algorithm=algorithm, names=names).run()
    pvalues = result.pvalues.to_dict()
    means = clean.mean(axis=0)
    ranks = means.rank(method="min")
    included = set(result.included)
    rows = []
    for model in clean.columns:
        rows.append(
            {
                "model": model,
                "metric": metric,
                "mean_loss": float(means[model]),
                "rank_mean_loss": int(ranks[model]),
                "mcs_included": model in included,
                "mcs_pvalue": float(pvalues.get(model, np.nan)),
                "n_dates": int(clean.shape[0]),
                "alpha": float(alpha),
                "bootstrap": int(bootstrap),
                "block_size": int(block_size),
                "algorithm": algorithm,
                "status": "ok",
            }
        )
    return pd.DataFrame(rows).sort_values(["mcs_included", "rank_mean_loss"], ascending=[False, True]).reset_index(drop=True)


def model_loss_summary(mse_df: pd.DataFrame, qlike_df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    if baseline not in mse_df.columns or baseline not in qlike_df.columns:
        raise KeyError(f"Baseline {baseline!r} must be present in prediction files")
    base_mse = mse_df[baseline].mean()
    base_qlike = qlike_df[baseline].mean()
    rows = []
    for model in mse_df.columns:
        mse = float(mse_df[model].mean())
        qlike = float(qlike_df[model].mean())
        rows.append(
            {
                "model": model,
                "mse": mse,
                "qlike": qlike,
                "mse_ratio_vs_HAR_M": mse / base_mse,
                "qlike_ratio_vs_HAR_M": qlike / base_qlike,
                **infer_model_metadata(model),
            }
        )
    return pd.DataFrame(rows).sort_values("qlike_ratio_vs_HAR_M").reset_index(drop=True)


def data_integrity_checks(truth: np.ndarray, preds: dict[str, np.ndarray], test_dates: np.ndarray, tickers: np.ndarray) -> pd.DataFrame:
    rows = [
        {
            "array": "truth",
            "shape": tuple(int(x) for x in truth.shape),
            "n_dates": int(len(test_dates)),
            "n_tickers": int(len(tickers)),
            "finite_share": float(np.isfinite(truth).mean()),
            "nonpositive_count": int(np.sum(truth <= 0)),
            "min": float(np.nanmin(truth)),
            "max": float(np.nanmax(truth)),
        }
    ]
    for model, pred in preds.items():
        rows.append(
            {
                "array": f"pred_{model}",
                "shape": tuple(int(x) for x in pred.shape),
                "n_dates": int(pred.shape[0]),
                "n_tickers": int(pred.shape[1]),
                "finite_share": float(np.isfinite(pred).mean()),
                "nonpositive_count": int(np.sum(pred <= 0)),
                "min": float(np.nanmin(pred)),
                "max": float(np.nanmax(pred)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    truth, test_dates, tickers, preds = load_structured_run(run_dir)
    diagnostics = run_dir / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)

    mse_df, qlike_df = per_date_loss_frames(truth, preds)
    loss_summary = model_loss_summary(mse_df, qlike_df, args.baseline)
    mcs_mse = run_mcs_table(mse_df, "MSE", args.alpha, args.bootstrap, args.block_size, args.algorithm, args.seed)
    mcs_qlike = run_mcs_table(qlike_df, "QLIKE", args.alpha, args.bootstrap, args.block_size, args.algorithm, args.seed)
    mcs_summary = pd.concat([mcs_mse, mcs_qlike], ignore_index=True)
    integrity = data_integrity_checks(truth, preds, test_dates, tickers)

    outputs = {
        "per_date_mse_losses.csv": mse_df,
        "per_date_qlike_losses.csv": qlike_df,
        "diagnostic_loss_summary.csv": loss_summary,
        "mcs_mse.csv": mcs_mse,
        "mcs_qlike.csv": mcs_qlike,
        "mcs_summary.csv": mcs_summary,
        "data_integrity_checks.csv": integrity,
    }
    for filename, df in outputs.items():
        df.to_csv(diagnostics / filename, index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "n_dates": int(truth.shape[0]),
        "n_tickers": int(truth.shape[1]),
        "n_models": int(len(preds)),
        "test_start": str(pd.to_datetime(test_dates)[0]),
        "test_end": str(pd.to_datetime(test_dates)[-1]),
        "baseline": args.baseline,
        "mcs_alpha": args.alpha,
        "mcs_bootstrap": args.bootstrap,
        "mcs_block_size": args.block_size,
        "mcs_algorithm": args.algorithm,
        "files": sorted(outputs),
    }
    (diagnostics / "recompute_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
