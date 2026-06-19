"""
Post-run diagnostics for Colab-generated GNNHAR outputs.

This script reads completed Google Drive run folders and adds Zhang-style
diagnostics without retraining any model:

- Hansen-Lunde-Nason Model Confidence Set (MCS), following Zhang's public-code
  interface: alpha=0.05, B=10000, block size=2, algorithm="SQ".
- Market-state split at the 90% high-volatility threshold.
- Prediction-level MAD / smoothing proxies, with hidden-state MAD used if the
  corresponding hidden representation arrays are present.
- Basic truth/prediction array integrity checks.
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
except ImportError:  # pragma: no cover - local script fallback
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from gnnhar.mcs import ModelConfidenceSet


def latest_run_dir(universe: str, output_root: Path) -> Path:
    base = output_root / universe
    if not base.exists():
        raise FileNotFoundError(f"No output directory exists for universe={universe}: {base}")
    runs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No run directories found under {base}")
    return runs[-1]


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


def load_run_arrays(universe: str, output_root: Path) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    out_dir = latest_run_dir(universe, output_root)
    truth = np.load(out_dir / "truth.npy")
    test_dates = np.load(out_dir / "test_dates.npy", allow_pickle=True)
    tickers = np.load(out_dir / "tickers.npy", allow_pickle=True)
    preds: dict[str, np.ndarray] = {}
    for path in sorted(out_dir.glob("pred_*.npy")):
        model = path.stem.replace("pred_", "")
        pred = np.load(path)
        if pred.shape == truth.shape:
            preds[model] = pred
        else:
            print(f"Skipping {model}: prediction shape {pred.shape} does not match truth {truth.shape}")
    if not preds:
        raise ValueError(f"No matching prediction arrays found in {out_dir}")
    return out_dir, truth, np.asarray(test_dates), np.asarray(tickers), preds


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
    random_seed: int,
) -> pd.DataFrame:
    clean = loss_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if clean.shape[0] < 5 or clean.shape[1] < 2:
        means = clean.mean(axis=0)
        return pd.DataFrame(
            {
                "model": list(clean.columns),
                "metric": metric,
                "mean_loss": means.values if clean.shape[1] else [],
                "rank_mean_loss": means.rank(method="min").values if clean.shape[1] else [],
                "mcs_included": False,
                "mcs_pvalue": np.nan,
                "n_dates": clean.shape[0],
                "alpha": alpha,
                "bootstrap": bootstrap,
                "block_size": block_size,
                "algorithm": algorithm,
                "status": "too_few_observations_or_models",
            }
        )

    np.random.seed(random_seed)
    model_names = np.array(clean.columns.astype(str).tolist(), dtype=object)
    result = ModelConfidenceSet(
        clean.values,
        alpha=alpha,
        B=bootstrap,
        w=block_size,
        algorithm=algorithm,
        names=model_names,
    ).run()
    pvalues = result.pvalues.to_dict()
    means = clean.mean(axis=0)
    ranks = means.rank(method="min")
    rows = []
    for model in clean.columns:
        pvalue = pvalues.get(model, np.nan)
        rows.append(
            {
                "model": model,
                "metric": metric,
                "mean_loss": float(means[model]),
                "rank_mean_loss": int(ranks[model]),
                "mcs_included": model in set(result.included),
                "mcs_pvalue": float(pvalue) if pd.notna(pvalue) else np.nan,
                "n_dates": int(clean.shape[0]),
                "alpha": alpha,
                "bootstrap": int(bootstrap),
                "block_size": int(block_size),
                "algorithm": algorithm,
                "status": "ok",
            }
        )
    return pd.DataFrame(rows).sort_values(["mcs_included", "rank_mean_loss"], ascending=[False, True]).reset_index(drop=True)


def model_loss_summary(mse_df: pd.DataFrame, qlike_df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    base_mse = mse_df[baseline].mean() if baseline in mse_df.columns else np.nan
    base_qlike = qlike_df[baseline].mean() if baseline in qlike_df.columns else np.nan
    rows = []
    for model in mse_df.columns:
        mse = float(mse_df[model].mean())
        qlike = float(qlike_df[model].mean())
        rows.append(
            {
                "model": model,
                "mse": mse,
                "qlike": qlike,
                "mse_ratio_vs_HAR_M": mse / base_mse if np.isfinite(base_mse) and base_mse != 0 else np.nan,
                "qlike_ratio_vs_HAR_M": qlike / base_qlike if np.isfinite(base_qlike) and base_qlike != 0 else np.nan,
                **infer_model_metadata(model),
            }
        )
    return pd.DataFrame(rows).sort_values("qlike_ratio_vs_HAR_M").reset_index(drop=True)


def market_state_proxy(universe: str, out_dir: Path, truth: np.ndarray, tickers: np.ndarray, quantile: float) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    ticker_upper = [str(t).upper() for t in tickers]
    for candidate in ["SPY", "SPX", "^GSPC", "SP500"]:
        if candidate in ticker_upper:
            idx = ticker_upper.index(candidate)
            market_rv = truth[:, idx]
            source = f"{candidate} RV from truth array"
            break
    else:
        market_rv = np.nanmean(truth, axis=1)
        source = "cross-sectional mean RV proxy from truth array"

    threshold = float(np.nanquantile(market_rv, quantile))
    masks = {
        "calm_bottom_90pct": market_rv < threshold,
        "turbulent_top_10pct": market_rv >= threshold,
    }
    metadata = {
        "universe": universe,
        "run_dir": str(out_dir),
        "source": source,
        "quantile": float(quantile),
        "threshold": threshold,
        "n_dates": int(len(market_rv)),
        "n_calm": int(np.sum(masks["calm_bottom_90pct"])),
        "n_turbulent": int(np.sum(masks["turbulent_top_10pct"])),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    return market_rv, masks, metadata


def regime_loss_table(mse_df: pd.DataFrame, qlike_df: pd.DataFrame, masks: dict[str, np.ndarray], baseline: str) -> pd.DataFrame:
    rows = []
    for regime, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            continue
        base_mse = mse_df.loc[mask, baseline].mean() if baseline in mse_df.columns else np.nan
        base_qlike = qlike_df.loc[mask, baseline].mean() if baseline in qlike_df.columns else np.nan
        for model in mse_df.columns:
            mse = float(mse_df.loc[mask, model].mean())
            qlike = float(qlike_df.loc[mask, model].mean())
            rows.append(
                {
                    "regime": regime,
                    "model": model,
                    "n_dates": int(mask.sum()),
                    "mse": mse,
                    "qlike": qlike,
                    "mse_ratio_vs_HAR_M": mse / base_mse if np.isfinite(base_mse) and base_mse != 0 else np.nan,
                    "qlike_ratio_vs_HAR_M": qlike / base_qlike if np.isfinite(base_qlike) and base_qlike != 0 else np.nan,
                    **infer_model_metadata(model),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["rank_qlike_within_regime"] = df.groupby("regime")["qlike"].rank(method="min")
        df["rank_mse_within_regime"] = df.groupby("regime")["mse"].rank(method="min")
        df = df.sort_values(["regime", "qlike_ratio_vs_HAR_M"]).reset_index(drop=True)
    return df


def average_pairwise_cosine_distance(matrix: np.ndarray, eps: float = 1e-12) -> float:
    matrix = np.asarray(matrix, dtype=float)
    valid = np.isfinite(matrix).all(axis=1)
    matrix = matrix[valid]
    n = matrix.shape[0]
    if n < 2:
        return np.nan
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    normalized = centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), eps)
    cosine = normalized @ normalized.T
    dist = 1.0 - cosine
    return float(np.nanmean(dist[~np.eye(n, dtype=bool)]))


def smoothing_diagnostics(preds: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for model, pred in preds.items():
        pred = np.clip(np.asarray(pred, dtype=float), 1e-10, None)
        rows.append(
            {
                "model": model,
                **infer_model_metadata(model),
                "forecast_mad_time_series": average_pairwise_cosine_distance(pred.T),
                "mean_cross_section_std": float(np.nanmean(np.nanstd(pred, axis=1))),
                "mean_cross_section_cv": float(np.nanmean(np.nanstd(pred, axis=1) / np.maximum(np.nanmean(pred, axis=1), 1e-10))),
                "diagnostic_level": "prediction_proxy",
                "note": "Prediction-level smoothing proxy; exact hidden-state MAD requires saved final GNN hidden representations.",
            }
        )
    return pd.DataFrame(rows).sort_values(["model_class", "uses_iv", "training_loss", "depth", "model"]).reset_index(drop=True)


def hidden_state_mad(out_dir: Path) -> pd.DataFrame:
    hidden_files = sorted(list(out_dir.glob("hidden_*.npy")) + list(out_dir.glob("repr_*.npy")) + list(out_dir.glob("embedding_*.npy")))
    rows = []
    for path in hidden_files:
        model = re.sub(r"^(hidden_|repr_|embedding_)", "", path.stem)
        arr = np.load(path)
        if arr.ndim == 3:
            value = float(np.nanmean([average_pairwise_cosine_distance(arr[t]) for t in range(arr.shape[0])]))
        elif arr.ndim == 2:
            value = average_pairwise_cosine_distance(arr)
        else:
            value = np.nan
        rows.append(
            {
                "model": model,
                "file": path.name,
                "hidden_state_mad": value,
                "array_shape": tuple(int(x) for x in arr.shape),
                "diagnostic_level": "hidden_state" if np.isfinite(value) else "unusable_hidden_array",
            }
        )
    return pd.DataFrame(rows)


def smoothing_depth_summary(smoothing_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if smoothing_df.empty:
        return pd.DataFrame()
    gnn = smoothing_df[smoothing_df["model_class"] == "GNNHAR"].copy()
    for (uses_iv, training_loss), grp in gnn.groupby(["uses_iv", "training_loss"], dropna=False):
        grp = grp.dropna(subset=["depth"]).sort_values("depth")
        if len(grp) < 2:
            continue
        for metric in ["forecast_mad_time_series", "mean_cross_section_cv"]:
            y = grp[metric].astype(float).values
            x = grp["depth"].astype(float).values
            corr = np.corrcoef(x, y)[0, 1] if np.isfinite(y).all() and np.std(y) > 0 else np.nan
            rows.append(
                {
                    "uses_iv": bool(uses_iv),
                    "training_loss": training_loss,
                    "metric": metric,
                    "depth_min": int(np.min(x)),
                    "depth_max": int(np.max(x)),
                    "n_depths": int(len(x)),
                    "pearson_corr_with_depth": float(corr) if np.isfinite(corr) else np.nan,
                    "first_depth_value": float(y[0]),
                    "last_depth_value": float(y[-1]),
                    "direction": "decreases_with_depth" if y[-1] < y[0] else "increases_with_depth" if y[-1] > y[0] else "flat",
                }
            )
    return pd.DataFrame(rows)


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


def update_manifest(out_dir: Path, saved_files: list[str], diagnostics_meta: dict) -> Path:
    manifest_path = out_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.setdefault("diagnostics", {})
    manifest["diagnostics"].update(diagnostics_meta)
    manifest["diagnostics"]["files"] = saved_files
    manifest["files"] = sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file())
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest_path


def run_universe(
    universe: str,
    output_root: Path,
    alpha: float,
    bootstrap: int,
    block_size: int,
    algorithm: str,
    seed: int,
    regime_quantile: float,
    baseline: str,
) -> dict:
    out_dir, truth, test_dates, tickers, preds = load_run_arrays(universe, output_root)
    diagnostic_dir = out_dir / "post_run_diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[str] = []

    integrity = data_integrity_checks(truth, preds, test_dates, tickers)
    mse_losses, qlike_losses = per_date_loss_frames(truth, preds)
    loss_summary = model_loss_summary(mse_losses, qlike_losses, baseline)

    market_rv, masks, regime_meta = market_state_proxy(universe, out_dir, truth, tickers, regime_quantile)
    regime_table = regime_loss_table(mse_losses, qlike_losses, masks, baseline)
    market_state = pd.DataFrame(
        {
            "test_date": pd.to_datetime(test_dates).astype(str),
            "market_rv_proxy": market_rv,
            "regime": np.where(masks["turbulent_top_10pct"], "turbulent_top_10pct", "calm_bottom_90pct"),
        }
    )

    print(f"{universe}: running MCS with B={bootstrap}, w={block_size}, algorithm={algorithm}")
    mcs_mse = run_mcs_table(mse_losses, "MSE", alpha, bootstrap, block_size, algorithm, seed)
    mcs_qlike = run_mcs_table(qlike_losses, "QLIKE", alpha, bootstrap, block_size, algorithm, seed)
    mcs_summary = pd.concat([mcs_mse, mcs_qlike], ignore_index=True)

    smoothing = smoothing_diagnostics(preds)
    hidden_mad = hidden_state_mad(out_dir)
    depth_summary = smoothing_depth_summary(smoothing)

    tables = {
        "data_integrity_checks.csv": integrity,
        "per_date_mse_losses.csv": mse_losses,
        "per_date_qlike_losses.csv": qlike_losses,
        "diagnostic_loss_summary.csv": loss_summary,
        "mcs_mse.csv": mcs_mse,
        "mcs_qlike.csv": mcs_qlike,
        "mcs_summary.csv": mcs_summary,
        "regime_loss_table.csv": regime_table,
        "market_state_series.csv": market_state,
        "mad_smoothing_diagnostics.csv": smoothing,
        "hidden_state_mad.csv": hidden_mad,
        "oversmoothing_depth_summary.csv": depth_summary,
    }
    for filename, table in tables.items():
        path = diagnostic_dir / filename
        table.to_csv(path, index=False)
        saved_files.append(str(path.relative_to(out_dir)))

    regime_path = diagnostic_dir / "regime_metadata.json"
    regime_path.write_text(json.dumps(regime_meta, indent=2, default=str))
    saved_files.append(str(regime_path.relative_to(out_dir)))

    diagnostics_meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mcs_alpha": alpha,
        "mcs_bootstrap": bootstrap,
        "mcs_block_size": block_size,
        "mcs_algorithm": algorithm,
        "regime_quantile": regime_quantile,
        "regime_source": regime_meta.get("source"),
        "baseline_model": baseline,
        "note": "MCS and regime analysis use saved out-of-sample forecasts. MAD uses hidden states if saved; otherwise prediction-level smoothing proxies are reported.",
    }
    manifest_path = diagnostic_dir / "post_run_diagnostics_manifest.json"
    manifest_path.write_text(json.dumps({**diagnostics_meta, "files": saved_files}, indent=2, default=str))
    saved_files.append(str(manifest_path.relative_to(out_dir)))
    update_manifest(out_dir, saved_files, diagnostics_meta)

    return {
        "universe": universe,
        "out_dir": str(out_dir),
        "diagnostic_dir": str(diagnostic_dir),
        "n_models": len(preds),
        "n_dates": int(truth.shape[0]),
        "n_tickers": int(truth.shape[1]),
        "mcs_qlike_included": mcs_qlike.loc[mcs_qlike["mcs_included"], "model"].tolist(),
        "regime_source": regime_meta.get("source"),
        "saved_files": saved_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--universes", nargs="+", default=["dow30", "sp100", "sp500"])
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--algorithm", default="SQ", choices=["SQ", "R"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--regime-quantile", type=float, default=0.90)
    parser.add_argument("--baseline", default="HAR_M")
    args = parser.parse_args()

    summaries = []
    for universe in args.universes:
        try:
            summary = run_universe(
                universe=universe,
                output_root=args.output_root,
                alpha=args.alpha,
                bootstrap=args.bootstrap,
                block_size=args.block_size,
                algorithm=args.algorithm,
                seed=args.seed,
                regime_quantile=args.regime_quantile,
                baseline=args.baseline,
            )
            summaries.append(summary)
            print(f"{universe}: wrote diagnostics to {summary['diagnostic_dir']}")
        except FileNotFoundError as exc:
            print(f"Skipping {universe}: {exc}")

    summary_df = pd.DataFrame(summaries)
    summary_dir = args.output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "post_run_diagnostics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary CSV: {summary_path}")
    if not summary_df.empty:
        print(summary_df[["universe", "n_models", "n_dates", "n_tickers", "diagnostic_dir"]].to_string(index=False))


if __name__ == "__main__":
    main()
