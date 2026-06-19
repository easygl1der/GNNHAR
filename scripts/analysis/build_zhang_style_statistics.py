#!/usr/bin/env python3
"""Build Zhang-style statistics from paper-ready GNNHAR outputs.

The input directory is the structured result tree produced by
``scripts/organize/paper_ready_results.py``.  This script does not retrain any
model.  It derives report-facing statistics from saved out-of-sample truth and
prediction arrays, and writes a separate ``zhang_style_statistics`` layer under
each universe.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy is available in the project env
    stats = None

try:
    from gnnhar.mcs import ModelConfidenceSet
except ImportError:  # pragma: no cover - local script fallback
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from gnnhar.mcs import ModelConfidenceSet


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_ROOT = REPO_ROOT / "outputs" / "paper_ready_20260617"
DRIVE_DATA_ROOT = REPO_ROOT / "data" / "google_drive_upload" / "GNNHAR_Research" / "data"
SCALE_DATA_ROOT = REPO_ROOT / "data" / "scale_experiment"
EPS = 1e-10

GRAPH_COLUMNS = [
    "graph_file",
    "loss_group",
    "n_nodes",
    "edges",
    "density",
    "degree_mean",
    "degree_median",
    "degree_min",
    "degree_max",
    "method",
    "alpha",
    "fallback",
    "train_rows",
    "diameter",
    "reachable_pair_share",
]

SHORTEST_PATH_COLUMNS = [
    "graph_file",
    "loss_group",
    "shortest_path_distance",
    "pair_count",
    "pair_share_all_pairs",
    "pair_share_reachable_pairs",
]

TICKER_ALIGNMENT_COLUMNS = [
    "panel",
    "available",
    "raw_source_dir",
    "raw_source_reason",
    "panel_file",
    "date_start",
    "date_end",
    "n_dates",
    "n_panel_tickers",
    "n_model_tickers",
    "n_matched_tickers",
    "n_missing_from_panel",
    "n_extra_in_panel",
    "missing_from_panel_sample",
    "extra_in_panel_sample",
]


ZHANG_STATISTICS = [
    {
        "component": "forecast_loss_ratios",
        "paper_reference": "Table 1 and Table 5",
        "zhang_source": "Summary_Results.py",
        "status_in_this_script": "implemented_from_saved_forecasts",
        "notes": "MSE and QLIKE losses are averaged over test dates and tickers, then normalized by HAR_M. Current saved forecasts cover the one-day horizon.",
    },
    {
        "component": "weekly_monthly_horizons",
        "paper_reference": "Table 1, Table 2, Table 3, Table 5",
        "zhang_source": "paper Section 4.1",
        "status_in_this_script": "scope_gap_current_saved_forecasts_are_one_day_only",
        "notes": "Zhang also reports one-week and one-month targets. These require h=5/h=22 target construction and saved forecasts, not post-run statistics from the current h=1 outputs.",
    },
    {
        "component": "model_confidence_set",
        "paper_reference": "Table 1, Table 2, Table 4, Table 5",
        "zhang_source": "MCS.py and Summary_Results.py",
        "status_in_this_script": "implemented_from_per_ticker_losses",
        "notes": "Uses Hansen-Lunde-Nason MCS with alpha=0.05, B=10000, block size=2, algorithm=SQ.",
    },
    {
        "component": "market_regime_losses",
        "paper_reference": "Table 2",
        "zhang_source": "Summary_Regime.py",
        "status_in_this_script": "implemented_from_saved_forecasts",
        "notes": "Uses SPY when present; otherwise uses cross-sectional mean RV proxy and records this difference.",
    },
    {
        "component": "alternative_validation_size",
        "paper_reference": "Table 4",
        "zhang_source": "paper Section 6.1",
        "status_in_this_script": "scope_gap_requires_new_rolling_runs",
        "notes": "The smaller-validation robustness table changes the rolling train/validation split, so it is not derivable from the current saved forecasts.",
    },
    {
        "component": "forecast_error_ratio_boxplots",
        "paper_reference": "Figure 5",
        "zhang_source": "BoxPlot_Error.py",
        "status_in_this_script": "implemented_as_source_tables_and_pngs",
        "notes": "Exports long-form source tables and grouped boxplot PNGs for all/calm/turbulent regimes.",
    },
    {
        "component": "fvu_nonlinearity",
        "paper_reference": "Table 3 and equation (11)",
        "zhang_source": "paper equation (11)",
        "status_in_this_script": "implemented_with_regime_split",
        "notes": "Computes FVU relative to HAR_M, plus optional incremental FVU against a matched predecessor.",
    },
    {
        "component": "multi_hop_dm",
        "paper_reference": "Figure 6 and Appendix E.1",
        "zhang_source": "paper Section 5.3 and Appendix E",
        "status_in_this_script": "implemented_when_models_exist",
        "notes": "Runs per-ticker and cross-sectional QLIKE DM tests for GNN depth and GHAR hop comparisons.",
    },
    {
        "component": "mad_smoothing",
        "paper_reference": "Figure 7 and equation (12)",
        "zhang_source": "paper equation (12)",
        "status_in_this_script": "implemented_when_graph_and_hidden_or_forecast_representations_exist",
        "notes": "Exact hidden-state MAD requires saved hidden states. Otherwise a clearly labeled forecast-level proxy is reported.",
    },
    {
        "component": "data_summary_and_correlation",
        "paper_reference": "Appendix A and Figure 1",
        "zhang_source": "paper data section",
        "status_in_this_script": "implemented_for_available_local_panels",
        "notes": "Exports RV summary statistics, missingness, Pearson/Spearman correlation summaries, and date alignment.",
    },
    {
        "component": "graph_structure",
        "paper_reference": "Appendix A.2",
        "zhang_source": "paper Table A.2",
        "status_in_this_script": "implemented_when_graph_matrices_exist",
        "notes": "Exports graph density, degree, and shortest path distance distributions for saved graph matrices.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--universes", nargs="+", default=["dow30", "sp100", "sp500"])
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--algorithm", choices=["SQ", "R"], default="SQ")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--regime-quantile", type=float, default=0.90)
    parser.add_argument("--baseline", default="HAR_M")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_universe(root: Path, universe: str) -> dict:
    ur = root / "universes" / universe
    if not ur.exists():
        raise FileNotFoundError(ur)
    truth = np.load(ur / "arrays" / "truth.npy")
    tickers = np.load(ur / "arrays" / "tickers.npy", allow_pickle=True).astype(str)
    test_dates = pd.to_datetime(np.load(ur / "arrays" / "test_dates.npy", allow_pickle=True))
    preds = {
        p.stem.removeprefix("pred_"): np.load(p)
        for p in sorted((ur / "predictions").glob("pred_*.npy"))
    }
    preds = {k: v for k, v in preds.items() if v.shape == truth.shape}
    if not preds:
        raise ValueError(f"{universe}: no prediction arrays match truth shape {truth.shape}")
    return {
        "universe": universe,
        "root": ur,
        "truth": truth.astype(float),
        "tickers": np.asarray(tickers, dtype=str),
        "test_dates": test_dates,
        "preds": {k: v.astype(float) for k, v in preds.items()},
    }


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def map_remote_data_dir(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path_text = str(path_value)
    marker = "/data/scale_experiment/"
    if marker in path_text:
        rel = path_text.split(marker, 1)[1].strip("/")
        return SCALE_DATA_ROOT / rel
    drive_marker = "/GNNHAR_Research/data/"
    if drive_marker in path_text:
        rel = path_text.split(drive_marker, 1)[1].strip("/")
        return DRIVE_DATA_ROOT / rel
    local = Path(path_text).expanduser()
    if local.exists():
        return local
    return None


def panel_match_score(raw_dir: Path, tickers: np.ndarray) -> tuple[int, int, bool]:
    model_tickers = set(map(str, tickers))
    required = [
        raw_dir / "merged_rv_data_filled.csv",
        raw_dir / "merged_iv_data_filled.csv",
        raw_dir / "daily_returns.csv",
    ]
    if not all(path.exists() for path in required):
        return (-1, -1, False)
    matched_total = 0
    missing_total = 0
    for path in required:
        try:
            df = pd.read_csv(path, nrows=1)
        except Exception:
            return (-1, -1, False)
        panel_tickers = set(c for c in df.columns if c != "Date")
        matched_total += len(model_tickers & panel_tickers)
        missing_total += len(model_tickers - panel_tickers)
    return (matched_total, -missing_total, True)


def autodl_metadata_data_dir(universe_root: Path) -> Path | None:
    for run_meta in [
        universe_root / "zhang_source" / "mse" / "run_metadata.json",
        universe_root / "zhang_source" / "qlike" / "run_metadata.json",
    ]:
        meta = read_json(run_meta)
        data_dir = map_remote_data_dir((meta.get("args") or {}).get("data_dir"))
        if data_dir is not None:
            return data_dir
    return None


def resolve_raw_source_dir(universe_root: Path, tickers: np.ndarray) -> tuple[Path | None, str]:
    universe = universe_root.name
    cfg = read_json(universe_root / "metadata" / "run_config.json")

    autodl_dir = autodl_metadata_data_dir(universe_root)
    if autodl_dir is not None and autodl_dir.exists():
        return autodl_dir, "autodl_run_metadata_args_data_dir"

    data_paths = cfg.get("data_paths") or {}
    mapped = map_remote_data_dir(data_paths.get("data_dir"))
    if mapped is not None and mapped.exists():
        return mapped, "run_config_data_paths_data_dir"

    candidates = [
        (SCALE_DATA_ROOT / universe, "scale_experiment_fallback"),
        (DRIVE_DATA_ROOT / universe, "google_drive_upload_fallback"),
    ]
    candidates = [(path, reason) for path, reason in candidates if path.exists()]
    if not candidates:
        return None, "no_local_raw_source_dir_found"
    scored = [
        (panel_match_score(path, tickers), path, reason)
        for path, reason in candidates
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    _, path, reason = scored[0]
    return path, reason


def infer_model_metadata(model: str) -> dict:
    depth_match = re.search(r"GNNHAR(\d+)L", model)
    hop_match = re.search(r"GHAR(\d+)H", model)
    if model.startswith("HAR"):
        family = "HAR"
    elif model.startswith("GHAR"):
        family = "GHAR"
    elif model.startswith("GNNHAR"):
        family = "GNNHAR"
    else:
        family = "OTHER"
    return {
        "family": family,
        "depth": int(depth_match.group(1)) if depth_match else np.nan,
        "hop": int(hop_match.group(1)) if hop_match else (1 if model.startswith("GHAR") else np.nan),
        "uses_iv": bool(model.endswith("_IV")),
        "estimation": "QLIKE" if "_Q" in model else "MSE" if "_M" in model else "unknown",
    }


def qlike_array(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.clip(np.asarray(y_true, dtype=float), EPS, None)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), EPS, None)
    ratio = y_true / y_pred
    return ratio - np.log(np.clip(ratio, EPS, None)) - 1.0


def loss_arrays(truth: np.ndarray, preds: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    mse = {}
    qlike = {}
    truth_clip = np.clip(truth, EPS, None)
    for model, pred in preds.items():
        pred_clip = np.clip(pred, EPS, None)
        mse[model] = (pred_clip - truth_clip) ** 2
        qlike[model] = qlike_array(truth_clip, pred_clip)
    return mse, qlike


def per_ticker_losses(
    losses: dict[str, np.ndarray],
    tickers: np.ndarray,
    metric: str,
    mask: np.ndarray | None = None,
) -> pd.DataFrame:
    rows = []
    use_mask = slice(None) if mask is None else np.asarray(mask, dtype=bool)
    for ticker_idx, ticker in enumerate(tickers):
        row = {"ticker": ticker}
        for model, arr in losses.items():
            row[model] = float(np.nanmean(arr[use_mask, ticker_idx]))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.insert(1, "metric", metric)
    return df


def per_date_losses(
    losses: dict[str, np.ndarray],
    test_dates: pd.DatetimeIndex,
    metric: str,
) -> pd.DataFrame:
    df = pd.DataFrame({model: np.nanmean(arr, axis=1) for model, arr in losses.items()})
    df.insert(0, "test_date", test_dates.astype(str))
    df.insert(1, "metric", metric)
    return df


def loss_ratio_summary(
    mse_by_ticker: pd.DataFrame,
    qlike_by_ticker: pd.DataFrame,
    baseline: str,
) -> pd.DataFrame:
    if baseline not in mse_by_ticker or baseline not in qlike_by_ticker:
        raise KeyError(f"baseline {baseline!r} missing in per-ticker losses")
    base_mse = float(mse_by_ticker[baseline].mean())
    base_qlike = float(qlike_by_ticker[baseline].mean())
    rows = []
    for model in [c for c in mse_by_ticker.columns if c not in {"ticker", "metric"}]:
        mse = float(mse_by_ticker[model].mean())
        qlike = float(qlike_by_ticker[model].mean())
        rows.append(
            {
                "model": model,
                **infer_model_metadata(model),
                "mse": mse,
                "qlike": qlike,
                "mse_ratio_vs_HAR_M": mse / base_mse if base_mse else np.nan,
                "qlike_ratio_vs_HAR_M": qlike / base_qlike if base_qlike else np.nan,
                "mse_gain_vs_HAR_M": 1.0 - mse / base_mse if base_mse else np.nan,
                "qlike_gain_vs_HAR_M": 1.0 - qlike / base_qlike if base_qlike else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("qlike_ratio_vs_HAR_M").reset_index(drop=True)


def run_mcs(loss_df: pd.DataFrame, metric: str, alpha: float, bootstrap: int, block_size: int, algorithm: str, seed: int) -> pd.DataFrame:
    model_cols = [c for c in loss_df.columns if c not in {"ticker", "metric"}]
    clean = loss_df[model_cols].replace([np.inf, -np.inf], np.nan).dropna(how="any")
    means = clean.mean(axis=0)
    ranks = means.rank(method="min")
    if clean.shape[0] < 5 or clean.shape[1] < 2:
        return pd.DataFrame(
            {
                "model": model_cols,
                "metric": metric,
                "mean_loss": [float(means.get(m, np.nan)) for m in model_cols],
                "rank_mean_loss": [float(ranks.get(m, np.nan)) for m in model_cols],
                "mcs_included": False,
                "mcs_pvalue": np.nan,
                "n_cross_section_units": clean.shape[0],
                "alpha": alpha,
                "bootstrap": bootstrap,
                "block_size": block_size,
                "algorithm": algorithm,
                "status": "too_few_units_or_models",
            }
        )
    np.random.seed(seed)
    names = np.array(clean.columns.astype(str).tolist(), dtype=object)
    try:
        result = ModelConfidenceSet(clean.values, alpha=alpha, B=bootstrap, w=block_size, algorithm=algorithm, names=names).run()
        pvalues = result.pvalues.to_dict()
        included = set(result.included)
        status = "ok"
    except Exception as exc:
        pvalues = {}
        included = set()
        status = f"failed:{type(exc).__name__}:{exc}"
    rows = []
    for model in clean.columns:
        rows.append(
            {
                "model": model,
                "metric": metric,
                "mean_loss": float(means[model]),
                "rank_mean_loss": float(ranks[model]),
                "mcs_included": model in included,
                "mcs_pvalue": float(pvalues.get(model, np.nan)) if model in pvalues else np.nan,
                "n_cross_section_units": int(clean.shape[0]),
                "alpha": alpha,
                "bootstrap": bootstrap,
                "block_size": block_size,
                "algorithm": algorithm,
                "status": status,
            }
        )
    return pd.DataFrame(rows).sort_values(["mcs_included", "rank_mean_loss"], ascending=[False, True]).reset_index(drop=True)


def market_state(
    universe_root: Path,
    raw_source_dir: Path | None,
    raw_source_reason: str,
    truth: np.ndarray,
    tickers: np.ndarray,
    test_dates: pd.DatetimeIndex,
    quantile: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict]:
    ticker_upper = [t.upper() for t in tickers.astype(str)]
    if "SPY" in ticker_upper:
        idx = ticker_upper.index("SPY")
        proxy = truth[:, idx]
        source = "SPY from truth array"
    else:
        proxy = np.nanmean(truth, axis=1)
        source = "cross-sectional mean RV proxy from truth array"
        raw_spy = raw_panel_path(raw_source_dir, "merged_rv_data_filled.csv")
        if raw_spy is not None and raw_spy.exists():
            try:
                raw = pd.read_csv(raw_spy)
                if "SPY" in raw.columns:
                    raw["Date"] = pd.to_datetime(raw["Date"])
                    mapped = raw.set_index("Date").reindex(test_dates)["SPY"].to_numpy(dtype=float)
                    if np.isfinite(mapped).sum() > len(mapped) // 2:
                        proxy = mapped
                        source = "SPY from local raw RV panel aligned to test dates"
            except Exception:
                pass
    threshold = float(np.nanquantile(proxy, quantile))
    masks = {
        "calm_bottom_90pct": proxy < threshold,
        "turbulent_top_10pct": proxy >= threshold,
    }
    series = pd.DataFrame(
        {
            "test_date": test_dates.astype(str),
            "market_rv_proxy": proxy,
            "regime": np.where(masks["turbulent_top_10pct"], "turbulent_top_10pct", "calm_bottom_90pct"),
        }
    )
    meta = {
        "source": source,
        "quantile": quantile,
        "threshold": threshold,
        "n_calm": int(np.sum(masks["calm_bottom_90pct"])),
        "n_turbulent": int(np.sum(masks["turbulent_top_10pct"])),
        "raw_source_dir": str(raw_source_dir) if raw_source_dir is not None else "",
        "raw_source_reason": raw_source_reason,
    }
    return series, masks, meta


def regime_loss_tables(
    mse_losses: dict[str, np.ndarray],
    qlike_losses: dict[str, np.ndarray],
    tickers: np.ndarray,
    masks: dict[str, np.ndarray],
    baseline: str,
    alpha: float,
    bootstrap: int,
    block_size: int,
    algorithm: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratio_rows = []
    mcs_rows = []
    for regime, mask in masks.items():
        mse_ticker = per_ticker_losses(mse_losses, tickers, "MSE", mask)
        qlike_ticker = per_ticker_losses(qlike_losses, tickers, "QLIKE", mask)
        for metric, table in [("MSE", mse_ticker), ("QLIKE", qlike_ticker)]:
            base = float(table[baseline].mean()) if baseline in table else np.nan
            for model in [c for c in table.columns if c not in {"ticker", "metric"}]:
                loss = float(table[model].mean())
                ratio_rows.append(
                    {
                        "regime": regime,
                        "metric": metric,
                        "model": model,
                        **infer_model_metadata(model),
                        "n_dates": int(np.asarray(mask).sum()),
                        "n_tickers": int(len(tickers)),
                        "loss": loss,
                        "ratio_vs_HAR_M": loss / base if np.isfinite(base) and base else np.nan,
                        "gain_vs_HAR_M": 1.0 - loss / base if np.isfinite(base) and base else np.nan,
                    }
                )
            mcs = run_mcs(table, metric, alpha, bootstrap, block_size, algorithm, seed)
            mcs.insert(0, "regime", regime)
            mcs_rows.append(mcs)
    return pd.DataFrame(ratio_rows), pd.concat(mcs_rows, ignore_index=True) if mcs_rows else pd.DataFrame()


def fvu_by_regime(
    preds: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    baseline: str,
) -> pd.DataFrame:
    if baseline not in preds:
        return pd.DataFrame()
    base = preds[baseline]
    rows = []
    for regime, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        for model, pred in preds.items():
            denom = np.sum((pred - np.nanmean(pred, axis=1, keepdims=True)) ** 2, axis=1)
            numer = np.sum((pred - base) ** 2, axis=1)
            fvu_t = np.divide(numer, np.maximum(denom, EPS))
            rows.append(
                {
                    "regime": regime,
                    "model": model,
                    **infer_model_metadata(model),
                    "n_dates": int(mask.sum()),
                    "fvu_mean_vs_HAR_M": float(np.nanmean(fvu_t[mask])),
                    "fvu_median_vs_HAR_M": float(np.nanmedian(fvu_t[mask])),
                    "fvu_q25_vs_HAR_M": float(np.nanquantile(fvu_t[mask], 0.25)),
                    "fvu_q75_vs_HAR_M": float(np.nanquantile(fvu_t[mask], 0.75)),
                }
            )
    return pd.DataFrame(rows).sort_values(["regime", "fvu_mean_vs_HAR_M"]).reset_index(drop=True)


def incremental_fvu(preds: dict[str, np.ndarray]) -> pd.DataFrame:
    pairs = []
    for suffix in ["M", "Q", "M_IV", "Q_IV"]:
        for base, cand in [
            (f"GHAR_{suffix}", f"GNNHAR1L_{suffix}"),
            (f"GNNHAR1L_{suffix}", f"GNNHAR2L_{suffix}"),
            (f"GNNHAR2L_{suffix}", f"GNNHAR3L_{suffix}"),
            (f"GNNHAR3L_{suffix}", f"GNNHAR4L_{suffix}"),
            (f"GNNHAR4L_{suffix}", f"GNNHAR5L_{suffix}"),
            (f"GHAR_{suffix}", f"GHAR2H_{suffix}"),
            (f"GHAR2H_{suffix}", f"GHAR3H_{suffix}"),
        ]:
            if base in preds and cand in preds:
                pairs.append((base, cand))
    rows = []
    for base, cand in pairs:
        p_base = preds[base]
        p_cand = preds[cand]
        denom = np.sum((p_cand - np.nanmean(p_cand, axis=1, keepdims=True)) ** 2, axis=1)
        numer = np.sum((p_cand - p_base) ** 2, axis=1)
        fvu_t = np.divide(numer, np.maximum(denom, EPS))
        rows.append(
            {
                "base_model": base,
                "candidate_model": cand,
                "comparison": f"{base} -> {cand}",
                "candidate_family": infer_model_metadata(cand)["family"],
                "candidate_depth": infer_model_metadata(cand)["depth"],
                "candidate_hop": infer_model_metadata(cand)["hop"],
                "uses_iv": infer_model_metadata(cand)["uses_iv"],
                "estimation": infer_model_metadata(cand)["estimation"],
                "incremental_fvu_mean": float(np.nanmean(fvu_t)),
                "incremental_fvu_median": float(np.nanmedian(fvu_t)),
                "incremental_fvu_q25": float(np.nanquantile(fvu_t, 0.25)),
                "incremental_fvu_q75": float(np.nanquantile(fvu_t, 0.75)),
            }
        )
    return pd.DataFrame(rows)


def dm_test_columns(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> tuple[np.ndarray, np.ndarray]:
    diff = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d_bar = np.nanmean(diff, axis=0)
    n_obs = diff.shape[0]
    centered = diff - d_bar
    gamma0 = np.nanmean(centered * centered, axis=0)
    if h > 1:
        gamma = gamma0.copy()
        for lag in range(1, h):
            cov = np.nanmean(centered[lag:] * centered[:-lag], axis=0)
            gamma += 2.0 * (1.0 - lag / h) * cov
    else:
        gamma = gamma0
    var_d = gamma / max(n_obs, 1)
    stat = d_bar / np.sqrt(np.maximum(var_d, EPS))
    corr = math.sqrt((n_obs + 1 - 2 * h + h * (h - 1) / max(n_obs, 1)) / max(n_obs, 1))
    stat_corr = stat * corr
    if stats is not None:
        pvalues = 2.0 * (1.0 - stats.t.cdf(np.abs(stat_corr), df=max(n_obs - 1, 1)))
    else:
        pvalues = np.array([math.erfc(abs(x) / math.sqrt(2.0)) for x in stat_corr])
    return stat_corr, pvalues


def candidate_pairs(preds: dict[str, np.ndarray]) -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    for suffix in ["M", "Q", "M_IV", "Q_IV"]:
        for base, cand, kind in [
            (f"HAR_{suffix}", f"GHAR_{suffix}", "graph_linear"),
            (f"GHAR_{suffix}", f"GNNHAR1L_{suffix}", "nonlinear_one_hop"),
            (f"GNNHAR1L_{suffix}", f"GNNHAR2L_{suffix}", "gnn_depth"),
            (f"GNNHAR2L_{suffix}", f"GNNHAR3L_{suffix}", "gnn_depth"),
            (f"GNNHAR3L_{suffix}", f"GNNHAR4L_{suffix}", "gnn_depth"),
            (f"GNNHAR4L_{suffix}", f"GNNHAR5L_{suffix}", "gnn_depth"),
            (f"GHAR_{suffix}", f"GHAR2H_{suffix}", "linear_multihop"),
            (f"GHAR2H_{suffix}", f"GHAR3H_{suffix}", "linear_multihop"),
        ]:
            if base in preds and cand in preds:
                pairs.append((base, cand, kind))
    return pairs


def pairwise_dm_tables(qlike_losses: dict[str, np.ndarray], tickers: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    ticker_rows = []
    summary_rows = []
    for base, cand, kind in candidate_pairs(qlike_losses):
        stat, pval = dm_test_columns(qlike_losses[base], qlike_losses[cand])
        mean_base = np.nanmean(qlike_losses[base], axis=0)
        mean_cand = np.nanmean(qlike_losses[cand], axis=0)
        for ticker, s, p, lb, lc in zip(tickers, stat, pval, mean_base, mean_cand):
            ticker_rows.append(
                {
                    "comparison_type": kind,
                    "base_model": base,
                    "candidate_model": cand,
                    "ticker": ticker,
                    "dm_stat_positive_favors_candidate": float(s),
                    "pvalue": float(p),
                    "mean_qlike_base": float(lb),
                    "mean_qlike_candidate": float(lc),
                    "candidate_loss_ratio_vs_base": float(lc / lb) if lb else np.nan,
                    "candidate_gain_vs_base": float(1.0 - lc / lb) if lb else np.nan,
                }
            )
        avg_base = float(np.nanmean(mean_base))
        avg_cand = float(np.nanmean(mean_cand))
        summary_rows.append(
            {
                "comparison_type": kind,
                "base_model": base,
                "candidate_model": cand,
                "cs_dm_stat_positive_favors_candidate": float(np.nanmean(stat)),
                "cs_pvalue_mean": float(np.nanmean(pval)),
                "ticker_p_median": float(np.nanmedian(pval)),
                "ticker_share_p_lt_0_05": float(np.nanmean(pval < 0.05)),
                "ticker_share_positive_dm": float(np.nanmean(stat > 0)),
                "mean_qlike_base": avg_base,
                "mean_qlike_candidate": avg_cand,
                "candidate_loss_ratio_vs_base": avg_cand / avg_base if avg_base else np.nan,
                "candidate_gain_vs_base": 1.0 - avg_cand / avg_base if avg_base else np.nan,
            }
        )
    return pd.DataFrame(ticker_rows), pd.DataFrame(summary_rows)


def boxplot_source_tables(
    truth: np.ndarray,
    preds: dict[str, np.ndarray],
    tickers: np.ndarray,
    test_dates: pd.DatetimeIndex,
    regime: pd.DataFrame,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    regime_lookup = regime.set_index("test_date")["regime"].to_dict()
    for model, pred in preds.items():
        meta = infer_model_metadata(model)
        error = pred - truth
        ratio = np.divide(pred, np.clip(truth, EPS, None))
        flat_date = np.repeat(test_dates.astype(str).to_numpy(), len(tickers))
        flat_ticker = np.tile(tickers, len(test_dates))
        frame = pd.DataFrame(
            {
                "model": model,
                "family": meta["family"],
                "depth": meta["depth"],
                "hop": meta["hop"],
                "uses_iv": meta["uses_iv"],
                "estimation": meta["estimation"],
                "test_date": flat_date,
                "ticker": flat_ticker,
                "regime": [regime_lookup.get(str(d), "unknown") for d in flat_date],
                "forecast_error": error.reshape(-1),
                "forecast_ratio": ratio.reshape(-1),
            }
        )
        records.append(frame)
    long_df = pd.concat(records, ignore_index=True)
    stats_rows = []
    group_cols = ["model", "family", "depth", "hop", "uses_iv", "estimation", "regime"]
    for keys, grp in long_df.groupby(group_cols, dropna=False):
        key = dict(zip(group_cols, keys))
        for metric in ["forecast_error", "forecast_ratio"]:
            values = grp[metric].replace([np.inf, -np.inf], np.nan).dropna()
            stats_rows.append(
                {
                    **key,
                    "metric": metric,
                    "n_obs": int(len(values)),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "q01": float(values.quantile(0.01)),
                    "q05": float(values.quantile(0.05)),
                    "q25": float(values.quantile(0.25)),
                    "median": float(values.quantile(0.50)),
                    "q75": float(values.quantile(0.75)),
                    "q95": float(values.quantile(0.95)),
                    "q99": float(values.quantile(0.99)),
                }
            )
    stats_df = pd.DataFrame(stats_rows)
    plot_boxplots(long_df, out_dir)
    return long_df, stats_df


def zhang_plot_models(models: Iterable[str]) -> list[str]:
    priority = [
        "HAR_M",
        "HAR_Q",
        "GHAR_M",
        "GHAR_Q",
        "GNNHAR1L_M",
        "GNNHAR1L_Q",
        "GNNHAR2L_M",
        "GNNHAR2L_Q",
        "GNNHAR3L_M",
        "GNNHAR3L_Q",
        "GNNHAR4L_M",
        "GNNHAR4L_Q",
        "GNNHAR5L_M",
        "GNNHAR5L_Q",
    ]
    available = set(models)
    return [m for m in priority if m in available]


def plot_boxplots(long_df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    models = zhang_plot_models(long_df["model"].unique())
    if not models:
        return
    regimes = [
        ("all", long_df),
        ("calm_bottom_90pct", long_df[long_df["regime"] == "calm_bottom_90pct"]),
        ("turbulent_top_10pct", long_df[long_df["regime"] == "turbulent_top_10pct"]),
    ]
    for metric, center in [("forecast_error", 0.0), ("forecast_ratio", 1.0)]:
        for regime_name, data in regimes:
            data = data[data["model"].isin(models)]
            if data.empty:
                continue
            values = [data.loc[data["model"] == model, metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy() for model in models]
            fig_h = max(4.5, 0.35 * len(models))
            plt.figure(figsize=(9.0, fig_h))
            try:
                plt.boxplot(values, orientation="horizontal", showfliers=False, tick_labels=models)
            except TypeError:
                plt.boxplot(values, vert=False, showfliers=False)
                plt.yticks(range(1, len(models) + 1), models)
            plt.axvline(center, color="grey", linestyle="--", linewidth=1)
            plt.xlabel(metric.replace("_", " "))
            plt.title(f"{metric.replace('_', ' ').title()} - {regime_name}")
            plt.tight_layout()
            plt.savefig(fig_dir / f"boxplot_{metric}_{regime_name}.png", dpi=220)
            plt.close()


def raw_panel_path(raw_source_dir: Path | None, name: str) -> Path | None:
    if raw_source_dir is None:
        return None
    return raw_source_dir / name


def ticker_alignment_audit(raw_source_dir: Path | None, raw_source_reason: str, tickers: np.ndarray) -> pd.DataFrame:
    rows = []
    model_tickers = set(map(str, tickers))
    for panel, filename in [
        ("local_raw_rv_panel", "merged_rv_data_filled.csv"),
        ("local_raw_return_panel", "daily_returns.csv"),
        ("local_raw_iv_panel", "merged_iv_data_filled.csv"),
    ]:
        path = raw_panel_path(raw_source_dir, filename)
        if path is None or not path.exists():
            rows.append(
                {
                    "panel": panel,
                    "available": False,
                    "raw_source_dir": str(raw_source_dir) if raw_source_dir is not None else "",
                    "raw_source_reason": raw_source_reason,
                    "panel_file": filename,
                    "n_model_tickers": len(model_tickers),
                }
            )
            continue
        df = pd.read_csv(path)
        if "Date" not in df.columns:
            rows.append(
                {
                    "panel": panel,
                    "available": False,
                    "raw_source_dir": str(raw_source_dir),
                    "raw_source_reason": raw_source_reason,
                    "panel_file": str(path),
                    "n_model_tickers": len(model_tickers),
                    "missing_from_panel_sample": "Date column missing",
                }
            )
            continue
        df["Date"] = pd.to_datetime(df["Date"])
        panel_tickers = set(c for c in df.columns if c != "Date")
        missing_from_panel = sorted(model_tickers - panel_tickers)
        extra_in_panel = sorted(panel_tickers - model_tickers)
        rows.append(
            {
                "panel": panel,
                "available": True,
                "raw_source_dir": str(raw_source_dir),
                "raw_source_reason": raw_source_reason,
                "panel_file": str(path),
                "date_start": str(df["Date"].min().date()),
                "date_end": str(df["Date"].max().date()),
                "n_dates": int(len(df)),
                "n_panel_tickers": int(len(panel_tickers)),
                "n_model_tickers": int(len(model_tickers)),
                "n_matched_tickers": int(len(model_tickers & panel_tickers)),
                "n_missing_from_panel": int(len(missing_from_panel)),
                "n_extra_in_panel": int(len(extra_in_panel)),
                "missing_from_panel_sample": ",".join(missing_from_panel[:25]),
                "extra_in_panel_sample": ",".join(extra_in_panel[:25]),
            }
        )
    return pd.DataFrame(rows, columns=TICKER_ALIGNMENT_COLUMNS)


def align_raw_panel(raw_source_dir: Path | None, tickers: np.ndarray, dates: pd.DatetimeIndex, name: str) -> pd.DataFrame | None:
    path = raw_panel_path(raw_source_dir, name)
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    available = [t for t in tickers if t in df.columns]
    if not available:
        return None
    out = df.set_index("Date").sort_index()[available].reindex(dates)
    out.index.name = "Date"
    return out


def describe_panel(df: pd.DataFrame, panel: str) -> pd.DataFrame:
    summary = df.describe(percentiles=[0.25, 0.5, 0.75]).T.rename(
        columns={"25%": "q25", "50%": "median", "75%": "q75"}
    )
    summary.insert(0, "ticker", summary.index)
    summary.insert(0, "panel", panel)
    return summary.reset_index(drop=True)


def data_summary_statistics(
    raw_source_dir: Path | None,
    raw_source_reason: str,
    truth: np.ndarray,
    tickers: np.ndarray,
    test_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth_df = pd.DataFrame(truth, index=test_dates, columns=tickers)
    raw_rv = align_raw_panel(raw_source_dir, tickers, test_dates, "merged_rv_data_filled.csv")
    raw_ret = align_raw_panel(raw_source_dir, tickers, test_dates, "daily_returns.csv")
    raw_iv = align_raw_panel(raw_source_dir, tickers, test_dates, "merged_iv_data_filled.csv")

    summary = describe_panel(truth_df, "truth_test_array")
    raw_summary = describe_panel(raw_rv, "local_raw_rv_aligned_to_test") if raw_rv is not None else pd.DataFrame(columns=summary.columns)

    missing_rows = []
    for name, df in [
        ("truth_test_array", truth_df),
        ("local_raw_rv_aligned_to_test", raw_rv),
        ("local_raw_return_aligned_to_test", raw_ret),
        ("local_raw_iv_aligned_to_test", raw_iv),
    ]:
        if df is None:
            missing_rows.append({"panel": name, "available": False})
            continue
        missing_rows.append(
            {
                "panel": name,
                "available": True,
                "n_dates": int(df.shape[0]),
                "n_tickers": int(df.shape[1]),
                "date_start": str(df.index.min().date()),
                "date_end": str(df.index.max().date()),
                "missing_cells": int(df.isna().sum().sum()),
                "missing_share": float(df.isna().mean().mean()),
                "finite_share": float(np.isfinite(df.to_numpy(dtype=float)).mean()),
            }
        )
    missing = pd.DataFrame(missing_rows)

    corr_rows = []
    for panel, df in [("rv_truth_test_array", truth_df), ("returns_local_aligned_to_test", raw_ret), ("iv_local_aligned_to_test", raw_iv)]:
        if df is None or df.shape[1] < 2:
            continue
        for method in ["pearson", "spearman"]:
            corr = df.corr(method=method).to_numpy(dtype=float)
            tri = corr[np.triu_indices_from(corr, k=1)]
            tri = tri[np.isfinite(tri)]
            if len(tri) == 0:
                continue
            corr_rows.append(
                {
                    "panel": panel,
                    "method": method,
                    "n_pairs": int(len(tri)),
                    "mean": float(np.mean(tri)),
                    "std": float(np.std(tri, ddof=1)) if len(tri) > 1 else np.nan,
                    "q05": float(np.quantile(tri, 0.05)),
                    "q25": float(np.quantile(tri, 0.25)),
                    "median": float(np.quantile(tri, 0.5)),
                    "q75": float(np.quantile(tri, 0.75)),
                    "q95": float(np.quantile(tri, 0.95)),
                }
            )
    corr_summary = pd.DataFrame(corr_rows)

    date_rows = []
    date_rows.append(
        {
            "source": "paper_ready_test_arrays",
            "date_start": str(test_dates.min().date()),
            "date_end": str(test_dates.max().date()),
            "n_dates": int(len(test_dates)),
            "n_tickers": int(len(tickers)),
        }
    )
    for source, filename in [
        ("local_raw_rv_panel", "merged_rv_data_filled.csv"),
        ("local_raw_return_panel", "daily_returns.csv"),
        ("local_raw_iv_panel", "merged_iv_data_filled.csv"),
    ]:
        path = raw_panel_path(raw_source_dir, filename)
        if path is not None and path.exists():
            raw_full = pd.read_csv(path)
            if "Date" in raw_full:
                raw_dates = pd.to_datetime(raw_full["Date"])
                date_rows.append(
                    {
                        "source": source,
                        "date_start": str(raw_dates.min().date()),
                        "date_end": str(raw_dates.max().date()),
                        "n_dates": int(len(raw_dates)),
                        "n_tickers": int(len([c for c in raw_full.columns if c != "Date"])),
                    }
                )
    date_alignment = pd.DataFrame(date_rows)
    ticker_alignment = ticker_alignment_audit(raw_source_dir, raw_source_reason, tickers)
    return summary, raw_summary, missing, corr_summary, date_alignment, ticker_alignment


def shortest_path_distribution(adj: np.ndarray) -> dict:
    binary = (np.asarray(adj) > 0).astype(bool)
    np.fill_diagonal(binary, False)
    n = binary.shape[0]
    counts: dict[int, int] = {}
    reachable = 0
    for start in range(n):
        dist = np.full(n, -1, dtype=int)
        dist[start] = 0
        q: deque[int] = deque([start])
        while q:
            node = q.popleft()
            for nb in np.flatnonzero(binary[node]):
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    q.append(int(nb))
        for d in dist[start + 1 :]:
            if d > 0:
                counts[int(d)] = counts.get(int(d), 0) + 1
                reachable += 1
    total_pairs = n * (n - 1) // 2
    return {
        "total_pairs": total_pairs,
        "reachable_pairs": reachable,
        "unreachable_pairs": total_pairs - reachable,
        "diameter": max(counts) if counts else np.nan,
        "counts": counts,
    }


def graph_statistics(universe_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    graph_dir = universe_root / "graphs"
    rows = []
    spd_rows = []
    if not graph_dir.exists():
        return pd.DataFrame(columns=GRAPH_COLUMNS), pd.DataFrame(columns=SHORTEST_PATH_COLUMNS)
    for path in sorted(graph_dir.rglob("*.npz")):
        z = np.load(path, allow_pickle=True)
        if "adjacency" not in z.files:
            continue
        adj = np.asarray(z["adjacency"], dtype=float)
        binary = adj > 0
        np.fill_diagonal(binary, False)
        n = adj.shape[0]
        edge_count = int(binary.sum() // 2)
        density = edge_count / (n * (n - 1) / 2) if n > 1 else np.nan
        degree = binary.sum(axis=1)
        info = {}
        if "info_json" in z.files:
            try:
                info = json.loads(str(z["info_json"]))
            except Exception:
                info = {}
        rel = path.relative_to(graph_dir)
        row = {
            "graph_file": str(rel),
            "loss_group": rel.parts[0] if len(rel.parts) > 1 else "",
            "n_nodes": n,
            "edges": edge_count,
            "density": density,
            "degree_mean": float(np.mean(degree)),
            "degree_median": float(np.median(degree)),
            "degree_min": int(np.min(degree)),
            "degree_max": int(np.max(degree)),
            "method": info.get("method"),
            "alpha": info.get("alpha"),
            "fallback": info.get("fallback"),
            "train_rows": info.get("train_rows"),
        }
        spd = shortest_path_distribution(binary)
        row["diameter"] = spd["diameter"]
        row["reachable_pair_share"] = spd["reachable_pairs"] / spd["total_pairs"] if spd["total_pairs"] else np.nan
        rows.append(row)
        total = spd["total_pairs"]
        for distance, count in sorted(spd["counts"].items()):
            spd_rows.append(
                {
                    "graph_file": str(rel),
                    "loss_group": row["loss_group"],
                    "shortest_path_distance": distance,
                    "pair_count": count,
                    "pair_share_all_pairs": count / total if total else np.nan,
                    "pair_share_reachable_pairs": count / spd["reachable_pairs"] if spd["reachable_pairs"] else np.nan,
                }
            )
        if spd["unreachable_pairs"]:
            spd_rows.append(
                {
                    "graph_file": str(rel),
                    "loss_group": row["loss_group"],
                    "shortest_path_distance": "unreachable",
                    "pair_count": spd["unreachable_pairs"],
                    "pair_share_all_pairs": spd["unreachable_pairs"] / total if total else np.nan,
                    "pair_share_reachable_pairs": np.nan,
                }
            )
    return pd.DataFrame(rows, columns=GRAPH_COLUMNS), pd.DataFrame(spd_rows, columns=SHORTEST_PATH_COLUMNS)


def forecast_mad_proxy(pred: np.ndarray, adj: np.ndarray | None = None) -> float:
    rep = np.asarray(pred, dtype=float).T
    rep = rep - np.nanmean(rep, axis=1, keepdims=True)
    norm = np.linalg.norm(rep, axis=1, keepdims=True)
    rep = rep / np.maximum(norm, EPS)
    dist = 1.0 - rep @ rep.T
    if adj is not None:
        mask = np.asarray(adj) > 0
        np.fill_diagonal(mask, False)
    else:
        mask = ~np.eye(dist.shape[0], dtype=bool)
    values = dist[mask]
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else np.nan


def mad_proxy_table(preds: dict[str, np.ndarray], graph_stats: pd.DataFrame, graph_dir: Path) -> pd.DataFrame:
    adj = None
    graph_note = "unmasked forecast-level proxy; exact hidden-state MAD requires saved hidden representations"
    if not graph_stats.empty:
        first_graph = graph_dir / str(graph_stats.iloc[0]["graph_file"])
        if first_graph.exists():
            try:
                adj = np.load(first_graph, allow_pickle=True)["adjacency"]
                graph_note = f"forecast-level proxy masked by first available graph {graph_stats.iloc[0]['graph_file']}; exact hidden-state MAD requires saved hidden representations"
            except Exception:
                adj = None
    rows = []
    for model, pred in preds.items():
        rows.append(
            {
                "model": model,
                **infer_model_metadata(model),
                "mad_forecast_proxy": forecast_mad_proxy(pred, adj),
                "diagnostic_level": "forecast_proxy",
                "note": graph_note,
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "uses_iv", "estimation", "depth", "model"]).reset_index(drop=True)


def existing_mad_diagnostics(universe_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    diag_dir = universe_root / "diagnostics"
    hidden_path = diag_dir / "hidden_state_mad.csv"
    hidden_available = hidden_path.exists() and hidden_path.stat().st_size > 1
    availability = pd.DataFrame(
        [
            {
                "diagnostic": "hidden_state_mad",
                "available": bool(hidden_available),
                "source_file": str(hidden_path.relative_to(universe_root)) if hidden_path.exists() else "",
                "note": (
                    "Exact hidden-state MAD is available."
                    if hidden_available
                    else "Exact Zhang Figure 7 hidden-state MAD is not available because saved hidden representations are absent."
                ),
            }
        ]
    )

    smoothing_path = diag_dir / "mad_smoothing_diagnostics.csv"
    if smoothing_path.exists() and smoothing_path.stat().st_size > 1:
        smoothing = pd.read_csv(smoothing_path)
    else:
        smoothing = pd.DataFrame(
            columns=[
                "model",
                "model_class",
                "depth",
                "uses_iv",
                "training_loss",
                "forecast_mad_time_series",
                "mean_cross_section_std",
                "mean_cross_section_cv",
                "diagnostic_level",
                "note",
            ]
        )

    oversmoothing_path = diag_dir / "oversmoothing_depth_summary.csv"
    if oversmoothing_path.exists() and oversmoothing_path.stat().st_size > 1:
        oversmoothing = pd.read_csv(oversmoothing_path)
    else:
        oversmoothing = pd.DataFrame(
            columns=[
                "uses_iv",
                "training_loss",
                "metric",
                "depth_min",
                "depth_max",
                "n_depths",
                "pearson_corr_with_depth",
                "first_depth_value",
                "last_depth_value",
                "direction",
            ]
        )
    return availability, smoothing, oversmoothing


def write_method_manifest(out_dir: Path, universe: str, context: dict, files: list[str]) -> None:
    manifest = {
        "universe": universe,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "zhang_statistics_reference": ZHANG_STATISTICS,
        "context": context,
        "files": sorted(files),
    }
    (out_dir / "zhang_style_statistics_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def save_csv(df: pd.DataFrame, path: Path, files: list[str], root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    files.append(str(path.relative_to(root)))


def process_universe(data: dict, args: argparse.Namespace) -> dict:
    ur = data["root"]
    out_dir = ur / "zhang_style_statistics"
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    truth = data["truth"]
    tickers = data["tickers"]
    test_dates = data["test_dates"]
    preds = data["preds"]
    raw_source_dir, raw_source_reason = resolve_raw_source_dir(ur, tickers)

    mse_losses, qlike_losses = loss_arrays(truth, preds)
    mse_ticker = per_ticker_losses(mse_losses, tickers, "MSE")
    qlike_ticker = per_ticker_losses(qlike_losses, tickers, "QLIKE")
    loss_summary = loss_ratio_summary(mse_ticker, qlike_ticker, args.baseline)
    mcs_mse = run_mcs(mse_ticker, "MSE", args.alpha, args.bootstrap, args.block_size, args.algorithm, args.seed)
    mcs_qlike = run_mcs(qlike_ticker, "QLIKE", args.alpha, args.bootstrap, args.block_size, args.algorithm, args.seed)

    save_csv(mse_ticker, out_dir / "losses" / "per_ticker_mse_losses.csv", files, ur)
    save_csv(qlike_ticker, out_dir / "losses" / "per_ticker_qlike_losses.csv", files, ur)
    save_csv(loss_summary, out_dir / "losses" / "loss_ratio_summary.csv", files, ur)
    save_csv(per_date_losses(mse_losses, test_dates, "MSE"), out_dir / "losses" / "per_date_mse_losses.csv", files, ur)
    save_csv(per_date_losses(qlike_losses, test_dates, "QLIKE"), out_dir / "losses" / "per_date_qlike_losses.csv", files, ur)
    save_csv(mcs_mse, out_dir / "mcs" / "mcs_mse_by_ticker.csv", files, ur)
    save_csv(mcs_qlike, out_dir / "mcs" / "mcs_qlike_by_ticker.csv", files, ur)
    save_csv(pd.concat([mcs_mse, mcs_qlike], ignore_index=True), out_dir / "mcs" / "mcs_summary_by_ticker.csv", files, ur)

    regime_series, masks, regime_meta = market_state(ur, raw_source_dir, raw_source_reason, truth, tickers, test_dates, args.regime_quantile)
    regime_loss, regime_mcs = regime_loss_tables(
        mse_losses,
        qlike_losses,
        tickers,
        masks,
        args.baseline,
        args.alpha,
        args.bootstrap,
        args.block_size,
        args.algorithm,
        args.seed,
    )
    save_csv(regime_series, out_dir / "regimes" / "market_state_series.csv", files, ur)
    save_csv(regime_loss, out_dir / "regimes" / "regime_loss_ratios_by_ticker.csv", files, ur)
    save_csv(regime_mcs, out_dir / "regimes" / "regime_mcs_by_ticker.csv", files, ur)
    (out_dir / "regimes").mkdir(parents=True, exist_ok=True)
    regime_meta_path = out_dir / "regimes" / "regime_metadata.json"
    regime_meta_path.write_text(json.dumps(regime_meta, indent=2, default=str), encoding="utf-8")
    files.append(str(regime_meta_path.relative_to(ur)))

    fvu_regime = fvu_by_regime(preds, masks, args.baseline)
    inc_fvu = incremental_fvu(preds)
    save_csv(fvu_regime, out_dir / "fvu" / "fvu_by_regime_vs_HAR_M.csv", files, ur)
    save_csv(inc_fvu, out_dir / "fvu" / "incremental_fvu_by_model_pair.csv", files, ur)

    dm_ticker, dm_summary = pairwise_dm_tables(qlike_losses, tickers)
    save_csv(dm_ticker, out_dir / "dm" / "pairwise_qlike_dm_by_ticker.csv", files, ur)
    save_csv(dm_summary, out_dir / "dm" / "pairwise_qlike_dm_summary.csv", files, ur)

    boxplot_long, boxplot_stats = boxplot_source_tables(truth, preds, tickers, test_dates, regime_series, out_dir / "boxplots")
    save_csv(boxplot_stats, out_dir / "boxplots" / "forecast_error_ratio_boxplot_statistics.csv", files, ur)
    save_csv(boxplot_long, out_dir / "boxplots" / "forecast_error_ratio_long.csv", files, ur)
    for fig in sorted((out_dir / "boxplots" / "figures").glob("*.png")):
        files.append(str(fig.relative_to(ur)))

    summary_stats, raw_summary_stats, missingness, corr_summary, date_alignment, ticker_alignment = data_summary_statistics(
        raw_source_dir,
        raw_source_reason,
        truth,
        tickers,
        test_dates,
    )
    save_csv(summary_stats, out_dir / "data" / "rv_summary_statistics_by_ticker.csv", files, ur)
    save_csv(raw_summary_stats, out_dir / "data" / "local_raw_rv_summary_statistics_by_ticker.csv", files, ur)
    save_csv(missingness, out_dir / "data" / "panel_missingness_and_integrity.csv", files, ur)
    save_csv(corr_summary, out_dir / "data" / "return_rv_correlation_summary.csv", files, ur)
    save_csv(date_alignment, out_dir / "data" / "date_alignment_audit.csv", files, ur)
    save_csv(ticker_alignment, out_dir / "data" / "ticker_alignment_audit.csv", files, ur)

    graph_stats, spd = graph_statistics(ur)
    save_csv(graph_stats, out_dir / "graphs" / "graph_structure_summary.csv", files, ur)
    save_csv(spd, out_dir / "graphs" / "shortest_path_distance_distribution.csv", files, ur)
    mad = mad_proxy_table(preds, graph_stats, ur / "graphs")
    save_csv(mad, out_dir / "mad" / "mad_forecast_proxy.csv", files, ur)
    mad_availability, smoothing, oversmoothing = existing_mad_diagnostics(ur)
    save_csv(mad_availability, out_dir / "mad" / "mad_availability.csv", files, ur)
    save_csv(smoothing, out_dir / "mad" / "mad_smoothing_diagnostics.csv", files, ur)
    save_csv(oversmoothing, out_dir / "mad" / "oversmoothing_depth_summary.csv", files, ur)

    coverage = pd.DataFrame(ZHANG_STATISTICS)
    coverage["universe"] = data["universe"]
    coverage["implemented_file_group"] = [
        "losses/loss_ratio_summary.csv",
        "scope_gap: current saved forecasts are one-day horizon only",
        "mcs/mcs_summary_by_ticker.csv",
        "regimes/regime_loss_ratios_by_ticker.csv",
        "scope_gap: smaller validation split requires a separate rolling run",
        "boxplots/forecast_error_ratio_boxplot_statistics.csv",
        "fvu/fvu_by_regime_vs_HAR_M.csv",
        "dm/pairwise_qlike_dm_summary.csv",
        "mad/mad_forecast_proxy.csv; mad/mad_smoothing_diagnostics.csv; mad/oversmoothing_depth_summary.csv",
        "data/rv_summary_statistics_by_ticker.csv",
        "graphs/graph_structure_summary.csv",
    ]
    save_csv(coverage, out_dir / "zhang_statistics_coverage.csv", files, ur)

    context = {
        "n_dates": int(truth.shape[0]),
        "n_tickers": int(truth.shape[1]),
        "n_models": int(len(preds)),
        "date_start": str(test_dates.min().date()),
        "date_end": str(test_dates.max().date()),
        "baseline": args.baseline,
        "raw_data_source_dir": str(raw_source_dir) if raw_source_dir is not None else "",
        "raw_data_source_reason": raw_source_reason,
        "regime": regime_meta,
        "mcs": {
            "alpha": args.alpha,
            "bootstrap": args.bootstrap,
            "block_size": args.block_size,
            "algorithm": args.algorithm,
        },
        "graph_matrices_available": int(len(graph_stats)),
        "exact_hidden_state_mad_available": bool((ur / "diagnostics" / "hidden_state_mad.csv").exists() and (ur / "diagnostics" / "hidden_state_mad.csv").stat().st_size > 1),
        "local_raw_alignment": ticker_alignment.to_dict(orient="records"),
    }
    write_method_manifest(out_dir, data["universe"], context, files)
    files.append(str((out_dir / "zhang_style_statistics_manifest.json").relative_to(ur)))
    return {
        "universe": data["universe"],
        "n_dates": context["n_dates"],
        "n_tickers": context["n_tickers"],
        "n_models": context["n_models"],
        "date_start": context["date_start"],
        "date_end": context["date_end"],
        "output_dir": str(out_dir),
        "file_count": len(files),
        "graph_matrices_available": context["graph_matrices_available"],
        "regime_source": regime_meta["source"],
        "raw_data_source_dir": context["raw_data_source_dir"],
        "raw_data_source_reason": context["raw_data_source_reason"],
        "local_raw_rv_matched_tickers": int(ticker_alignment.loc[ticker_alignment["panel"] == "local_raw_rv_panel", "n_matched_tickers"].fillna(0).max()) if "n_matched_tickers" in ticker_alignment else 0,
        "local_raw_rv_missing_tickers": int(ticker_alignment.loc[ticker_alignment["panel"] == "local_raw_rv_panel", "n_missing_from_panel"].fillna(0).max()) if "n_missing_from_panel" in ticker_alignment else 0,
        "exact_hidden_state_mad_available": context["exact_hidden_state_mad_available"],
    }


def write_root_summary(result_root: Path, summaries: list[dict]) -> None:
    def read_optional_csv(path: Path) -> pd.DataFrame:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"
        table = df.copy()
        for col in table.columns:
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else str(x))
        widths = {
            col: max(len(str(col)), int(table[col].map(len).max()) if len(table) else 0)
            for col in table.columns
        }
        header = "| " + " | ".join(str(col).ljust(widths[col]) for col in table.columns) + " |"
        sep = "| " + " | ".join("-" * widths[col] for col in table.columns) + " |"
        rows = []
        for _, row in table.iterrows():
            rows.append("| " + " | ".join(str(row[col]).ljust(widths[col]) for col in table.columns) + " |")
        return "\n".join([header, sep] + rows)

    out = result_root / "zhang_style_statistics"
    out.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out / "universe_statistics_summary.csv", index=False)
    coverage = pd.DataFrame(ZHANG_STATISTICS)
    coverage.to_csv(out / "zhang_statistics_reference_coverage.csv", index=False)

    loss_rows = []
    best_rows = []
    mcs_rows = []
    dm_rows = []
    fvu_rows = []
    alignment_rows = []
    graph_rows = []
    mad_rows = []
    for summary in summaries:
        universe = summary["universe"]
        base = result_root / "universes" / universe / "zhang_style_statistics"

        loss = read_optional_csv(base / "losses" / "loss_ratio_summary.csv")
        if not loss.empty:
            loss.insert(0, "universe", universe)
            loss_rows.append(loss)
            for metric_col, gain_col in [
                ("mse_ratio_vs_HAR_M", "mse_gain_vs_HAR_M"),
                ("qlike_ratio_vs_HAR_M", "qlike_gain_vs_HAR_M"),
            ]:
                if metric_col in loss:
                    best = loss.loc[loss[metric_col].idxmin()].to_dict()
                    best_rows.append(
                        {
                            "universe": universe,
                            "selection_metric": metric_col,
                            "best_model": best.get("model"),
                            "loss_ratio_vs_HAR_M": best.get(metric_col),
                            "gain_vs_HAR_M": best.get(gain_col),
                            "family": best.get("family"),
                            "depth": best.get("depth"),
                            "hop": best.get("hop"),
                            "uses_iv": best.get("uses_iv"),
                            "estimation": best.get("estimation"),
                        }
                    )

        mcs = read_optional_csv(base / "mcs" / "mcs_summary_by_ticker.csv")
        if not mcs.empty:
            mcs.insert(0, "universe", universe)
            mcs_rows.append(mcs)

        dm = read_optional_csv(base / "dm" / "pairwise_qlike_dm_summary.csv")
        if not dm.empty:
            dm.insert(0, "universe", universe)
            dm_rows.append(dm)

        fvu = read_optional_csv(base / "fvu" / "fvu_by_regime_vs_HAR_M.csv")
        if not fvu.empty:
            fvu.insert(0, "universe", universe)
            fvu_rows.append(fvu)

        alignment = read_optional_csv(base / "data" / "ticker_alignment_audit.csv")
        if not alignment.empty:
            alignment.insert(0, "universe", universe)
            alignment_rows.append(alignment)

        graph = read_optional_csv(base / "graphs" / "graph_structure_summary.csv")
        if not graph.empty:
            graph.insert(0, "universe", universe)
            graph_rows.append(graph)

        mad = read_optional_csv(base / "mad" / "oversmoothing_depth_summary.csv")
        if not mad.empty:
            mad.insert(0, "universe", universe)
            mad_rows.append(mad)

    if loss_rows:
        pd.concat(loss_rows, ignore_index=True).to_csv(out / "cross_universe_loss_ratio_summary.csv", index=False)
    if best_rows:
        pd.DataFrame(best_rows).to_csv(out / "cross_universe_best_models.csv", index=False)
    if mcs_rows:
        mcs_all = pd.concat(mcs_rows, ignore_index=True)
        mcs_all.to_csv(out / "cross_universe_mcs_summary_by_ticker.csv", index=False)
        mcs_rollup = (
            mcs_all.groupby(["universe", "metric", "model"], dropna=False)
            .agg(
                mean_loss=("mean_loss", "mean"),
                mean_rank=("rank_mean_loss", "mean"),
                share_mcs_included=("mcs_included", "mean"),
                mean_mcs_pvalue=("mcs_pvalue", "mean"),
            )
            .reset_index()
            .sort_values(["universe", "metric", "mean_rank"])
        )
        mcs_rollup.to_csv(out / "cross_universe_mcs_rollup.csv", index=False)
    if dm_rows:
        pd.concat(dm_rows, ignore_index=True).to_csv(out / "cross_universe_pairwise_qlike_dm_summary.csv", index=False)
    if fvu_rows:
        pd.concat(fvu_rows, ignore_index=True).to_csv(out / "cross_universe_fvu_by_regime_vs_HAR_M.csv", index=False)
    if alignment_rows:
        pd.concat(alignment_rows, ignore_index=True).to_csv(out / "cross_universe_ticker_alignment_audit.csv", index=False)
    if graph_rows:
        graph_all = pd.concat(graph_rows, ignore_index=True)
        graph_all.to_csv(out / "cross_universe_graph_structure_summary.csv", index=False)
        graph_rollup = (
            graph_all.groupby(["universe", "loss_group"], dropna=False)
            .agg(
                graph_count=("graph_file", "count"),
                n_nodes_mean=("n_nodes", "mean"),
                edges_mean=("edges", "mean"),
                density_mean=("density", "mean"),
                degree_mean=("degree_mean", "mean"),
                diameter_mean=("diameter", "mean"),
                reachable_pair_share_mean=("reachable_pair_share", "mean"),
            )
            .reset_index()
        )
        graph_rollup.to_csv(out / "cross_universe_graph_rollup.csv", index=False)
    if mad_rows:
        pd.concat(mad_rows, ignore_index=True).to_csv(out / "cross_universe_oversmoothing_depth_summary.csv", index=False)

    readme_lines = [
        "# Zhang-Style Statistics Layer",
        "",
        "This folder is generated from saved out-of-sample forecasts in `paper_ready_20260617`.",
        "It follows Zhang et al.'s public code organization where possible:",
        "",
        "- `losses/`: per-ticker MSE/QLIKE loss matrices and HAR_M-normalized loss ratios.",
        "- `mcs/`: Hansen-Lunde-Nason MCS tables using per-ticker losses.",
        "- `regimes/`: calm/turbulent split using SPY when available, otherwise a recorded cross-sectional mean RV proxy.",
        "- `boxplots/`: source tables and PNGs for forecast error and forecast ratio boxplots.",
        "- `fvu/`: FVU tables relative to HAR_M and incremental FVU for matched model pairs.",
        "- `dm/`: per-ticker and cross-sectional QLIKE DM tests for graph, depth, and multi-hop comparisons.",
        "- `mad/`: exact hidden-state MAD when available; otherwise explicitly labeled forecast-level proxies.",
        "- `data/`: RV summary statistics, missingness, date alignment, and return/RV correlation summaries.",
        "- `graphs/`: graph density and shortest path distance summaries when saved graph matrices exist.",
        "",
        "Important limitations: current saved runs cover the one-day horizon, so one-week and one-month Zhang tables require separate horizon-specific runs. Current saved runs also do not include hidden representations, so exact Zhang Figure 7 hidden-state MAD cannot be recovered without rerunning models with hidden-state saving enabled.",
        "",
    ]
    readme_lines.append(markdown_table(summary_df))
    (out / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summaries = []
    for universe in args.universes:
        data = load_universe(args.result_root, universe)
        summary = process_universe(data, args)
        summaries.append(summary)
        print(json.dumps(summary, indent=2))
    write_root_summary(args.result_root, summaries)
    print(f"Root summary: {args.result_root / 'zhang_style_statistics'}")


if __name__ == "__main__":
    main()
