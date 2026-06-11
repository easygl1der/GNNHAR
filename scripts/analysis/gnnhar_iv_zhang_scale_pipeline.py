#!/usr/bin/env python3
"""Zhang-style rolling GHAR/GNNHAR scale experiment with an IV extension.

This script is separate from ``gnnhar_iv_scale_pipeline.py`` on purpose.  The
older scale script is a static split screen.  This file follows the rolling
structure in chaozhang-ox/GNNHAR more closely:

* every forecast origin uses up to 1000 prior observations;
* the next 22 observations are forecast as one block;
* neural models reserve the latest 22 pre-origin observations as validation;
* the graph is recomputed from pre-origin returns for each block;
* the Zhang graph is the non-zero pattern of the GLASSO precision matrix.

The IV variants are an explicit extension.  They append IV HAR lags as node
features and, for linear GHAR+IV, graph-aggregated IV lags.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.gnnhar_iv_pipeline import (  # noqa: E402
    EPS,
    ModelRun,
    build_iv_decomposition,
    build_mcs_table,
    build_regime_table,
    evaluate_runs,
    mse_loss,
    qlike_loss,
    save_tables,
)


@dataclass
class RollingPanel:
    universe: str
    dates: pd.DatetimeIndex
    tickers: List[str]
    target: np.ndarray
    rv_features: np.ndarray
    iv_features: np.ndarray
    fake_iv_features: np.ndarray
    returns: pd.DataFrame
    coverage: pd.DataFrame
    raw_shapes: Dict[str, Tuple[int, int]]
    source_date_range: Tuple[str, str]


@dataclass
class BlockInfo:
    block_id: int
    origin_index: int
    origin_date: str
    train_start_date: str
    valid_start_date: str
    test_end_date: str
    train_size: int
    valid_size: int
    test_size: int
    graph_method: str
    graph_alpha: float
    graph_edges: int
    graph_density: float
    graph_train_rows: int
    graph_fallback: Optional[str]


def parse_args() -> argparse.Namespace:
    default_output = (
        "/content/drive/MyDrive/GNNHAR-colab-runs/zhang-scale"
        if Path("/content").exists()
        else "outputs/zhang-scale"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-name", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--returns-file", required=True)
    parser.add_argument("--output-dir", default=default_output)
    parser.add_argument("--coverage-threshold", type=float, default=0.98)
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument("--fill-limit", type=int, default=5)
    parser.add_argument("--lookback", type=int, default=1000)
    parser.add_argument("--window", type=int, default=22)
    parser.add_argument("--valid-len", type=int, default=22)
    parser.add_argument("--first-origin-frac", type=float, default=0.70)
    parser.add_argument("--first-origin-index", type=int, default=-1)
    parser.add_argument("--max-blocks", type=int, default=0)
    parser.add_argument("--block-stride", type=int, default=22)
    parser.add_argument("--graph-method", choices=["glasso_cv", "glasso", "corr"], default="glasso_cv")
    parser.add_argument("--glasso-alpha", type=float, default=0.05)
    parser.add_argument("--glasso-alpha-grid", default="0.01,0.03,0.05,0.08,0.1,0.2")
    parser.add_argument("--glasso-cv-folds", type=int, default=3)
    parser.add_argument("--glasso-max-iter", type=int, default=600)
    parser.add_argument("--glasso-tol", type=float, default=1e-4)
    parser.add_argument("--max-neighbors", type=int, default=0)
    parser.add_argument("--zhang-exact", action="store_true")
    parser.add_argument("--models", default="HAR,GHAR,HAR+IV,GHAR+IV,GNNHAR1L,GNNHAR1L-IV")
    parser.add_argument("--hidden-grid", default="9,16")
    parser.add_argument("--lr-grid", default="0.001,0.0003")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-nn", type=int, default=1)
    parser.add_argument("--ensemble-screen-percentile", type=float, default=50.0)
    parser.add_argument("--loss", choices=["MSE", "QLIKE"], default="MSE")
    parser.add_argument("--mcs-bootstrap", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--prediction-floor-quantile", type=float, default=0.001)
    parser.add_argument("--prediction-floor-value", type=float, default=0.0)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def parse_csv_items(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_float_grid(text: str) -> List[float]:
    return [float(item) for item in parse_csv_items(text)]


def parse_int_grid(text: str) -> List[int]:
    return [int(float(item)) for item in parse_csv_items(text)]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def read_wide_csv(path: Path, positive: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"date": "Date"})
    if "Date" not in df.columns:
        raise ValueError(f"{path} must contain Date or date")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if positive:
        df = df.where(df > 0)
    return df


def fill_internal_gaps(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    out = df.copy()
    kwargs = {"method": "time", "limit_direction": "both"}
    if limit > 0:
        kwargs["limit"] = limit
    for col in out.columns:
        series = out[col]
        first = series.first_valid_index()
        last = series.last_valid_index()
        if first is None or last is None:
            continue
        out.loc[first:last, col] = series.loc[first:last].interpolate(**kwargs)
    return out


def lag_average(df: pd.DataFrame, lags: int) -> pd.DataFrame:
    return sum(df.shift(i) for i in range(1, lags + 1)) / float(lags)


def build_feature_tensor(source: pd.DataFrame) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    pieces = [lag_average(source, 1), lag_average(source, 5), lag_average(source, 22)]
    feature_df = pd.concat(pieces, axis=1, keys=["d", "w", "m"]).dropna(axis=0, how="any")
    dates = feature_df.index
    arr = np.stack([feature_df[key].to_numpy(dtype=np.float32) for key in ["d", "w", "m"]], axis=2)
    return pd.DatetimeIndex(dates), arr


def select_tickers(
    rv: pd.DataFrame,
    iv: pd.DataFrame,
    returns: pd.DataFrame,
    threshold: float,
    max_tickers: int,
) -> Tuple[List[str], pd.DataFrame]:
    tickers = sorted(set(rv.columns) & set(iv.columns) & set(returns.columns))
    common_dates = rv.index.intersection(iv.index).intersection(returns.index)
    if not len(common_dates):
        raise ValueError("RV, IV, and returns do not share any dates")
    coverage = pd.DataFrame(
        {
            "rv_coverage": rv.loc[common_dates, tickers].notna().mean(),
            "iv_coverage": iv.loc[common_dates, tickers].notna().mean(),
            "returns_coverage": returns.loc[common_dates, tickers].notna().mean(),
        }
    )
    coverage["min_coverage"] = coverage.min(axis=1)
    coverage = coverage.sort_values(["min_coverage", "iv_coverage", "rv_coverage"], ascending=False)
    selected = coverage.index[coverage["min_coverage"] >= threshold].tolist()
    if max_tickers > 0:
        selected = selected[:max_tickers]
    selected = sorted(selected)
    if len(selected) < 5:
        raise ValueError(f"Only {len(selected)} tickers passed coverage threshold {threshold}")
    return selected, coverage


def load_panel(args: argparse.Namespace) -> RollingPanel:
    data_dir = Path(args.data_dir)
    rv_raw = read_wide_csv(data_dir / "merged_rv_data_filled.csv", positive=True)
    iv_raw = read_wide_csv(data_dir / "merged_iv_data_filled.csv", positive=True)
    returns_raw = read_wide_csv(Path(args.returns_file), positive=False)
    tickers, coverage = select_tickers(rv_raw, iv_raw, returns_raw, args.coverage_threshold, args.max_tickers)
    common_dates = rv_raw.index.intersection(iv_raw.index).intersection(returns_raw.index)
    rv = fill_internal_gaps(rv_raw.loc[common_dates, tickers], args.fill_limit)
    iv = fill_internal_gaps(iv_raw.loc[common_dates, tickers], args.fill_limit)
    returns = fill_internal_gaps(returns_raw.loc[common_dates, tickers], args.fill_limit)

    rv_dates, rv_features = build_feature_tensor(rv)
    iv_dates, iv_features = build_feature_tensor(iv)
    dates = rv_dates.intersection(iv_dates).intersection(rv.index).intersection(returns.index)
    rv_features = rv_features[rv_dates.get_indexer(dates)]
    iv_features = iv_features[iv_dates.get_indexer(dates)]
    target = rv.loc[dates, tickers].to_numpy(dtype=np.float32)

    valid = np.isfinite(target).all(axis=1)
    valid &= np.isfinite(rv_features).all(axis=(1, 2))
    valid &= np.isfinite(iv_features).all(axis=(1, 2))
    returns_aligned = returns.loc[dates, tickers]
    valid &= np.isfinite(returns_aligned.to_numpy(dtype=float)).all(axis=1)

    dates = pd.DatetimeIndex(dates[valid])
    target = target[valid]
    rv_features = rv_features[valid]
    iv_features = iv_features[valid]
    returns_aligned = returns_aligned.loc[dates, tickers]
    if len(dates) < args.valid_len + args.window + 80:
        raise ValueError(f"Only {len(dates)} valid dates remain after alignment")

    rng = np.random.default_rng(args.seed)
    fake_iv_features = iv_features[rng.permutation(len(dates))].copy()
    return RollingPanel(
        universe=args.universe_name,
        dates=dates,
        tickers=tickers,
        target=target,
        rv_features=rv_features,
        iv_features=iv_features,
        fake_iv_features=fake_iv_features,
        returns=returns_aligned,
        coverage=coverage,
        raw_shapes={
            "rv": (int(rv_raw.shape[0]), int(rv_raw.shape[1])),
            "iv": (int(iv_raw.shape[0]), int(iv_raw.shape[1])),
            "returns": (int(returns_raw.shape[0]), int(returns_raw.shape[1])),
        },
        source_date_range=(str(common_dates.min().date()), str(common_dates.max().date())),
    )


def prediction_floor(panel: RollingPanel, train_indices: np.ndarray, args: argparse.Namespace) -> float:
    values = panel.target[train_indices]
    positive = values[np.isfinite(values) & (values > 0)]
    if len(positive) == 0:
        return EPS
    return max(EPS, float(np.quantile(positive, args.prediction_floor_quantile)), float(args.prediction_floor_value))


def standardized_return_values(returns: pd.DataFrame) -> np.ndarray:
    values = returns.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="all").fillna(0.0).to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = values - values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    return values / np.where(scale < EPS, 1.0, scale)


def normalize_binary_adjacency(mask: np.ndarray, max_neighbors: int) -> Tuple[np.ndarray, int, float]:
    adj = np.asarray(mask, dtype=float).copy()
    np.fill_diagonal(adj, 0.0)
    adj = np.maximum(adj, adj.T)
    if max_neighbors > 0 and max_neighbors < adj.shape[0] - 1:
        keep = np.zeros_like(adj, dtype=bool)
        for i in range(adj.shape[0]):
            idx = np.flatnonzero(adj[i] > 0)
            if len(idx) > max_neighbors:
                idx = idx[:max_neighbors]
            keep[i, idx] = True
        keep = np.logical_or(keep, keep.T)
        adj = np.where(keep, adj, 0.0)
    n_edges = int(np.count_nonzero(np.triu(adj > 0, 1)))
    if n_edges == 0:
        corr = np.ones_like(adj) - np.eye(adj.shape[0])
        return normalize_binary_adjacency(corr, max(1, min(max_neighbors or 5, adj.shape[0] - 1)))
    degrees = adj.sum(axis=1)
    d_inv = np.diag(1.0 / np.sqrt(degrees + EPS))
    normalized = d_inv @ adj @ d_inv
    density = n_edges / max(1.0, adj.shape[0] * (adj.shape[0] - 1) / 2.0)
    return normalized.astype(np.float32), n_edges, float(density)


def normalize_weight_adjacency(weights: np.ndarray, max_neighbors: int) -> Tuple[np.ndarray, int, float]:
    adj = np.asarray(weights, dtype=float).copy()
    adj = np.nan_to_num(np.abs(adj), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(adj, 0.0)
    adj = np.maximum(adj, adj.T)
    if max_neighbors > 0 and max_neighbors < adj.shape[0] - 1:
        keep = np.zeros_like(adj, dtype=bool)
        for i in range(adj.shape[0]):
            row = adj[i].copy()
            row[i] = 0.0
            nonzero = np.flatnonzero(row > 0)
            if len(nonzero) == 0:
                continue
            k = min(max_neighbors, len(nonzero))
            idx = nonzero[np.argpartition(row[nonzero], -k)[-k:]]
            keep[i, idx] = True
        keep = np.logical_or(keep, keep.T)
        adj = np.where(keep, adj, 0.0)
    n_edges = int(np.count_nonzero(np.triu(adj > 0, 1)))
    if n_edges == 0:
        corr = np.ones_like(adj) - np.eye(adj.shape[0])
        return normalize_weight_adjacency(corr, max(1, min(max_neighbors or 5, adj.shape[0] - 1)))
    degrees = adj.sum(axis=1)
    d_inv = np.diag(1.0 / np.sqrt(degrees + EPS))
    normalized = d_inv @ adj @ d_inv
    density = n_edges / max(1.0, adj.shape[0] * (adj.shape[0] - 1) / 2.0)
    return normalized.astype(np.float32), n_edges, float(density)


def fit_precision_graph(values: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, float, str, Optional[str]]:
    fallback: Optional[str] = None
    method = args.graph_method
    if method == "corr":
        corr = np.abs(np.corrcoef(values, rowvar=False))
        return corr, float("nan"), "corr", None

    try:
        if method == "glasso_cv":
            from sklearn.covariance import GraphicalLassoCV

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = GraphicalLassoCV(
                    cv=args.glasso_cv_folds,
                    max_iter=args.glasso_max_iter,
                    tol=args.glasso_tol,
                    n_jobs=None,
                ).fit(values)
            return model.precision_, float(model.alpha_), "glasso_cv", None
        from sklearn.covariance import GraphicalLasso

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GraphicalLasso(
                alpha=args.glasso_alpha,
                max_iter=args.glasso_max_iter,
                tol=args.glasso_tol,
            ).fit(values)
        return model.precision_, float(args.glasso_alpha), "glasso", None
    except Exception as exc:
        fallback = f"{type(exc).__name__}: {exc}"
        if args.zhang_exact:
            raise

    # Robust empirical fallback: shrink returns before trying fixed alpha values.
    try:
        from sklearn.covariance import GraphicalLasso, LedoitWolf

        cov = LedoitWolf().fit(values).covariance_
        for alpha in parse_float_grid(args.glasso_alpha_grid):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = GraphicalLasso(
                        alpha=alpha,
                        covariance="precomputed",
                        max_iter=args.glasso_max_iter,
                        tol=max(args.glasso_tol, 1e-4),
                    ).fit(cov)
                return model.precision_, float(alpha), "glasso_shrunk", fallback
            except Exception as inner_exc:
                fallback = f"{fallback}; shrunk alpha {alpha} failed: {type(inner_exc).__name__}: {inner_exc}"
    except TypeError:
        # Older scikit-learn does not support covariance='precomputed'.
        pass
    except Exception as exc:
        fallback = f"{fallback}; shrinkage retry failed: {type(exc).__name__}: {exc}"

    corr = np.abs(np.corrcoef(values, rowvar=False))
    return corr, float("nan"), f"{method}_fallback_corr", f"{fallback}; used absolute correlation graph"


def build_block_adjacency(
    panel: RollingPanel,
    origin: int,
    train_start: int,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, object]]:
    ret_window = panel.returns.iloc[train_start:origin]
    values = standardized_return_values(ret_window)
    precision, alpha, method, fallback = fit_precision_graph(values, args)
    if method.startswith("glasso"):
        if args.zhang_exact:
            mask = np.asarray(np.abs(precision) > 0, dtype=float)
        else:
            mask = np.asarray(np.abs(precision) > 1e-10, dtype=float)
        adj, n_edges, density = normalize_binary_adjacency(mask, args.max_neighbors)
    else:
        adj, n_edges, density = normalize_weight_adjacency(precision, args.max_neighbors)
    info = {
        "method": method,
        "alpha": alpha,
        "edges": n_edges,
        "density": density,
        "train_rows": int(values.shape[0]),
        "fallback": fallback,
    }
    return adj, info


def apply_graph(features: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    return np.einsum("ij,tjf->tif", adjacency, features)


def linear_hop_depth(model: str) -> int:
    if model.startswith("GHAR2H"):
        return 2
    if model.startswith("GHAR3H"):
        return 3
    return 1


def adjacency_powers(adjacency: np.ndarray, hop_depth: int) -> List[np.ndarray]:
    powers: List[np.ndarray] = []
    base = np.asarray(adjacency, dtype=np.float32)
    current = base.copy()
    for depth in range(1, hop_depth + 1):
        if depth > 1:
            current = current @ base
            np.fill_diagonal(current, 0.0)
            max_abs = float(np.max(np.abs(current))) if current.size else 0.0
            if max_abs > 1.0:
                current = current / max_abs
        powers.append(current.astype(np.float32).copy())
    return powers


def make_linear_design(panel: RollingPanel, model: str, adjacency: Optional[np.ndarray]) -> np.ndarray:
    blocks = [panel.rv_features]
    use_iv = "+IV" in model
    use_fake = "fakeIV" in model
    iv = panel.fake_iv_features if use_fake else panel.iv_features
    if use_iv or use_fake:
        blocks.append(iv)
    if model.startswith("GHAR"):
        if adjacency is None:
            raise ValueError("GHAR design requires adjacency")
        for graph in adjacency_powers(adjacency, linear_hop_depth(model)):
            blocks.append(apply_graph(panel.rv_features, graph))
            if use_iv or use_fake:
                blocks.append(apply_graph(iv, graph))
    return np.concatenate(blocks, axis=2)


def fit_linear_block(
    design: np.ndarray,
    target: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    pred_floor: float,
) -> np.ndarray:
    from sklearn.linear_model import LinearRegression

    x_train = design[train_idx].reshape(-1, design.shape[-1])
    y_train = target[train_idx].reshape(-1)
    model = LinearRegression()
    model.fit(x_train, y_train)
    pred = model.predict(design[test_idx].reshape(-1, design.shape[-1])).reshape(len(test_idx), target.shape[1])
    for ticker_idx in range(target.shape[1]):
        ticker_train = target[train_idx, ticker_idx]
        ticker_floor = float(np.nanmin(ticker_train[ticker_train > 0])) if np.any(ticker_train > 0) else pred_floor
        pred[:, ticker_idx] = np.where(pred[:, ticker_idx] <= 0, max(pred_floor, ticker_floor), pred[:, ticker_idx])
    return np.clip(pred, pred_floor, None).astype(np.float32)


def fit_torch_linear_block(
    design: np.ndarray,
    target: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    test_idx: np.ndarray,
    loss_name: str,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    pred_floor: float,
) -> np.ndarray:
    import torch
    import torch.nn as nn

    x_scaled, y_scaled, y_stats = standardize_features(design, train_idx, valid_idx, test_idx, target, loss_name)
    n_train = len(train_idx)
    n_valid = len(valid_idx)
    test_local = np.arange(n_train + n_valid, x_scaled.shape[0])
    train_local = np.arange(n_train, dtype=np.int64)
    valid_local = np.arange(n_train, n_train + n_valid, dtype=np.int64)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_t = torch.tensor(x_scaled, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_scaled, dtype=torch.float32, device=device)
    model = nn.Linear(design.shape[-1], 1, bias=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    batch_size = max(1, min(batch_size, n_train))
    y_mean = y_stats["y_mean"]
    y_std = y_stats["y_std"]
    use_qlike = loss_name.upper() == "QLIKE"

    def forward(values: torch.Tensor) -> torch.Tensor:
        return model(values).squeeze(-1)

    def criterion(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        if use_qlike:
            pred_orig = torch.exp(pred * y_std + y_mean).clamp_min(EPS)
            truth_orig = torch.exp(truth * y_std + y_mean).clamp_min(EPS)
            ratio = truth_orig / pred_orig
            return (ratio - torch.log(ratio) - 1.0).mean()
        return ((truth - pred) ** 2).mean()

    best_state = None
    best_valid = float("inf")
    stale = 0
    patience = max(20, epochs // 6)
    valid_tensor = torch.tensor(valid_local, dtype=torch.long, device=device)
    for _epoch in range(epochs):
        order = train_local.copy()
        rng.shuffle(order)
        model.train()
        for start in range(0, len(order), batch_size):
            batch = torch.tensor(order[start : start + batch_size], dtype=torch.long, device=device)
            optimizer.zero_grad()
            loss = criterion(forward(x_t[batch]), y_t[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = float(criterion(forward(x_t[valid_tensor]), y_t[valid_tensor]).detach().cpu().item())
        if valid_loss + 1e-9 < best_valid:
            best_valid = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(test_local, dtype=torch.long, device=device)
        pred_scaled = forward(x_t[test_tensor]).detach().cpu().numpy()
    if use_qlike:
        pred = np.exp(pred_scaled * y_std + y_mean)
    else:
        pred = pred_scaled * y_std + y_mean
    return np.clip(pred, pred_floor, None).astype(np.float32)


def standardize_features(
    x: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    test_idx: np.ndarray,
    y: np.ndarray,
    loss_name: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    train_x = x[train_idx].reshape(-1, x.shape[-1])
    mean = train_x.mean(axis=0, keepdims=True)
    std = np.where(train_x.std(axis=0, keepdims=True) < EPS, 1.0, train_x.std(axis=0, keepdims=True))
    x_scaled = (x[np.r_[train_idx, valid_idx, test_idx]] - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)

    y_block = y[np.r_[train_idx, valid_idx, test_idx]]
    y_work = np.log(np.clip(y_block, EPS, None)) if loss_name.upper() == "QLIKE" else y_block
    train_y = y_work[: len(train_idx)]
    y_mean = float(train_y.mean())
    y_std = float(train_y.std())
    if y_std < EPS:
        y_std = 1.0
    y_scaled = (y_work - y_mean) / y_std
    return x_scaled.astype(np.float32), y_scaled.astype(np.float32), {"y_mean": y_mean, "y_std": y_std}


def train_gnn_block(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    adjacency: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    test_idx: np.ndarray,
    hidden: int,
    lr: float,
    epochs: int,
    batch_size: int,
    num_nn: int,
    screen_percentile: float,
    loss_name: str,
    seed: int,
    pred_floor: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    import torch
    import torch.nn as nn

    class GraphConvLayer(nn.Module):
        def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(in_features, out_features))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))
            self.bias = nn.Parameter(torch.ones(1, out_features)) if bias else None

        def forward(self, node_feature: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
            h = torch.matmul(node_feature, self.weight)
            out = torch.matmul(adj, h)
            if self.bias is not None:
                out = out + self.bias
            return out

    class ZhangGNNHAR(nn.Module):
        def __init__(self, in_features: int, hidden_features: int, layers: int) -> None:
            super().__init__()
            self.linear1 = nn.Linear(in_features, 1, bias=True)
            self.gcn1 = GraphConvLayer(in_features, hidden_features, bias=False)
            self.extra = nn.ModuleList(
                [GraphConvLayer(hidden_features, hidden_features, bias=False) for _ in range(max(0, layers - 1))]
            )
            self.mlp1 = nn.Linear(hidden_features, 1, bias=False)
            self.relu = nn.ReLU()

        def forward(self, node_feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
            h1 = self.linear1(node_feat)
            h2 = self.relu(self.gcn1(node_feat, adj))
            for layer in self.extra:
                h2 = self.relu(layer(h2, adj))
            h2 = self.mlp1(h2)
            return self.relu(h1 + h2).squeeze(-1)

    layer_match = re.search(r"GNNHAR(\d+)L", model_name)
    layers = int(layer_match.group(1)) if layer_match else 1

    x_scaled, y_scaled, y_stats = standardize_features(x, train_idx, valid_idx, test_idx, y, loss_name)
    n_train = len(train_idx)
    n_valid = len(valid_idx)
    test_local = np.arange(n_train + n_valid, x_scaled.shape[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_t = torch.tensor(x_scaled, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_scaled, dtype=torch.float32, device=device)
    adj_t = torch.tensor(adjacency, dtype=torch.float32, device=device)
    train_local = np.arange(n_train, dtype=np.int64)
    valid_local = np.arange(n_train, n_train + n_valid, dtype=np.int64)
    batch_size = max(1, min(batch_size, n_train))
    rng = np.random.default_rng(seed)

    use_qlike = loss_name.upper() == "QLIKE"
    y_mean = y_stats["y_mean"]
    y_std = y_stats["y_std"]

    def criterion(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        if use_qlike:
            pred_orig = torch.exp(pred * y_std + y_mean).clamp_min(EPS)
            truth_orig = torch.exp(truth * y_std + y_mean).clamp_min(EPS)
            ratio = truth_orig / pred_orig
            return (ratio - torch.log(ratio) - 1.0).mean()
        return ((truth - pred) ** 2).mean()

    pred_list: List[np.ndarray] = []
    valid_losses: List[float] = []
    train_histories: List[Dict[str, object]] = []
    for model_i in range(max(1, num_nn)):
        torch.manual_seed(seed + model_i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + model_i)
        model = ZhangGNNHAR(x.shape[-1], hidden, layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        best_state = None
        best_valid = float("inf")
        stale = 0
        patience = max(20, epochs // 6)
        valid_trace: List[float] = []
        train_trace: List[float] = []
        for _epoch in range(epochs):
            order = train_local.copy()
            rng.shuffle(order)
            batch_losses = []
            model.train()
            for start in range(0, len(order), batch_size):
                batch = torch.tensor(order[start : start + batch_size], dtype=torch.long, device=device)
                optimizer.zero_grad()
                pred = model(x_t[batch], adj_t)
                loss = criterion(pred, y_t[batch])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu().item()))
            model.eval()
            with torch.no_grad():
                valid_tensor = torch.tensor(valid_local, dtype=torch.long, device=device)
                valid_loss = float(criterion(model(x_t[valid_tensor], adj_t), y_t[valid_tensor]).detach().cpu().item())
            valid_trace.append(valid_loss)
            train_trace.append(float(np.mean(batch_losses)))
            if valid_loss + 1e-9 < best_valid:
                best_valid = valid_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_tensor = torch.tensor(test_local, dtype=torch.long, device=device)
            pred_scaled = model(x_t[test_tensor], adj_t).detach().cpu().numpy()
        if use_qlike:
            pred = np.exp(pred_scaled * y_std + y_mean)
        else:
            pred = pred_scaled * y_std + y_mean
        pred = np.clip(pred, pred_floor, None).astype(np.float32)
        pred_list.append(pred)
        valid_losses.append(best_valid)
        train_histories.append({"model_index": model_i, "best_valid": best_valid, "epochs_run": len(valid_trace)})

    threshold = np.percentile(valid_losses, screen_percentile)
    selected = [idx for idx, loss in enumerate(valid_losses) if loss <= threshold]
    if not selected:
        selected = [int(np.argmin(valid_losses))]
    pred = np.stack([pred_list[idx] for idx in selected], axis=0).mean(axis=0)
    info = {
        "hidden": hidden,
        "lr": lr,
        "layers": layers,
        "num_nn": int(max(1, num_nn)),
        "selected_ensemble": selected,
        "valid_loss": float(np.mean([valid_losses[idx] for idx in selected])),
        "all_valid_losses": [float(value) for value in valid_losses],
        "histories": train_histories,
    }
    return pred.astype(np.float32), info


def tune_and_fit_gnn_block(
    model_name: str,
    design: np.ndarray,
    panel: RollingPanel,
    adjacency: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    test_idx: np.ndarray,
    hidden_grid: Sequence[int],
    lr_grid: Sequence[float],
    args: argparse.Namespace,
    block_seed: int,
    pred_floor: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    best_pred: Optional[np.ndarray] = None
    best_info: Optional[Dict[str, object]] = None
    records: List[Dict[str, object]] = []
    best_key = (float("inf"), float("inf"), float("inf"))
    for hidden in hidden_grid:
        for lr in lr_grid:
            pred, info = train_gnn_block(
                model_name=model_name,
                x=design,
                y=panel.target,
                adjacency=adjacency,
                train_idx=train_idx,
                valid_idx=valid_idx,
                test_idx=test_idx,
                hidden=hidden,
                lr=lr,
                epochs=args.epochs,
                batch_size=args.batch_size,
                num_nn=args.num_nn,
                screen_percentile=args.ensemble_screen_percentile,
                loss_name=args.loss,
                seed=block_seed + hidden + int(lr * 1_000_000),
                pred_floor=pred_floor,
            )
            valid_loss = float(info["valid_loss"])
            record = dict(info)
            record.update({"model": model_name, "selected": False})
            records.append(record)
            key = (valid_loss, hidden, lr)
            if key < best_key:
                best_key = key
                best_pred = pred
                best_info = record
    if best_pred is None or best_info is None:
        raise RuntimeError(f"No GNN fit succeeded for {model_name}")
    for record in records:
        record["selected"] = (
            record["model"] == best_info["model"]
            and record["hidden"] == best_info["hidden"]
            and abs(record["lr"] - best_info["lr"]) < 1e-15
        )
    best_info = dict(best_info)
    best_info["grid_records"] = records
    return best_pred, best_info


def build_origins(n_obs: int, args: argparse.Namespace) -> List[int]:
    if args.first_origin_index >= 0:
        first = args.first_origin_index
    else:
        first = max(args.lookback, args.valid_len + 5, int(n_obs * args.first_origin_frac))
    first = min(max(first, args.valid_len + 5), n_obs - args.window)
    origins = list(range(first, n_obs - 1, args.block_stride))
    origins = [origin for origin in origins if origin + 1 <= n_obs and len(range(origin, min(origin + args.window, n_obs))) > 0]
    if args.max_blocks > 0:
        origins = origins[: args.max_blocks]
    if not origins:
        raise ValueError("No rolling forecast origins were generated")
    return origins


def model_family(name: str) -> str:
    if name.startswith("GNNHAR"):
        return "GNNHAR"
    if name.startswith("GHAR"):
        return "GHAR"
    return "HAR"


def model_iv_channel(name: str) -> str:
    if "fakeIV" in name:
        return "fake"
    if "+IV" in name or name.endswith("-IV"):
        return "real"
    return "none"


def model_adjacency(name: str) -> str:
    if name.startswith("GHAR"):
        return f"Rolling GLASSO-{linear_hop_depth(name)}hop"
    if name.startswith("GNNHAR"):
        return "Rolling GLASSO"
    return "Identity"


def empty_prediction(panel: RollingPanel) -> np.ndarray:
    return np.full(panel.target.shape, np.nan, dtype=np.float32)


def save_predictions(output_dir: Path, panel: RollingPanel, runs: List[ModelRun], test_mask: np.ndarray) -> None:
    test_idx = np.flatnonzero(test_mask)
    arrays = {
        "truth": panel.target[test_idx].astype(np.float32),
        "dates": panel.dates[test_idx].astype(str).to_numpy(),
        "tickers": np.asarray(panel.tickers),
    }
    for run in runs:
        key = "pred_" + run.name.replace("+", "plus").replace("-", "_")
        arrays[key] = run.prediction[test_idx].astype(np.float32)
        pred_df = pd.DataFrame(run.prediction[test_idx], index=panel.dates[test_idx], columns=panel.tickers)
        pred_df.to_csv(output_dir / "predictions" / f"{run.name.replace('+', 'plus').replace('-', '_')}.csv")
    np.savez_compressed(output_dir / "predictions_test.npz", **arrays)
    truth_df = pd.DataFrame(panel.target[test_idx], index=panel.dates[test_idx], columns=panel.tickers)
    truth_df.to_csv(output_dir / "predictions" / "truth.csv")


def save_figures(panel: RollingPanel, runs: List[ModelRun], loss_table: pd.DataFrame, output_dir: Path, test_mask: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table = loss_table.sort_values("test_qlike").head(min(14, len(loss_table)))
    plt.figure(figsize=(10, 4.8))
    plt.bar(table["model"], 1.0 - table["qlike_ratio_vs_har"], color="#596f62")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("QLIKE improvement vs HAR")
    plt.title(f"{panel.universe}: rolling Zhang-style test losses")
    plt.tight_layout()
    plt.savefig(fig_dir / "rolling_qlike_gain_vs_har.png", dpi=220)
    plt.close()

    test_idx = np.flatnonzero(test_mask)
    truth = panel.target[test_idx]
    top_names = table["model"].head(min(8, len(table))).tolist()
    predictions = {run.name: run.prediction[test_idx] for run in runs}
    plt.figure(figsize=(10, 4.8))
    plt.boxplot(
        [np.ravel(np.abs(truth - predictions[name])) for name in top_names],
        tick_labels=top_names,
        showfliers=False,
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("absolute forecast error")
    plt.title("Rolling forecast error distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "rolling_error_boxplot.png", dpi=220)
    plt.close()


def write_report(
    output_dir: Path,
    panel: RollingPanel,
    args: argparse.Namespace,
    block_table: pd.DataFrame,
    loss_table: pd.DataFrame,
    iv_decomposition: pd.DataFrame,
) -> None:
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    best = loss_table.sort_values("test_qlike").iloc[0]
    fallback_blocks = int(block_table["graph_fallback"].notna().sum()) if "graph_fallback" in block_table else 0
    lines = [
        f"# Zhang-style Rolling Scale Experiment: {panel.universe}",
        "",
        "## Design",
        "",
        "This run uses Zhang's rolling forecast structure rather than the earlier static split. For each forecast origin, the graph is recomputed from pre-origin returns over a window of at most 1000 observations; neural models use the last 22 pre-origin observations as validation and forecast the next 22 trading days. Linear HAR/GHAR baselines use `LinearRegression` and the same non-positive forecast replacement rule used in Zhang's public `GHAR.py`.",
        "",
        "The IV models are an extension: IV HAR lags are appended to the node features, and GHAR+IV also includes graph-aggregated IV lags. These rows should be interpreted as the user's research extension, not as part of Zhang's original baseline.",
        "",
        "## Data",
        "",
        f"- Assets selected: {len(panel.tickers)}",
        f"- Valid aligned dates after HAR lags: {panel.dates.min().date()} to {panel.dates.max().date()}",
        f"- Source common date range: {panel.source_date_range[0]} to {panel.source_date_range[1]}",
        f"- Coverage threshold: {args.coverage_threshold:.3f}",
        f"- Rolling blocks: {len(block_table)}",
        f"- Forecasted dates: {int(loss_table.attrs.get('n_test_dates', 0))}",
        "",
        "## Graph Audit",
        "",
        f"- Requested graph method: {args.graph_method}",
        f"- Blocks with graph fallback: {fallback_blocks} / {len(block_table)}",
        f"- Median graph density: {float(block_table['graph_density'].median()):.6g}",
        f"- Median graph edges: {float(block_table['graph_edges'].median()):.1f}",
        "",
        "## Result",
        "",
        f"The best model by rolling test QLIKE is `{best['model']}` with QLIKE {best['test_qlike']:.6g} and MSE {best['test_mse']:.6g}.",
        "",
        "The complete ranking is in `tables/model_losses.csv`. The graph log is in `tables/graph_blocks.csv`, which is the first place to check before interpreting S&P 500 scale effects.",
        "",
        "## IV Decomposition",
        "",
    ]
    if iv_decomposition.empty:
        lines.append("IV decomposition is unavailable because the corresponding fake-IV controls were not run.")
    else:
        for _, row in iv_decomposition.iterrows():
            lines.append(
                f"- {row['family']}: total QLIKE improvement {row['total_iv_improvement']:.6g}; "
                f"real-IV information gain {row['genuine_information_gain']:.6g}; "
                f"parameter-expansion gain {row['parameter_expansion_gain']:.6g}."
            )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "If S&P 500 still has graph fallback blocks, do not describe its graph result as a clean GLASSO scale result. If the graph succeeds but gains shrink, the likely explanations to test next are graph density/neighbor caps, sector heterogeneity, noisy IV coverage, and GNN hyperparameter capacity rather than data-pairing failure.",
        ]
    )
    (report_dir / "zhang_rolling_scale_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_metadata(
    output_dir: Path,
    panel: RollingPanel,
    args: argparse.Namespace,
    block_table: pd.DataFrame,
    test_mask: np.ndarray,
) -> None:
    excluded = panel.coverage.index.difference(panel.tickers).tolist()
    metadata = {
        "args": vars(args),
        "implementation": "zhang_style_rolling",
        "zhang_reference": {
            "repo": "https://github.com/chaozhang-ox/GNNHAR",
            "rolling_window": "origin every 22 days; up to 1000 prior observations; 22 validation observations for neural models",
            "graph": "GraphicalLassoCV precision nonzero pattern with symmetric degree normalization",
            "linear_baseline": "sklearn LinearRegression with non-positive forecast replacement by ticker training minimum",
            "iv_extension": "appends IV HAR lags and graph-aggregated IV lags; not in original Zhang baseline",
        },
        "panel": {
            "universe": panel.universe,
            "raw_shapes": panel.raw_shapes,
            "source_date_range": panel.source_date_range,
            "selected_tickers": len(panel.tickers),
            "excluded_tickers": excluded,
            "date_start": str(panel.dates.min().date()),
            "date_end": str(panel.dates.max().date()),
            "n_dates": int(len(panel.dates)),
        },
        "rolling": {
            "n_blocks": int(len(block_table)),
            "n_test_dates": int(test_mask.sum()),
            "fallback_blocks": int(block_table["graph_fallback"].notna().sum()) if "graph_fallback" in block_table else 0,
            "graph_methods": block_table["graph_method"].value_counts(dropna=False).to_dict()
            if "graph_method" in block_table
            else {},
        },
        "tickers": panel.tickers,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    panel.coverage.to_csv(output_dir / "ticker_coverage.csv")


def run(args: argparse.Namespace) -> None:
    if args.fast:
        args.max_blocks = args.max_blocks or 1
        args.epochs = min(args.epochs, 40)
        args.num_nn = min(args.num_nn, 1)
        args.hidden_grid = parse_csv_items(args.hidden_grid)[0]
        args.lr_grid = parse_csv_items(args.lr_grid)[0]
        args.mcs_bootstrap = min(args.mcs_bootstrap, 20)
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)

    panel = load_panel(args)
    models = parse_csv_items(args.models)
    hidden_grid = parse_int_grid(args.hidden_grid)
    lr_grid = parse_float_grid(args.lr_grid)
    origins = build_origins(len(panel.dates), args)

    predictions = {model: empty_prediction(panel) for model in models}
    block_rows: List[BlockInfo] = []
    gnn_rows: List[Dict[str, object]] = []
    test_mask = np.zeros(len(panel.dates), dtype=bool)

    start_time = time.time()
    for block_id, origin in enumerate(origins):
        train_start = max(origin - args.lookback, 0)
        valid_start = max(origin - args.valid_len, train_start + 1)
        train_idx = np.arange(train_start, valid_start, dtype=np.int64)
        valid_idx = np.arange(valid_start, origin, dtype=np.int64)
        test_end = min(origin + args.window, len(panel.dates))
        test_idx = np.arange(origin, test_end, dtype=np.int64)
        if len(train_idx) < 20 or len(valid_idx) < 1 or len(test_idx) < 1:
            continue
        adj, graph_info = build_block_adjacency(panel, origin, train_start, args)
        pred_floor = prediction_floor(panel, train_idx, args)
        block_rows.append(
            BlockInfo(
                block_id=block_id,
                origin_index=int(origin),
                origin_date=str(panel.dates[origin].date()),
                train_start_date=str(panel.dates[train_start].date()),
                valid_start_date=str(panel.dates[valid_start].date()),
                test_end_date=str(panel.dates[test_idx[-1]].date()),
                train_size=int(len(train_idx)),
                valid_size=int(len(valid_idx)),
                test_size=int(len(test_idx)),
                graph_method=str(graph_info["method"]),
                graph_alpha=float(graph_info["alpha"]),
                graph_edges=int(graph_info["edges"]),
                graph_density=float(graph_info["density"]),
                graph_train_rows=int(graph_info["train_rows"]),
                graph_fallback=graph_info.get("fallback"),
            )
        )
        test_mask[test_idx] = True

        design_cache: Dict[str, np.ndarray] = {}
        for model_name in models:
            if model_name.startswith("GNNHAR"):
                if "fakeIV" in model_name:
                    base = "GNN_FAKE_IV"
                elif model_name.endswith("-IV") or "+IV" in model_name:
                    base = "GNN_IV"
                else:
                    base = "GNN"
                if base not in design_cache:
                    if base == "GNN_IV":
                        design_cache[base] = np.concatenate([panel.rv_features, panel.iv_features], axis=2)
                    elif base == "GNN_FAKE_IV":
                        design_cache[base] = np.concatenate([panel.rv_features, panel.fake_iv_features], axis=2)
                    else:
                        design_cache[base] = panel.rv_features
                pred, info = tune_and_fit_gnn_block(
                    model_name=model_name,
                    design=design_cache[base],
                    panel=panel,
                    adjacency=adj,
                    train_idx=train_idx,
                    valid_idx=valid_idx,
                    test_idx=test_idx,
                    hidden_grid=hidden_grid,
                    lr_grid=lr_grid,
                    args=args,
                    block_seed=args.seed + block_id * 1000,
                    pred_floor=pred_floor,
                )
                predictions[model_name][test_idx] = pred
                for record in info.get("grid_records", []):
                    row = dict(record)
                    row.update({"block_id": block_id, "origin_date": str(panel.dates[origin].date())})
                    row.pop("histories", None)
                    row.pop("all_valid_losses", None)
                    row.pop("selected_ensemble", None)
                    gnn_rows.append(row)
            else:
                if model_name not in design_cache:
                    design_cache[model_name] = make_linear_design(panel, model_name, adj)
                if args.loss.upper() == "QLIKE":
                    pred = fit_torch_linear_block(
                        design=design_cache[model_name],
                        target=panel.target,
                        train_idx=train_idx,
                        valid_idx=valid_idx,
                        test_idx=test_idx,
                        loss_name=args.loss,
                        epochs=args.epochs,
                        lr=lr_grid[0],
                        batch_size=args.batch_size,
                        seed=args.seed + block_id * 1000 + len(model_name),
                        pred_floor=pred_floor,
                    )
                else:
                    pred = fit_linear_block(design_cache[model_name], panel.target, train_idx, test_idx, pred_floor)
                predictions[model_name][test_idx] = pred
        elapsed = time.time() - start_time
        print(
            json.dumps(
                {
                    "block": block_id,
                    "origin": str(panel.dates[origin].date()),
                    "test_end": str(panel.dates[test_idx[-1]].date()),
                    "graph": graph_info["method"],
                    "edges": graph_info["edges"],
                    "fallback": bool(graph_info.get("fallback")),
                    "elapsed_sec": round(elapsed, 2),
                }
            )
        )

    if not test_mask.any():
        raise RuntimeError("No test predictions were generated")

    split = {"train": np.flatnonzero(~test_mask), "valid": np.array([], dtype=np.int64), "test": np.flatnonzero(test_mask)}
    eval_panel = type(
        "EvalPanel",
        (),
        {
            "dates": panel.dates,
            "tickers": panel.tickers,
            "target": panel.target,
            "split": split,
        },
    )()
    runs = [
        ModelRun(
            name=name,
            family=model_family(name),
            iv_channel=model_iv_channel(name),
            adjacency=model_adjacency(name),
            estimation=args.loss.upper(),
            prediction=np.clip(predictions[name], EPS, None),
        )
        for name in models
    ]
    loss_table, losses = evaluate_runs(runs, eval_panel)
    loss_table.attrs["n_test_dates"] = int(test_mask.sum())
    ratio_table = loss_table[
        [
            "model",
            "estimation",
            "mse_ratio_vs_har",
            "qlike_ratio_vs_har",
            "mse_ratio_vs_har_iv",
            "qlike_ratio_vs_har_iv",
        ]
    ].copy()
    dm_table = pd.DataFrame()
    try:
        from scripts.analysis.gnnhar_iv_scale_pipeline import build_dm_table

        dm_table = build_dm_table(losses["qlike"])
    except Exception as exc:
        dm_table = pd.DataFrame([{"note": f"DM table unavailable: {type(exc).__name__}: {exc}"}])
    mcs_table = build_mcs_table(losses["qlike"], args.mcs_bootstrap)
    iv_decomposition = build_iv_decomposition(loss_table)
    regime_table = build_regime_table(runs, eval_panel)
    block_table = pd.DataFrame([asdict(row) for row in block_rows])
    gnn_table = pd.DataFrame(gnn_rows)

    tables = {
        "model_losses": loss_table,
        "loss_ratios": ratio_table,
        "graph_blocks": block_table,
        "gnn_tuning": gnn_table,
        "dm_tests": dm_table,
        "mcs_results": mcs_table,
        "iv_decomposition": iv_decomposition,
        "regime_results": regime_table,
    }
    save_tables(tables, output_dir)
    save_predictions(output_dir, panel, runs, test_mask)
    if not args.skip_figures:
        save_figures(panel, runs, loss_table, output_dir, test_mask)
    write_report(output_dir, panel, args, block_table, loss_table, iv_decomposition)
    save_metadata(output_dir, panel, args, block_table, test_mask)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "universe": panel.universe,
                "n_tickers": len(panel.tickers),
                "n_blocks": len(block_table),
                "n_test_dates": int(test_mask.sum()),
                "best_model": str(loss_table.iloc[0]["model"]),
                "best_test_qlike": float(loss_table.iloc[0]["test_qlike"]),
                "fallback_blocks": int(block_table["graph_fallback"].notna().sum()),
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
