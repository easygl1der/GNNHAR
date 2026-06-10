#!/usr/bin/env python3
"""Run GNNHAR-IV scale experiments on S&P 100 / S&P 500 VolVue panels."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.gnnhar_iv_pipeline import (  # noqa: E402
    EPS,
    ModelRun,
    PanelData,
    apply_graph,
    build_iv_decomposition,
    build_mcs_table,
    build_regime_table,
    chronological_split,
    evaluate_runs,
    fit_linear,
    lag_average,
    make_design,
    mse_loss,
    qlike_loss,
    save_tables,
    score_on_split,
)


@dataclass
class GraphInfo:
    method: str
    alpha: float
    max_neighbors: int
    n_nodes: int
    n_edges: int
    density: float
    train_rows: int
    fallback: Optional[str] = None


def parse_args() -> argparse.Namespace:
    default_output = (
        "/content/drive/MyDrive/GNNHAR-colab-runs/scale-experiment"
        if Path("/content").exists()
        else "outputs/scale-experiment"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-name", required=True, help="Label used in reports, e.g. sp100 or sp500")
    parser.add_argument("--data-dir", required=True, help="Directory with merged_rv_data_filled.csv and merged_iv_data_filled.csv")
    parser.add_argument("--returns-file", required=True, help="Wide daily returns CSV aligned by Date")
    parser.add_argument("--output-dir", default=default_output)
    parser.add_argument("--coverage-threshold", type=float, default=0.98)
    parser.add_argument("--max-tickers", type=int, default=0, help="Optional cap after coverage ranking; 0 keeps all passing tickers")
    parser.add_argument("--fill-limit", type=int, default=5, help="Internal interpolation limit; 0 means unlimited within each listed interval")
    parser.add_argument("--adjacency-method", choices=["glasso", "glasso_cv", "corr"], default="glasso")
    parser.add_argument("--glasso-alpha", type=float, default=0.05)
    parser.add_argument("--glasso-max-iter", type=int, default=500)
    parser.add_argument("--max-neighbors", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--mcs-bootstrap", type=int, default=100)
    parser.add_argument("--gnn-depths", default="1,2,3", help="Comma-separated GNN layer counts")
    parser.add_argument("--skip-qlike-training", action="store_true", default=True)
    parser.add_argument("--include-qlike-training", dest="skip_qlike_training", action="store_false")
    parser.add_argument("--no-fake-iv", action="store_true")
    parser.add_argument("--no-random-graph", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Short smoke run")
    return parser.parse_args()


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


def read_positive_wide_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"date": "Date"})
    if "Date" not in df.columns:
        raise ValueError(f"{path} must contain Date or date")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    df = df.where(df > 0)
    return df


def read_returns_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"date": "Date"})
    if "Date" not in df.columns:
        raise ValueError(f"{path} must contain Date or date")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def fill_internal_gaps(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    out = df.copy()
    interpolate_kwargs = {"method": "time", "limit_direction": "both"}
    if limit > 0:
        interpolate_kwargs["limit"] = limit
    for col in out.columns:
        series = out[col]
        first = series.first_valid_index()
        last = series.last_valid_index()
        if first is None or last is None:
            continue
        window = series.loc[first:last]
        out.loc[first:last, col] = window.interpolate(**interpolate_kwargs)
    return out


def build_features(source: pd.DataFrame) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    pieces = [lag_average(source, 1), lag_average(source, 5), lag_average(source, 22)]
    feature_df = pd.concat(pieces, axis=1, keys=["d", "w", "m"]).dropna(axis=0, how="any")
    dates = feature_df.index
    arr = np.stack(
        [feature_df[key].to_numpy(dtype=np.float32) for key in ["d", "w", "m"]],
        axis=2,
    )
    return dates, arr


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
        raise ValueError("RV, IV, and returns do not share dates")
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
        raise ValueError(
            f"Only {len(selected)} tickers pass coverage threshold {threshold}; "
            "lower --coverage-threshold or inspect the input panel."
        )
    return selected, coverage


def load_scale_panel(args: argparse.Namespace) -> Tuple[PanelData, Dict[str, object], pd.DataFrame]:
    data_dir = Path(args.data_dir)
    rv_raw = read_positive_wide_csv(data_dir / "merged_rv_data_filled.csv")
    iv_raw = read_positive_wide_csv(data_dir / "merged_iv_data_filled.csv")
    returns_raw = read_returns_csv(Path(args.returns_file))

    tickers, coverage = select_tickers(
        rv_raw,
        iv_raw,
        returns_raw,
        threshold=args.coverage_threshold,
        max_tickers=args.max_tickers,
    )
    common_dates = rv_raw.index.intersection(iv_raw.index).intersection(returns_raw.index)
    rv = fill_internal_gaps(rv_raw.loc[common_dates, tickers], args.fill_limit)
    iv = fill_internal_gaps(iv_raw.loc[common_dates, tickers], args.fill_limit)
    returns = fill_internal_gaps(returns_raw.loc[common_dates, tickers], args.fill_limit)

    rv_dates, rv_features = build_features(rv)
    iv_dates, iv_features = build_features(iv)
    dates = rv_dates.intersection(iv_dates).intersection(rv.index)
    rv_features = rv_features[rv_dates.get_indexer(dates)]
    iv_features = iv_features[iv_dates.get_indexer(dates)]
    target = rv.loc[dates, tickers].to_numpy(dtype=np.float32)

    valid = np.isfinite(target).all(axis=1)
    valid &= np.isfinite(rv_features).all(axis=(1, 2))
    valid &= np.isfinite(iv_features).all(axis=(1, 2))
    dates = pd.DatetimeIndex(dates[valid])
    rv_features = rv_features[valid]
    iv_features = iv_features[valid]
    target = target[valid]
    if len(dates) < 200:
        raise ValueError(f"Only {len(dates)} valid dates after HAR lag construction and filtering")

    rng = np.random.default_rng(args.seed)
    fake_iv_features = iv_features[rng.permutation(len(dates))].copy()
    returns_aligned = returns.loc[returns.index.intersection(dates), tickers]

    panel = PanelData(
        dates=dates,
        tickers=tickers,
        target=target,
        rv_features=rv_features,
        iv_features=iv_features,
        fake_iv_features=fake_iv_features,
        returns=returns_aligned,
        split=chronological_split(len(dates)),
    )
    info = {
        "universe": args.universe_name,
        "raw_tickers": int(len(set(rv_raw.columns) & set(iv_raw.columns) & set(returns_raw.columns))),
        "selected_tickers": int(len(tickers)),
        "excluded_tickers": coverage.index.difference(tickers).tolist(),
        "coverage_threshold": float(args.coverage_threshold),
        "max_tickers": int(args.max_tickers),
        "fill_limit": int(args.fill_limit),
        "n_dates_after_lags": int(len(dates)),
        "date_start": str(dates.min().date()),
        "date_end": str(dates.max().date()),
    }
    return panel, info, coverage


def standardize_returns(returns: pd.DataFrame, train_dates: Iterable[pd.Timestamp]) -> Tuple[np.ndarray, int]:
    train_returns = returns.loc[returns.index.intersection(pd.DatetimeIndex(train_dates))]
    train_returns = train_returns.dropna(axis=0, how="all")
    train_returns = train_returns.fillna(0.0)
    values = train_returns.to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = values - values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    values = values / np.where(scale < EPS, 1.0, scale)
    return values, int(values.shape[0])


def sparsify_symmetric(weights: np.ndarray, max_neighbors: int) -> Tuple[np.ndarray, int]:
    weights = np.asarray(weights, dtype=float).copy()
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(weights, 0.0)
    n_nodes = weights.shape[0]
    keep = np.zeros_like(weights, dtype=bool)
    if max_neighbors > 0 and max_neighbors < n_nodes - 1:
        for i in range(n_nodes):
            row = weights[i].copy()
            row[i] = 0.0
            if np.count_nonzero(row) == 0:
                continue
            idx = np.argpartition(row, -max_neighbors)[-max_neighbors:]
            keep[i, idx[row[idx] > 0]] = True
    else:
        keep = weights > 0
    keep = np.logical_or(keep, keep.T)
    sparse = np.where(keep, weights, 0.0)
    sparse = np.maximum(sparse, sparse.T)
    np.fill_diagonal(sparse, 0.0)
    n_edges = int(np.count_nonzero(np.triu(sparse > 0, 1)))
    if n_edges == 0:
        corr = np.ones_like(sparse) - np.eye(n_nodes)
        sparse, n_edges = sparsify_symmetric(corr, max(1, min(max_neighbors, n_nodes - 1)))
    degrees = sparse.sum(axis=1)
    d_inv = np.diag(1.0 / np.sqrt(degrees + EPS))
    normalized = d_inv @ sparse @ d_inv
    return normalized.astype(np.float32), n_edges


def build_adjacency(
    returns: pd.DataFrame,
    train_dates: Iterable[pd.Timestamp],
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, GraphInfo]:
    values, train_rows = standardize_returns(returns, train_dates)
    method = args.adjacency_method
    alpha = float(args.glasso_alpha)
    fallback = None
    try:
        if method == "corr":
            raw_weights = np.abs(np.corrcoef(values, rowvar=False))
            alpha = float("nan")
        elif method == "glasso_cv":
            from sklearn.covariance import GraphicalLassoCV

            model = GraphicalLassoCV(cv=3, max_iter=args.glasso_max_iter).fit(values)
            raw_weights = np.abs(model.precision_)
            alpha = float(model.alpha_)
        else:
            from sklearn.covariance import GraphicalLasso

            model = GraphicalLasso(alpha=args.glasso_alpha, max_iter=args.glasso_max_iter).fit(values)
            raw_weights = np.abs(model.precision_)
    except Exception as exc:
        fallback = f"{type(exc).__name__}: {exc}; used absolute correlation graph"
        raw_weights = np.abs(np.corrcoef(values, rowvar=False))
        method = f"{method}_fallback_corr"
        alpha = float("nan")

    np.fill_diagonal(raw_weights, 0.0)
    normalized, n_edges = sparsify_symmetric(raw_weights, args.max_neighbors)
    adj = pd.DataFrame(normalized, index=returns.columns, columns=returns.columns)
    density = n_edges / max(1.0, (len(returns.columns) * (len(returns.columns) - 1) / 2.0))
    info = GraphInfo(
        method=method,
        alpha=alpha,
        max_neighbors=int(args.max_neighbors),
        n_nodes=int(len(returns.columns)),
        n_edges=int(n_edges),
        density=float(density),
        train_rows=train_rows,
        fallback=fallback,
    )
    adj.attrs.update(asdict(info))
    return adj, info


def random_adjacency_like(n_nodes: int, n_edges: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    possible = np.array(np.triu_indices(n_nodes, 1)).T
    n_edges = max(1, min(int(n_edges), len(possible)))
    chosen = possible[rng.choice(len(possible), size=n_edges, replace=False)]
    mat = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    mat[chosen[:, 0], chosen[:, 1]] = 1.0
    mat = mat + mat.T
    degrees = mat.sum(axis=1)
    d_inv = np.diag(1.0 / np.sqrt(degrees + EPS)).astype(np.float32)
    return d_inv @ mat @ d_inv


def fit_gnn_sparse(
    name: str,
    iv_channel: str,
    adjacency_name: str,
    x: np.ndarray,
    y: np.ndarray,
    adjacency: np.ndarray,
    split: Dict[str, np.ndarray],
    layers: int,
    estimation: str,
    epochs: int,
    hidden: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> ModelRun:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    class SparseGNNHARNet(nn.Module):
        def __init__(self, in_features: int, hidden_features: int, n_layers: int) -> None:
            super().__init__()
            dims = [in_features] + [hidden_features] * max(n_layers - 1, 0)
            self.weights = nn.ModuleList([nn.Linear(dims[i] * 2, hidden_features) for i in range(n_layers)])
            self.out = nn.Linear(hidden_features + in_features, 1)

        @staticmethod
        def aggregate(adj_sparse: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
            batch, n_nodes, n_features = values.shape
            flat = values.permute(1, 0, 2).reshape(n_nodes, batch * n_features)
            aggregated = torch.sparse.mm(adj_sparse, flat)
            return aggregated.reshape(n_nodes, batch, n_features).permute(1, 0, 2)

        def forward(self, values: torch.Tensor, adj_sparse: torch.Tensor) -> torch.Tensor:
            h = values
            for layer in self.weights:
                neighbor_h = self.aggregate(adj_sparse, h)
                h = functional.relu(layer(torch.cat([h, neighbor_h], dim=-1)))
            return self.out(torch.cat([h, values], dim=-1)).squeeze(-1)

    train_x = x[split["train"]].reshape(-1, x.shape[-1])
    x_mean = train_x.mean(axis=0, keepdims=True)
    x_std = np.where(train_x.std(axis=0, keepdims=True) < EPS, 1.0, train_x.std(axis=0, keepdims=True))
    x_scaled = (x - x_mean.reshape(1, 1, -1)) / x_std.reshape(1, 1, -1)

    use_qlike = estimation.upper() == "QLIKE"
    y_work = np.log(np.clip(y, EPS, None)) if use_qlike else y
    train_y_work = y_work[split["train"]]
    y_mean = float(train_y_work.mean())
    y_std = float(train_y_work.std())
    if y_std < EPS:
        y_std = 1.0
    y_scaled = (y_work - y_mean) / y_std

    graph = np.asarray(adjacency, dtype=np.float32).copy()
    graph = graph + np.eye(graph.shape[0], dtype=np.float32)
    degree = graph.sum(axis=1)
    d_inv = np.diag(1.0 / np.sqrt(degree + EPS)).astype(np.float32)
    graph = d_inv @ graph @ d_inv
    rows, cols = np.nonzero(np.abs(graph) > 0)
    values = graph[rows, cols].astype(np.float32)

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_t = torch.tensor(x_scaled, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_scaled, dtype=torch.float32, device=device)
    adj_t = torch.sparse_coo_tensor(
        torch.tensor(np.vstack([rows, cols]), dtype=torch.long, device=device),
        torch.tensor(values, dtype=torch.float32, device=device),
        size=graph.shape,
        device=device,
    ).coalesce()

    model = SparseGNNHARNet(x.shape[-1], hidden, layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    train_idx = np.asarray(split["train"], dtype=np.int64)
    valid_idx = np.asarray(split["valid"], dtype=np.int64)
    batch_size = max(1, min(int(batch_size), len(train_idx)))
    rng = np.random.default_rng(seed)
    best_state = None
    best_valid = float("inf")
    patience = max(12, epochs // 6)
    stale = 0

    def criterion(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        if use_qlike:
            pred_orig = torch.exp(pred * y_std + y_mean).clamp_min(EPS)
            truth_orig = torch.exp(truth * y_std + y_mean).clamp_min(EPS)
            ratio = truth_orig / pred_orig
            return (ratio - torch.log(ratio) - 1.0).mean()
        return ((truth - pred) ** 2).mean()

    def predict_scaled(indices: np.ndarray, chunk: int = 128) -> np.ndarray:
        chunks = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(indices), chunk):
                idx = torch.tensor(indices[start : start + chunk], dtype=torch.long, device=device)
                chunks.append(model(x_t[idx], adj_t).detach().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    for _epoch in range(epochs):
        order = train_idx.copy()
        rng.shuffle(order)
        model.train()
        for start in range(0, len(order), batch_size):
            idx = torch.tensor(order[start : start + batch_size], dtype=torch.long, device=device)
            optimizer.zero_grad()
            pred = model(x_t[idx], adj_t)
            loss = criterion(pred, y_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        valid_pred = torch.tensor(predict_scaled(valid_idx), dtype=torch.float32, device=device)
        valid_truth = y_t[torch.tensor(valid_idx, dtype=torch.long, device=device)]
        valid_loss = criterion(valid_pred, valid_truth).item()
        if valid_loss + 1e-9 < best_valid:
            best_valid = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    all_idx = np.arange(x.shape[0], dtype=np.int64)
    pred_scaled = predict_scaled(all_idx)
    if use_qlike:
        pred = np.exp(pred_scaled * y_std + y_mean)
    else:
        pred = pred_scaled * y_std + y_mean
    pred = np.clip(pred, EPS, None)
    return ModelRun(name, "GNNHAR", iv_channel, adjacency_name, estimation.upper(), pred.astype(np.float32))


def build_dm_table(losses: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("HAR", "GHAR"),
        ("HAR+IV", "GHAR+IV"),
        ("GHAR", "GNNHAR1L"),
        ("GHAR+IV", "GNNHAR1L-IV"),
        ("HAR+IV", "HAR+fakeIV"),
        ("GNNHAR1L", "GNNHAR1L-IV"),
        ("GNNHAR2L", "GNNHAR2L-IV"),
        ("GNNHAR3L", "GNNHAR3L-IV"),
    ]
    rows = []
    for first, second in pairs:
        if first not in losses or second not in losses:
            continue
        diff = (losses[first] - losses[second]).dropna().to_numpy()
        if len(diff) < 3 or float(np.std(diff, ddof=1)) < EPS:
            stat, pvalue = float("nan"), float("nan")
        else:
            stat = float(diff.mean() / (diff.std(ddof=1) / math.sqrt(len(diff))))
            try:
                from scipy.stats import t

                pvalue = float(2.0 * (1.0 - t.cdf(abs(stat), df=len(diff) - 1)))
            except Exception:
                pvalue = float(math.erfc(abs(stat) / math.sqrt(2.0)))
        rows.append(
            {
                "comparison": f"{first} vs {second}",
                "loss_a": first,
                "loss_b": second,
                "mean_loss_a": float(losses[first].mean()),
                "mean_loss_b": float(losses[second].mean()),
                "dm_stat_positive_favors_b": stat,
                "pvalue": pvalue,
            }
        )
    return pd.DataFrame(rows)


def save_scale_figures(
    panel: PanelData,
    adjacency: pd.DataFrame,
    runs: List[ModelRun],
    loss_table: pd.DataFrame,
    iv_decomposition: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    graph = adjacency.to_numpy()
    degrees = (np.abs(graph) > 0).sum(axis=1)

    plt.figure(figsize=(7, 4))
    plt.hist(degrees, bins=min(30, max(5, int(degrees.max()) + 1)), color="#3f6f8f", edgecolor="white")
    plt.xlabel("nonzero graph neighbors")
    plt.ylabel("assets")
    plt.title("GLASSO graph degree distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "graph_degree_distribution.png", dpi=200)
    plt.close()

    plot_table = loss_table.sort_values("test_qlike").head(min(14, len(loss_table)))
    plt.figure(figsize=(10, 5))
    plt.bar(plot_table["model"], 1.0 - plot_table["qlike_ratio_vs_har"], color="#586f4e")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("QLIKE improvement vs HAR")
    plt.title("Out-of-sample QLIKE gains")
    plt.tight_layout()
    plt.savefig(fig_dir / "qlike_gain_vs_har.png", dpi=200)
    plt.close()

    if not iv_decomposition.empty:
        idx = np.arange(len(iv_decomposition))
        width = 0.35
        plt.figure(figsize=(8, 4.5))
        plt.bar(idx - width / 2, iv_decomposition["genuine_information_gain"], width, label="real IV over fake IV")
        plt.bar(idx + width / 2, iv_decomposition["parameter_expansion_gain"], width, label="parameter expansion")
        plt.xticks(idx, iv_decomposition["family"])
        plt.ylabel("QLIKE reduction")
        plt.title("IV contribution decomposition")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "iv_decomposition.png", dpi=200)
        plt.close()

    test_idx = panel.split["test"]
    truth = panel.target[test_idx]
    test_dates = panel.dates[test_idx].astype(str).to_numpy()
    predictions = {run.name: run.prediction[test_idx] for run in runs}
    np.savez_compressed(
        output_dir / "predictions_test.npz",
        truth=truth.astype(np.float32),
        dates=test_dates,
        tickers=np.asarray(panel.tickers),
        **{f"pred_{run.name.replace('+', 'plus').replace('-', '_')}": pred.astype(np.float32) for run, pred in zip(runs, predictions.values())},
    )


def write_scale_report(
    output_dir: Path,
    args: argparse.Namespace,
    panel: PanelData,
    graph_info: GraphInfo,
    panel_info: Dict[str, object],
    loss_table: pd.DataFrame,
    iv_decomposition: pd.DataFrame,
    dm_table: pd.DataFrame,
) -> None:
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    best = loss_table.iloc[0]
    lines = [
        f"# GNNHAR-IV Scale Experiment: {args.universe_name}",
        "",
        "## Design",
        "",
        "This run evaluates whether graph-based volatility forecasting gains persist or increase when the asset universe expands beyond the original Dow 30 setting. The dependent variable is 30-day close-to-close realized volatility. The exogenous option-market channel is 30-day mean implied volatility. HAR uses own-asset daily, weekly, and monthly lag averages; GHAR adds graph-aggregated lag features; GNNHAR learns nonlinear graph message passing over the same HAR inputs.",
        "",
        "Fake IV is generated by permuting IV dates. This keeps the IV scale and dimension but destroys the timing relation, so real-IV gains can be separated from gains caused only by additional parameters.",
        "",
        "## Data",
        "",
        f"- Universe: {args.universe_name}",
        f"- Selected assets: {len(panel.tickers)} out of {panel_info['raw_tickers']} common RV/IV/return tickers",
        f"- Date range after HAR lags: {panel.dates.min().date()} to {panel.dates.max().date()}",
        f"- Split: {len(panel.split['train'])} train, {len(panel.split['valid'])} validation, {len(panel.split['test'])} test dates",
        f"- Coverage threshold: {args.coverage_threshold:.3f}",
        f"- Graph method: {graph_info.method}, alpha={graph_info.alpha:.6g}, edges={graph_info.n_edges}, density={graph_info.density:.6g}",
        "",
        "## Main Result",
        "",
        f"The best model by test QLIKE is `{best['model']}` with QLIKE {best['test_qlike']:.6g} and MSE {best['test_mse']:.6g}.",
        "",
        "The complete ranking is stored in `tables/model_losses.csv`; ratios relative to HAR and HAR+IV are stored in `tables/loss_ratios.csv`.",
        "",
        "## IV Contribution",
        "",
    ]
    if iv_decomposition.empty:
        lines.append("The IV decomposition is unavailable because one or more fake-IV comparison models were not run.")
    else:
        for _, row in iv_decomposition.iterrows():
            lines.append(
                f"- {row['family']}: total QLIKE improvement {row['total_iv_improvement']:.6g}; "
                f"genuine information gain {row['genuine_information_gain']:.6g}; "
                f"parameter expansion gain {row['parameter_expansion_gain']:.6g}."
            )
    lines.extend(["", "## Statistical Tests", ""])
    if dm_table.empty:
        lines.append("No DM comparison rows were available.")
    else:
        for _, row in dm_table.iterrows():
            lines.append(
                f"- {row['comparison']}: statistic {row['dm_stat_positive_favors_b']:.4g}, "
                f"p-value {row['pvalue']:.4g}."
            )
    if graph_info.fallback:
        lines.extend(["", "## Graph Note", "", graph_info.fallback])
    (report_dir / "scale_experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    panel: PanelData,
    panel_info: Dict[str, object],
    coverage: pd.DataFrame,
    graph_info: GraphInfo,
) -> None:
    metadata = {
        "args": vars(args),
        "panel": panel_info,
        "graph": asdict(graph_info),
        "n_dates": len(panel.dates),
        "n_tickers": len(panel.tickers),
        "tickers": panel.tickers,
        "date_start": str(panel.dates.min().date()),
        "date_end": str(panel.dates.max().date()),
        "split_sizes": {key: int(len(value)) for key, value in panel.split.items()},
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    coverage.to_csv(output_dir / "ticker_coverage.csv")


def parse_depths(text: str) -> List[int]:
    depths = []
    for item in text.split(","):
        item = item.strip()
        if item:
            depths.append(int(item))
    return sorted(set(depths))


def main() -> None:
    args = parse_args()
    if args.fast:
        args.epochs = min(args.epochs, 20)
        args.mcs_bootstrap = min(args.mcs_bootstrap, 40)
        args.gnn_depths = "1"
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel, panel_info, coverage = load_scale_panel(args)
    adjacency, graph_info = build_adjacency(panel.returns, panel.dates[panel.split["train"]], args)
    glasso_np = adjacency.to_numpy(dtype=np.float32)
    random_np = random_adjacency_like(len(panel.tickers), graph_info.n_edges, args.seed + 1)

    designs: Dict[str, np.ndarray] = {
        "HAR": make_design(panel, None, None),
        "GHAR": make_design(panel, glasso_np, None),
        "HAR_IV": make_design(panel, None, panel.iv_features),
        "GHAR_IV": make_design(panel, glasso_np, panel.iv_features),
    }
    if not args.no_random_graph:
        designs["GHAR_RANDOM"] = make_design(panel, random_np, None)
        designs["GHAR_IV_RANDOM"] = make_design(panel, random_np, panel.iv_features)
    if not args.no_fake_iv:
        designs["HAR_FAKE"] = make_design(panel, None, panel.fake_iv_features)
        designs["GHAR_FAKE"] = make_design(panel, glasso_np, panel.fake_iv_features)

    runs: List[ModelRun] = [
        fit_linear("HAR", "HAR", "none", "Identity", designs["HAR"], panel),
        fit_linear("GHAR", "GHAR", "none", "GLASSO", designs["GHAR"], panel),
        fit_linear("HAR+IV", "HAR", "real", "Identity", designs["HAR_IV"], panel),
        fit_linear("GHAR+IV", "GHAR", "real", "GLASSO", designs["GHAR_IV"], panel),
    ]
    if not args.no_random_graph:
        runs.extend(
            [
                fit_linear("GHAR_RANDOM", "GHAR", "none", "Random", designs["GHAR_RANDOM"], panel),
                fit_linear("GHAR+IV-random", "GHAR", "real", "Random", designs["GHAR_IV_RANDOM"], panel),
            ]
        )
    if not args.no_fake_iv:
        runs.extend(
            [
                fit_linear("HAR+fakeIV", "HAR", "fake", "Identity", designs["HAR_FAKE"], panel),
                fit_linear("GHAR+fakeIV", "GHAR", "fake", "GLASSO", designs["GHAR_FAKE"], panel),
            ]
        )

    gnn_specs: List[Tuple[str, str, str, np.ndarray, np.ndarray, int, str]] = []
    for depth in parse_depths(args.gnn_depths):
        gnn_specs.append((f"GNNHAR{depth}L", "none", "GLASSO", designs["HAR"], glasso_np, depth, "MSE"))
        gnn_specs.append((f"GNNHAR{depth}L-IV", "real", "GLASSO", designs["HAR_IV"], glasso_np, depth, "MSE"))
    if not args.no_random_graph and 3 in parse_depths(args.gnn_depths):
        gnn_specs.append(("GNNHAR3L-IV-random", "real", "Random", designs["HAR_IV"], random_np, 3, "MSE"))
    if not args.no_fake_iv:
        gnn_specs.append(("GNNHAR1L-IV+fakeIV", "fake", "GLASSO", designs["HAR_FAKE"], glasso_np, 1, "MSE"))
    if not args.skip_qlike_training:
        gnn_specs.extend(
            [
                ("GNNHAR1L-QLIKE", "none", "GLASSO", designs["HAR"], glasso_np, 1, "QLIKE"),
                ("GNNHAR1L-IV-QLIKE", "real", "GLASSO", designs["HAR_IV"], glasso_np, 1, "QLIKE"),
            ]
        )

    tuning_records = []
    for offset, (name, iv_channel, adjacency_name, design, graph, layers, estimation) in enumerate(gnn_specs):
        run = fit_gnn_sparse(
            name=name,
            iv_channel=iv_channel,
            adjacency_name=adjacency_name,
            x=design,
            y=panel.target,
            adjacency=graph,
            split=panel.split,
            layers=layers,
            estimation=estimation,
            epochs=args.epochs,
            hidden=args.hidden,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed + 100 + offset,
        )
        runs.append(run)
        valid_mse, valid_qlike = score_on_split(run, panel, "valid")
        test_mse, test_qlike = score_on_split(run, panel, "test")
        tuning_records.append(
            {
                "model": name,
                "layers": layers,
                "estimation": estimation,
                "hidden": args.hidden,
                "lr": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "valid_mse": valid_mse,
                "valid_qlike": valid_qlike,
                "test_mse_diagnostic": test_mse,
                "test_qlike_diagnostic": test_qlike,
            }
        )

    loss_table, losses = evaluate_runs(runs, panel)
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
    dm_table = build_dm_table(losses["qlike"])
    mcs_table = build_mcs_table(losses["qlike"], args.mcs_bootstrap)
    iv_decomposition = build_iv_decomposition(loss_table)
    regime_table = build_regime_table(runs, panel)

    tables = {
        "model_losses": loss_table,
        "loss_ratios": ratio_table,
        "dm_tests": dm_table,
        "mcs_results": mcs_table,
        "iv_decomposition": iv_decomposition,
        "regime_results": regime_table,
        "gnn_runs": pd.DataFrame(tuning_records),
    }
    save_tables(tables, output_dir)
    if not args.skip_figures:
        save_scale_figures(panel, adjacency, runs, loss_table, iv_decomposition, output_dir)
    write_scale_report(output_dir, args, panel, graph_info, panel_info, loss_table, iv_decomposition, dm_table)
    save_metadata(output_dir, args, panel, panel_info, coverage, graph_info)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "universe": args.universe_name,
                "n_tickers": len(panel.tickers),
                "n_dates": len(panel.dates),
                "best_model": str(loss_table.iloc[0]["model"]),
                "best_test_qlike": float(loss_table.iloc[0]["test_qlike"]),
                "graph_density": graph_info.density,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
