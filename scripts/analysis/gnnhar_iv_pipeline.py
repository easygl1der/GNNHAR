#!/usr/bin/env python3
"""Run a Colab-friendly GNNHAR-IV empirical analysis pipeline.

The script is intentionally self-contained because the legacy scripts in this
repository keep command-line parsing and local path constants at import time.
It follows Zhang et al.'s HAR/GHAR/GNNHAR evaluation layout and adds real-IV
and fake-IV comparisons for the Dow 30 data in this repository.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EPS = 1e-8


@dataclass
class PanelData:
    dates: pd.DatetimeIndex
    tickers: List[str]
    target: np.ndarray
    rv_features: np.ndarray
    iv_features: np.ndarray
    fake_iv_features: np.ndarray
    returns: pd.DataFrame
    split: Dict[str, np.ndarray]


@dataclass
class ModelRun:
    name: str
    family: str
    iv_channel: str
    adjacency: str
    estimation: str
    prediction: np.ndarray


def parse_args() -> argparse.Namespace:
    default_output = (
        "/content/drive/MyDrive/GNNHAR-colab-runs/outputs"
        if Path("/content").exists()
        else "outputs/gnnhar_iv"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="experiments/dow30/data")
    parser.add_argument("--output-dir", default=default_output)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--fast", action="store_true", help="short smoke-test run")
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mcs-bootstrap", type=int, default=300)
    parser.add_argument("--skip-qlike-training", action="store_true")
    parser.add_argument("--tune-gnn", action="store_true", help="select GNN hyperparameters on the validation split")
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


def read_wide_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return df


def lag_average(df: pd.DataFrame, lags: int) -> pd.DataFrame:
    return sum(df.shift(i) for i in range(1, lags + 1)) / float(lags)


def build_features(source: pd.DataFrame) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    pieces = [lag_average(source, 1), lag_average(source, 5), lag_average(source, 22)]
    feature_df = pd.concat(pieces, axis=1, keys=["d", "w", "m"]).dropna()
    dates = feature_df.index
    arr = np.stack(
        [feature_df[key].to_numpy(dtype=np.float32) for key in ["d", "w", "m"]],
        axis=2,
    )
    return dates, arr


def chronological_split(n_obs: int) -> Dict[str, np.ndarray]:
    train_end = int(n_obs * 0.70)
    valid_end = int(n_obs * 0.85)
    return {
        "train": np.arange(0, train_end),
        "valid": np.arange(train_end, valid_end),
        "test": np.arange(valid_end, n_obs),
    }


def load_panel(data_dir: Path, seed: int) -> PanelData:
    rv = read_wide_csv(data_dir / "merged_rv_data_filled.csv")
    iv = read_wide_csv(data_dir / "merged_iv_data_filled.csv")
    returns = read_wide_csv(data_dir / "dow30_daily_returns_2021_2026.csv")
    tickers = sorted(set(rv.columns) & set(iv.columns) & set(returns.columns))
    rv = rv[tickers]
    iv = iv[tickers]
    returns = returns[tickers]
    common_dates = rv.index.intersection(iv.index)
    rv = rv.loc[common_dates]
    iv = iv.loc[common_dates]
    rv_dates, rv_features = build_features(rv)
    iv_dates, iv_features = build_features(iv)
    dates = rv_dates.intersection(iv_dates)
    date_pos_rv = rv_dates.get_indexer(dates)
    date_pos_iv = iv_dates.get_indexer(dates)
    rv_features = rv_features[date_pos_rv]
    iv_features = iv_features[date_pos_iv]
    target = rv.loc[dates, tickers].to_numpy(dtype=np.float32)

    rng = np.random.default_rng(seed)
    fake_order = rng.permutation(len(dates))
    fake_iv_features = iv_features[fake_order].copy()

    return PanelData(
        dates=dates,
        tickers=tickers,
        target=target,
        rv_features=rv_features,
        iv_features=iv_features,
        fake_iv_features=fake_iv_features,
        returns=returns.loc[returns.index.intersection(dates), tickers],
        split=chronological_split(len(dates)),
    )


def glasso_adjacency(returns: pd.DataFrame, train_dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    from sklearn.covariance import GraphicalLasso, GraphicalLassoCV

    train_returns = returns.loc[returns.index.intersection(pd.DatetimeIndex(train_dates))]
    train_returns = train_returns.replace([np.inf, -np.inf], np.nan).dropna()
    train_values = train_returns.to_numpy(dtype=float)
    train_values = np.nan_to_num(train_values, nan=0.0, posinf=0.0, neginf=0.0)
    train_values = train_values - train_values.mean(axis=0, keepdims=True)
    train_scale = train_values.std(axis=0, keepdims=True)
    train_values = train_values / np.where(train_scale < EPS, 1.0, train_scale)
    try:
        cov = GraphicalLassoCV(cv=5, max_iter=1000).fit(train_values)
        precision = cov.precision_
        alpha = float(cov.alpha_)
    except Exception:
        cov = GraphicalLasso(alpha=0.01, max_iter=1000).fit(train_values)
        precision = cov.precision_
        alpha = 0.01

    connected = (np.abs(precision) > 1e-10).astype(float)
    np.fill_diagonal(connected, 0.0)
    degrees = connected.sum(axis=1)
    d_inv = np.diag(1.0 / np.sqrt(degrees + EPS))
    normalized = d_inv @ connected @ d_inv
    adj = pd.DataFrame(normalized, index=returns.columns, columns=returns.columns)
    adj.attrs["alpha"] = alpha
    adj.attrs["density"] = float(connected.mean())
    return adj


def random_adjacency(n_nodes: int, seed: int, density: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    upper = rng.random((n_nodes, n_nodes)) < density
    mat = np.triu(upper, 1).astype(float)
    mat = mat + mat.T
    degrees = mat.sum(axis=1)
    return np.diag(1.0 / np.sqrt(degrees + EPS)) @ mat @ np.diag(1.0 / np.sqrt(degrees + EPS))


def apply_graph(features: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    return np.einsum("ij,tjf->tif", adjacency, features)


def make_design(
    panel: PanelData,
    adjacency: Optional[np.ndarray],
    iv: Optional[np.ndarray],
) -> np.ndarray:
    blocks = [panel.rv_features]
    if iv is not None:
        blocks.append(iv)
    if adjacency is not None:
        blocks.append(apply_graph(panel.rv_features, adjacency))
        if iv is not None:
            blocks.append(apply_graph(iv, adjacency))
    return np.concatenate(blocks, axis=2)


def flatten_split(x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return x[idx].reshape(-1, x.shape[-1]), y[idx].reshape(-1)


def fit_linear(name: str, family: str, iv_channel: str, adjacency_name: str, x: np.ndarray, panel: PanelData) -> ModelRun:
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_x, train_y = flatten_split(x, panel.target, panel.split["train"])
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 25)))
    model.fit(train_x, train_y)
    pred = model.predict(x.reshape(-1, x.shape[-1])).reshape(panel.target.shape)
    pred = np.clip(pred, EPS, None)
    return ModelRun(name, family, iv_channel, adjacency_name, "MSE", pred.astype(np.float32))


def qlike_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.clip(y_true, EPS, None)
    y_pred = np.clip(y_pred, EPS, None)
    ratio = y_true / y_pred
    return ratio - np.log(ratio) - 1.0


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (y_true - y_pred) ** 2


def fit_gnn(
    name: str,
    iv_channel: str,
    x: np.ndarray,
    y: np.ndarray,
    adjacency: np.ndarray,
    split: Dict[str, np.ndarray],
    layers: int,
    estimation: str,
    epochs: int,
    hidden: int,
    lr: float,
    seed: int,
) -> ModelRun:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    class GNNHARNet(nn.Module):
        def __init__(self, in_features: int, hidden_features: int, n_layers: int) -> None:
            super().__init__()
            dims = [in_features] + [hidden_features] * max(n_layers - 1, 0)
            self.weights = nn.ModuleList(
                [nn.Linear(dims[i] * 2, hidden_features) for i in range(n_layers)]
            )
            self.out = nn.Linear(hidden_features + in_features, 1)

        def forward(self, values: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
            h = values
            for layer in self.weights:
                neighbor_h = torch.matmul(adj, h)
                h = functional.relu(layer(torch.cat([h, neighbor_h], dim=-1)))
            return self.out(torch.cat([h, values], dim=-1)).squeeze(-1)

    train_x = x[split["train"]].reshape(-1, x.shape[-1])
    x_mean = train_x.mean(axis=0, keepdims=True)
    x_std = train_x.std(axis=0, keepdims=True)
    x_std = np.where(x_std < EPS, 1.0, x_std)
    x_scaled = (x - x_mean.reshape(1, 1, -1)) / x_std.reshape(1, 1, -1)

    use_qlike = estimation.upper() == "QLIKE"
    y_work = np.log(np.clip(y, EPS, None)) if use_qlike else y
    train_y_work = y_work[split["train"]]
    y_mean = float(train_y_work.mean())
    y_std = float(train_y_work.std())
    if y_std < EPS:
        y_std = 1.0
    y_scaled = (y_work - y_mean) / y_std

    # Preserve own-asset HAR information during graph propagation. The GLASSO
    # adjacency is intentionally zero-diagonal for spillover features, but a
    # neural message-passing layer needs self-loops to avoid discarding the
    # node's own lagged volatility at every layer.
    graph = np.asarray(adjacency, dtype=np.float32).copy()
    graph = graph + np.eye(graph.shape[0], dtype=np.float32)
    degree = graph.sum(axis=1)
    graph = np.diag(1.0 / np.sqrt(degree + EPS)).astype(np.float32) @ graph @ np.diag(
        1.0 / np.sqrt(degree + EPS)
    ).astype(np.float32)

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_t = torch.tensor(x_scaled, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_scaled, dtype=torch.float32, device=device)
    adj_t = torch.tensor(graph, dtype=torch.float32, device=device)
    train_idx = torch.tensor(split["train"], dtype=torch.long, device=device)
    valid_idx = torch.tensor(split["valid"], dtype=torch.long, device=device)
    model = GNNHARNet(x.shape[-1], hidden, layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state = None
    best_valid = float("inf")
    patience = max(25, epochs // 8)
    stale = 0

    def criterion(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        if use_qlike:
            pred_orig = torch.exp(pred * y_std + y_mean).clamp_min(EPS)
            truth_orig = torch.exp(truth * y_std + y_mean).clamp_min(EPS)
            ratio = truth_orig / pred_orig
            return (ratio - torch.log(ratio) - 1.0).mean()
        return ((truth - pred) ** 2).mean()

    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(x_t[train_idx], adj_t)
        loss = criterion(pred, y_t[train_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_pred = model(x_t[valid_idx], adj_t)
            valid_loss = criterion(valid_pred, y_t[valid_idx]).item()
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
    model.eval()
    with torch.no_grad():
        pred_scaled = model(x_t, adj_t).detach().cpu().numpy()
    if use_qlike:
        pred = np.exp(pred_scaled * y_std + y_mean)
    else:
        pred = pred_scaled * y_std + y_mean
    pred = np.clip(pred, EPS, None)
    return ModelRun(name, "GNNHAR", iv_channel, "GLASSO", estimation.upper(), pred.astype(np.float32))


def score_on_split(run: ModelRun, panel: PanelData, split_name: str) -> Tuple[float, float]:
    idx = panel.split[split_name]
    y = panel.target[idx]
    pred = run.prediction[idx]
    return float(mse_loss(y, pred).mean()), float(qlike_loss(y, pred).mean())


def tune_gnn_run(
    name: str,
    iv_channel: str,
    x: np.ndarray,
    y: np.ndarray,
    adjacency: np.ndarray,
    panel: PanelData,
    layers: int,
    estimation: str,
    base_epochs: int,
    base_hidden: int,
    base_lr: float,
    seed: int,
    fast: bool,
) -> Tuple[ModelRun, pd.DataFrame]:
    if fast:
        hidden_grid = sorted(set([base_hidden, 16, 32]))
        lr_grid = sorted(set([base_lr, 1e-3]))
        epoch_grid = [base_epochs]
    else:
        hidden_grid = sorted(set([base_hidden, 8, 16, 32, 64]))
        lr_grid = sorted(set([base_lr, 3e-4, 1e-3, 3e-3]))
        epoch_grid = sorted(set([base_epochs, max(300, base_epochs), max(500, base_epochs)]))
    records = []
    best_run = None
    best_score = float("inf")
    best_key = None

    for hidden in hidden_grid:
        for lr in lr_grid:
            for epochs in epoch_grid:
                candidate = fit_gnn(
                    name=name,
                    iv_channel=iv_channel,
                    x=x,
                    y=y,
                    adjacency=adjacency,
                    split=panel.split,
                    layers=layers,
                    estimation=estimation,
                    epochs=epochs,
                    hidden=hidden,
                    lr=lr,
                    seed=seed,
                )
                valid_mse, valid_qlike = score_on_split(candidate, panel, "valid")
                test_mse, test_qlike = score_on_split(candidate, panel, "test")
                score = valid_qlike if estimation.upper() == "QLIKE" else valid_mse
                records.append(
                    {
                        "model": name,
                        "layers": layers,
                        "estimation": estimation.upper(),
                        "hidden": hidden,
                        "lr": lr,
                        "epochs": epochs,
                        "seed": seed,
                        "valid_mse": valid_mse,
                        "valid_qlike": valid_qlike,
                        "test_mse_diagnostic": test_mse,
                        "test_qlike_diagnostic": test_qlike,
                        "selected": False,
                    }
                )
                key = (score, hidden, lr, epochs)
                if key < (best_score, *(best_key or (float("inf"), float("inf"), float("inf")))):
                    best_score = score
                    best_key = (hidden, lr, epochs)
                    best_run = candidate

    table = pd.DataFrame(records)
    if best_key is not None:
        hidden, lr, epochs = best_key
        mask = (
            (table["hidden"] == hidden)
            & (table["lr"] == lr)
            & (table["epochs"] == epochs)
            & (table["model"] == name)
        )
        table.loc[mask, "selected"] = True
    if best_run is None:
        raise RuntimeError(f"GNN tuning failed for {name}")
    return best_run, table


def evaluate_runs(runs: List[ModelRun], panel: PanelData) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    test_idx = panel.split["test"]
    y = panel.target[test_idx]
    records = []
    losses_mse = {}
    losses_qlike = {}
    for run in runs:
        pred = run.prediction[test_idx]
        mse = mse_loss(y, pred)
        qlike = qlike_loss(y, pred)
        losses_mse[run.name] = mse.mean(axis=1)
        losses_qlike[run.name] = qlike.mean(axis=1)
        records.append(
            {
                "model": run.name,
                "family": run.family,
                "iv_channel": run.iv_channel,
                "adjacency": run.adjacency,
                "estimation": run.estimation,
                "test_mse": float(mse.mean()),
                "test_qlike": float(qlike.mean()),
            }
        )
    table = pd.DataFrame(records).sort_values("test_qlike").reset_index(drop=True)
    har_mse = float(table.loc[table["model"] == "HAR", "test_mse"].iloc[0])
    har_qlike = float(table.loc[table["model"] == "HAR", "test_qlike"].iloc[0])
    hariv_mse = float(table.loc[table["model"] == "HAR+IV", "test_mse"].iloc[0])
    hariv_qlike = float(table.loc[table["model"] == "HAR+IV", "test_qlike"].iloc[0])
    table["mse_ratio_vs_har"] = table["test_mse"] / har_mse
    table["qlike_ratio_vs_har"] = table["test_qlike"] / har_qlike
    table["mse_ratio_vs_har_iv"] = table["test_mse"] / hariv_mse
    table["qlike_ratio_vs_har_iv"] = table["test_qlike"] / hariv_qlike
    return table, {
        "mse": pd.DataFrame(losses_mse, index=panel.dates[test_idx]),
        "qlike": pd.DataFrame(losses_qlike, index=panel.dates[test_idx]),
    }


def dm_test(loss_a: pd.Series, loss_b: pd.Series) -> Tuple[float, float]:
    diff = (loss_a - loss_b).dropna().to_numpy()
    n = len(diff)
    if n < 3 or float(np.std(diff, ddof=1)) < EPS:
        return float("nan"), float("nan")
    stat = float(diff.mean() / (diff.std(ddof=1) / math.sqrt(n)))
    try:
        from scipy.stats import t

        pvalue = float(2.0 * (1.0 - t.cdf(abs(stat), df=n - 1)))
    except Exception:
        pvalue = float(math.erfc(abs(stat) / math.sqrt(2.0)))
    return stat, pvalue


def build_dm_table(losses: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("HAR", "GHAR"),
        ("GHAR", "GNNHAR1L"),
        ("GHAR+IV", "GNNHAR1L-IV"),
        ("HAR+IV", "HAR+fakeIV"),
        ("GNNHAR1L-IV", "GNNHAR2L-IV"),
    ]
    records = []
    for first, second in pairs:
        if first not in losses or second not in losses:
            continue
        stat, pvalue = dm_test(losses[first], losses[second])
        records.append(
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
    return pd.DataFrame(records)


def build_mcs_table(losses: pd.DataFrame, bootstrap: int) -> pd.DataFrame:
    try:
        from src.gnnhar.mcs import ModelConfidenceSet

        mcs = ModelConfidenceSet(losses, alpha=0.05, B=bootstrap, w=5, algorithm="SQ").run()
        pvalues = mcs.pvalues
        rows = []
        for model in losses.columns:
            rows.append(
                {
                    "model": model,
                    "included_at_5pct": model in set(mcs.included),
                    "mcs_pvalue": float(pvalues.get(model, np.nan)),
                }
            )
        return pd.DataFrame(rows).sort_values(["included_at_5pct", "mcs_pvalue"], ascending=[False, False])
    except Exception as exc:
        return pd.DataFrame(
            [{"model": model, "included_at_5pct": np.nan, "mcs_pvalue": np.nan, "note": str(exc)} for model in losses.columns]
        )


def build_iv_decomposition(loss_table: pd.DataFrame) -> pd.DataFrame:
    families = [
        ("HAR", "HAR", "HAR+IV", "HAR+fakeIV"),
        ("GHAR", "GHAR", "GHAR+IV", "GHAR+fakeIV"),
        ("GNNHAR1L", "GNNHAR1L", "GNNHAR1L-IV", "GNNHAR1L-IV+fakeIV"),
    ]
    rows = []
    lookup = loss_table.set_index("model")
    for family, base, real_iv, fake_iv in families:
        if not {base, real_iv, fake_iv}.issubset(lookup.index):
            continue
        base_loss = float(lookup.loc[base, "test_qlike"])
        real_loss = float(lookup.loc[real_iv, "test_qlike"])
        fake_loss = float(lookup.loc[fake_iv, "test_qlike"])
        rows.append(
            {
                "family": family,
                "baseline_qlike": base_loss,
                "real_iv_qlike": real_loss,
                "fake_iv_qlike": fake_loss,
                "total_iv_improvement": base_loss - real_loss,
                "genuine_information_gain": fake_loss - real_loss,
                "parameter_expansion_gain": base_loss - fake_loss,
            }
        )
    return pd.DataFrame(rows)


def build_regime_table(runs: List[ModelRun], panel: PanelData) -> pd.DataFrame:
    test_idx = panel.split["test"]
    y = panel.target[test_idx]
    market_rv = y.mean(axis=1)
    threshold = float(np.quantile(market_rv, 0.75))
    records = []
    for regime, mask in {
        "calm": market_rv < threshold,
        "volatile_top_quartile": market_rv >= threshold,
    }.items():
        for run in runs:
            pred = run.prediction[test_idx][mask]
            truth = y[mask]
            records.append(
                {
                    "regime": regime,
                    "model": run.name,
                    "n_dates": int(mask.sum()),
                    "mse": float(mse_loss(truth, pred).mean()),
                    "qlike": float(qlike_loss(truth, pred).mean()),
                }
            )
    return pd.DataFrame(records).sort_values(["regime", "qlike"])


def save_tables(tables: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(table_dir / f"{name}.csv", index=False)
        try:
            table.to_latex(table_dir / f"{name}.tex", index=False, float_format="%.6g")
        except Exception:
            pass


def save_figures(
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

    plt.figure(figsize=(8, 7))
    plt.imshow(adjacency.to_numpy(), cmap="viridis")
    plt.colorbar(label="normalized GLASSO weight")
    plt.xticks(range(len(panel.tickers)), panel.tickers, rotation=90, fontsize=7)
    plt.yticks(range(len(panel.tickers)), panel.tickers, fontsize=7)
    plt.title("GLASSO adjacency")
    plt.tight_layout()
    plt.savefig(fig_dir / "glasso_adjacency_heatmap.png", dpi=200)
    plt.close()

    best_names = loss_table.head(min(10, len(loss_table)))["model"].tolist()
    y = panel.target[panel.split["test"]]
    predictions = {run.name: run.prediction[panel.split["test"]] for run in runs}
    test_dates = panel.dates[panel.split["test"]].astype(str).to_numpy()
    np.savez_compressed(
        output_dir / "predictions_test.npz",
        truth=y.astype(np.float32),
        dates=test_dates,
        tickers=np.asarray(panel.tickers),
        **{f"pred_{run.name.replace('+', 'plus').replace('-', '_')}": predictions[run.name].astype(np.float32) for run in runs},
    )

    plt.figure(figsize=(10, 5))
    plt.boxplot(
        [np.ravel(np.abs(y - predictions[name])) for name in best_names],
        tick_labels=best_names,
        showfliers=False,
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("absolute forecast error")
    plt.title("Forecast error distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "forecast_error_boxplot.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.boxplot(
        [np.ravel(predictions[name] / np.clip(y, EPS, None)) for name in best_names],
        tick_labels=best_names,
        showfliers=False,
    )
    plt.axhline(1.0, color="black", linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("forecast / realized RV")
    plt.title("Forecast ratio distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "forecast_ratio_boxplot.png", dpi=200)
    plt.close()

    plot_table = loss_table.sort_values("test_qlike").head(12)
    plt.figure(figsize=(10, 5))
    plt.bar(plot_table["model"], plot_table["qlike_ratio_vs_har"])
    plt.axhline(1.0, color="black", linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("QLIKE ratio vs HAR")
    plt.title("Out-of-sample QLIKE comparison")
    plt.tight_layout()
    plt.savefig(fig_dir / "model_comparison_bar.png", dpi=200)
    plt.close()

    if not iv_decomposition.empty:
        idx = np.arange(len(iv_decomposition))
        width = 0.35
        plt.figure(figsize=(8, 5))
        plt.bar(idx - width / 2, iv_decomposition["genuine_information_gain"], width, label="genuine IV")
        plt.bar(idx + width / 2, iv_decomposition["parameter_expansion_gain"], width, label="parameter expansion")
        plt.xticks(idx, iv_decomposition["family"])
        plt.ylabel("QLIKE reduction")
        plt.title("IV contribution decomposition")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "iv_decomposition.png", dpi=200)
        plt.close()

    selected = [name for name in ["HAR", "HAR+IV", "GHAR", "GHAR+IV", "GNNHAR1L", "GNNHAR1L-IV"] if name in predictions]
    market_rv = y.mean(axis=1)
    volatile_threshold = float(np.quantile(market_rv, 0.75))
    regimes = [
        ("Full test period", np.ones(len(market_rv), dtype=bool)),
        ("Calm days", market_rv < volatile_threshold),
        ("Volatile days", market_rv >= volatile_threshold),
    ]

    def horizontal_boxplot(ax, values: List[np.ndarray], title: str, reference: float, xlabel: str) -> None:
        ax.boxplot(values, vert=False, tick_labels=selected, showfliers=False, widths=0.65)
        ax.axvline(reference, color="0.35", linestyle="--", linewidth=1.0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="x", alpha=0.25)

    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharey=False)
    for row, (title, mask) in enumerate(regimes):
        errors = [np.ravel(predictions[name][mask] - y[mask]) for name in selected]
        ratios = [np.ravel(predictions[name][mask] / np.clip(y[mask], EPS, None)) for name in selected]
        horizontal_boxplot(axes[row, 0], errors, f"{title}: forecast errors", 0.0, "forecast - realized RV")
        horizontal_boxplot(axes[row, 1], ratios, f"{title}: forecast ratios", 1.0, "forecast / realized RV")
    fig.suptitle("Grouped forecast error and ratio boxplots", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(fig_dir / "zhang_style_error_ratio_boxplots.png", dpi=220)
    plt.close(fig)

    qq_models = [name for name in ["HAR", "HAR+IV", "GHAR", "GHAR+IV"] if name in predictions]
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.5))
    axes_flat = axes.ravel()
    try:
        from scipy.stats import norm

        probs = (np.arange(1, 1001) - 0.5) / 1000.0
        theoretical = norm.ppf(probs)
    except Exception:
        probs = (np.arange(1, 1001) - 0.5) / 1000.0
        theoretical = np.quantile(np.random.default_rng(12345).standard_normal(500000), probs)
    for ax, name in zip(axes_flat, qq_models):
        residuals = np.ravel(predictions[name] - y)
        residuals = (residuals - residuals.mean()) / max(residuals.std(ddof=1), EPS)
        empirical = np.quantile(residuals, probs)
        ax.scatter(theoretical, empirical, s=5, alpha=0.55)
        xlim = [float(theoretical.min()), float(theoretical.max())]
        ax.plot(xlim, xlim, color="black", linewidth=1)
        ax.set_xlim(xlim)
        ax.set_title(name)
        ax.set_xlabel("Normal quantile")
        ax.set_ylabel("Residual quantile")
        ax.grid(alpha=0.25)
    for ax in axes_flat[len(qq_models):]:
        ax.axis("off")
    fig.suptitle("Q-Q plots of standardized forecast residuals", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_dir / "residual_qq_plots.png", dpi=220)
    plt.close(fig)

    phase_models = [name for name in ["HAR", "HAR+IV", "GHAR", "GHAR+IV"] if name in predictions]
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.5))
    axes_flat = axes.ravel()
    truth_flat = np.ravel(y)
    lo, hi = np.quantile(truth_flat, [0.01, 0.99])
    for ax, name in zip(axes_flat, phase_models):
        pred_flat = np.ravel(predictions[name])
        ax.hexbin(truth_flat, pred_flat, gridsize=35, cmap="Blues", mincnt=1)
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(name)
        ax.set_xlabel("Realized RV")
        ax.set_ylabel("Forecast RV")
        ax.grid(alpha=0.2)
    for ax in axes_flat[len(phase_models):]:
        ax.axis("off")
    fig.suptitle("Forecast-realization phase plots", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_dir / "forecast_phase_plots.png", dpi=220)
    plt.close(fig)


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    panel: PanelData,
    adjacency: pd.DataFrame,
    loss_table: pd.DataFrame,
    iv_decomposition: pd.DataFrame,
    dm_table: pd.DataFrame,
) -> None:
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    best = loss_table.iloc[0]
    lines = [
        "# GNNHAR-IV Dow 30 Empirical Report",
        "",
        "## Design",
        "",
        "This run extends Zhang et al.'s GNNHAR realized-volatility forecasting design by adding implied volatility (IV) as an exogenous information channel. The baseline HAR uses daily, weekly, and monthly RV lags. GHAR and GNNHAR add GLASSO-based cross-sectional spillover features estimated from returns in the training window.",
        "",
        "The IV extension compares real-IV models with fake-IV controls. Fake IV is generated by date-permuting the IV feature tensor, preserving scale and dimensionality while breaking the timing relation with realized volatility. This separates genuine IV information from parameter expansion.",
        "",
        "All models are evaluated out of sample with MSE and QLIKE forecast losses. The `estimation` column records the training criterion, so training loss and forecast loss remain distinct.",
        "",
        "## Data",
        "",
        f"- Assets: {len(panel.tickers)} Dow 30 tickers",
        f"- Dates after HAR lag construction: {panel.dates.min().date()} to {panel.dates.max().date()}",
        f"- Split: {len(panel.split['train'])} train, {len(panel.split['valid'])} validation, {len(panel.split['test'])} test dates",
        f"- GLASSO alpha: {adjacency.attrs.get('alpha', float('nan')):.6g}",
        f"- GLASSO edge density: {adjacency.attrs.get('density', float('nan')):.6g}",
        "",
        "## Main Result",
        "",
        f"The best model by test QLIKE is `{best['model']}` with QLIKE {best['test_qlike']:.6g} and MSE {best['test_mse']:.6g}.",
        "",
        "See `tables/model_losses.csv` for the complete ranking and `tables/loss_ratios.csv` for ratios relative to HAR and HAR+IV.",
        "",
        "## IV Contribution",
        "",
    ]
    if iv_decomposition.empty:
        lines.append("The IV decomposition table was not available because one or more comparison models were missing.")
    else:
        for _, row in iv_decomposition.iterrows():
            lines.append(
                f"- {row['family']}: total QLIKE improvement {row['total_iv_improvement']:.6g}; "
                f"genuine information gain {row['genuine_information_gain']:.6g}; "
                f"parameter expansion gain {row['parameter_expansion_gain']:.6g}."
            )
    lines.extend(
        [
            "",
            "## Statistical Evaluation",
            "",
            "The MCS table uses the Hansen, Lunde, and Nason model confidence set procedure at the 5 percent level with block bootstrap. DM tests report positive statistics when the second named model has lower average loss than the first.",
            "",
        ]
    )
    if not dm_table.empty:
        for _, row in dm_table.iterrows():
            lines.append(
                f"- {row['comparison']}: statistic {row['dm_stat_positive_favors_b']:.4g}, p-value {row['pvalue']:.4g}."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Zhang et al.'s GNNHAR logic is preserved through HAR lags, GLASSO spillover structure, one-, two-, and three-layer graph models, MSE/QLIKE forecast losses, relative loss ratios, MCS tests, DM comparisons, and regime-stratified evaluation. The IV channel is an extension rather than a replacement: it adds option-market information to the same RV forecasting framework.",
            "",
            "Multi-hop graph layers can help when volatility shocks propagate through the GLASSO network, but deeper models may also over-smooth cross-sectional signals or become unstable under QLIKE training. The fake-IV benchmark is therefore important: real-IV gains should exceed fake-IV gains before being interpreted as informational.",
            "",
            "## Limitations",
            "",
            "- The Dow 30 sample is smaller than Zhang et al.'s S&P 100 setting.",
            "- IV availability and cleaning choices can affect the economic interpretation.",
            "- Additional IV features increase model dimension and can overfit.",
            "- QLIKE-trained neural models can be numerically less stable than MSE-trained models.",
            "- Colab hardware, CUDA availability, and random seeds can create small run-to-run differences.",
            "",
            "## Reproduction",
            "",
            "The run configuration is stored in `run_metadata.json`. The Colab launcher clones the GitHub branch, mounts Drive, and writes tables, figures, and this report to the configured Drive output directory.",
        ]
    )
    (report_dir / "gnnhar_iv_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_metadata(output_dir: Path, args: argparse.Namespace, panel: PanelData) -> None:
    metadata = {
        "args": vars(args),
        "n_dates": len(panel.dates),
        "tickers": panel.tickers,
        "date_start": str(panel.dates.min().date()),
        "date_end": str(panel.dates.max().date()),
        "split_sizes": {key: int(len(value)) for key, value in panel.split.items()},
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.fast:
        args.epochs = min(args.epochs, 40)
        args.mcs_bootstrap = min(args.mcs_bootstrap, 60)

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = load_panel(data_dir, args.seed)
    glasso = glasso_adjacency(panel.returns, panel.dates[panel.split["train"]])
    glasso_np = glasso.to_numpy(dtype=np.float32)
    random_np = random_adjacency(len(panel.tickers), args.seed + 1, glasso.attrs.get("density", 0.15)).astype(np.float32)

    designs = {
        "HAR": make_design(panel, None, None),
        "GHAR": make_design(panel, glasso_np, None),
        "GHAR_RANDOM": make_design(panel, random_np, None),
        "HAR_IV": make_design(panel, None, panel.iv_features),
        "GHAR_IV": make_design(panel, glasso_np, panel.iv_features),
        "HAR_FAKE": make_design(panel, None, panel.fake_iv_features),
        "GHAR_FAKE": make_design(panel, glasso_np, panel.fake_iv_features),
        "GNN_IV_FAKE": make_design(panel, glasso_np, panel.fake_iv_features),
    }

    runs: List[ModelRun] = [
        fit_linear("HAR", "HAR", "none", "Identity", designs["HAR"], panel),
        fit_linear("GHAR", "GHAR", "none", "GLASSO", designs["GHAR"], panel),
        fit_linear("GHAR_RANDOM", "GHAR", "none", "Random", designs["GHAR_RANDOM"], panel),
        fit_linear("HAR+IV", "HAR", "real", "Identity", designs["HAR_IV"], panel),
        fit_linear("GHAR+IV", "GHAR", "real", "GLASSO", designs["GHAR_IV"], panel),
        fit_linear("HAR+fakeIV", "HAR", "fake", "Identity", designs["HAR_FAKE"], panel),
        fit_linear("GHAR+fakeIV", "GHAR", "fake", "GLASSO", designs["GHAR_FAKE"], panel),
    ]

    gnn_specs = [
        ("GNNHAR1L", "none", designs["HAR"], 1, "MSE"),
        ("GNNHAR2L", "none", designs["HAR"], 2, "MSE"),
        ("GNNHAR3L", "none", designs["HAR"], 3, "MSE"),
        ("GNNHAR1L-IV", "real", designs["HAR_IV"], 1, "MSE"),
        ("GNNHAR2L-IV", "real", designs["HAR_IV"], 2, "MSE"),
        ("GNNHAR3L-IV", "real", designs["HAR_IV"], 3, "MSE"),
        ("GNNHAR1L-IV+fakeIV", "fake", designs["HAR_FAKE"], 1, "MSE"),
    ]
    if not args.skip_qlike_training:
        gnn_specs.extend(
            [
                ("GNNHAR1L-QLIKE", "none", designs["HAR"], 1, "QLIKE"),
                ("GNNHAR1L-IV-QLIKE", "real", designs["HAR_IV"], 1, "QLIKE"),
            ]
        )

    tuning_tables = []
    for offset, (name, iv_channel, design, layers, estimation) in enumerate(gnn_specs):
        if args.tune_gnn:
            run, tuning_table = tune_gnn_run(
                name=name,
                iv_channel=iv_channel,
                x=design,
                y=panel.target,
                adjacency=glasso_np,
                panel=panel,
                layers=layers,
                estimation=estimation,
                base_epochs=args.epochs,
                base_hidden=args.hidden,
                base_lr=args.lr,
                seed=args.seed + offset,
                fast=args.fast,
            )
            runs.append(run)
            tuning_tables.append(tuning_table)
        else:
            runs.append(
                fit_gnn(
                    name=name,
                    iv_channel=iv_channel,
                    x=design,
                    y=panel.target,
                    adjacency=glasso_np,
                    split=panel.split,
                    layers=layers,
                    estimation=estimation,
                    epochs=args.epochs,
                    hidden=args.hidden,
                    lr=args.lr,
                    seed=args.seed + offset,
                )
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
    mcs_table = build_mcs_table(losses["qlike"], args.mcs_bootstrap)
    dm_table = build_dm_table(losses["qlike"])
    iv_decomposition = build_iv_decomposition(loss_table)
    regime_table = build_regime_table(runs, panel)

    tables = {
        "model_losses": loss_table,
        "loss_ratios": ratio_table,
        "mcs_results": mcs_table,
        "dm_tests": dm_table,
        "iv_decomposition": iv_decomposition,
        "regime_results": regime_table,
    }
    if tuning_tables:
        tables["gnn_tuning"] = pd.concat(tuning_tables, ignore_index=True)
    save_tables(tables, output_dir)
    save_figures(panel, glasso, runs, loss_table, iv_decomposition, output_dir)
    write_report(output_dir, args, panel, glasso, loss_table, iv_decomposition, dm_table)
    save_metadata(output_dir, args, panel)
    print(f"Wrote GNNHAR-IV outputs to {output_dir}")


if __name__ == "__main__":
    main()
