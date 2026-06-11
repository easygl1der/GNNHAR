#!/usr/bin/env python3
"""Summarize Zhang-style rolling scale experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Universe output directories")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def get_loss(losses: pd.DataFrame, model: str, metric: str) -> Optional[float]:
    lookup = losses.set_index("model")
    if model not in lookup.index:
        return None
    return float(lookup.loc[model, metric])


def improvement(losses: pd.DataFrame, base: str, improved: str, metric: str) -> Optional[float]:
    base_loss = get_loss(losses, base, metric)
    improved_loss = get_loss(losses, improved, metric)
    if base_loss is None or improved_loss is None or abs(base_loss) < 1e-15:
        return None
    return 1.0 - improved_loss / base_loss


def best_model(losses: pd.DataFrame, prefix: str, metric: str, iv: Optional[bool]) -> Optional[str]:
    table = losses[losses["model"].astype(str).str.startswith(prefix)].copy()
    if iv is True:
        table = table[table["model"].astype(str).str.contains("IV")]
    elif iv is False:
        table = table[~table["model"].astype(str).str.contains("IV")]
    table = table[~table["model"].astype(str).str.contains("fake", case=False, regex=False)]
    if table.empty:
        return None
    return str(table.sort_values(metric).iloc[0]["model"])


def read_run(path: Path, label: str) -> Dict[str, object]:
    losses = pd.read_csv(path / "tables" / "model_losses.csv")
    graph = pd.read_csv(path / "tables" / "graph_blocks.csv")
    metadata_path = path / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    panel = metadata.get("panel", {})
    rolling = metadata.get("rolling", {})
    return {
        "label": label,
        "path": path,
        "losses": losses,
        "graph": graph,
        "metadata": metadata,
        "n_assets": int(panel.get("selected_tickers", 0)),
        "n_dates": int(panel.get("n_dates", 0)),
        "n_test_dates": int(rolling.get("n_test_dates", 0)),
        "fallback_blocks": int(rolling.get("fallback_blocks", graph["graph_fallback"].notna().sum())),
        "n_blocks": int(rolling.get("n_blocks", len(graph))),
    }


def summarize(runs: List[Dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, object]] = []
    gain_rows: List[Dict[str, object]] = []
    graph_rows: List[Dict[str, object]] = []
    for run in runs:
        losses = run["losses"]
        graph = run["graph"]
        best_qlike = losses.sort_values("test_qlike").iloc[0]
        best_mse = losses.sort_values("test_mse").iloc[0]
        best_gnn = best_model(losses, "GNNHAR", "test_qlike", iv=False)
        best_gnn_iv = best_model(losses, "GNNHAR", "test_qlike", iv=True)
        row = {
            "universe": run["label"],
            "path": str(run["path"]),
            "n_assets": run["n_assets"],
            "n_dates": run["n_dates"],
            "n_test_dates": run["n_test_dates"],
            "n_blocks": run["n_blocks"],
            "fallback_blocks": run["fallback_blocks"],
            "best_qlike_model": best_qlike["model"],
            "best_test_qlike": float(best_qlike["test_qlike"]),
            "best_mse_model": best_mse["model"],
            "best_test_mse": float(best_mse["test_mse"]),
            "best_gnn_qlike_model": best_gnn,
            "best_gnn_iv_qlike_model": best_gnn_iv,
            "har_qlike": get_loss(losses, "HAR", "test_qlike"),
            "ghar_qlike": get_loss(losses, "GHAR", "test_qlike"),
            "har_iv_qlike": get_loss(losses, "HAR+IV", "test_qlike"),
            "ghar_iv_qlike": get_loss(losses, "GHAR+IV", "test_qlike"),
        }
        summary_rows.append(row)

        comparisons = [
            ("Graph gain without IV: GHAR vs HAR", "HAR", "GHAR"),
            ("Graph gain with IV: GHAR+IV vs HAR+IV", "HAR+IV", "GHAR+IV"),
            ("IV gain in HAR: HAR+IV vs HAR", "HAR", "HAR+IV"),
            ("IV gain in GHAR: GHAR+IV vs GHAR", "GHAR", "GHAR+IV"),
        ]
        if best_gnn:
            comparisons.append((f"Best non-IV GNN vs GHAR: {best_gnn}", "GHAR", best_gnn))
        if best_gnn_iv:
            comparisons.append((f"Best IV GNN vs GHAR+IV: {best_gnn_iv}", "GHAR+IV", best_gnn_iv))
        if best_gnn and best_gnn_iv:
            comparisons.append((f"IV gain in best GNN: {best_gnn_iv} vs {best_gnn}", best_gnn, best_gnn_iv))
        for name, base, improved in comparisons:
            gain_rows.append(
                {
                    "universe": run["label"],
                    "n_assets": run["n_assets"],
                    "comparison": name,
                    "base_model": base,
                    "improved_model": improved,
                    "mse_gain": improvement(losses, base, improved, "test_mse"),
                    "qlike_gain": improvement(losses, base, improved, "test_qlike"),
                }
            )

        graph_rows.append(
            {
                "universe": run["label"],
                "n_assets": run["n_assets"],
                "n_blocks": len(graph),
                "fallback_blocks": int(graph["graph_fallback"].notna().sum()),
                "median_edges": float(graph["graph_edges"].median()),
                "median_density": float(graph["graph_density"].median()),
                "methods": ";".join(f"{k}:{v}" for k, v in graph["graph_method"].value_counts(dropna=False).items()),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(gain_rows), pd.DataFrame(graph_rows)


def save_figures(summary: pd.DataFrame, gains: pd.DataFrame, graph: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    graph_gain = gains[gains["comparison"].str.startswith("Graph gain without IV")].sort_values("n_assets")
    graph_iv_gain = gains[gains["comparison"].str.startswith("Graph gain with IV")].sort_values("n_assets")
    plt.figure(figsize=(7.5, 4.5))
    if not graph_gain.empty:
        plt.plot(graph_gain["n_assets"], graph_gain["qlike_gain"] * 100.0, marker="o", label="GHAR vs HAR")
    if not graph_iv_gain.empty:
        plt.plot(graph_iv_gain["n_assets"], graph_iv_gain["qlike_gain"] * 100.0, marker="s", label="GHAR+IV vs HAR+IV")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("number of assets")
    plt.ylabel("QLIKE improvement (%)")
    plt.title("Zhang-style rolling graph gains")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "rolling_scale_graph_gains.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.5, 4.5))
    plt.bar(graph["universe"], graph["fallback_blocks"])
    plt.ylabel("fallback blocks")
    plt.title("Rolling graph construction fallback audit")
    plt.tight_layout()
    plt.savefig(fig_dir / "rolling_graph_fallback_blocks.png", dpi=220)
    plt.close()


def write_report(summary: pd.DataFrame, gains: pd.DataFrame, graph: pd.DataFrame, output_dir: Path) -> None:
    def to_markdown_simple(table: pd.DataFrame) -> str:
        if table.empty:
            return "_No rows._"
        string_table = table.copy()
        for col in string_table.columns:
            string_table[col] = string_table[col].map(
                lambda value: "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
            )
        header = "| " + " | ".join(string_table.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(string_table.columns)) + " |"
        rows = [
            "| " + " | ".join(str(value).replace("|", "/") for value in row) + " |"
            for row in string_table.to_numpy()
        ]
        return "\n".join([header, sep] + rows)

    lines = [
        "# Zhang-style Rolling Scale Summary",
        "",
        "This summary reads only corrected rolling outputs from `gnnhar_iv_zhang_scale_pipeline.py`. Interpret scale gains only after checking `graph_audit.csv`; if a large universe has many fallback blocks, it is not a clean GLASSO scale test.",
        "",
        "## Universe Summary",
        "",
        to_markdown_simple(summary),
        "",
        "## Graph Audit",
        "",
        to_markdown_simple(graph),
        "",
        "## Gains",
        "",
        to_markdown_simple(gains),
        "",
    ]
    (output_dir / "rolling_scale_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = [Path(item) for item in args.run_dirs]
    labels = args.labels if args.labels else [path.name for path in paths]
    if len(labels) != len(paths):
        raise ValueError("--labels length must match --run-dirs")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [read_run(path, label) for path, label in zip(paths, labels)]
    summary, gains, graph = summarize(runs)
    summary.to_csv(output_dir / "rolling_scale_summary.csv", index=False)
    gains.to_csv(output_dir / "rolling_scale_gains.csv", index=False)
    graph.to_csv(output_dir / "graph_audit.csv", index=False)
    save_figures(summary, gains, graph, output_dir)
    write_report(summary, gains, graph, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "runs": labels}, indent=2))


if __name__ == "__main__":
    main()
