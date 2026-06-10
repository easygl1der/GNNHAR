#!/usr/bin/env python3
"""Summarize Dow30, S&P100, and S&P500 GNNHAR-IV scale results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dow30-output", required=True)
    parser.add_argument("--sp100-output", required=True)
    parser.add_argument("--sp500-output", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--compile-tex", action="store_true")
    return parser.parse_args()


def read_metadata(path: Path) -> Dict[str, object]:
    metadata_path = path / "run_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def read_run(label: str, path_text: str) -> Dict[str, object]:
    path = Path(path_text)
    losses = pd.read_csv(path / "tables" / "model_losses.csv")
    metadata = read_metadata(path)
    if "panel" in metadata:
        panel = metadata.get("panel", {})
        n_assets = int(panel.get("selected_tickers", metadata.get("n_tickers", 0)))
        date_start = str(panel.get("date_start", metadata.get("date_start", "")))
        date_end = str(panel.get("date_end", metadata.get("date_end", "")))
    else:
        tickers = metadata.get("tickers", [])
        n_assets = int(len(tickers) if tickers else metadata.get("n_tickers", 30))
        date_start = str(metadata.get("date_start", ""))
        date_end = str(metadata.get("date_end", ""))
    return {
        "label": label,
        "path": path,
        "losses": losses,
        "metadata": metadata,
        "n_assets": n_assets,
        "n_dates": int(metadata.get("n_dates", 0)),
        "date_start": date_start,
        "date_end": date_end,
    }


def loss_lookup(losses: pd.DataFrame, metric: str) -> pd.Series:
    return losses.set_index("model")[metric]


def get_loss(losses: pd.DataFrame, model: str, metric: str) -> Optional[float]:
    lookup = loss_lookup(losses, metric)
    if model not in lookup.index:
        return None
    return float(lookup.loc[model])


def improvement(losses: pd.DataFrame, base: str, improved: str, metric: str) -> Optional[float]:
    base_loss = get_loss(losses, base, metric)
    improved_loss = get_loss(losses, improved, metric)
    if base_loss is None or improved_loss is None or base_loss == 0:
        return None
    return 1.0 - improved_loss / base_loss


def best_model(losses: pd.DataFrame, prefix: str, metric: str, contains_iv: Optional[bool]) -> Optional[str]:
    table = losses.copy()
    table = table[table["model"].astype(str).str.startswith(prefix)]
    if contains_iv is True:
        table = table[table["model"].astype(str).str.contains("IV")]
    elif contains_iv is False:
        table = table[~table["model"].astype(str).str.contains("IV")]
    table = table[~table["model"].astype(str).str.contains("fake", case=False, regex=False)]
    table = table[~table["model"].astype(str).str.contains("random", case=False, regex=False)]
    if table.empty:
        return None
    return str(table.sort_values(metric).iloc[0]["model"])


def add_gain_row(rows: List[Dict[str, object]], run: Dict[str, object], name: str, base: str, improved: str) -> None:
    losses = run["losses"]
    rows.append(
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


def build_summary(runs: Iterable[Dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, object]] = []
    gain_rows: List[Dict[str, object]] = []
    for run in runs:
        losses = run["losses"]
        best_qlike = losses.sort_values("test_qlike").iloc[0]
        best_mse = losses.sort_values("test_mse").iloc[0]
        best_gnn = best_model(losses, "GNNHAR", "test_qlike", contains_iv=False)
        best_gnn_iv = best_model(losses, "GNNHAR", "test_qlike", contains_iv=True)
        summary_rows.append(
            {
                "universe": run["label"],
                "n_assets": run["n_assets"],
                "n_dates": run["n_dates"],
                "date_start": run["date_start"],
                "date_end": run["date_end"],
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
        )
        add_gain_row(gain_rows, run, "Graph gain without IV: GHAR vs HAR", "HAR", "GHAR")
        add_gain_row(gain_rows, run, "Graph gain with IV: GHAR+IV vs HAR+IV", "HAR+IV", "GHAR+IV")
        add_gain_row(gain_rows, run, "IV gain in HAR: HAR+IV vs HAR", "HAR", "HAR+IV")
        add_gain_row(gain_rows, run, "IV gain in GHAR: GHAR+IV vs GHAR", "GHAR", "GHAR+IV")
        if best_gnn:
            add_gain_row(gain_rows, run, f"Best non-IV GNN vs GHAR: {best_gnn}", "GHAR", best_gnn)
        if best_gnn_iv:
            add_gain_row(gain_rows, run, f"Best IV GNN vs GHAR+IV: {best_gnn_iv}", "GHAR+IV", best_gnn_iv)
        if best_gnn and best_gnn_iv:
            add_gain_row(gain_rows, run, f"IV gain in best GNN: {best_gnn_iv} vs {best_gnn}", best_gnn, best_gnn_iv)
    return pd.DataFrame(summary_rows), pd.DataFrame(gain_rows)


def save_figures(summary: pd.DataFrame, gains: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    graph_gain = gains[gains["comparison"].str.startswith("Graph gain without IV")].sort_values("n_assets")
    iv_gain = gains[gains["comparison"].str.startswith("IV gain in GHAR")].sort_values("n_assets")

    plt.figure(figsize=(7.5, 4.5))
    if not graph_gain.empty:
        plt.plot(graph_gain["n_assets"], graph_gain["qlike_gain"] * 100.0, marker="o", label="GHAR vs HAR")
    if not iv_gain.empty:
        plt.plot(iv_gain["n_assets"], iv_gain["qlike_gain"] * 100.0, marker="s", label="GHAR+IV vs GHAR")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("number of assets")
    plt.ylabel("QLIKE improvement (%)")
    plt.title("Scale experiment gains")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "scale_gains_by_assets.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8.5, 4.8))
    table = summary.sort_values("n_assets")
    plt.plot(table["n_assets"], table["har_qlike"], marker="o", label="HAR")
    plt.plot(table["n_assets"], table["ghar_qlike"], marker="o", label="GHAR")
    plt.plot(table["n_assets"], table["ghar_iv_qlike"], marker="o", label="GHAR+IV")
    plt.xlabel("number of assets")
    plt.ylabel("test QLIKE")
    plt.title("Out-of-sample QLIKE by universe")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "scale_qlike_by_assets.png", dpi=220)
    plt.close()


def latex_escape(text: object) -> str:
    value = str(text)
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def format_percent(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    return f"{float(value) * 100.0:.2f}\\%"


def write_latex_report(summary: pd.DataFrame, gains: pd.DataFrame, output_dir: Path) -> Path:
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    tex_path = report_dir / "scale_experiment_report.tex"
    graph_gain = gains[gains["comparison"].str.startswith("Graph gain without IV")].copy()
    iv_gain = gains[gains["comparison"].str.startswith("IV gain in GHAR")].copy()

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{amsmath}",
        r"\usepackage{hyperref}",
        r"\title{Scale Effects in GHAR and GNNHAR-IV Volatility Forecasting}",
        r"\author{Volatility Research Pipeline}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\section{Research Question}",
        (
            r"We test whether graph structure becomes more valuable as the cross-sectional universe expands. "
            r"For each universe, the target is 30-day realized volatility and the exogenous option-market "
            r"channel is 30-day mean implied volatility. The key estimand is the out-of-sample loss reduction "
            r"\(1 - L_{\mathrm{enhanced}} / L_{\mathrm{baseline}}\), reported for MSE and QLIKE."
        ),
        r"\section{Universe-Level Results}",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Best models and baseline losses by universe}",
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Universe & Assets & Dates & HAR QLIKE & GHAR+IV QLIKE & Best QLIKE model \\",
        r"\midrule",
    ]
    for _, row in summary.sort_values("n_assets").iterrows():
        lines.append(
            f"{latex_escape(row['universe'])} & {int(row['n_assets'])} & {int(row['n_dates'])} & "
            f"{float(row['har_qlike']):.6g} & {float(row['ghar_iv_qlike']):.6g} & "
            f"{latex_escape(row['best_qlike_model'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            r"\section{Scale Gains}",
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Selected QLIKE gains by universe}",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Universe & Assets & GHAR over HAR & GHAR+IV over GHAR \\",
            r"\midrule",
        ]
    )
    merged = summary[["universe", "n_assets"]].merge(
        graph_gain[["universe", "qlike_gain"]].rename(columns={"qlike_gain": "graph_gain"}),
        on="universe",
        how="left",
    )
    merged = merged.merge(
        iv_gain[["universe", "qlike_gain"]].rename(columns={"qlike_gain": "iv_gain"}),
        on="universe",
        how="left",
    )
    for _, row in merged.sort_values("n_assets").iterrows():
        lines.append(
            f"{latex_escape(row['universe'])} & {int(row['n_assets'])} & "
            f"{format_percent(row['graph_gain'])} & {format_percent(row['iv_gain'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            r"\section{Interpretation}",
            (
                r"A positive GHAR-over-HAR gain means the estimated asset graph adds forecasting information "
                r"beyond own-asset HAR lags. A positive GHAR+IV-over-GHAR gain means the implied-volatility "
                r"channel contributes information beyond the graph. The GNN rows in the CSV output evaluate "
                r"whether nonlinear message passing improves on the linear graph benchmark."
            ),
            r"\section{Artifacts}",
            r"The machine-readable tables are \texttt{scale\_summary.csv} and \texttt{scale\_gains.csv}. "
            r"The main figures are saved under \texttt{figures/}.",
            r"\end{document}",
        ]
    )
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tex_path


def maybe_compile(tex_path: Path) -> None:
    import subprocess

    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", tex_path.name],
        cwd=str(tex_path.parent),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        read_run("Dow30", args.dow30_output),
        read_run("S&P100", args.sp100_output),
        read_run("S&P500", args.sp500_output),
    ]
    summary, gains = build_summary(runs)
    summary.to_csv(output_dir / "scale_summary.csv", index=False)
    gains.to_csv(output_dir / "scale_gains.csv", index=False)
    save_figures(summary, gains, output_dir)
    tex_path = write_latex_report(summary, gains, output_dir)
    if args.compile_tex:
        maybe_compile(tex_path)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary_csv": str(output_dir / "scale_summary.csv"),
                "gains_csv": str(output_dir / "scale_gains.csv"),
                "tex_report": str(tex_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
