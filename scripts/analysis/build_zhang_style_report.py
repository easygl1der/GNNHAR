#!/usr/bin/env python3
"""Generate a LaTeX report from the Zhang-style statistics layer."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_ROOT = REPO_ROOT / "outputs" / "paper_ready_20260617"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "zhang_style_statistics_20260618"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def latex_escape(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for src, dst in repl.items():
        text = text.replace(src, dst)
    return text


def fmt_num(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return latex_escape(value)


def fmt_pct(value: object, digits: int = 1) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{100.0 * float(value):.{digits}f}\\%"
    except Exception:
        return latex_escape(value)


def fmt_int(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(int(float(value)))


def bool_mark(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        truth = value.strip().lower() in {"true", "1", "yes"}
    else:
        truth = bool(value)
    return "yes" if truth else "no"


def tabular(df: pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
    """Return a booktabs tabular. columns entries are (source, label, formatter)."""

    def apply_fmt(value: object, formatter: str) -> str:
        if formatter == "num2":
            return fmt_num(value, 2)
        if formatter == "num3":
            return fmt_num(value, 3)
        if formatter == "num4":
            return fmt_num(value, 4)
        if formatter == "num1":
            return fmt_num(value, 1)
        if formatter == "pct1":
            return fmt_pct(value, 1)
        if formatter == "int":
            return fmt_int(value)
        if formatter == "bool":
            return bool_mark(value)
        return latex_escape(value)

    align = "l" + "r" * (len(columns) - 1)
    header = " & ".join(latex_escape(label) for _, label, _ in columns) + r" \\"
    rows = [rf"\begin{{tabular}}{{{align}}}", r"\toprule", header, r"\midrule"]
    if df.empty:
        rows.append(r"\multicolumn{" + str(len(columns)) + r"}{c}{No rows available.} \\")
    else:
        for _, row in df.iterrows():
            rows.append(
                " & ".join(apply_fmt(row.get(src), formatter) for src, _, formatter in columns)
                + r" \\"
            )
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(rows)


def table_float(caption: str, label: str, body: str, note: str | None = None, size: str = "scriptsize") -> str:
    note_text = ""
    if note:
        note_text = "\n" + rf"\vspace{{2pt}}\begin{{minipage}}{{0.98\textwidth}}\footnotesize {note}\end{{minipage}}"
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            rf"\caption{{{latex_escape(caption)}}}",
            rf"\label{{{label}}}",
            rf"\{size}",
            rf"\resizebox{{\textwidth}}{{!}}{{%",
            body,
            r"}",
            note_text,
            r"\end{table}",
        ]
    )


def make_loss_table(loss: pd.DataFrame, mcs: pd.DataFrame, universe: str) -> pd.DataFrame:
    sub = loss[loss["universe"] == universe].copy()
    if sub.empty:
        return sub
    mcs_sub = mcs[mcs["universe"] == universe].copy()
    if not mcs_sub.empty:
        mse = mcs_sub[mcs_sub["metric"] == "MSE"][["model", "share_mcs_included", "mean_mcs_pvalue"]]
        ql = mcs_sub[mcs_sub["metric"] == "QLIKE"][["model", "share_mcs_included", "mean_mcs_pvalue"]]
        mse = mse.rename(columns={"share_mcs_included": "mcs_mse", "mean_mcs_pvalue": "mcs_mse_p"})
        ql = ql.rename(columns={"share_mcs_included": "mcs_qlike", "mean_mcs_pvalue": "mcs_qlike_p"})
        sub = sub.merge(mse, on="model", how="left").merge(ql, on="model", how="left")
    sub["iv"] = sub["uses_iv"].map(lambda x: "yes" if bool(x) else "no")
    sub["depth_or_hop"] = sub.apply(
        lambda row: (
            f"L{int(row['depth'])}"
            if pd.notna(row.get("depth"))
            else f"H{int(row['hop'])}" if pd.notna(row.get("hop")) else ""
        ),
        axis=1,
    )
    sub["mcs_mse_in"] = sub["mcs_mse"].map(lambda x: bool(float(x) > 0) if pd.notna(x) else False)
    sub["mcs_qlike_in"] = sub["mcs_qlike"].map(lambda x: bool(float(x) > 0) if pd.notna(x) else False)
    return sub.sort_values(["qlike_ratio_vs_HAR_M", "mse_ratio_vs_HAR_M"]).reset_index(drop=True)


def best_by_group(df: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    clean = df.dropna(subset=[value_col]).copy()
    if clean.empty:
        return clean
    idx = clean.groupby(group_cols, dropna=False)[value_col].idxmin()
    return clean.loc[idx].sort_values(group_cols).reset_index(drop=True)


def selected_dm(dm: pd.DataFrame, universe: str) -> pd.DataFrame:
    sub = dm[dm["universe"] == universe].copy()
    if sub.empty:
        return sub
    order = {"graph_linear": 0, "nonlinear_one_hop": 1, "linear_multihop": 2, "gnn_depth": 3}
    sub["order"] = sub["comparison_type"].map(order).fillna(9)
    return sub.sort_values(["order", "base_model", "candidate_model"]).drop(columns=["order"]).reset_index(drop=True)


def write_report(result_root: Path, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    stats_root = result_root / "zhang_style_statistics"
    summary = read_csv(stats_root / "universe_statistics_summary.csv")
    alignment = read_csv(stats_root / "cross_universe_ticker_alignment_audit.csv")
    loss = read_csv(stats_root / "cross_universe_loss_ratio_summary.csv")
    best = read_csv(stats_root / "cross_universe_best_models.csv")
    mcs = read_csv(stats_root / "cross_universe_mcs_rollup.csv")
    dm = read_csv(stats_root / "cross_universe_pairwise_qlike_dm_summary.csv")
    fvu = read_csv(stats_root / "cross_universe_fvu_by_regime_vs_HAR_M.csv")
    graph = read_csv(stats_root / "cross_universe_graph_rollup.csv")
    coverage = read_csv(stats_root / "zhang_statistics_reference_coverage.csv")

    parts: list[str] = []
    parts.append(
        r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=0.85in]{geometry}
\usepackage{amsmath,amssymb,booktabs,array,graphicx,float,hyperref}
\usepackage{caption}
\usepackage{microtype}
\hypersetup{colorlinks=true,linkcolor=blue!55!black,urlcolor=blue!55!black,citecolor=blue!55!black}
\setlength{\parskip}{0.55em}
\setlength{\parindent}{0pt}
\title{\textbf{Zhang-Style Forecast Evaluation for Dow 30, S\&P 100, and S\&P 500 GNNHAR Runs}}
\author{Generated from local Volatility outputs}
\date{June 18, 2026}
\begin{document}
\maketitle

\begin{abstract}
This report summarizes the post-run evidence generated from the saved Dow 30, S\&P 100,
and S\&P 500 GNNHAR forecasts.  The objective is not to retrain models, but to read the
local prediction arrays and compute the statistics used in Zhang et al.'s GNNHAR paper:
HAR-normalized MSE and QLIKE loss ratios, model confidence sets, calm-versus-turbulent
regime tables, pairwise QLIKE Diebold-Mariano comparisons, FVU nonlinearity diagnostics,
forecast error and forecast ratio boxplot sources, graph summaries, and smoothing diagnostics.
The S\&P 500 analysis is now tied to the actual AutoDL run source rather than the smaller
Google Drive upload panel.
\end{abstract}
"""
    )

    parts.append(
        r"""\section{Notation}
Let \(v_{i,t}\) denote the realized variance or realized volatility target for ticker \(i\) on
date \(t\), and let \(\widehat v_{i,t}^{(m)}\) be the forecast from model \(m\).
The benchmark model is \(HAR_{\mathrm{M}}\), the HAR model estimated under MSE.  The loss ratios are
\[
R^{MSE}_m =
\frac{\frac{1}{NT}\sum_{i,t}(v_{i,t}-\widehat v_{i,t}^{(m)})^2}
{\frac{1}{NT}\sum_{i,t}(v_{i,t}-\widehat v_{i,t}^{(HAR_{\mathrm{M}})})^2},
\quad
R^{QL}_m =
\frac{\frac{1}{NT}\sum_{i,t}QL(v_{i,t},\widehat v_{i,t}^{(m)})}
{\frac{1}{NT}\sum_{i,t}QL(v_{i,t},\widehat v_{i,t}^{(HAR_{\mathrm{M}})})}.
\]
The QLIKE loss is
\[
QL(y,\widehat y)=\frac{y}{\widehat y}-\log\!\left(\frac{y}{\widehat y}\right)-1.
\]
A ratio below one means lower average forecast loss than \(HAR_{\mathrm{M}}\).  \(GHARkH\) denotes a
linear graph-HAR model using \(k\)-hop neighbors; \(GNNHARkL\) denotes a \(k\)-layer graph
neural HAR model.  The suffix \(_IV\) indicates that lagged implied volatility is included.
MCS denotes the Hansen--Lunde--Nason model confidence set at the \(5\%\) level.  DM denotes
the QLIKE Diebold--Mariano comparison; in the generated files, a positive statistic favors the
candidate model listed in the comparison.
"""
    )

    parts.append(r"\section{Data Source Alignment and the 397-versus-449 Issue}")
    parts.append(
        "The earlier discrepancy came from comparing the S\\&P 500 AutoDL forecasts against the "
        "wrong raw panel.  The formal AutoDL run used the local equivalent of "
        r"\texttt{data/scale\_experiment/sp500}, whose RV, IV, and return panels each contain "
        "503 tickers before coverage filtering.  The model run selected 449 tickers.  The smaller "
        "397-column RV/IV panel belongs to the Google Drive upload copy and is not the correct "
        "source for the formal S\\&P 500 run.  After rebinding the analysis script to the AutoDL "
        "metadata source, all 449 model tickers match the S\\&P 500 RV, IV, and return panels."
    )
    parts.append(
        table_float(
            "Run-level sample summary",
            "tab:sample-summary",
            tabular(
                summary,
                [
                    ("universe", "Universe", "str"),
                    ("n_dates", "Test dates", "int"),
                    ("n_tickers", "Tickers", "int"),
                    ("n_models", "Models", "int"),
                    ("date_start", "Test start", "str"),
                    ("date_end", "Test end", "str"),
                    ("raw_data_source_reason", "Raw source rule", "str"),
                    ("local_raw_rv_missing_tickers", "RV missing", "int"),
                ],
            ),
            note=r"The S\&P 500 source rule is now AutoDL run metadata. Dow 30 and S\&P 100 retain the run-configured Drive source because those runs point there and match their model tickers.",
        )
    )
    parts.append(
        table_float(
            "Ticker alignment audit",
            "tab:ticker-alignment",
            tabular(
                alignment,
                [
                    ("universe", "Universe", "str"),
                    ("panel", "Panel", "str"),
                    ("raw_source_reason", "Source rule", "str"),
                    ("n_panel_tickers", "Panel tickers", "int"),
                    ("n_model_tickers", "Model tickers", "int"),
                    ("n_matched_tickers", "Matched", "int"),
                    ("n_missing_from_panel", "Missing", "int"),
                    ("n_extra_in_panel", "Extra", "int"),
                ],
            ),
            note=r"Extra tickers are available in the raw panel but excluded by the modeling coverage filter; they are not missing model inputs.",
        )
    )

    parts.append(r"\section{Implemented Zhang-Style Statistics}")
    parts.append(
        table_float(
            "Coverage of Zhang-style statistics",
            "tab:coverage",
            tabular(
                coverage,
                [
                    ("component", "Component", "str"),
                    ("paper_reference", "Paper reference", "str"),
                    ("status_in_this_script", "Status", "str"),
                ],
            ),
            note="The current saved forecasts are one-day horizon forecasts. Zhang's one-week and one-month tables require separate horizon-specific target construction and reruns. Exact hidden-state MAD requires saved hidden representations; current diagnostics include forecast-level smoothing proxies.",
        )
    )

    parts.append(r"\section{Main Loss-Ratio Evidence}")
    parts.append(
        "Tables below report the complete model set available for each universe in the current "
        "one-day-horizon saved forecasts.  These tables correspond to Zhang's Table 1 and Table 5 "
        "style, with the added IV variants and the S\\&P 500 extension."
    )
    loss_columns = [
        ("model", "Model", "str"),
        ("family", "Family", "str"),
        ("depth_or_hop", "Depth/hop", "str"),
        ("iv", "IV", "str"),
        ("estimation", "Est.", "str"),
        ("mse_ratio_vs_HAR_M", "MSE ratio", "num3"),
        ("qlike_ratio_vs_HAR_M", "QL ratio", "num3"),
        ("mcs_mse_in", "MCS MSE", "bool"),
        ("mcs_qlike_in", "MCS QL", "bool"),
    ]
    for universe in ["dow30", "sp100", "sp500"]:
        tab = make_loss_table(loss, mcs, universe)
        parts.append(
            table_float(
                f"Complete loss ratios for {universe.upper()}",
                f"tab:{universe}-loss",
                tabular(tab, loss_columns),
                note=r"Ratios are relative to \(HAR_{\mathrm{M}}\). MCS columns indicate inclusion in the model confidence set based on per-ticker losses.",
                size="tiny",
            )
        )

    parts.append(
        table_float(
            "Best models by universe and evaluation loss",
            "tab:best-models",
            tabular(
                best,
                [
                    ("universe", "Universe", "str"),
                    ("selection_metric", "Selection metric", "str"),
                    ("best_model", "Best model", "str"),
                    ("loss_ratio_vs_HAR_M", "Ratio", "num3"),
                    ("gain_vs_HAR_M", "Gain", "pct1"),
                    ("family", "Family", "str"),
                    ("uses_iv", "IV", "bool"),
                ],
            ),
            note=r"The best QLIKE model differs by universe: Dow 30 favors a nonlinear IV model, S\&P 100 favors HAR with IV, and S\&P 500 favors GHAR with IV.",
        )
    )

    parts.append(r"\section{MCS and DM Interpretation}")
    parts.append(
        "The MCS results are stricter than simple loss ranking.  Dow 30 has a clear QLIKE winner "
        r"\(GNNHAR2L\_Q\_IV\).  S\&P 100 has a broad IV-trained QLIKE confidence set, indicating "
        "that several IV variants are statistically hard to separate.  S\\&P 500 is much more "
        r"linear in the current run: \texttt{GHAR\_M\_IV} is the QLIKE MCS winner, while most QLIKE-trained "
        "neural specifications have very poor loss ratios."
    )
    for universe in ["dow30", "sp100", "sp500"]:
        sub = selected_dm(dm, universe)
        parts.append(
            table_float(
                f"QLIKE DM comparison summary for {universe.upper()}",
                f"tab:{universe}-dm",
                tabular(
                    sub,
                    [
                        ("comparison_type", "Type", "str"),
                        ("base_model", "Base", "str"),
                        ("candidate_model", "Candidate", "str"),
                        ("candidate_loss_ratio_vs_base", "Cand./base", "num3"),
                        ("candidate_gain_vs_base", "Gain", "pct1"),
                        ("ticker_share_p_lt_0_05", "p<0.05 share", "pct1"),
                        ("ticker_share_positive_dm", "Positive share", "pct1"),
                    ],
                ),
                note="A candidate/base ratio below one means the candidate has lower average QLIKE than the base.  Positive DM share is the share of tickers where the DM statistic favors the candidate.",
                size="tiny",
            )
        )

    parts.append(r"\section{Regime, FVU, and Graph Diagnostics}")
    regime_best_rows = []
    for universe in ["dow30", "sp100", "sp500"]:
        regime_path = result_root / "universes" / universe / "zhang_style_statistics" / "regimes" / "regime_loss_ratios_by_ticker.csv"
        reg = read_csv(regime_path)
        if not reg.empty:
            winners = best_by_group(reg, "ratio_vs_HAR_M", ["regime", "metric"])
            winners.insert(0, "universe", universe)
            regime_best_rows.append(winners)
    regime_best = pd.concat(regime_best_rows, ignore_index=True) if regime_best_rows else pd.DataFrame()
    parts.append(
        table_float(
            "Best regime-specific loss ratios",
            "tab:regime-best",
            tabular(
                regime_best,
                [
                    ("universe", "Universe", "str"),
                    ("regime", "Regime", "str"),
                    ("metric", "Metric", "str"),
                    ("model", "Best model", "str"),
                    ("ratio_vs_HAR_M", "Ratio", "num3"),
                    ("gain_vs_HAR_M", "Gain", "pct1"),
                    ("n_dates", "Dates", "int"),
                ],
            ),
            note="The current regime split uses the cross-sectional mean RV proxy because SPY is not present in these saved truth arrays or raw panels.  Zhang uses SPY RV, so this is method-aligned but not data-identical.",
        )
    )

    fvu_best = best_by_group(fvu, "fvu_mean_vs_HAR_M", ["universe", "regime"])
    parts.append(
        table_float(
            "Lowest FVU relative to HAR_M by universe and regime",
            "tab:fvu-best",
            tabular(
                fvu_best,
                [
                    ("universe", "Universe", "str"),
                    ("regime", "Regime", "str"),
                    ("model", "Model", "str"),
                    ("fvu_mean_vs_HAR_M", "Mean FVU", "num4"),
                    ("fvu_median_vs_HAR_M", "Median FVU", "num4"),
                    ("uses_iv", "IV", "bool"),
                    ("estimation", "Est.", "str"),
                ],
            ),
            note=r"FVU is an effect-size diagnostic for how far a model's forecast function moves away from \(HAR_{\mathrm{M}}\); it is not a significance test.",
        )
    )

    parts.append(
        table_float(
            "Available graph rollup",
            "tab:graph-rollup",
            tabular(
                graph,
                [
                    ("universe", "Universe", "str"),
                    ("loss_group", "Loss group", "str"),
                    ("graph_count", "Graphs", "int"),
                    ("n_nodes_mean", "Nodes", "num1"),
                    ("edges_mean", "Edges", "num1"),
                    ("density_mean", "Density", "num3"),
                    ("degree_mean", "Mean degree", "num2"),
                    ("diameter_mean", "Diameter", "num2"),
                ],
            ),
            note=r"Only the S\&P 500 AutoDL package currently contains graph matrices in the paper-ready tree; Dow 30 and S\&P 100 forecasts remain analyzable, but graph structural summaries require saved graph matrices.",
        )
    )

    parts.append(r"\section{Comparison with Zhang et al.}")
    parts.append(
        r"""Zhang et al. find that GHAR improves on HAR, GNNHAR1L or GNNHAR2L often improves further,
QLIKE training is useful especially under turbulent regimes, and very deep GNNHAR can deteriorate
because of over-smoothing.  Our Dow 30 result is directionally close once IV is included:
\texttt{GNNHAR2L\_Q\_IV} is the best QLIKE model and improves roughly \(11.1\%\) over \(HAR_{\mathrm{M}}\).
The S\&P 100 run is weaker for graph nonlinearity: IV helps, but the best QLIKE model is
\(HAR_Q\_IV\), and several IV graph/neural variants are statistically close by MCS.
The S\&P 500 AutoDL run is different again.  It is internally consistent over 449 selected
tickers, but the best QLIKE model is \texttt{GHAR\_M\_IV}, and QLIKE-trained neural models perform
poorly in the saved run.  This means the large-universe extension does not currently show a
stronger GNN improvement just because the graph has more nodes.  It instead suggests that
large-universe GNN training is more sensitive to optimization, scaling, and possibly over-smoothing.
"""
    )

    parts.append(r"\section{Current Scope Gaps}")
    parts.append(
        r"""The current local outputs support the one-day Zhang-style analysis.  The following pieces
cannot be reconstructed exactly from the saved forecasts alone:
\begin{itemize}
\item one-week and one-month horizon tables, because those require horizon-specific targets and reruns;
\item Zhang's exact SPY regime split, unless a SPY RV series is added and aligned;
\item exact hidden-state MAD from Zhang's Figure 7, because hidden representations were not saved;
\item the smaller-validation robustness table, because that changes the rolling train-validation split.
\end{itemize}
The generated CSV layer records these limitations explicitly in
\texttt{outputs/paper\_ready\_20260617/zhang\_style\_statistics/zhang\_statistics\_reference\_coverage.csv}.
"""
    )

    parts.append(r"\section{Generated Files}")
    parts.append(
        r"""The complete machine-readable results are stored under
\texttt{outputs/paper\_ready\_20260617/zhang\_style\_statistics/} and under each universe's
\texttt{zhang\_style\_statistics/} folder.  The most important cross-universe files are:
\begin{itemize}
\item \texttt{cross\_universe\_loss\_ratio\_summary.csv}
\item \texttt{cross\_universe\_mcs\_rollup.csv}
\item \texttt{cross\_universe\_pairwise\_qlike\_dm\_summary.csv}
\item \texttt{cross\_universe\_fvu\_by\_regime\_vs\_HAR\_M.csv}
\item \texttt{cross\_universe\_ticker\_alignment\_audit.csv}
\item \texttt{cross\_universe\_graph\_rollup.csv}
\end{itemize}
"""
    )
    parts.append(r"\end{document}")

    tex_path = report_dir / "zhang_style_statistics_report.tex"
    tex_path.write_text("\n\n".join(parts), encoding="utf-8")
    return tex_path


def main() -> None:
    args = parse_args()
    tex_path = write_report(args.result_root, args.report_dir)
    print(tex_path)


if __name__ == "__main__":
    main()
