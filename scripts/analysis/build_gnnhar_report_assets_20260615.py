from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "gnnhar_results_20260615"
TABLE_DIR = REPORT_DIR / "tables"


DOW30_LOSS = "outputs/colab-runs/20260610T151242Z/tables/loss_ratios.csv"
DOW30_DM = "outputs/colab-runs/20260610T151242Z/tables/dm_tests.csv"
DOW30_MCS = "outputs/colab-runs/20260610T151242Z/tables/mcs_results.csv"
DOW30_META = "outputs/colab-runs/20260610T151242Z/run_metadata.json"

SP100_SCALE_LOSS = "outputs/colab-scale-runs/scale-20260610T201904Z/sp100/tables/loss_ratios.csv"
SP100_SCALE_DM = "outputs/colab-scale-runs/scale-20260610T201904Z/sp100/tables/dm_tests.csv"
SP100_SCALE_MCS = "outputs/colab-scale-runs/scale-20260610T201904Z/sp100/tables/mcs_results.csv"
SP100_SCALE_META = "outputs/colab-scale-runs/scale-20260610T201904Z/sp100/run_metadata.json"

SP500_LOSS = "outputs/colab-scale-runs/scale-20260610T201904Z/sp500/tables/loss_ratios.csv"
SP500_DM = "outputs/colab-scale-runs/scale-20260610T201904Z/sp500/tables/dm_tests.csv"
SP500_MCS = "outputs/colab-scale-runs/scale-20260610T201904Z/sp500/tables/mcs_results.csv"
SP500_META = "outputs/colab-scale-runs/scale-20260610T201904Z/sp500/run_metadata.json"


def mkdirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path)


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def tex(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    replacements = {
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
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def fmt(value, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    value = float(value)
    if 0 < abs(value) < 0.00005:
        return r"$<10^{-4}$"
    return f"{value:.{digits}f}"


def fmt_pct(value, digits: int = 2) -> str:
    if pd.isna(value):
        return ""
    return f"{100 * float(value):.{digits}f}\\%"


def write_table(
    df: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
    align: str | None = None,
    numeric_cols: Iterable[str] = (),
    pct_cols: Iterable[str] = (),
    latex_cols: Iterable[str] = (),
    size: str = r"\small",
    fit_width: bool = False,
) -> None:
    numeric_cols = set(numeric_cols)
    pct_cols = set(pct_cols)
    latex_cols = set(latex_cols)
    align = align or ("l" * len(df.columns))
    lines = [r"\begin{table}[H]", r"\centering", size]
    if fit_width:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.extend(
        [
            r"\begin{tabular}{" + align + r"}",
            r"\toprule",
            " & ".join(str(c) for c in df.columns) + r" \\",
            r"\midrule",
        ]
    )
    for _, row in df.iterrows():
        vals: list[str] = []
        for col in df.columns:
            if col in pct_cols:
                vals.append(fmt_pct(row[col]))
            elif col in numeric_cols:
                vals.append(fmt(row[col]))
            elif col in latex_cols:
                vals.append(str(row[col]))
            else:
                vals.append(tex(row[col]))
        lines.append(" & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if fit_width:
        lines.append(r"}")
    lines += [rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def current_sp100_a100_full() -> pd.DataFrame:
    rows = [
        ("HAR_Q_IV", 0.980672, 0.932787),
        ("GHAR_Q_IV", 0.981155, 0.933495),
        ("GNNHAR2L_Q_IV", 0.982333, 0.933495),
        ("GNNHAR5L_Q_IV", 0.980863, 0.934228),
        ("GNNHAR1L_Q_IV", 0.981388, 0.934858),
        ("GNNHAR4L_Q_IV", 0.985826, 0.936106),
        ("GNNHAR3L_Q_IV", 0.981460, 0.940106),
        ("GHAR_Q", 1.002762, 0.976368),
        ("GNNHAR5L_Q", 1.003253, 0.976786),
        ("GNNHAR3L_Q", 1.003032, 0.977386),
        ("GHAR_M_IV", 0.964891, 0.977778),
        ("GNNHAR4L_Q", 1.002622, 0.978495),
        ("HAR_M_IV", 0.964803, 0.979188),
        ("GNNHAR2L_Q", 1.003510, 0.979452),
        ("HAR_Q", 1.005618, 0.980136),
        ("GNNHAR1L_Q", 1.003929, 0.980192),
        ("GNNHAR1L_M_IV", 0.971581, 0.986787),
        ("GNNHAR3L_M_IV", 0.967609, 0.990027),
        ("GHAR_M", 0.997465, 0.992531),
        ("GNNHAR5L_M_IV", 0.979556, 0.993644),
        ("GNNHAR4L_M_IV", 0.978923, 0.995530),
        ("GNNHAR1L_M", 0.999951, 0.996787),
        ("GNNHAR2L_M", 1.000086, 0.999458),
        ("GNNHAR3L_M", 1.001130, 0.999559),
        ("HAR_M", 1.000000, 1.000000),
        ("GNNHAR4L_M", 1.004042, 1.003234),
        ("GNNHAR5L_M", 1.003038, 1.010496),
        ("GNNHAR2L_M_IV", 0.988759, 1.012166),
    ]
    return pd.DataFrame(rows, columns=["model", "mse_ratio", "qlike_ratio"])


def sp100_label(name: str) -> str:
    iv = name.endswith("_IV")
    base = name[:-3] if iv else name
    parts = base.split("_")
    criterion = parts[-1]
    stem = "_".join(parts[:-1])
    superscript = r"^{IV}" if iv else ""
    return rf"\(\mathrm{{{stem}}}_{{{criterion}}}{superscript}\)"


def legacy_label(name: str) -> str:
    label = name
    suffix = ""
    if "+IV" in label or "-IV" in label:
        suffix = r"^{IV}"
    label = (
        label.replace("+IV", "")
        .replace("-IV", "")
        .replace("+fakeIV", "")
        .replace("-QLIKE", "_Q")
        .replace("-random", "")
        .replace("_RANDOM", "")
    )
    if "_" in label:
        stem, criterion = label.split("_", 1)
        return rf"\(\mathrm{{{stem}}}_{{{criterion}}}{suffix}\)"
    return rf"\(\mathrm{{{label}}}{suffix}\)"


def sp100_loss_table(rows: list[str]) -> pd.DataFrame:
    df = current_sp100_a100_full().set_index("model").loc[rows].reset_index()
    return pd.DataFrame(
        {
            "Model": df["model"].map(sp100_label),
            "EC": df["model"].str.extract(r"_(M|Q)(?:_IV)?$", expand=False).map(lambda x: rf"\({x}\)"),
            r"\(\rho_M\)": df["mse_ratio"],
            r"\(\rho_Q\)": df["qlike_ratio"],
            r"\(1-\rho_Q\)": 1.0 - df["qlike_ratio"],
        }
    )


def sp100_block(name: str) -> str:
    if "4L" in name or "5L" in name:
        return "depth extension"
    if name.endswith("_IV"):
        return "IV extension"
    return "strict core"


def sp100_full_loss_table() -> pd.DataFrame:
    df = current_sp100_a100_full().copy()
    df["block"] = df["model"].map(sp100_block)
    block_order = {"strict core": 0, "IV extension": 1, "depth extension": 2}
    df["block_order"] = df["block"].map(block_order)
    df["criterion"] = df["model"].str.extract(r"_(M|Q)(?:_IV)?$", expand=False)
    df = df.sort_values(["block_order", "criterion", "model"]).reset_index(drop=True)
    return pd.DataFrame(
        {
            "Model": df["model"].map(sp100_label),
            "Block": df["block"],
            "EC": df["criterion"].map(lambda x: rf"\({x}\)"),
            r"\(\rho_M\)": df["mse_ratio"],
            r"\(\rho_Q\)": df["qlike_ratio"],
            r"\(1-\rho_Q\)": 1.0 - df["qlike_ratio"],
        }
    )


def legacy_loss_table(path: str, rows: list[str]) -> pd.DataFrame:
    df = read_csv(path).set_index("model").loc[rows].reset_index()
    return pd.DataFrame(
        {
            "Model": df["model"].map(legacy_label),
            r"\(\rho_M\)": df["mse_ratio_vs_har"],
            r"\(\rho_Q\)": df["qlike_ratio_vs_har"],
            r"\(1-\rho_Q\)": 1.0 - df["qlike_ratio_vs_har"],
        }
    )


def legacy_block(name: str) -> str:
    lowered = name.lower()
    if "fakeiv" in lowered:
        return "diagnostic placebo"
    if "random" in lowered:
        return "diagnostic random graph"
    if "+iv" in lowered or "-iv" in lowered:
        return "IV extension"
    if "qlike" in lowered:
        return "QLIKE-trained extension"
    return "strict core"


def legacy_full_loss_table(path: str) -> pd.DataFrame:
    df = read_csv(path).copy()
    df["Block"] = df["model"].map(legacy_block)
    block_order = {
        "strict core": 0,
        "QLIKE-trained extension": 1,
        "IV extension": 2,
        "diagnostic placebo": 3,
        "diagnostic random graph": 4,
    }
    df["block_order"] = df["Block"].map(block_order)
    df = df.sort_values(["block_order", "qlike_ratio_vs_har", "model"]).reset_index(drop=True)
    return pd.DataFrame(
        {
            "Model": df["model"],
            "Block": df["Block"],
            "Est.": df["estimation"],
            r"\(\rho_M\)": df["mse_ratio_vs_har"],
            r"\(\rho_Q\)": df["qlike_ratio_vs_har"],
            r"\(1-\rho_Q\)": 1.0 - df["qlike_ratio_vs_har"],
            r"\(\rho_Q^{IV}\)": df["qlike_ratio_vs_har_iv"],
        }
    )


def dm_row(universe: str, path: str, comparison: str, question: str) -> dict:
    row = read_csv(path).query("comparison == @comparison").iloc[0]
    ratio = float(row["mean_loss_b"] / row["mean_loss_a"])
    gain = 1.0 - ratio
    pvalue = float(row["pvalue"])
    dm = float(row["dm_stat_positive_favors_b"])
    if gain > 0 and pvalue < 0.05:
        conclusion = "improves"
    elif gain < 0 and pvalue < 0.05:
        conclusion = "deteriorates"
    else:
        conclusion = "not significant"
    return {
        "Universe": universe,
        "Question": question,
        "Comparison": comparison,
        r"\(L_B/L_A\)": ratio,
        r"\(1-L_B/L_A\)": gain,
        "DM": dm,
        r"\(p\)": pvalue,
        "Result": conclusion,
    }


def build_matched_dm_summary() -> pd.DataFrame:
    rows = [
        dm_row("Dow30", DOW30_DM, "HAR vs GHAR", "graph linearity"),
        dm_row("Dow30", DOW30_DM, "GHAR vs GNNHAR1L", "nonlinearity"),
        dm_row("Dow30", DOW30_DM, "GHAR+IV vs GNNHAR1L-IV", "nonlinearity with IV"),
        dm_row("SP100 scale", SP100_SCALE_DM, "HAR vs GHAR", "graph linearity"),
        dm_row("SP100 scale", SP100_SCALE_DM, "GHAR vs GNNHAR1L", "nonlinearity"),
        dm_row("SP100 scale", SP100_SCALE_DM, "GHAR+IV vs GNNHAR1L-IV", "nonlinearity with IV"),
        dm_row("SP500 scale", SP500_DM, "HAR vs GHAR", "graph linearity"),
        dm_row("SP500 scale", SP500_DM, "GHAR vs GNNHAR1L", "nonlinearity"),
        dm_row("SP500 scale", SP500_DM, "GHAR+IV vs GNNHAR1L-IV", "nonlinearity with IV"),
    ]
    return pd.DataFrame(rows)


def build_current_sp100_depth_dm() -> pd.DataFrame:
    rows = [
        ("GNNHAR1L_M", "GNNHAR2L_M", "No", "M", 1.002679, -0.002679, -1.206600, 0.227586, "not significant"),
        ("GNNHAR1L_Q", "GNNHAR2L_Q", "No", "Q", 0.999244, 0.000756, 0.636820, 0.524242, "not significant"),
        ("GNNHAR1L_M_IV", "GNNHAR2L_M_IV", "Yes", "M", 1.025720, -0.025720, -2.632840, 0.008468, "deteriorates"),
        ("GNNHAR2L_M_IV", "GNNHAR3L_M_IV", "Yes", "M", 0.978127, 0.021873, 2.540777, 0.011061, "improves"),
        ("GNNHAR2L_Q_IV", "GNNHAR3L_Q_IV", "Yes", "Q", 1.007080, -0.007080, -2.086680, 0.036917, "deteriorates"),
    ]
    df = pd.DataFrame(
        rows,
        columns=["Base", "Candidate", "IV", "EC", r"\(L_B/L_A\)", r"\(1-L_B/L_A\)", "DM", r"\(p\)", "Result"],
    )
    df["Base"] = df["Base"].map(sp100_label)
    df["Candidate"] = df["Candidate"].map(sp100_label)
    df["EC"] = df["EC"].map(lambda x: rf"\({x}\)")
    return df


def build_current_sp100_robustness_dm() -> pd.DataFrame:
    rows = [
        ("linear GHAR 1-hop", "linear GHAR 1+2-hop", "No", "M", 1.005537, -0.005537, -4.064740, 0.000048, "deteriorates"),
        ("linear GHAR 1-hop", "linear GHAR 1+2+3-hop", "No", "M", 1.008175, -0.008175, -4.076247, 0.000046, "deteriorates"),
        ("GNNHAR1L_Q", "GNNHAR5L_Q", "No", "Q", 0.996524, 0.003476, 2.223569, 0.026177, "improves"),
        ("GNNHAR4L_Q", "GNNHAR5L_Q", "No", "Q", 0.998253, 0.001747, 2.034750, 0.041876, "improves"),
    ]
    out = []
    for base, cand, iv, ec, ratio, gain, dm, pvalue, result in rows:
        out.append(
            {
                "Base": sp100_label(base) if base.startswith("GNNHAR") else base,
                "Candidate": sp100_label(cand) if cand.startswith("GNNHAR") else cand,
                "IV": iv,
                "EC": rf"\({ec}\)",
                r"\(L_B/L_A\)": ratio,
                r"\(1-L_B/L_A\)": gain,
                "DM": dm,
                r"\(p\)": pvalue,
                "Result": result,
            }
        )
    return pd.DataFrame(out)


def build_metadata_summary() -> pd.DataFrame:
    rows = []
    for universe, source, path in [
        ("Dow30", "completed earlier full stack", DOW30_META),
        ("SP100 scale", "short-budget scale artifact", SP100_SCALE_META),
        ("SP500 scale", "short-budget scale artifact", SP500_META),
    ]:
        meta = load_json(path)
        panel = meta.get("panel", {})
        split = meta.get("split_sizes", {})
        args = meta.get("args", {})
        rows.append(
            {
                "Universe": universe,
                r"\(N\)": meta.get("n_tickers") or len(meta.get("tickers", [])) or panel.get("selected_tickers"),
                "Dates": f"{meta.get('date_start') or panel.get('date_start')}--{meta.get('date_end') or panel.get('date_end')}",
                "Test days": split.get("test"),
                "Training budget": f"epochs={args.get('epochs')}, hidden={args.get('hidden')}",
                "Status": source,
            }
        )
    rows.append(
        {
            "Universe": "SP100 A100",
            r"\(N\)": 91,
            "Dates": "2021-06-09--2026-06-09",
            "Test days": 234,
            "Training budget": "epochs=5000, hidden=9",
            "Status": "current Colab long run",
        }
    )
    return pd.DataFrame(rows)


def mcs_summary(path: str) -> str:
    df = read_csv(path)
    df = df[~df["model"].str.contains("fakeIV|random|RANDOM", regex=True)]
    included = df[df["included_at_5pct"].astype(bool)]["model"].tolist()
    return ", ".join(included[:6]) + (" ..." if len(included) > 6 else "")


def build_mcs_set_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Universe": "Dow30", "Models retained in 5\\% MCS": mcs_summary(DOW30_MCS)},
            {"Universe": "SP100 scale", "Models retained in 5\\% MCS": mcs_summary(SP100_SCALE_MCS)},
            {"Universe": "SP500 scale", "Models retained in 5\\% MCS": mcs_summary(SP500_MCS)},
        ]
    )


def write_all() -> None:
    mkdirs()

    metadata = build_metadata_summary()
    metadata.to_csv(TABLE_DIR / "run_metadata_summary.csv", index=False)
    write_table(
        metadata,
        TABLE_DIR / "run_metadata_summary.tex",
        "Available runs.  The SP100 A100 run is the latest long-run notebook result; SP500 is not yet a matched A100 long run.",
        "tab:run-metadata",
        align="lrlrll",
        fit_width=True,
    )

    dow30_full = legacy_full_loss_table(DOW30_LOSS)
    dow30_full.to_csv(TABLE_DIR / "dow30_full_loss_summary.csv", index=False)
    write_table(
        dow30_full,
        TABLE_DIR / "dow30_full_loss_summary.tex",
        r"Dow30 full available model comparison.  Diagnostic placebo and random-graph rows are included for completeness but are not part of the strict Zhang-core comparison.",
        "tab:dow30-full-loss",
        align="lllrrrr",
        numeric_cols=[r"\(\rho_M\)", r"\(\rho_Q\)", r"\(\rho_Q^{IV}\)"],
        pct_cols=[r"\(1-\rho_Q\)"],
        size=r"\scriptsize",
        fit_width=True,
    )

    dow30_core = legacy_loss_table(DOW30_LOSS, ["HAR", "GHAR", "GNNHAR1L", "GNNHAR2L", "GNNHAR3L", "GNNHAR1L-QLIKE"])
    dow30_core.to_csv(TABLE_DIR / "dow30_core_loss_summary.csv", index=False)
    write_table(
        dow30_core,
        TABLE_DIR / "dow30_core_loss_summary.tex",
        r"Dow30 strict-core loss ratios relative to HAR.  The QLIKE-trained row is only available for the legacy one-layer GNN artifact.",
        "tab:dow30-core-loss",
        align="lrrr",
        numeric_cols=[r"\(\rho_M\)", r"\(\rho_Q\)"],
        pct_cols=[r"\(1-\rho_Q\)"],
        latex_cols=["Model"],
    )

    dow30_iv = legacy_loss_table(DOW30_LOSS, ["HAR+IV", "GHAR+IV", "GNNHAR1L-IV", "GNNHAR2L-IV", "GNNHAR3L-IV", "GNNHAR1L-IV-QLIKE"])
    dow30_iv.to_csv(TABLE_DIR / "dow30_iv_loss_summary.csv", index=False)
    write_table(
        dow30_iv,
        TABLE_DIR / "dow30_iv_loss_summary.tex",
        r"Dow30 IV-augmented extension loss ratios relative to HAR.",
        "tab:dow30-iv-loss",
        align="lrrr",
        numeric_cols=[r"\(\rho_M\)", r"\(\rho_Q\)"],
        pct_cols=[r"\(1-\rho_Q\)"],
        latex_cols=["Model"],
    )

    sp100_core_rows = [
        "HAR_M",
        "GHAR_M",
        "GNNHAR1L_M",
        "GNNHAR2L_M",
        "GNNHAR3L_M",
        "HAR_Q",
        "GHAR_Q",
        "GNNHAR1L_Q",
        "GNNHAR2L_Q",
        "GNNHAR3L_Q",
    ]
    sp100_core = sp100_loss_table(sp100_core_rows)
    sp100_core.to_csv(TABLE_DIR / "sp100_a100_core_loss_summary.csv", index=False)
    write_table(
        sp100_core,
        TABLE_DIR / "sp100_a100_core_loss_summary.tex",
        r"Current SP100 A100 strict Zhang-core loss ratios relative to \(HAR_M\).",
        "tab:sp100-a100-core-loss",
        align="llrrr",
        numeric_cols=[r"\(\rho_M\)", r"\(\rho_Q\)"],
        pct_cols=[r"\(1-\rho_Q\)"],
        latex_cols=["Model", "EC"],
    )

    sp100_full = sp100_full_loss_table()
    sp100_full.to_csv(TABLE_DIR / "sp100_a100_full_loss_summary.csv", index=False)
    write_table(
        sp100_full,
        TABLE_DIR / "sp100_a100_full_loss_summary.tex",
        r"Current SP100 A100 full available model comparison.  Ratios are relative to \(HAR_M\).",
        "tab:sp100-a100-full-loss",
        align="lllrrr",
        numeric_cols=[r"\(\rho_M\)", r"\(\rho_Q\)"],
        pct_cols=[r"\(1-\rho_Q\)"],
        latex_cols=["Model", "EC"],
        size=r"\scriptsize",
        fit_width=True,
    )

    sp100_iv_rows = [
        "HAR_M_IV",
        "GHAR_M_IV",
        "GNNHAR1L_M_IV",
        "GNNHAR2L_M_IV",
        "GNNHAR3L_M_IV",
        "HAR_Q_IV",
        "GHAR_Q_IV",
        "GNNHAR1L_Q_IV",
        "GNNHAR2L_Q_IV",
        "GNNHAR3L_Q_IV",
    ]
    sp100_iv = sp100_loss_table(sp100_iv_rows)
    sp100_iv.to_csv(TABLE_DIR / "sp100_a100_iv_loss_summary.csv", index=False)
    write_table(
        sp100_iv,
        TABLE_DIR / "sp100_a100_iv_loss_summary.tex",
        r"Current SP100 A100 IV-augmented extension loss ratios relative to \(HAR_M\).",
        "tab:sp100-a100-iv-loss",
        align="llrrr",
        numeric_cols=[r"\(\rho_M\)", r"\(\rho_Q\)"],
        pct_cols=[r"\(1-\rho_Q\)"],
        latex_cols=["Model", "EC"],
    )

    sp100_depth = sp100_loss_table(["GNNHAR4L_M", "GNNHAR5L_M", "GNNHAR4L_Q", "GNNHAR5L_Q", "GNNHAR4L_M_IV", "GNNHAR5L_M_IV", "GNNHAR4L_Q_IV", "GNNHAR5L_Q_IV"])
    sp100_depth.to_csv(TABLE_DIR / "sp100_a100_depth_extension_loss_summary.csv", index=False)
    write_table(
        sp100_depth,
        TABLE_DIR / "sp100_a100_depth_extension_loss_summary.tex",
        r"Current SP100 A100 four- and five-layer depth extensions.  These are outside Zhang's main DJIA core but follow his larger-universe extension logic.",
        "tab:sp100-a100-depth-loss",
        align="llrrr",
        numeric_cols=[r"\(\rho_M\)", r"\(\rho_Q\)"],
        pct_cols=[r"\(1-\rho_Q\)"],
        latex_cols=["Model", "EC"],
    )

    sp500_core = legacy_loss_table(SP500_LOSS, ["HAR", "GHAR", "GNNHAR1L", "GNNHAR2L", "GNNHAR3L"])
    sp500_core.to_csv(TABLE_DIR / "sp500_core_loss_summary.csv", index=False)
    write_table(
        sp500_core,
        TABLE_DIR / "sp500_core_loss_summary.tex",
        r"SP500 scale-run strict-core loss ratios relative to HAR.  This is a short-budget scale artifact, not the latest A100 protocol.",
        "tab:sp500-core-loss",
        align="lrrr",
        numeric_cols=[r"\(\rho_M\)", r"\(\rho_Q\)"],
        pct_cols=[r"\(1-\rho_Q\)"],
        latex_cols=["Model"],
    )

    sp500_full = legacy_full_loss_table(SP500_LOSS)
    sp500_full.to_csv(TABLE_DIR / "sp500_full_loss_summary.csv", index=False)
    write_table(
        sp500_full,
        TABLE_DIR / "sp500_full_loss_summary.tex",
        r"SP500 scale-run full available model comparison.  Diagnostic placebo and random-graph rows are included for completeness.",
        "tab:sp500-full-loss",
        align="lllrrrr",
        numeric_cols=[r"\(\rho_M\)", r"\(\rho_Q\)", r"\(\rho_Q^{IV}\)"],
        pct_cols=[r"\(1-\rho_Q\)"],
        size=r"\scriptsize",
        fit_width=True,
    )

    sp500_iv = legacy_loss_table(SP500_LOSS, ["HAR+IV", "GHAR+IV", "GNNHAR1L-IV", "GNNHAR2L-IV", "GNNHAR3L-IV"])
    sp500_iv.to_csv(TABLE_DIR / "sp500_iv_loss_summary.csv", index=False)
    write_table(
        sp500_iv,
        TABLE_DIR / "sp500_iv_loss_summary.tex",
        r"SP500 scale-run IV extension loss ratios relative to HAR.",
        "tab:sp500-iv-loss",
        align="lrrr",
        numeric_cols=[r"\(\rho_M\)", r"\(\rho_Q\)"],
        pct_cols=[r"\(1-\rho_Q\)"],
        latex_cols=["Model"],
    )

    matched_dm = build_matched_dm_summary()
    matched_dm.to_csv(TABLE_DIR / "matched_dm_summary.csv", index=False)
    write_table(
        matched_dm,
        TABLE_DIR / "matched_dm_summary.tex",
        r"Matched QLIKE DM tests in available completed artifacts.  \(B\) is the second model in the comparison; positive \(1-L_B/L_A\) favors \(B\).",
        "tab:matched-dm",
        align="lllrrrrl",
        numeric_cols=[r"\(L_B/L_A\)", "DM", r"\(p\)"],
        pct_cols=[r"\(1-L_B/L_A\)"],
        fit_width=True,
        size=r"\scriptsize",
    )

    depth_dm = build_current_sp100_depth_dm()
    depth_dm.to_csv(TABLE_DIR / "sp100_a100_depth_dm_core.csv", index=False)
    write_table(
        depth_dm,
        TABLE_DIR / "sp100_a100_depth_dm_core.tex",
        r"Current SP100 A100 GNN depth DM tests.  Positive \(1-L_B/L_A\) favors the candidate model.",
        "tab:sp100-a100-depth-dm",
        align="llllrrrrl",
        numeric_cols=[r"\(L_B/L_A\)", "DM", r"\(p\)"],
        pct_cols=[r"\(1-L_B/L_A\)"],
        latex_cols=["Base", "Candidate", "EC"],
        fit_width=True,
        size=r"\scriptsize",
    )

    robustness_dm = build_current_sp100_robustness_dm()
    robustness_dm.to_csv(TABLE_DIR / "sp100_a100_robustness_dm.csv", index=False)
    write_table(
        robustness_dm,
        TABLE_DIR / "sp100_a100_robustness_dm.tex",
        r"SP100 A100 robustness DM tests for linear multi-hop GHAR and four-/five-layer GNN extensions.",
        "tab:sp100-a100-robustness-dm",
        align="llllrrrrl",
        numeric_cols=[r"\(L_B/L_A\)", "DM", r"\(p\)"],
        pct_cols=[r"\(1-L_B/L_A\)"],
        latex_cols=["Base", "Candidate", "EC"],
        fit_width=True,
        size=r"\scriptsize",
    )

    mcs = build_mcs_set_summary()
    mcs.to_csv(TABLE_DIR / "mcs_set_summary.csv", index=False)
    write_table(
        mcs,
        TABLE_DIR / "mcs_set_summary.tex",
        r"Model Confidence Set summaries after excluding diagnostic random-graph and fake-IV rows.",
        "tab:mcs-summary",
        align="ll",
    )

    read_csv(DOW30_DM).assign(universe="Dow30").to_csv(TABLE_DIR / "dow30_dm_tests.csv", index=False)
    read_csv(SP100_SCALE_DM).assign(universe="SP100 scale").to_csv(TABLE_DIR / "sp100_scale_dm_tests.csv", index=False)
    read_csv(SP500_DM).assign(universe="SP500 scale").to_csv(TABLE_DIR / "sp500_dm_tests.csv", index=False)
    current_sp100_a100_full().to_csv(TABLE_DIR / "current_sp100_a100_loss_ratios.csv", index=False)

    print(f"Wrote report assets under {TABLE_DIR}")


if __name__ == "__main__":
    write_all()
