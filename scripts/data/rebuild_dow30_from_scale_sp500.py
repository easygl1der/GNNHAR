from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "data" / "scale_experiment" / "sp500"
TARGET_ROOT = (
    REPO_ROOT / "data" / "google_drive_upload" / "GNNHAR_Research" / "data" / "dow30"
)
AUDIT_ROOT = REPO_ROOT / "data" / "google_drive_upload" / "GNNHAR_Research" / "data_audit"

FILES = {
    "rv": "merged_rv_data_filled.csv",
    "iv": "merged_iv_data_filled.csv",
    "ret": "daily_returns.csv",
}


def read_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df.columns = df.columns.astype(str)
    return df.sort_index(axis=1).apply(pd.to_numeric, errors="coerce")


def clean_panel(df: pd.DataFrame, *, positive: bool) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan).sort_index()
    if positive:
        df = df.where(df > 0)
    # Zhang's public code uses forward fill. Backfill is only used here for
    # leading gaps that cannot be filled by ffill within the target interval.
    return df.ffill().bfill()


def audit_panel(df: pd.DataFrame, path: Path) -> dict:
    arr = df.to_numpy(dtype=float)
    return {
        "path": str(path),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
        "nan_cells": int(df.isna().sum().sum()),
        "nonfinite_cells": int((~np.isfinite(arr)).sum()),
        "columns": list(df.columns),
    }


def backup_existing_target() -> Path | None:
    if not TARGET_ROOT.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = TARGET_ROOT.parent / f"dow30_backup_before_sp500_rebuild_{stamp}"
    shutil.copytree(TARGET_ROOT, backup_dir)
    return backup_dir


def main() -> None:
    source = {kind: read_panel(SOURCE_ROOT / filename) for kind, filename in FILES.items()}

    current_dow = read_panel(TARGET_ROOT / FILES["rv"])
    dow_tickers = list(current_dow.columns)
    missing = {
        kind: sorted(set(dow_tickers) - set(df.columns)) for kind, df in source.items()
    }
    if any(missing.values()):
        raise ValueError(f"Missing dow30 tickers in source panels: {missing}")

    common_index = source["rv"].index.intersection(source["iv"].index).intersection(
        source["ret"].index
    )
    if len(common_index) == 0:
        raise ValueError("No common dates across source RV/IV/returns panels.")

    rebuilt = {
        "rv": clean_panel(source["rv"].loc[common_index, dow_tickers], positive=True),
        "iv": clean_panel(source["iv"].loc[common_index, dow_tickers], positive=True),
        "ret": clean_panel(source["ret"].loc[common_index, dow_tickers], positive=False),
    }

    bad = {
        kind: {
            "nan_cells": int(df.isna().sum().sum()),
            "nonfinite_cells": int((~np.isfinite(df.to_numpy(dtype=float))).sum()),
        }
        for kind, df in rebuilt.items()
    }
    if any(v["nan_cells"] or v["nonfinite_cells"] for v in bad.values()):
        raise ValueError(f"Rebuilt panels still contain missing/nonfinite values: {bad}")

    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    backup_dir = backup_existing_target()

    output_paths = {}
    for kind, filename in FILES.items():
        path = TARGET_ROOT / filename
        rebuilt[kind].to_csv(path)
        output_paths[kind] = path

    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(SOURCE_ROOT),
        "target_root": str(TARGET_ROOT),
        "backup_dir": str(backup_dir) if backup_dir else None,
        "method": (
            "Rebuilt dow30 by selecting current dow30 tickers from the complete "
            "scale_experiment sp500 panels; cleaned with positive filter for RV/IV, "
            "then ffill and bfill."
        ),
        "tickers": dow_tickers,
        "panels": {
            kind: audit_panel(rebuilt[kind], output_paths[kind]) for kind in FILES
        },
        "source_files": {kind: str(SOURCE_ROOT / filename) for kind, filename in FILES.items()},
    }
    audit_path = AUDIT_ROOT / "dow30_rebuild_from_scale_sp500_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))

    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
