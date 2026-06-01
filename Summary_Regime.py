"""Compatibility entry point for regime summaries.

The implementation now lives in ``scripts/summarize_regimes.py``.
"""

from runpy import run_path
from pathlib import Path


if __name__ == "__main__":
    run_path(str(Path(__file__).resolve().parent / "scripts" / "summarize_regimes.py"), run_name="__main__")

