"""Compatibility entry point for result summaries.

The implementation now lives in ``scripts/summarize_results.py``.
"""

from runpy import run_path
from pathlib import Path


if __name__ == "__main__":
    run_path(str(Path(__file__).resolve().parent / "scripts" / "summarize_results.py"), run_name="__main__")

