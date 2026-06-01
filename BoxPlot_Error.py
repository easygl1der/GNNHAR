"""Compatibility entry point for forecast error plots.

The implementation now lives in ``scripts/plot_error_boxplots.py``.
"""

from runpy import run_path
from pathlib import Path


if __name__ == "__main__":
    run_path(str(Path(__file__).resolve().parent / "scripts" / "plot_error_boxplots.py"), run_name="__main__")
