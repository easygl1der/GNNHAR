"""Compatibility entry point for the linear HAR/GHAR models.

The implementation now lives in ``src/gnnhar/linear.py``.
"""

from runpy import run_module


if __name__ == "__main__":
    run_module("src.gnnhar.linear", run_name="__main__")

