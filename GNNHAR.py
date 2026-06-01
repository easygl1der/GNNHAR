"""Compatibility entry point for the PyTorch GNNHAR models.

The implementation now lives in ``src/gnnhar/models.py``.
"""

from runpy import run_module


if __name__ == "__main__":
    run_module("src.gnnhar.models", run_name="__main__")

