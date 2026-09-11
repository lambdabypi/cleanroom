"""Streamlit dashboard. Entry point is `app.py`, launched by `cleanroom ui`."""

from pathlib import Path

APP_PATH = Path(__file__).resolve().parent / "app.py"

__all__ = ["APP_PATH"]
