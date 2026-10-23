"""Shared asset locations. Third-party robot data stays outside the Python package."""
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
