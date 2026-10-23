"""Backward-compatible training CLI; implementation lives in ``supermimic.rl.runner``."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supermimic.rl.runner import main, run

__all__ = ["main", "run"]

if __name__ == "__main__":
    main()
