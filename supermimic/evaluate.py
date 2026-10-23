"""Backward-compatible evaluation CLI; see ``supermimic.rl.evaluation``."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supermimic.rl.evaluation import evaluate, main, run

__all__ = ["evaluate", "main", "run"]

if __name__ == "__main__":
    main()
