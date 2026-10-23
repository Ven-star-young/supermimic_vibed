"""Compatibility import; implementation lives in ``supermimic.assets.g1.g1_constants``."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module("supermimic.assets.g1.g1_constants")
