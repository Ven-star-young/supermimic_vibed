"""Compatibility alias for :mod:`supermimic.rl.models`.

The alias preserves assignments to ``networks.AMP_DTYPE`` in existing scripts.
"""

import sys

from .rl import models as _models

sys.modules[__name__] = _models
