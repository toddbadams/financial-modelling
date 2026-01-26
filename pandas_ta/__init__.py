"""Provide a lightweight bridge so ``import pandas_ta`` keeps working on Python 3.11."""

import pandas_ta_classic as _pandas_ta_classic
from pandas_ta_classic import *  # noqa: F401,F403

__version__ = getattr(_pandas_ta_classic, "__version__", "0.0.0")
__all__ = getattr(_pandas_ta_classic, "__all__", [])
