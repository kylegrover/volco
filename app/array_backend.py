"""Array backend selector.

Uses CuPy when a CUDA device is detected, falling back to NumPy otherwise.
The selected module is exposed as ``np`` so callers can do::

    from app.array_backend import np

    arr = np.zeros((10, 10), dtype=np.int8)

This makes the array backend swappable without changing call-sites: just swap
the import and all numpy calls transparently run on GPU when CuPy is active.
"""

import logging

_logger = logging.getLogger(__name__)

try:
    import cupy as np  # type: ignore[import]

    # Confirm that at least one CUDA device is accessible; if not, fall back.
    np.cuda.Device(0).use()
    _BACKEND = "cupy"
    _logger.info("Array backend: CuPy (CUDA device detected)")
except ImportError:
    import numpy as np  # type: ignore[assignment]

    _BACKEND = "numpy"
    _logger.debug("Array backend: NumPy (CuPy not installed)")
except Exception as _exc:
    import numpy as np  # type: ignore[assignment]

    _BACKEND = "numpy"
    _logger.info("Array backend: NumPy (CuPy unavailable: %s)", _exc)

__all__ = ["np", "_BACKEND"]
