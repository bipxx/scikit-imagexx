"""Exact decoded RGB resource for the Chelsea sample image."""

from hashlib import sha256
from importlib.resources import files

import numpy as np

from ._compat import fallback


_SHAPE = (300, 451, 3)
_SIZE = 405900
_SHA256 = "416b729128bfb2c3d1eb69bf9b1734a796293abc17939267b2dc94f8a5784031"
_RESOURCE = files(__package__).joinpath("_data_assets").joinpath("chelsea.rgb")


def chelsea():
    """Return a fresh RGB image of Chelsea, the cat photographed by Stefan van der Walt."""
    try:
        raw = _RESOURCE.read_bytes()
    except FileNotFoundError:
        return fallback("data", "chelsea")
    if len(raw) != _SIZE or sha256(raw).hexdigest() != _SHA256:
        return fallback("data", "chelsea")
    return np.frombuffer(raw, dtype=np.uint8).reshape(_SHAPE).copy()


cat = chelsea
__all__ = ["chelsea", "cat"]
