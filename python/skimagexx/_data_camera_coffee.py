"""Exact packaged camera and coffee sample arrays."""

from hashlib import sha256
from importlib.resources import files
from math import prod

import numpy as np

from ._compat import fallback


_PACKAGE = files(__package__).joinpath("_data_assets")
_ASSETS = {
    "camera": ("camera.gray", (512, 512),
               "5cb24482a53416f99052258be2b1ee38cd31c559a70c8a8b321cba231b332e21"),
    "coffee": ("coffee.rgb", (400, 600, 3),
               "0ce2b51640b9c95f19617f03eabf40c3f0368589cc1ee1190b70966165ac184f"),
}


def _read_asset(name):
    filename, shape, expected = _ASSETS[name]
    try:
        raw = _PACKAGE.joinpath(filename).read_bytes()
    except FileNotFoundError:
        return fallback("data", name)
    if len(raw) != prod(shape) or sha256(raw).hexdigest() != expected:
        return fallback("data", name)
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()


def camera():
    """Return a fresh grayscale camera image (512 × 512, uint8)."""
    return _read_asset("camera")


def coffee():
    """Return a fresh color coffee cup image (400 × 600 × 3, uint8)."""
    return _read_asset("coffee")


__all__ = ["camera", "coffee"]
