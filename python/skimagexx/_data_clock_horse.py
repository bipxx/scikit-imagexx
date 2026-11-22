"""Owned local resources for the exact clock and horse sample arrays."""

from hashlib import sha256
from importlib.resources import files

import numpy as np

from ._compat import fallback


_PACKAGE = files(__package__).joinpath("_data_assets")
_ASSETS = {
    "clock": ("clock.gray", (300, 400), np.uint8,
              "ad313afa739ea86c00ce55d190f1fa284c1982e9bc70eb4c9b21ac29c5c7a85c"),
    "horse": ("horse.bool", (328, 400), np.bool_,
              "1c40c41499d11d4864907189c6d278bf0cb47e7dbf889ae1f5fb0359659a24ab"),
}


def _read_asset(name):
    filename, shape, dtype, expected = _ASSETS[name]
    try:
        raw = _PACKAGE.joinpath(filename).read_bytes()
    except FileNotFoundError:
        return fallback("data", name)
    if len(raw) != shape[0] * shape[1] or sha256(raw).hexdigest() != expected:
        return fallback("data", name)
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def clock():
    """Return a fresh grayscale image of a motion-blurred wall clock."""
    return _read_asset("clock")


def horse():
    """Return a fresh Boolean array of the horse silhouette."""
    return _read_asset("horse")


__all__ = ["clock", "horse"]
