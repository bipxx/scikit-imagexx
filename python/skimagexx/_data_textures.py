"""Exact packaged grayscale texture samples."""

from hashlib import sha256
from importlib.resources import files

import numpy as np

from ._compat import fallback


_PACKAGE = files(__package__).joinpath("_data_assets")
_TEXTURES = {
    "brick": ((512, 512), "664a145c5253f0d66db1a12776785f0ea35a44cc7447ffc933f6d6118dc58643"),
    "grass": ((512, 512), "b18dae4c68bf850a7a7b28a29d1846c76be890665117b57fd125fe29c4d4ede6"),
    "gravel": ((512, 512), "3d51ad45f789cd8b98534b7af6bce774e499ead45421135afd757358c7230009"),
    "text": ((172, 448), "6705caed21e6281799a52591c27498da5526cace39f2b6af3141b2ff11e2e517"),
}


def _texture(name):
    shape, expected = _TEXTURES[name]
    try:
        raw = _PACKAGE.joinpath(name + ".gray").read_bytes()
    except FileNotFoundError:
        return fallback("data", name)
    if len(raw) != shape[0] * shape[1] or sha256(raw).hexdigest() != expected:
        return fallback("data", name)
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()


def brick():
    """Return a fresh grayscale image of a brick wall."""
    return _texture("brick")


def grass():
    """Return a fresh grayscale image of grass."""
    return _texture("grass")


def gravel():
    """Return a fresh grayscale image of gravel."""
    return _texture("gravel")


def text():
    """Return a fresh grayscale image of text used for corner detection."""
    return _texture("text")


__all__ = ["brick", "grass", "gravel", "text"]
