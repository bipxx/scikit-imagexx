"""Sample arrays from verified local scikit-image files, with fresh native storage."""

from os import path, stat

import numpy as np

from ._compat import fallback
from ._core import data_loaders_a_copy

try:
    from skimage.data import _fetchers
except ModuleNotFoundError as error:
    if error.name != "skimage":
        raise
    _fetchers = None


_FILES = {
    "astronaut": "astronaut.png",
    "brain": "brain.tiff",
    "cell": "cell.png",
    "cells3d": "cells3d.tif",
    "coins": "coins.png",
    "colorwheel": "color.png",
    "eagle": "eagle.png",
    "hubble_deep_field": "hubble_deep_field.jpg",
    "human_mitosis": "mitosis.tif",
    "immunohistochemistry": "ihc.png",
    "kidney": "kidney.tif",
    "lbp_frontal_face_cascade_filename": "lbpcascade_frontalface_opencv.xml",
    "lfw_subset": "lfw_subset.npy",
}

_verified = {}
_decoded = {}


def _stamp(filename):
    try:
        info = stat(filename)
    except OSError:
        return None
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _verified_local(name):
    """Use the provider's cache-before-distribution precedence and registry hash."""
    key = "data/" + _FILES[name]
    expected = _fetchers.registry.get(key)
    if expected is None:
        return None
    provider = _fetchers._image_fetcher
    root = path.dirname(_fetchers.data_dir) if provider is None else str(provider.abspath)
    roots = (root, _fetchers._DISTRIBUTION_DIR)
    if roots[0] == roots[1]:
        roots = roots[:1]
    for directory in roots:
        filename = path.join(directory, key)
        stamp = _stamp(filename)
        if stamp is None:
            continue
        identity = filename, stamp, expected
        previous = _verified.get((name, directory == root))
        if previous is not None and previous[0] == identity:
            valid = previous[1]
        else:
            valid = _fetchers._has_hash(filename, expected)
            if stamp != _stamp(filename):
                return None
            _verified[name, directory == root] = identity, valid
        if valid:
            return identity
    return None


def _read_image(filename):
    from skimage import io
    from skimage.io.manage_plugins import plugin_store

    plugins = plugin_store["imread"]
    if filename.lower().endswith((".tif", ".tiff")):
        from skimage.io._plugins import tifffile_plugin

        default = tifffile_plugin.imread
        reader = next((reader for name, reader in plugins if name == "tifffile"), default)
    else:
        from skimage.io._plugins import imageio_plugin

        default = imageio_plugin.imread
        reader = plugins[0][1] if plugins else None
    return (io.imread, reader) if reader is default else None


def _snapshot(value):
    if type(value) is not np.ndarray or not value.size or value.dtype.hasobject:
        return None
    owner = value if value.flags.owndata else value.base
    if (type(owner) is not np.ndarray or not owner.flags.owndata
            or not owner.flags.aligned or not owner.flags.writeable
            or owner.dtype is not value.dtype or owner.size != value.size):
        return None
    stride = owner.itemsize
    for dimension, actual in zip(reversed(owner.shape), reversed(owner.strides)):
        if actual != stride:
            return None
        stride *= dimension
    shape = axes = None
    if owner is not value:
        if owner.ctypes.data != value.ctypes.data:
            return None
        if value.flags.c_contiguous:
            shape = value.shape
            view = owner.reshape(shape)
        elif value.ndim >= 3 and value.ndim == owner.ndim:
            # skimage.io.imread moves planar RGB(A) channels to the last axis.
            axes = (*range(value.ndim - 3), -2, -1, -3)
            view = owner.transpose(axes)
        else:
            return None
        if (view.shape, view.strides, view.flags.num) != (
                value.shape, value.strides, value.flags.num):
            return None
    return (owner.tobytes(), owner.dtype, owner.shape), shape, axes


def _copy_snapshot(snapshot):
    arguments, shape, axes = snapshot
    owner = data_loaders_a_copy(*arguments)
    if axes is not None:
        return owner.transpose(axes)
    if shape is not None:
        return owner.reshape(shape)
    return owner


def _array(name):
    identity = _verified_local(name) if _fetchers is not None else None
    if identity is None:
        return fallback("data", name)
    filename, stamp, _ = identity
    is_npy = name == "lfw_subset"
    decoder = (np.load,) if is_npy else _read_image(filename)
    if decoder is None:
        return fallback("data", name)
    key = identity, decoder
    cached = _decoded.get(name)
    if cached is not None and cached[0] == key:
        return _copy_snapshot(cached[1])

    value = decoder[0](filename) if is_npy else decoder[0](filename, as_gray=False)
    snapshot = _snapshot(value)
    if snapshot is None:
        return value
    if stamp == _stamp(filename):
        _decoded[name] = key, snapshot
    return _copy_snapshot(snapshot)


def astronaut():
    return _array("astronaut")


def brain():
    return _array("brain")


def cell():
    return _array("cell")


def cells3d():
    return _array("cells3d")


def coins():
    return _array("coins")


def colorwheel():
    return _array("colorwheel")


def eagle():
    return _array("eagle")


def hubble_deep_field():
    return _array("hubble_deep_field")


def human_mitosis():
    return _array("human_mitosis")


def immunohistochemistry():
    return _array("immunohistochemistry")


def kidney():
    return _array("kidney")


def lbp_frontal_face_cascade_filename():
    name = "lbp_frontal_face_cascade_filename"
    identity = _verified_local(name) if _fetchers is not None else None
    return identity[0] if identity is not None else fallback("data", name)


def lfw_subset():
    return _array("lfw_subset")


if _fetchers is not None:
    # Preserve Pooch's full registry traversal, cache setup and path restoration.
    download_all = _fetchers.download_all
else:
    def download_all(directory=None):
        return fallback("data", "download_all", directory)


__all__ = [*_FILES, "download_all"]
