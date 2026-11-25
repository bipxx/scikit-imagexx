"""Load fixed bundled samples and retain upstream fetching for remote data."""

from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np

from . import _core
from ._compat import fallback


_STEREO_DISPARITY = "motorcycle_disp.npz"


def _local_payload(fetchers, filename):
    """Use the same cache-before-distribution order and digest as stock."""
    expected = fetchers.registry[f"data/{filename}"]
    shipped = Path(fetchers.__file__).with_name(filename)
    cached = Path(fetchers.data_dir) / filename
    paths = (shipped,) if cached == shipped else (cached, shipped)
    for path in paths:
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            continue
        if sha256(payload).hexdigest() == expected:
            return payload
    return None


@lru_cache(maxsize=9)
def _decoded(filename, payload):
    # Cache immutable decoded pixels keyed by the exact verified encoded bytes.
    if filename == _STEREO_DISPARITY:
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            image = archive["arr_0"]
    else:
        from skimage.io import imread

        image = imread(BytesIO(payload))
    return image.tobytes(), image.dtype, image.shape


def _bundled(name, *filenames):
    materialize = getattr(_core, "data_loaders_b_materialize", None)
    if materialize is None:
        return fallback("data", name)
    try:
        from skimage.data import _fetchers
    except ModuleNotFoundError as error:
        if error.name != "skimage":
            raise
        return fallback("data", name)

    from skimage.io._plugins import imageio_plugin
    from skimage.io.manage_plugins import plugin_store

    readers = plugin_store["imread"]
    if not readers or readers[0][1] is not imageio_plugin.imread:
        return fallback("data", name)

    results = []
    for filename in filenames:
        payload = _local_payload(_fetchers, filename)
        if payload is None:
            return fallback("data", name)
        pixels, dtype, shape = _decoded(filename, payload)
        if filename == _STEREO_DISPARITY:
            # NumPy's NPY reader returns a 2D view of a flat owning buffer.
            array = materialize(pixels, dtype, (shape[0] * shape[1],))
            results.append(array.reshape(shape))
        else:
            results.append(materialize(pixels, dtype, shape))
    if len(results) == 1:
        return results[0]
    disparity, left, right = results
    return left, right, disparity


def lily():
    """Load the lily of the valley fluorescence image."""
    return fallback("data", "lily")


def logo():
    """Return the scikit-image logo."""
    return _bundled("logo", "logo.png")


def microaneurysms():
    """Return the grayscale microaneurysm image."""
    return _bundled("microaneurysms", "microaneurysms.png")


def moon():
    """Return the grayscale image of the lunar surface."""
    return _bundled("moon", "moon.png")


def nickel_solidification():
    """Load x-radiographs of a solidifying nickel alloy."""
    return fallback("data", "nickel_solidification")


def page():
    """Return the grayscale scanned page."""
    return _bundled("page", "page.png")


def palisades_of_vogt():
    """Load the corneal microscopy stack."""
    return fallback("data", "palisades_of_vogt")


def protein_transport():
    """Load the fluorescence microscopy sequence of protein transport."""
    return fallback("data", "protein_transport")


def retina():
    """Return the color retinal image."""
    return _bundled("retina", "retina.jpg")


def rocket():
    """Return the color photo of a Falcon 9 launch."""
    return _bundled("rocket", "rocket.jpg")


def skin():
    """Load the microscopy image of epidermis and dermis."""
    return fallback("data", "skin")


def stereo_motorcycle():
    """Return the left image, right image and ground-truth disparities."""
    return _bundled(
        "stereo_motorcycle", _STEREO_DISPARITY, "motorcycle_left.png", "motorcycle_right.png"
    )


def vortex():
    """Load the first PIV Challenge Case B1 image pair."""
    return fallback("data", "vortex")
