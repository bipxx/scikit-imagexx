"""Native finite floating-point morphological snakes and gradient preprocessing."""

import numpy as np

from . import _core
from ._compat import fallback
from ._level_set import checkerboard_level_set, disk_level_set


# Match the public callable defaults by identity when stock is installed. This
# leaf is loaded with segmentation, not when importing the package itself.
try:
    from skimage.segmentation import morphsnakes as _stock
except ModuleNotFoundError as error:
    if error.name != 'skimage':
        raise
    _stock = None

_chan_callback = (lambda x: None) if _stock is None else _stock.morphological_chan_vese.__defaults__[-1]
_geodesic_callback = (
    (lambda x: None) if _stock is None
    else _stock.morphological_geodesic_active_contour.__defaults__[-1]
)
_local_reverse = False
_LIMIT = 8 * 1024 * 1024


def _phase(iterations, smoothing):
    if not iterations or not smoothing:
        return False
    global _local_reverse
    odd = bool((iterations & 1) and (smoothing & 1))
    if _stock is None:
        reverse = _local_reverse
        _local_reverse ^= odd
        return reverse

    # The pinned implementation shares one alternating process-wide iterator.
    # Consume its phase when available so supported and fallback calls compose.
    # The stock callables are inspected for order and never executed here.
    stages = _stock._curvop.funcs
    stage = next(stages)
    if not odd:
        next(stages)
    return stage.__code__.co_names[0] == 'inf_sup'


def _ordinary_array(array):
    if (type(array) is not np.ndarray or array.dtype.metadata is not None
            or not array.dtype.isnative or not array.flags.c_contiguous
            or not array.flags.aligned or not array.flags.writeable):
        return False
    base = array.base
    return base is None or (
        type(base) is np.ndarray and base.flags.owndata
        and (base.flags.c_contiguous or base.flags.f_contiguous)
    )


def _image(image, minimum):
    return (
        _ordinary_array(image) and image.dtype.char in 'fd'
        and image.ndim in (2, 3) and min(image.shape) >= minimum and image.size <= _LIMIT
    )


def _scalar(value, low, high):
    return type(value) in (int, float) and low <= value <= high


def _contour(image, iterations, smoothing, initial):
    return (
        _image(image, 2) and type(iterations) is int and 0 <= iterations <= 4096
        and type(smoothing) is int and 0 <= smoothing <= 128
        and (type(initial) is str and initial in ('disk', 'checkerboard')
             or _ordinary_array(initial) and initial.shape == image.shape
             and initial.dtype.kind in 'biuf' and initial.dtype.itemsize <= 8)
        and _core.morphological_snakes_supported(image)
    )


def _initial_mask(initial, shape):
    if type(initial) is str:
        if initial == 'disk':
            return disk_level_set(shape)
        return checkerboard_level_set(shape)
    return np.int8(initial > 0)


def inverse_gaussian_gradient(image, alpha=100.0, sigma=5.0):
    """Return the inverse of the nearest-border Gaussian gradient magnitude."""
    if (_image(image, 1) and _scalar(alpha, 0, 1e12) and _scalar(sigma, 0.5, 16)
            and _core.morphological_snakes_supported(image)):
        return _core.inverse_gaussian_gradient(image, alpha, sigma)
    return fallback('segmentation', 'inverse_gaussian_gradient', image, alpha=alpha, sigma=sigma)


def morphological_chan_vese(
    image, num_iter, init_level_set='checkerboard', smoothing=1,
    lambda1=1, lambda2=1, iter_callback=_chan_callback,
):
    """Evolve a binary level set using the two Chan-Vese region averages."""
    if (iter_callback is _chan_callback and _scalar(lambda1, 0, 1e12)
            and _scalar(lambda2, 0, 1e12) and _contour(image, num_iter, smoothing, init_level_set)):
        initial = _initial_mask(init_level_set, image.shape)
        return _core.morphological_chan_vese(
            image, initial, num_iter, smoothing, lambda1, lambda2, _phase(num_iter, smoothing)
        )
    return fallback(
        'segmentation', 'morphological_chan_vese', image, num_iter, init_level_set,
        smoothing, lambda1, lambda2, iter_callback,
    )


def morphological_geodesic_active_contour(
    gimage, num_iter, init_level_set='disk', smoothing=1,
    threshold='auto', balloon=0, iter_callback=_geodesic_callback,
):
    """Evolve a binary level set along edges of a preprocessed image."""
    if (iter_callback is _geodesic_callback and _scalar(balloon, -1e12, 1e12)
            and (balloon == 0 or abs(balloon) >= 1e-12)
            and (type(threshold) is str and threshold == 'auto' or _scalar(threshold, -1e12, 1e12))
            and _contour(gimage, num_iter, smoothing, init_level_set)):
        initial = _initial_mask(init_level_set, gimage.shape)
        cutoff = np.percentile(gimage, 40) if threshold == 'auto' else threshold
        if balloon:
            scaled = cutoff / np.abs(balloon)
            # NumPy 1 and 2 promote float arrays compared against scalars
            # differently. Preserve the live comparison precision.
            if np.result_type(gimage, scaled) == np.dtype('f'):
                scaled = np.float32(scaled)
            cutoff = float(scaled)
        return _core.morphological_geodesic_active_contour(
            gimage, initial, num_iter, smoothing, (balloon > 0) - (balloon < 0),
            cutoff, _phase(num_iter, smoothing),
        )
    return fallback(
        'segmentation', 'morphological_geodesic_active_contour', gimage, num_iter,
        init_level_set, smoothing, threshold, balloon, iter_callback,
    )
