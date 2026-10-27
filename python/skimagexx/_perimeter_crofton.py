"""Owned Crofton counts and weighted sum with explicit type providers."""
import numpy as np
from . import _core


def _coefficients(directions):
    if directions == 2:
        return [0, np.pi / 2, 0, 0, 0, np.pi / 2, 0, 0,
                np.pi / 2, np.pi, 0, 0, np.pi / 2, np.pi, 0, 0]
    return [0, np.pi / 4 * (1 + 1 / np.sqrt(2)),
            np.pi / (4 * np.sqrt(2)), np.pi / (2 * np.sqrt(2)),
            0, np.pi / 4 * (1 + 1 / np.sqrt(2)), 0,
            np.pi / (4 * np.sqrt(2)), np.pi / 4, np.pi / 2,
            np.pi / (4 * np.sqrt(2)), np.pi / (4 * np.sqrt(2)),
            np.pi / 4, np.pi / 2, 0, 0]


def perimeter_crofton(image, directions=4):
    """Calculate a 2D Crofton perimeter using the global optional workers.

    Ordinary real arrays use owned threshold, stencil and histogram stages.
    Other input types retain NumPy/SciPy primitive preparation. Coefficient
    construction and scalar boxing follow the original library.
    """
    if image.ndim != 2:
        raise NotImplementedError('`perimeter_crofton` supports 2D images only')
    workers = _core.get_num_threads()
    if (type(image) is np.ndarray and image.dtype.kind in 'biuf'
            and image.dtype.itemsize in (1, 2, 4, 8)):
        h = _core.crofton_counts(image, workers)
    else:
        from scipy import ndimage as ndi
        binary = np.pad((image > 0).astype(np.uint8), pad_width=1, mode='constant')
        codes = ndi.convolve(binary, np.array([[0, 0, 0], [0, 1, 4], [0, 2, 8]]),
                             mode='constant', cval=0)
        h = np.bincount(codes.ravel(), minlength=16)
    # The source compares directions only after the histogram is complete.
    coefs = _coefficients(directions)
    return np.float64(_core.crofton_perimeter(h, np.asarray(coefs), workers))
