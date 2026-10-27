"""Source binary/padding protocols and owned Euler configuration arithmetic."""
import numpy as np
from . import _core

# Exact Ohser configuration tables from scikit-image0.26.0 _regionprops_utils.py.
# The original source and tables are pinned by the original-first contract.
_EULER_COEFS2D_4 = [0, 1, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0]
_EULER_COEFS2D_8 = [0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, -1, 0]
_EULER_COEFS3D_26 = np.array([
    0, 1, 1, 0, 1, 0, -2, -1, 1, -2, 0, -1, 0, -1, -1, 0,
    1, 0, -2, -1, -2, -1, -1, -2, -6, -3, -3, -2, -3, -2, 0, -1,
    1, -2, 0, -1, -6, -3, -3, -2, -2, -1, -1, -2, -3, 0, -2, -1,
    0, -1, -1, 0, -3, -2, 0, -1, -3, 0, -2, -1, 0, 1, 1, 0,
    1, -2, -6, -3, 0, -1, -3, -2, -2, -1, -3, 0, -1, -2, -2, -1,
    0, -1, -3, -2, -1, 0, 0, -1, -3, 0, 0, 1, -2, -1, 1, 0,
    -2, -1, -3, 0, -3, 0, 0, 1, -1, 4, 0, 3, 0, 3, 1, 2,
    -1, -2, -2, -1, -2, -1, 1, 0, 0, 3, 1, 2, 1, 2, 2, 1,
    1, -6, -2, -3, -2, -3, -1, 0, 0, -3, -1, -2, -1, -2, -2, -1,
    -2, -3, -1, 0, -1, 0, 4, 3, -3, 0, 0, 1, 0, 1, 3, 2,
    0, -3, -1, -2, -3, 0, 0, 1, -1, 0, 0, -1, -2, 1, -1, 0,
    -1, -2, -2, -1, 0, 1, 3, 2, -2, 1, -1, 0, 1, 2, 2, 1,
    0, -3, -3, 0, -1, -2, 0, 1, -1, 0, -2, 1, 0, -1, -1, 0,
    -1, -2, 0, 1, -2, -1, 3, 2, -2, 1, 1, 2, -1, 0, 2, 1,
    -1, 0, -2, 1, -2, 1, 1, 2, -2, 3, -1, 2, -1, 2, 0, 1,
    0, -1, -1, 0, -1, 0, 2, 1, -1, 2, 0, 1, 0, 1, 1, 0,
])


def _configuration(ndim, connectivity):
    if ndim == 2:
        config = np.array([[0, 0, 0], [0, 1, 4], [0, 2, 8]])
        if connectivity == 1:
            coefs = _EULER_COEFS2D_4
        else:
            coefs = _EULER_COEFS2D_8
        bins = 16
    else:
        if connectivity == 2:
            raise NotImplementedError('For 3D images, Euler number is implemented for connectivities 1 and 3 only')
        config = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 1, 4], [0, 2, 8]],
                           [[0, 0, 0], [0, 16, 64], [0, 32, 128]]])
        if connectivity == 1:
            coefs = _EULER_COEFS3D_26[::-1]
        else:
            coefs = _EULER_COEFS3D_26
        bins = 256
    return config, coefs, bins


def euler_number(image, connectivity=None):
    """Compute Euler characteristic with explicit binary/padding providers.

    Configuration counts and bounded exact LUT arithmetic are owned C++.
    The private binding validates/copies binary bytes once before native work.
    Snum above2**53, unsupported ranks and unassessed subclass preparation
    retain literal primitive operations, without owned weighted credit.
    """
    if type(image) is np.ndarray:
        _core.fft_check(image)
    image = (image > 0).astype(int)
    image = np.pad(image, pad_width=1, mode='constant')
    if connectivity is None:
        connectivity = image.ndim
    config, coefs, bins = _configuration(image.ndim, connectivity)
    workers = _core.get_num_threads()
    if type(image) is np.ndarray and image.ndim in (2, 3):
        # Keep the source's padded image/coefs objects intact for callbacks and
        # witnesses. This separate byte copy crosses the native buffer boundary.
        prepared = np.ascontiguousarray(image, dtype=np.uint8)
        volume = image.ndim == 3
        h = _core.euler_counts(prepared, volume, workers)
        coefficients = np.ascontiguousarray(coefs, dtype=np.int8)
        owned = _core.euler_weighted(h, coefficients, volume, workers)
        if owned is not None:
            if volume:
                return int(owned)
            return np.result_type(np.asarray(coefs).dtype, h.dtype).type(owned)
    else:
        from scipy import ndimage as ndi
        XF = ndi.convolve(image, config, mode='constant', cval=0)
        h = np.bincount(XF.ravel(), minlength=bins)
    if image.ndim == 2:
        return coefs @ h
    return int(0.125 * coefs @ h)
