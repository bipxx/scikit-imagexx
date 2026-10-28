"""Owned order-0/1 sampling with the scikit-image 0.26 window prelude.

SciPy's one-dimensional get_window and NumPy coordinate construction and
clipping remain explicit providers. Only radial interpolation is C++; higher
orders and mixed-empty coordinate arrays are outside this initial native scope.
"""
import functools
import operator
import warnings

import numpy as np

from . import _core


def _safe_as_int(val, atol=1e-3):
    if isinstance(val, np.ndarray):
        # Shape and output_shape may be caller-owned views. Validate their
        # storage before the first NumPy read, using the existing boundary.
        _core.fft_check(val)
    mod = np.asarray(val) % 1
    if mod.ndim == 0:
        if mod > 0.5:
            mod = 1 - mod
    else:
        mod[mod > 0.5] = 1 - mod[mod > 0.5]
    if not np.allclose(mod, 0, atol=atol):
        raise ValueError(f'Integer argument required but received {val}, check inputs.')
    return np.round(val).astype(np.int64)


def _warp_window(image, inverse_map, map_args=None, output_shape=None, order=None,
                 mode='constant', cval=0.0, clip=True, preserve_range=False):
    """The coordinate-array window path, with owned sampling instead of warp."""
    if map_args is None:
        map_args = {}
    if image.size == 0:
        raise ValueError("Cannot warp empty image with dimensions", image.shape)
    if order is None:
        order = 1
    elif order < 0 or order > 5:
        raise ValueError("Spline interpolation order has to be in the range 0-5.")
    if order > 0:
        # get_window already supplies float64. Both source conversions leave
        # it unchanged, but preserve the option's source-ordered truth test.
        bool(preserve_range)
    input_shape = np.array(image.shape)
    if output_shape is None:
        output_shape = input_shape
    else:
        output_shape = _safe_as_int(output_shape)
    if order == 2:
        warnings.warn(
            "Bi-quadratic interpolation behavior has changed due "
            "to a bug in the implementation of scikit-image. "
            "The new version now serves as a wrapper "
            "around SciPy's interpolation functions, which itself "
            "is not verified to be a correct implementation. Until "
            "skimage's implementation is fixed, we recommend "
            "to use bi-linear or bi-cubic interpolation instead."
        )
    # SciPy's C entry also requires an integer order, including after the
    # range test has admitted a floating NaN or a numerically integral float.
    order = operator.index(order)
    if order not in (0, 1):
        raise NotImplementedError("window sampling currently supports only orders 0 and 1")
    if inverse_map[0].size == 0:
        raise NotImplementedError("mixed-empty window shapes are outside the native scope")

    workers = _core.get_num_threads()
    warped = _core.window_sample(image.reshape(-1), inverse_map[0], order, workers)

    # Preserve grid-constant clipping, including positive-minimum windows:
    # cval expands the range only if it actually occurs in the sampled output.
    if clip:
        min_val = np.min(image)
        if np.isnan(min_val):
            min_func = np.nanmin
            max_func = np.nanmax
            min_val = min_func(image)
        else:
            min_func = np.min
            max_func = np.max
        max_val = max_func(image)
        preserve_cval = (
            mode == 'constant'
            and not min_val <= cval <= max_val
            and min_func(warped) <= cval <= max_func(warped)
        )
        if preserve_cval:
            cval = image.dtype.type(cval)
            min_val = min(min_val, cval)
            max_val = max(max_val, cval)
        min_val, max_val = np.asarray(min_val), np.asarray(max_val)
        np.clip(warped, min_val, max_val, out=warped)
    return warped


def window(window_type, shape, warp_kwargs=None):
    """Return a window using provider coordinates and owned order-0/1 sampling."""
    if np.isscalar(shape):
        shape = (_safe_as_int(shape),)
    else:
        shape = tuple(_safe_as_int(shape))
    if any(s < 0 for s in shape):
        raise ValueError("invalid shape")
    ndim = len(shape)
    if ndim <= 0:
        raise ValueError("Number of dimensions must be greater than zero")
    max_size = functools.reduce(max, shape)
    from scipy.signal import get_window
    w = get_window(window_type, max_size, fftbins=False)
    w = np.reshape(w, (-1,) + (1,) * (ndim - 1))

    L = [np.arange(s, dtype=np.float32) * (max_size / s) for s in shape]
    center = (max_size / 2) - 0.5
    dist = 0
    for g in np.meshgrid(*L, sparse=True, indexing='ij'):
        g -= center
        dist = dist + g * g
    dist = np.sqrt(dist)
    coords = np.zeros((ndim,) + dist.shape, dtype=np.float32)
    coords[0] = dist + center
    if warp_kwargs is None:
        warp_kwargs = {}
    try:
        return _warp_window(w, coords, mode='constant', cval=0.0, **warp_kwargs)
    except TypeError as error:
        # Python includes the callee in argument-binding errors. Translate
        # only those prefixes; do not spoof a module or hide provider errors.
        message = str(error)
        prefixes = (
            (__name__ + "._warp_window() got multiple values for keyword argument ",
             "skimage.transform._warps.warp() got multiple values for keyword argument "),
            (__name__ + "._warp_window() argument after ** ",
             "skimage.transform._warps.warp() argument after ** "),
            ("_warp_window() got an unexpected keyword argument ",
             "warp() got an unexpected keyword argument "),
        )
        for before, after in prefixes:
            if message.startswith(before):
                raise TypeError(after + message[len(before):]) from None
        raise
