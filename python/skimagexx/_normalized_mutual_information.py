"""Native ordinary NMI counts/entropy with explicit provider boundaries.

Small NumPy edge preparation retains installed 1.x/2.x dtype semantics.
Exceptional inputs remain visible compatibility fallback until native closure.
"""
import warnings
import numpy as np
from . import _core
from ._compat import fallback

_REAL = {np.bool_, np.uint8, np.int8, np.uint16, np.int16, np.uint32,
         np.int32, np.uint64, np.int64, np.float16, np.float32, np.float64}


def _prepare(image0, image1, bins):
    ordinary = lambda x: type(x) in (np.ndarray, np.memmap) or isinstance(x, np.generic)
    if not ordinary(image0) or not ordinary(image1): return None
    if image0.ndim != image1.ndim:
        raise ValueError('NMI requires images of same number of dimensions. '
                         f'Got {image0.ndim}D for `image0` and {image1.ndim}D for `image1`.')
    for image in (image0, image1):
        if image.dtype.type not in _REAL: return None
    if (warnings.showwarning is not warnings._showwarning_orig or
            any(value not in ('warn', 'raise', 'ignore') for value in np.geterr().values())):
        return None
    if type(bins) is int: dimensions = (bins, bins)
    elif type(bins) in (list, tuple) and len(bins) == 2: dimensions = bins
    else: return None
    if any(type(value) is not int or not 1 <= value <= 4096 for value in dimensions): return None
    if dimensions[0] * dimensions[1] > 1048576: return None
    a, b = np.asarray(image0), np.asarray(image1)
    admitted, samples, bounds = _core.nmi_prepare(a, b)
    if not admitted: return None
    dtype = np.result_type(a.dtype, b.dtype)
    edges, widths = [], []
    # Probe only the tiny edge vectors with flags suppressed. Unsupported edge
    # arithmetic is delegated once, preserving original warnings and callbacks.
    with np.errstate(all='ignore'):
        for axis, count in enumerate(dimensions):
            low, high = dtype.type(bounds[2 * axis]), dtype.type(bounds[2 * axis + 1])
            if low == high: low, high = low - 0.5, high + 0.5
            edge = np.linspace(low, high, count + 1)
            width = np.diff(edge)
            edge64, width64 = edge.astype(np.float64, copy=False), width.astype(np.float64, copy=False)
            if (not np.all(np.isfinite(edge)) or not np.all(np.isfinite(width)) or
                    np.any(width64 < 1e-100) or np.any(width64 > 1e100)):
                return None
            edges.append(edge64)
            widths.append(width64)
    return a, b, edges[0], edges[1], widths[0], widths[1]


def normalized_mutual_information(image0, image1, *, bins=100):
    prepared = _prepare(image0, image1, bins)
    if prepared is not None:
        values = _core.nmi_statistics(*prepared)
        if values is not None:
            h0, h1, joint = (np.float64(value) for value in values)
            return (h0 + h1) / joint
    return fallback('metrics', 'normalized_mutual_information', image0, image1, bins=bins)
