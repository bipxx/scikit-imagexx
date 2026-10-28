"""Literal polygon protocol with owned prepared B-spline arithmetic."""
import numpy as np
from . import _core

_SUBDIVISION_MASKS = {
    1: ([1, 1], [1, 1]),
    2: ([3, 1], [1, 3]),
    3: ([1, 6, 1], [0, 4, 4]),
    4: ([5, 10, 1], [1, 10, 5]),
    5: ([1, 15, 15, 1], [0, 6, 20, 6]),
    6: ([7, 35, 21, 1], [1, 21, 35, 7]),
    7: ([1, 28, 70, 28, 1], [0, 8, 56, 56, 8]),
}
_DEGREES = (bool, int, float, np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64)


def subdivide_polygon(coords, degree=2, preserve_ends=False):
    # scikit-image0.26 _polygon.py, BSD-3-Clause; see third_party/scikit-image-LICENSE.txt.
    if degree not in _SUBDIVISION_MASKS:
        raise ValueError("Invalid B-Spline degree. Only degree 1 - 7 is " "supported.")

    circular = np.all(coords[0, :] == coords[-1, :])

    method = 'valid'
    if circular:
        # remove last coordinate because of wrapping
        coords = coords[:-1, :]
        # circular convolution by wrapping boundaries
        method = 'same'

    native_result = None
    if (type(coords) is np.ndarray and type(preserve_ends) is bool
            and type(degree) in _DEGREES and coords.dtype.kind in 'biuf'
            and coords.dtype.itemsize <= 8 and coords.ndim == 2
            and coords.shape[1] == 2 and coords.shape[0] >= 1):
        prepared = np.require(coords, dtype=np.float64, requirements=('C', 'A'))
        native_result = _core.subdivide_polygon_apply(
            prepared, int(degree), bool(circular), preserve_ends, _core.get_num_threads())
        if native_result is not None:
            return native_result

    mask_even, mask_odd = _SUBDIVISION_MASKS[degree]
    # divide by total weight
    mask_even = np.array(mask_even, float) / (2**degree)
    mask_odd = np.array(mask_odd, float) / (2**degree)

    from scipy import signal
    even = signal.convolve2d(
        coords.T, np.atleast_2d(mask_even), mode=method, boundary='wrap'
    )
    odd = signal.convolve2d(
        coords.T, np.atleast_2d(mask_odd), mode=method, boundary='wrap'
    )

    out = np.zeros((even.shape[1] + odd.shape[1], 2))
    out[1::2] = even.T
    out[::2] = odd.T

    if circular:
        # close polygon
        out = np.vstack([out, out[0, :]])

    if preserve_ends and not circular:
        out = np.vstack([coords[0, :], out, coords[-1, :]])

    return out
