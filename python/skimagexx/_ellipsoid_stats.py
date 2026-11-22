"""Native scalar ellipsoid volume and Carlson surface area."""
import numpy as np

from . import _core
from ._compat import fallback

_SCALARS = frozenset((
    bool, int, float, np.bool_, np.int8, np.uint8, np.int16, np.uint16,
    np.int32, np.uint32, np.int64, np.uint64, np.float16, np.float32,
    np.float64, np.longdouble,
))


def _surface_integral(x, y, z):
    dtype = np.result_type(x, y, z)
    if dtype in (np.dtype('float16'), np.dtype('float32'), np.dtype('float64')):
        values = float(x), float(y), float(z)
        # Positive bounded arguments need no exceptional conversion or ufunc.
        if all(2.0**-192 <= value <= 2.0**192 for value in values):
            if dtype == np.dtype('float64'):
                scalar = np.float64
            else:
                scalar = np.float32
                if all(2.0**-126 <= value <= float(np.finfo(np.float32).max) for value in values):
                    values = tuple(float(scalar(value)) for value in values)
                else:
                    scalar = None
            if scalar is not None:
                native = _core._ellipsoid_stats_rg(*values)
                if native is not None:
                    return scalar(native)
    # Keep the exact SciPy exceptional, extended-type and warning behavior
    # once NumPy/Python have evaluated the original typed scalar expressions.
    from scipy.special import elliprg

    return elliprg(x, y, z)


def ellipsoid_stats(a, b, c):
    """Return the volume and surface area for three positive semi-axes.

    The finite scalar Carlson integral is computed directly. Extreme and
    exceptional integral arguments retain the SciPy special-function provider.
    Array and arbitrary object protocols use the configured compatibility path.
    """
    result = _core._ellipsoid_stats_builtin(a, b, c)
    if result is not None:
        volume, surface = result
        return volume, np.float64(surface)
    if type(a) not in _SCALARS or type(b) not in _SCALARS or type(c) not in _SCALARS:
        return fallback('draw', 'ellipsoid_stats', a, b, c)
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError('Parameters a, b, and c must all be > 0')
    volume = 4 / 3.0 * np.pi * a * b * c
    surface = 3 * volume * _surface_integral(1 / a**2, 1 / b**2, 1 / c**2)
    return volume, surface
