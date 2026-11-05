"""Owned ellipsoid grid for ordinary scalar inputs; explicit protocol provider.

BSD source attribution: third_party/scikit-image-LICENSE.txt. Grid arithmetic
and coordinate allocation live in the C++ prepared plan, not in this boundary.
"""
from . import _core


def ellipsoid(a, b, c, spacing=(1.0, 1.0, 1.0), levelset=False):
    native_result = None
    # Preserve original short-circuit comparison/conversion/truth order for
    # unknown scalar and container protocols: do not invoke them then replay.
    if (any(type(v) not in (int, float) for v in (a, b, c))
            or type(spacing) not in (tuple, list) or type(levelset) is not bool):
        from skimage.draw.draw3d import ellipsoid as provider
        return provider(a, b, c, spacing=spacing, levelset=levelset)
    # NumPy int64 addition/object promotion happens before the source float()
    # division. Only this exact integer range keeps every bounded sum/coordinate
    # exactly representable in int64 and float64; large ints retain the provider.
    if any(type(v) is int and abs(v) > 2**50 for v in (a, b, c, *spacing)):
        from skimage.draw.draw3d import ellipsoid as provider
        return provider(a, b, c, spacing=spacing, levelset=levelset)
    if (a <= 0) or (b <= 0) or (c <= 0):
        raise ValueError('Parameters a, b, and c must all be > 0')
    # The captured safe builtin malformed-spacing partition has no callbacks,
    # warnings or state effects. Keep axis validation ahead of these errors.
    if (type(spacing) is tuple and len(spacing) in (1, 2, 4)
            and all(0 < v <= 8 for v in (a, b, c))
            and all(type(v) in (int, float) and .25 <= v <= 2 for v in spacing)):
        if len(spacing) == 1:
            raise IndexError('tuple index out of range')
        raise ValueError(f'operands could not be broadcast together with shapes (3,) ({len(spacing)},) ')
    if len(spacing) != 3 or any(type(v) not in (int, float) for v in spacing):
        from skimage.draw.draw3d import ellipsoid as provider
        return provider(a, b, c, spacing=spacing, levelset=levelset)
    values = (a, b, c, *spacing)
    # Bound conversions before float() so enormous Python integers retain the
    # untouched provider's object-arithmetic/errors rather than overflow here.
    if not all(2.0**-128 <= v <= 2.0**128 for v in values):
        from skimage.draw.draw3d import ellipsoid as provider
        return provider(a, b, c, spacing=spacing, levelset=levelset)
    workers = _core.get_num_threads()
    native_result = _core.draw_ellipsoid(tuple(float(v) for v in (a,b,c)),
        tuple(float(v) for v in spacing), levelset, workers)
    if native_result is not None:
        return native_result
    from skimage.draw.draw3d import ellipsoid as provider
    return provider(a, b, c, spacing=spacing, levelset=levelset)
