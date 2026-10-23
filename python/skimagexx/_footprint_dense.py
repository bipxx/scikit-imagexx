"""Private dense footprint pipeline. Decomposition is deliberately unimplemented.

All geometry is generated in C++ without coordinate grids. NumPy supplies 1-D
coordinate arithmetic and dtype construction, including unusual dtypes. Arrays
without object references are filled directly in their final allocation; object
dtypes require a native Boolean mask and NumPy's reference-aware conversion.
"""
import numpy as np
from . import _core
_TRUE = np.ones(1, dtype=bool)
_FALSE = np.zeros(1, dtype=bool)
_TRUE.flags.writeable = _FALSE.flags.writeable = False


def _generate(shape, dtype, kind, costs=None, limit=0.):
    if costs is None:
        costs = np.empty(0, dtype=np.float64)
    else:
        costs = np.ascontiguousarray(costs, dtype=np.float64)
    if kind == 0:
        # np.ones supplies the dtype's exact scalar (e.g. object int, U3 '1').
        # Allocation precedes atom conversion just as in np.ones(shape, dtype).
        output = np.empty(shape, dtype=dtype)
        # Resolve caller dtype metadata only during the original-equivalent allocation.
        one = np.ones((), dtype=output.dtype)
        zero = np.zeros((), dtype=output.dtype)
    else:
        # np.array(bool_mask, dtype) parses dtype even for empty geometry.
        dtype = np.dtype(bool if dtype is None else dtype)
        provider = dtype.hasobject or (dtype.kind in 'SUV' and dtype.itemsize == 0)
        output = np.empty(shape, dtype=bool if provider else dtype)
        one = np.array(_TRUE, dtype=output.dtype)
        zero = np.array(_FALSE, dtype=output.dtype)
    if output.dtype.hasobject:
        # Rectangle has no geometry; fill retains Python reference ownership.
        output.fill(1)
        return output
    _core.footprint_dense_into(output, costs, kind, float(np.asarray(limit).item()), zero, one)
    if kind != 0 and provider:
        return np.array(output, dtype=dtype)
    return output


def _dense_only(decomposition):
    if decomposition is not None:
        raise NotImplementedError('private dense footprints only support decomposition=None')


def footprint_rectangle(shape, *, dtype=np.uint8, decomposition=None):
    _dense_only(decomposition)
    # Preserve validation order, including non-iterable or fractional shapes.
    any(width % 2 == 0 for width in shape)
    return _generate(shape, dtype, 0)


def diamond(radius, dtype=np.uint8, *, decomposition=None):
    _dense_only(decomposition)
    coordinates = np.arange(0, radius * 2 + 1)
    costs = np.abs(coordinates - radius)
    return _generate((coordinates.size,) * 2, dtype, 1, costs, radius)


def disk(radius, dtype=np.uint8, *, strict_radius=True, decomposition=None):
    _dense_only(decomposition)
    coordinates = np.arange(-radius, radius + 1)
    if not strict_radius:
        radius += .5
    return _generate((coordinates.size,) * 2, dtype, 2, coordinates ** 2, radius ** 2)


def ball(radius, dtype=np.uint8, *, strict_radius=True, decomposition=None):
    _dense_only(decomposition)
    n = 2 * radius + 1
    # Sparse grid normalization reproduces mgrid's unusual negative/fractional
    # sample counts and errors while allocating only three 1-D vectors.
    slices = (slice(-radius, radius, n * 1j),) * 3
    coordinates = np.ogrid[slices][0].reshape(-1)
    if not strict_radius:
        radius += .5
    return _generate((coordinates.size,) * 3, dtype, 3, coordinates ** 2, radius * radius)
