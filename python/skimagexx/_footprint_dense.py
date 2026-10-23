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
_NUMERIC_DTYPE_CLASSES = frozenset(type(np.dtype(name)) for name in
    ('bool','uint8','int8','uint16','int16','uint32','int32','uint64','int64',
     'float16','float32','float64','complex64','complex128'))
_NUMERIC_SCALAR_CLASSES = (bool,int,float,complex,np.bool_,np.uint8,np.int8,
    np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,
    np.float32,np.float64,np.complex64,np.complex128)
_NUMERIC_DTYPE_CLASS_IDS = frozenset(map(id, _NUMERIC_DTYPE_CLASSES))
_NUMERIC_SCALAR_CLASS_IDS = frozenset(map(id, _NUMERIC_SCALAR_CLASSES))


def _inert_numeric_dtype(dtype):
    # Never inspect/normalize custom dtype providers: their conversion hooks
    # and invocation order belong to the original general path. This caches
    # neither dtype instances nor scalar atoms.
    if id(type(dtype)) in _NUMERIC_DTYPE_CLASS_IDS:
        descriptor = dtype
    elif dtype is None or id(dtype) in _NUMERIC_SCALAR_CLASS_IDS:
        descriptor = np.dtype(dtype)
    else:
        return False
    if id(type(descriptor)) not in _NUMERIC_DTYPE_CLASS_IDS:
        return False
    if descriptor.fields is not None or descriptor.subdtype is not None:
        return False
    return (descriptor.kind == 'b' and descriptor.itemsize == 1 or
            descriptor.kind in 'iu' and descriptor.itemsize in (1,2,4,8) or
            descriptor.kind == 'f' and descriptor.itemsize in (2,4,8) or
            descriptor.kind == 'c' and descriptor.itemsize in (8,16))


def _generate(shape, dtype, kind, costs=None, limit=0.):
    if kind == 0 and _inert_numeric_dtype(dtype):
        output = np.empty(shape, dtype=dtype)
        _core.footprint_rectangle_numeric_into(output)
        return output
    if costs is None:
        costs = np.empty(0, dtype=np.float64)
    else:
        costs = np.ascontiguousarray(costs, dtype=np.float64)
    if kind == 0:
        # np.ones supplies the dtype's exact scalar (e.g. object int, U3 '1').
        # Allocation precedes atom conversion just as in np.ones(shape, dtype).
        output = np.empty(shape, dtype=dtype)
        # The original np.ones resolves a caller dtype provider once, during
        # allocation. Further atom work must use that resolved descriptor.
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
