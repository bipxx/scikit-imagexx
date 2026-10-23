"""Private exact decomposition tuples, using native masks and ordered plans.

Pinned sphere counts and finite numeric cross plans live in C++. NumPy provides
index validation, dtype construction and rotated views. Non-numeric cross sums
retain NumPy's conversion/warning rules; they never call upstream generators.
"""
import operator
import warnings
import numpy as np
from . import _core
from . import _footprint_dense as dense

_PROBES = {2: np.empty((251, 0), dtype=np.uint8),
           3: np.empty((101, 0), dtype=np.uint8)}


def _element(dtype, dimensions, kind, axis=0, direction=0, vertical=0, horizontal=0,
             literal=False):
    shape = (2 * vertical + 1, 2 * horizontal + 1) if kind == 2 else (3,) * dimensions
    if literal:
        mask = _element(np.uint8, dimensions, kind, axis, direction)
        # A literal 2-D T uses integer-list conversion, distinct from zeros
        # plus assignment for Unicode, object and generic-unit datetime dtypes.
        return np.array(mask.tolist(), dtype=dtype)
    output = np.empty(shape, dtype=dtype)
    if output.dtype.kind not in 'biufc':
        output = np.zeros(shape, dtype=dtype)
        if kind == 0:
            face = [slice(None)] * dimensions; face[axis] = slice(direction, direction + 1)
            output[tuple(face)] = 1
            stem = [slice(1, 2)] * dimensions; stem[axis] = slice(None)
            output[tuple(stem)] = 1
        elif kind == 1:
            for dim in range(dimensions):
                stem = [slice(1, 2)] * dimensions; stem[dim] = slice(None)
                output[tuple(stem)] = 1
        else:
            if horizontal: output[vertical, :] = 1
            if vertical: output[:, horizontal] = 1
        return output
    zero, one = np.zeros((), dtype=output.dtype), np.ones((), dtype=output.dtype)
    _core.footprint_element_into(output, zero, one, kind, dimensions, axis, direction,
                                 int(vertical), int(horizontal))
    return output


def _sphere(radius, dimensions, dtype):
    if radius == 1:
        function = dense.disk if dimensions == 2 else dense.ball
        return ((function(1, dtype=dtype, strict_radius=False), 1),)
    maximum = _PROBES[dimensions].shape[0]
    if radius > maximum:
        raise ValueError(f'precomputed {dimensions}D decomposition unavailable for radius > {maximum}')
    selected = _PROBES[dimensions][radius]
    if selected.ndim != 1:
        # Preserve NumPy advanced-index shapes and Python unpacking failures.
        a, b, c = selected
        del a, b, c
    t, diamond, square = (np.uint8(n) for n in _core.footprint_sphere_counts(
        dimensions, operator.index(radius)))
    sequence = []
    if t > 0:
        if dimensions == 2:
            first = _element(dtype, 2, 0, literal=True)
            series = (first, np.rot90(first, 1), np.rot90(first, 2), np.rot90(first, 3))
        else:
            series = tuple(_element(dtype, dimensions, 0, axis, direction)
                           for axis in range(dimensions) for direction in [0, 2])
        sequence.extend((element, t) for element in series)
    if diamond > 0:
        sequence.append((_element(dtype, dimensions, 1), diamond))
    if square > 0:
        sequence.append((dense.footprint_rectangle((3,) * dimensions, dtype=dtype), square))
    return tuple(sequence)


def footprint_rectangle(shape, *, dtype=np.uint8, decomposition=None):
    has_even_width = any(width % 2 == 0 for width in shape)
    if decomposition == 'sequence' and has_even_width:
        warnings.warn("decomposition='sequence' is only supported for uneven footprints, "
                      "falling back to decomposition='separable'", stacklevel=2)
        decomposition = 'sequence_fallback'
    def partial(dim, width):
        target = (1,) * dim + (width,) + (1,) * (len(shape) - dim - 1)
        return dense._generate(target, dtype, 0), 1
    if decomposition is None:
        return dense._generate(shape, dtype, 0)
    if decomposition in ('separable', 'sequence_fallback'):
        return tuple(partial(dim, width) for dim, width in enumerate(shape))
    if decomposition == 'sequence':
        minimum = min(shape)
        repetitions = 1 + (minimum - 3) // 2
        sequence = [(dense._generate((3,) * len(shape), dtype, 0), repetitions)]
        for dim, width in enumerate(shape):
            if width > minimum:
                sequence.append(partial(dim, width - minimum + 1))
        return tuple(sequence)
    raise ValueError(f'Unrecognized decomposition: {decomposition}')


def diamond(radius, dtype=np.uint8, *, decomposition=None):
    if decomposition is None:
        return dense.diamond(radius, dtype=dtype)
    if decomposition == 'sequence':
        footprint = dense.diamond(1, dtype=dtype)
        return ((footprint, 1 + ((2 * radius + 1) - footprint.shape[0]) // 2),)
    raise ValueError(f'Unrecognized decomposition: {decomposition}')


def _crosses(radius, dtype, strict_radius):
    # Sparse native counts apply to numeric Boolean geometry. Unusual dtypes
    # must undergo the original integer-sum conversion first (including errors).
    coordinates = np.arange(-radius, radius + 1)
    if not strict_radius:
        radius += .5
    costs = np.ascontiguousarray(coordinates ** 2, dtype=np.float64)
    limit = float(np.asarray(radius ** 2).item())
    resolved = np.dtype(bool if dtype is None else dtype)
    if resolved.kind in 'biuf':
        plans = _core.footprint_cross_costs(costs, limit)
    else:
        mask = np.empty((coordinates.size,) * 2, dtype=bool)
        zero, one = np.array(False), np.array(True)
        _core.footprint_dense_into(mask, costs, 2, limit, zero, one)
        footprint = np.array(mask, dtype=dtype)
        quadrant = footprint[footprint.shape[0] // 2:, footprint.shape[1] // 2:]
        sums = np.ascontiguousarray(quadrant.sum(0, dtype=int))
        plans = _core.footprint_cross_sums(sums, quadrant.shape[0])
    return tuple((_element(np.uint8, 2, 2, vertical=v, horizontal=h), n)
                 for v, h, n in plans)


def disk(radius, dtype=np.uint8, *, strict_radius=True, decomposition=None):
    if decomposition is None:
        return dense.disk(radius, dtype=dtype, strict_radius=strict_radius)
    elif decomposition == 'sequence':
        sequence = _sphere(radius, 2, dtype)
    elif decomposition == 'crosses':
        sequence = _crosses(radius, dtype, strict_radius)
    return sequence


def ball(radius, dtype=np.uint8, *, strict_radius=True, decomposition=None):
    if decomposition is None:
        return dense.ball(radius, dtype=dtype, strict_radius=strict_radius)
    elif decomposition == 'sequence':
        sequence = _sphere(radius, 3, dtype)
    else:
        raise ValueError(f'Unrecognized decomposition: {decomposition}')
    return sequence
