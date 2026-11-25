"""Dask overlap adapter for the public array utility."""

from functools import partial

import numpy as np

from ._compat import fallback


def _default_chunks(shape, workers):
    from math import ceil

    subdivisions = ceil(workers ** (1.0 / len(shape)))
    chunks = []
    used = 1
    for length in shape:
        regular, remainder = divmod(length, subdivisions)
        if used < workers and regular:
            chunks.append((regular,) * (subdivisions - 1) + (regular + remainder,))
        else:
            chunks.append((length,))
        used *= subdivisions
    return tuple(chunks)


def _call_ufunc(function, block):
    return function(block)


def apply_parallel(
    function,
    array,
    chunks=None,
    depth=0,
    mode=None,
    extra_arguments=(),
    extra_keywords=None,
    *,
    dtype=None,
    compute=None,
    channel_axis=None,
):
    """Apply a callback to overlapping array chunks using Dask."""
    try:
        import dask.array as da
    except ImportError:
        raise RuntimeError("Could not import 'dask'.  Please install using 'pip install dask'")

    is_dask = isinstance(array, da.Array)
    if (
        (not is_dask and type(array) is not np.ndarray)
        or array.dtype.metadata is not None
        or (isinstance(dtype, np.dtype) and dtype.metadata is not None)
    ):
        return fallback(
            'util', 'apply_parallel', function, array, chunks, depth, mode,
            extra_arguments, extra_keywords, dtype=dtype, compute=compute,
            channel_axis=channel_axis,
        )
    if compute is None:
        compute = not is_dask
    default_keywords = extra_keywords is None
    if default_keywords:
        extra_keywords = {}
    if channel_axis is not None:
        channel_axis = channel_axis % array.ndim

    if chunks is None:
        from multiprocessing import cpu_count

        try:
            workers = cpu_count()
        except NotImplementedError:
            workers = 4
        shape = array.shape
        if channel_axis is None:
            chunks = _default_chunks(shape, workers)
        else:
            spatial_shape = shape[:channel_axis] + shape[channel_axis + 1:]
            spatial_chunks = _default_chunks(spatial_shape, workers)
            chunks = (
                spatial_chunks[:channel_axis] + (shape[channel_axis],) + spatial_chunks[channel_axis:]
            )
    elif channel_axis is not None and len(chunks) == array.ndim - 1:
        chunks = list(chunks)
        chunks.insert(channel_axis, array.shape[channel_axis])
        chunks = tuple(chunks)

    if mode == 'wrap':
        mode = 'periodic'
    elif mode == 'symmetric':
        mode = 'reflect'
    elif mode == 'edge':
        mode = 'nearest'
    elif mode is None:
        mode = 'reflect'
    if channel_axis is not None:
        depth = [depth] * (array.ndim - 1) if np.isscalar(depth) else list(depth)
        if len(depth) == array.ndim - 1:
            depth.insert(channel_axis, 0)
        depth = tuple(depth)

    def wrapped_func(block):
        return function(block, *extra_arguments, **extra_keywords)

    if is_dask:
        result = array.map_overlap(wrapped_func, depth, boundary=mode, dtype=dtype)
    else:
        blocks = da.from_array(array, chunks=chunks)
        callback = wrapped_func
        if (
            compute is True
            and isinstance(function, np.ufunc)
            and getattr(np, function.__name__, None) is function
            and type(extra_arguments) is tuple and not extra_arguments
            and default_keywords
        ):
            # A stable top-level adapter lets Dask tokenize standard ufuncs
            # without serializing a dynamic closure for each eager call.
            callback = partial(_call_ufunc, function)
        if type(depth) is int and depth == 0 and type(mode) is str:
            # Dask's zero-depth overlap for one array delegates to map_blocks.
            result = blocks.map_blocks(callback, dtype=dtype)
        else:
            # The array constructed here has no second input to align.
            result = blocks.map_overlap(
                callback, depth, boundary=mode, dtype=dtype, align_arrays=False,
            )
    return result.compute() if compute else result
