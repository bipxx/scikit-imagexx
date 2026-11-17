"""Bounded padded-copy regressions; real execution requires a reviewed core/F1.

The direct native boundary keeps signed/gapped descriptors visible, while the
public empty cases exercise the existing early return. No original calls.
"""
import builtins
import ctypes
import importlib
import inspect
import math

import pytest
from test_flood_fill_public import target, genuine_global_worker_and_caller


CASES = (
    ((5,), 'C', False), ((5,), 'reverse', False),
    ((3, 4), 'C', False), ((3, 4), 'F', True),
    ((3, 4), 'reverse', False), ((3, 4), 'step', False),
    ((2, 3, 4), 'C', False), ((2, 3, 4), 'F', True),
    ((2, 3, 4), 'outer-reverse', False),
    ((2, 3, 4), 'outer-step', True),
    ((1, 4, 1), 'C', False), ((1, 4, 1), 'F', True),
)


@pytest.fixture
def checked_call():
    from util_views_fenv import FEnv, backend
    libc, helper = backend()

    def state():
        value = FEnv()
        assert libc.fegetenv(ctypes.byref(value)) == 0
        return value

    def call(operation, *args):
        outer = state()
        before = state()
        try:
            try:
                return operation(*args)
            finally:
                after = state()
                assert (before.control, before.status, before.mxcsr) == (
                    after.control, after.status, after.mxcsr)
        finally:
            assert helper.reference_fenv_restore(ctypes.byref(outer), 32) == 0
            assert bytes(state()) == bytes(outer)
    return call


@pytest.mark.parametrize('dtype', ('uint8', 'float32'))
@pytest.mark.parametrize('tolerance', (False, True))
@pytest.mark.parametrize('shape,layout,fortran', CASES)
def test_native_padded_copy_geometry_and_FP(target, checked_call, monkeypatch,
                                          dtype, tolerance, shape, layout, fortran):
    np, module = target
    core = module._core if hasattr(module, '_core') else importlib.import_module('skimagexx._core')
    assert inspect.isbuiltin(core.flood_mask)
    physical = (tuple(2 * n for n in shape) if layout == 'step' else
                tuple(n if axis == 0 else 2 * n for axis, n in enumerate(shape))
                if layout == 'outer-step' else shape)
    backing = np.full(physical, 1, dtype=dtype, order='F' if fortran else 'C')
    image = backing
    if layout == 'reverse':
        image = image[tuple(slice(None, None, -1) for _ in shape)]
    elif layout == 'step':
        image = image[tuple(slice(None, None, 2) for _ in shape)]
    elif layout == 'outer-reverse':
        image = image[::-1]
    elif layout == 'outer-step':
        image = image[(slice(None),) + (slice(None, None, 2),) * (len(shape) - 1)]
    image.flat[-1] = 2
    footprint = np.ones((3,) * len(shape), dtype=np.uint8)
    before = backing.tobytes(order='A')
    pointer = image.ctypes.data
    policy, handler, buffer = np.geterr(), np.geterrcall(), np.getbufsize()
    expected = np.full(tuple(n + 2 for n in shape), 2, dtype=np.uint8,
                       order='F' if fortran else 'C')
    interior = tuple(slice(1, n + 1) for n in shape)
    expected[interior] = 1
    expected[tuple(n for n in shape)] = 0
    original_import = builtins.__import__

    def refused(name, *args, **kwargs):
        assert name != 'skimage' and not name.startswith('skimage.'), name
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, '__import__', refused)
        actual = checked_call(core.flood_mask, image, [0] * len(shape),
                              footprint, fortran, tolerance, 1, 1)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert actual.strides == expected.strides
    assert actual.tobytes(order='A') == expected.tobytes(order='A')
    assert actual.size == math.prod(n + 2 for n in shape)
    assert not np.shares_memory(actual, image)
    assert backing.tobytes(order='A') == before and image.ctypes.data == pointer
    assert np.geterr() == policy and np.geterrcall() is handler and np.getbufsize() == buffer


@pytest.mark.parametrize('dtype', ('uint8', 'float32'))
@pytest.mark.parametrize('order', ('C', 'F'))
def test_public_empty_keeps_early_return_and_FP(target, checked_call, dtype, order):
    np, module = target
    image = np.empty((0, 3), dtype=dtype, order=order)
    mask = checked_call(module.flood, image, (999, 999))
    filled = checked_call(module.flood_fill, image, (999, 999), 7)
    assert mask.shape == image.shape and mask.dtype == np.dtype(bool)
    assert filled.shape == image.shape and filled.dtype == image.dtype
    assert mask.tobytes() == filled.tobytes() == image.tobytes() == b''
    assert filled is not image
