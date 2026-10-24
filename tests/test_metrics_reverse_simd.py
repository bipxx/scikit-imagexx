"""Reverse transport: exact matched-core observations, plus original parity.

No timing. A same-toolchain frozen control is required for NaN payload checks;
this diagnostic does not extend the public original's NaN payload contract.
"""
import importlib.util
import os
from pathlib import Path
import struct

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('metrics_reverse_helpers', Path(__file__).with_name('test_metrics_reductions.py'))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


@pytest.fixture(autouse=True, params=[1, 4])
def workers(request):
    previous = helpers.native._core.get_num_threads()
    helpers.native._core.set_num_threads(request.param)
    yield
    helpers.native._core.set_num_threads(previous)


@pytest.fixture(scope='module')
def control():
    path = os.environ.get('SKIMAGEXX_METRICS_PROVEN_BASELINE')
    if not path:
        pytest.skip('exact payload diagnostic requires explicit matched frozen core')
    spec = importlib.util.spec_from_file_location('metrics_reverse_control._metrics_load_core', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    result.set_num_threads(1)
    return result


def bits(value):
    if isinstance(value, tuple):
        return tuple(bits(item) for item in value)
    if isinstance(value, np.generic):
        return type(value), value.dtype.str, value.tobytes()
    return struct.pack('=d', value) if isinstance(value, float) else value


def arrays(dtype, width, layout, pattern):
    rng = np.random.default_rng(129 + width)
    shape = (3, width)
    a = np.ldexp(rng.uniform(-1, 1, shape), rng.integers(-12, 12, shape)).astype(dtype)
    b = np.ldexp(rng.uniform(-1, 1, shape), rng.integers(-12, 12, shape)).astype(dtype)
    if pattern == 'zeros':
        a.ravel()[::3] = -0.; a.ravel()[1::5] = 0.
    if pattern == 'exceptional':
        word = np.uint32 if dtype is np.float32 else np.uint64
        payloads = ([0x7f800001, 0xff800003, 0x7fc00011, 0xffc00017, 0x7f800000, 0xff800000]
                    if dtype is np.float32 else
                    [0x7ff0000000000001, 0xfff0000000000003, 0x7ff8000000000011,
                     0xfff8000000000017, 0x7ff0000000000000, 0xfff0000000000000])
        a.view(word).ravel()[::3] = np.resize(np.array(payloads, word), a.size // 3 + bool(a.size % 3))
        b.ravel()[::7] = np.inf
    if layout == 'unaligned':
        x = np.ndarray(shape, dtype, buffer=bytearray(a.nbytes + 1), offset=1); x[:] = a; a = x
        x = np.ndarray(shape, dtype, buffer=bytearray(b.nbytes + 1), offset=1); x[:] = b; b = x
    if layout == 'endian':
        a = a.astype(a.dtype.newbyteorder('>')); b = b.astype(b.dtype.newbyteorder('>'))
    a = a[::-1, ::-1]
    if layout != 'mixed':
        b = b[::-1, ::-1]
    return a, b


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('layout', ['reverse', 'mixed', 'unaligned', 'endian'])
@pytest.mark.parametrize('pattern', ['finite', 'zeros', 'exceptional'])
def test_matched_reverse_lanes_stages_tree_and_flags(control, dtype, layout, pattern):
    for width in (1, 3, 4, 5, 7, 8, 9, 31, 32, 33, 127, 128, 129, 257, 513):
        a, b = arrays(dtype, width, layout, pattern)
        before = a.tobytes(), b.tobytes()
        for mode in range(4):
            for capture in (False, True):
                actual = helpers.native._core.metrics_reduce(a, b, dtype is np.float32, mode, capture, False)
                expected = control.metrics_reduce(a, b, dtype is np.float32, mode, capture, False)
                assert {k: bits(v) for k, v in actual.items()} == {k: bits(v) for k, v in expected.items()}
        for operation in range(4):
            actual, flags = helpers.native._core.metrics_stage(a, b, dtype is np.float32, operation)
            expected, expected_flags = control.metrics_stage(a, b, dtype is np.float32, operation)
            assert flags == expected_flags
            assert actual.dtype == expected.dtype and actual.shape == expected.shape
            assert actual.strides == expected.strides and actual.tobytes() == expected.tobytes()
        assert before == (a.tobytes(), b.tobytes())


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('policy', ['ignore', 'warn', 'raise'])
@pytest.mark.parametrize('pattern', ['finite', 'zeros', 'exceptional'])
def test_reverse_original_policy_values_and_warnings(dtype, policy, pattern):
    for layout in ('reverse', 'mixed', 'unaligned', 'endian'):
        a, b = arrays(dtype, 129, layout, pattern)
        with np.errstate(all=policy):
            for name, options in helpers.FUNCTIONS:
                helpers.check(name, a, b, **options)
