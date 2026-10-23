"""Stepped binary16 load parity, with original warnings and input ownership."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('metrics_half_load_helpers', Path(__file__).with_name('test_metrics_reductions.py'))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


@pytest.fixture(autouse=True, params=[1, 4])
def workers(request):
    previous = helpers.native._core.get_num_threads()
    helpers.native._core.set_num_threads(request.param)
    yield
    helpers.native._core.set_num_threads(previous)


def stepped(values, step, swap):
    dtype = np.dtype('>f2' if swap else 'f2')
    count = values.size
    stride = abs(step) * 2
    owner = bytearray((count - 1) * stride + 3)
    result = np.ndarray((count,), dtype, buffer=owner, offset=1 + ((count - 1) * stride if step < 0 else 0), strides=(step * 2,))
    result[:] = values
    return result


@pytest.mark.parametrize('step', [2, -2, 3, -3, 5, -5])
@pytest.mark.parametrize('swap', [False, True])
@pytest.mark.parametrize('second_dtype', [np.float16, np.float32, np.float64])
def test_every_finite_encoding_in_exact_strided_allocation(step, swap, second_dtype):
    words = np.arange(65536, dtype=np.uint16)
    finite = words[(words & 0x7c00) != 0x7c00].view(np.float16)
    first = stepped(finite, step, swap)
    second = np.zeros(first.shape, second_dtype)
    for name, options in helpers.FUNCTIONS:
        helpers.check(name, first, second, **options)


@pytest.mark.parametrize('step', [2, -3])
@pytest.mark.parametrize('swap', [False, True])
@pytest.mark.parametrize('policy', ['warn', 'raise', 'ignore'])
@pytest.mark.parametrize('pattern', ['nan-payloads', 'signed-zero', 'finite-extreme'])
def test_exception_fallback_and_zero_signs(step, swap, policy, pattern):
    bits = {'nan-payloads': [0x7c01, 0x7e03, 0xfc01, 0x7c00, 0xfc00],
            'signed-zero': [0, 0x8000, 0, 0x8000],
            'finite-extreme': [0x7bff, 0xfbff, 1, 0x8001]}[pattern]
    first = stepped(np.resize(np.array(bits, np.uint16), 129).view(np.float16), step, swap)
    second = np.zeros(first.shape, np.float16)
    with np.errstate(all=policy):
        for name, options in helpers.FUNCTIONS:
            helpers.check(name, first, second, **options)
