"""Simulate the caller's 32-bit promotion rule; this is not Windows qualification."""
import sys

import numpy as np
import pytest
import skimage.measure as original
import skimage.measure.block as original_module
import skimagexx as xx
from skimagexx import _compat


@pytest.mark.parametrize('dtype', ('bool', 'int32', 'uint32'))
@pytest.mark.parametrize('shape,order', (((6, 2), 'F'), ((4, 6), 'F'), ((2, 6), 'C')),
                         ids=('F-unit-axis', 'F-matrix', 'C-unit-axis'))
def test_default_sum_simulated_integer_width(dtype, shape, order, monkeypatch):
    image = np.empty(shape, dtype=dtype, order=order)
    values = {'bool': (True, False, True, True),
              'int32': (2**31 - 1, 2**31 - 2, -2**31 + 3, -17),
              'uint32': (2**32 - 1, 2**32 - 2, 17, 3)}[dtype]
    for index in np.ndindex(shape):
        image[index] = values[(index[0] * shape[1] + index[1]) % len(values)]
    result_dtype = np.uint32 if dtype == 'uint32' else np.int32
    # The genuine original result, including its iterator-selected strides,
    # is computed before patching the platform-default type or any traps.
    expected = original.block_reduce(image, block_size=2,
                                     func_kwargs={'dtype': result_dtype})
    before = (image.dtype, image.shape, image.strides, image.flags.writeable,
              image.flags.owndata, type(image.base), image.tobytes())
    prior_width = np.int_
    prior_workers = xx.get_num_threads()
    prior_fallback = xx.set_fallback_enabled(False)
    prior_profile = sys.getprofile()
    returns = []

    def deny(*args, **kwargs):
        raise AssertionError('default integer sum forwarded to an upstream reducer')

    def profile(frame, event, function):
        if event == 'c_return' and function is xx._core.block_reduce_sum:
            returns.append('block_reduce_sum')

    try:
        xx.set_num_threads(1)
        with monkeypatch.context() as patch:
            patch.setattr(np, 'int_', np.int32)
            patch.setattr(np, 'sum', deny)
            patch.setattr(original, 'block_reduce', deny)
            patch.setattr(original_module, 'block_reduce', deny)
            patch.setattr(_compat, 'fallback', deny)
            sys.setprofile(profile)
            try:
                actual = xx.measure.block_reduce(image, block_size=2)
            finally:
                sys.setprofile(prior_profile)
        assert np.int_ is prior_width
        assert returns == ['block_reduce_sum']
        assert type(actual) is type(expected) is np.ndarray
        assert actual.dtype == expected.dtype == np.dtype(result_dtype)
        assert actual.dtype.metadata == expected.dtype.metadata
        assert actual.shape == expected.shape and actual.strides == expected.strides
        assert actual.flags.owndata == expected.flags.owndata is True
        assert actual.flags.writeable == expected.flags.writeable
        assert actual.flags.c_contiguous == expected.flags.c_contiguous
        assert actual.flags.f_contiguous == expected.flags.f_contiguous
        assert actual.base is expected.base is None
        assert actual.tobytes() == expected.tobytes()
        assert not np.shares_memory(actual, image)
        assert before == (image.dtype, image.shape, image.strides, image.flags.writeable,
                          image.flags.owndata, type(image.base), image.tobytes())
        assert xx.get_num_threads() == 1 and _compat._fallback_enabled is False
    finally:
        sys.setprofile(prior_profile)
        xx.set_num_threads(prior_workers)
        xx.set_fallback_enabled(prior_fallback)
