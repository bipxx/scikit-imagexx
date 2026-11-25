"""Fixed one-worker public call adapter for phase cross correlation."""

from functools import partial
import inspect
import sys
import warnings

import numpy as np
from scipy.fft import fftn
from scipy.ndimage import fourier_shift


def _cases():
    rng = np.random.default_rng(20260806)
    for shape, dtype, factor, fourier in [
        ((96, 128), np.float32, 1, False),
        ((96, 128), np.float64, 8, False),
        ((128, 96), np.float32, 1, True),
        ((79, 127), np.float64, 16, True),
    ]:
        reference = rng.normal(size=shape).astype(dtype)
        if fourier:
            reference = fftn(reference)
            offset = (5.25, -9.5) if factor > 1 else (5, -9)
            moving = fourier_shift(reference, offset)
        else:
            moving = np.roll(reference, (5, -9), axis=(0, 1))
        name = f"{'fourier' if fourier else 'real'}_{dtype.__name__}_{shape[0]}x{shape[1]}_upsample{factor}"
        options = dict(upsample_factor=factor, space="fourier" if fourier else "real")
        yield name, np.ascontiguousarray(reference), np.ascontiguousarray(moving), options


def public_call_pairs(original, candidate, native_entry):
    """Validate fixed workloads and yield untimed stock/candidate callables."""
    assert inspect.signature(original) == inspect.signature(candidate)
    assert inspect.isbuiltin(native_entry) and native_entry.__module__ == "skimagexx._core"
    original_code = inspect.unwrap(original).__code__
    for name, reference, moving, options in _cases():
        snapshots = (reference.tobytes(), moving.tobytes())
        stock_call = partial(original, reference, moving, **options)
        native_call = partial(candidate, reference, moving, **options)
        expected = stock_call()
        calls = []

        def profile(frame, event, value):
            if event == "call" and frame.f_code is original_code:
                raise AssertionError(f"{name} entered the stock public function")
            if event == "c_return" and value is native_entry:
                calls.append(value)

        previous = sys.getprofile()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sys.setprofile(profile)
            try:
                actual = native_call()
            finally:
                sys.setprofile(previous)
        assert calls == [native_entry]
        assert [type(x) for x in actual] == [type(x) for x in expected]
        assert actual[0].dtype == expected[0].dtype
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_allclose(actual[1:], expected[1:], rtol=1e-6, atol=1e-6)
        assert snapshots == (reference.tobytes(), moving.tobytes())
        yield name, stock_call, native_call
