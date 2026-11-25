"""Fixed one-worker public BRIEF extraction cases."""

from functools import partial
import inspect
import sys
import warnings

import numpy as np


def _cases():
    for name, shape, dtype, count, options in (
        ("normal-f32-256", (384, 512), np.float32, 8192, {"sigma": 1}),
        ("uniform-f64-512", (320, 384), np.float64, 4096,
         {"descriptor_size": 512, "mode": "uniform", "sigma": 0, "rng": 37}),
    ):
        rng = np.random.default_rng(131)
        image = rng.random(shape, dtype=dtype)
        keypoints = rng.integers((26, 26), np.asarray(shape) - 26, size=(count, 2))
        yield name, image, keypoints, options


def public_call_pairs(original, candidate, native_entry):
    """Verify public outputs and native dispatch, then yield reusable extract calls."""
    assert inspect.signature(original) == inspect.signature(candidate)
    assert inspect.signature(original.extract) == inspect.signature(candidate.extract)
    assert inspect.isbuiltin(native_entry)
    for name, image, keypoints, options in _cases():
        stock = original(**options)
        native = candidate(**options)
        reference = partial(stock.extract, image, keypoints)
        accelerated = partial(native.extract, image, keypoints)
        assert reference() is None
        calls = []

        def profile(frame, event, value):
            if event == "call" and frame.f_code is original.extract.__code__:
                raise AssertionError(f"{name} entered the original BRIEF extract")
            if event == "c_return" and value is native_entry:
                calls.append(value)

        previous = sys.getprofile()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sys.setprofile(profile)
            try:
                assert accelerated() is None
            finally:
                sys.setprofile(previous)
        assert calls == [native_entry]
        for field in ("mask", "descriptors"):
            np.testing.assert_array_equal(getattr(native, field), getattr(stock, field))
        yield name, reference, accelerated
