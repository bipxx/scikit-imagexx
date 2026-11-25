"""One-thread public SIFT call pairs for the sustained sampler."""

from functools import partial
import inspect
import sys

import numpy as np


def _constructor(factory):
    return factory()


def _deltas(instance):
    return instance.deltas


def _result(instance, image, method):
    getattr(instance, method)(image)
    return instance.descriptors if method == "extract" else instance.keypoints


def public_call_pairs(original, candidate, native_entry):
    from skimage.data import camera, coins

    assert inspect.signature(original) == inspect.signature(candidate)
    assert inspect.isbuiltin(native_entry)
    stock, native = original(), candidate()
    assert stock.n_octaves == native.n_octaves
    np.testing.assert_array_equal(stock.deltas, native.deltas)
    yield "init", partial(_constructor, original), partial(_constructor, candidate)
    yield "deltas", partial(_deltas, stock), partial(_deltas, native)

    for name, image in (("camera-u8", camera()), ("coins-f32", coins().astype(np.float32) / 255)):
        for method in ("detect", "extract", "detect_and_extract"):
            stock, native = original(), candidate()
            if method == "extract":
                stock.detect(image)
                native.detect(image)
            stock_call = partial(_result, stock, image, method)
            native_call = partial(_result, native, image, method)
            expected = stock_call()
            calls = []

            def profile(frame, event, arg):
                if event == "call" and frame.f_code is getattr(original, method).__code__:
                    raise AssertionError("native SIFT called the original implementation")
                if event == "c_return" and arg is native_entry:
                    calls.append(arg)

            previous = sys.getprofile()
            try:
                sys.setprofile(profile)
                actual = native_call()
            finally:
                sys.setprofile(previous)
            assert calls == [native_entry]
            np.testing.assert_array_equal(actual, expected)
            if method == "detect_and_extract":
                np.testing.assert_array_equal(native.descriptors, stock.descriptors)
            yield f"{name}-{method}", stock_call, native_call
