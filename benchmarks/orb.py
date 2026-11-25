"""Prepared public ORB calls with one-worker stock verification outside timing."""

from functools import partial
import inspect
import sys

import numpy as np


METHOD = dict(workers=1, samples=7, minimum_sample_ns=100_000_000, minimum_warmup_ns=250_000_000)


def _call(instance, method, image, args):
    result = getattr(instance, method)(image, *args)
    assert result is None
    return instance


def _equal(original, candidate):
    for name in ("keypoints", "scales", "orientations", "responses", "descriptors", "mask_"):
        a, b = getattr(original, name, None), getattr(candidate, name, None)
        if a is None:
            assert b is None
        else:
            assert a.dtype == b.dtype and a.shape == b.shape
            assert a.flags.owndata == b.flags.owndata and a.strides == b.strides
            if name in ("orientations", "responses"):
                np.testing.assert_allclose(a, b, atol=2e-7, rtol=2e-6)
            else:
                np.testing.assert_array_equal(a, b)


def public_call_pairs(original, candidate, native_entries):
    """Verify and yield constructor, detect, extract, and combined calls."""
    assert inspect.signature(original) == inspect.signature(candidate)
    assert all(inspect.isbuiltin(entry) for entry in native_entries)
    stock_init, native_init = partial(original, n_keypoints=350), partial(candidate, n_keypoints=350)
    expected, actual = stock_init(), native_init()
    assert vars(expected) == vars(actual)
    yield "constructor", stock_init, native_init

    from skimage import data

    rng = np.random.default_rng(17141)
    cases = (("coins_uint8", data.coins()), ("noise_float32", rng.random((286, 315), dtype=np.float32)))
    for label, image in cases:
        known = stock_init()
        known.detect(image)
        inputs = (known.keypoints, known.scales, known.orientations)
        source = image.tobytes(), tuple(value.tobytes() for value in inputs)
        for method in ("detect", "extract", "detect_and_extract"):
            args = inputs if method == "extract" else ()
            stock_call = partial(_call, stock_init(), method, image, args)
            native_call = partial(_call, native_init(), method, image, args)
            expected = stock_call()
            calls = set()
            stock_code = getattr(original, method).__code__

            def profile(frame, event, value):
                assert event != "call" or frame.f_code is not stock_code
                if event == "c_return" and value in native_entries:
                    calls.add(value.__name__)

            previous = sys.getprofile()
            sys.setprofile(profile)
            try:
                actual = native_call()
            finally:
                sys.setprofile(previous)
            required = {"orb_image", "orb_reduce"}
            if method != "extract":
                required.add("orb_detect_octave")
            if method != "detect":
                required.add("orb_extract_octave")
            assert calls == required
            _equal(expected, actual)
            assert source == (image.tobytes(), tuple(value.tobytes() for value in inputs))
            yield f"{label}.{method}", stock_call, native_call
