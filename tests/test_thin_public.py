"""Result, storage, iteration and compatibility checks against frozen thin."""

import inspect
import sys
import warnings

import numpy as np
import pytest

import skimagexx as xx
from skimagexx import _thin
from skimagexx._compat import FallbackWarning

from fixtures.thin_reference_v1 import contract, controls, fixed_cases, snapshot


_CONTRACT = contract()
_FALLBACK = {"negative-cap", "fractional-cap", "strided", "noncanonical-bool", "empty", "rank-one"}
_CALLS = {name: (value, limit) for name, value, limit in (*fixed_cases(), *controls())}


@pytest.fixture
def public():
    target = xx.morphology.thin
    assert target is _thin.thin
    assert str(inspect.signature(target)) == _CONTRACT["signature"]
    fallback = xx.set_fallback_enabled(False)
    threads = xx.get_num_threads()
    xx.set_num_threads(1)
    try:
        yield target
    finally:
        xx.set_num_threads(threads)
        xx.set_fallback_enabled(fallback)


@pytest.mark.parametrize("name", [name for name in _CALLS if name not in _FALLBACK])
def test_thin_native(name, public, monkeypatch):
    from skimage import morphology as original

    def forbidden(*args, **kwargs):
        pytest.fail("native thin delegated to the original public function")

    monkeypatch.setattr(original, "thin", forbidden)
    entry = xx._core.thin_binary
    assert inspect.isbuiltin(entry) and entry.__module__ == "skimagexx._core"
    returns = []

    def profile(frame, event, value):
        if event == "c_return" and value is entry:
            returns.append(value)

    image, max_num_iter = _CALLS[name]
    expected = _CONTRACT["observations"][name]
    assert snapshot(image) == expected["input"]
    previous = sys.getprofile()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sys.setprofile(profile)
        try:
            result = public(image, max_num_iter=max_num_iter)
        finally:
            sys.setprofile(previous)
    assert returns == [entry] and xx.get_num_threads() == 1
    assert snapshot(result) == expected["output"]
    assert not np.shares_memory(result, image)
    assert snapshot(image) == expected["input"]


@pytest.mark.parametrize("name", [name for name in _CALLS if name in _FALLBACK])
def test_thin_compatibility(name, public, monkeypatch):
    from skimage import morphology as original

    image, limit = _CALLS[name]
    with pytest.raises(NotImplementedError, match="fallback is disabled"):
        public(image, max_num_iter=limit)
    calls = []
    marker = object()

    def provider(value, max_num_iter=None):
        calls.append((value, max_num_iter))
        return marker

    monkeypatch.setattr(original, "thin", provider)
    xx.set_fallback_enabled(True)
    with pytest.warns(FallbackWarning, match="skimage.morphology.thin"):
        assert public(image, max_num_iter=limit) is marker
    assert len(calls) == 1 and calls[0][0] is image and calls[0][1] == limit


def test_thin_public_bounds(public):
    image = np.ones((4, 8), bool)
    unsupported = (
        (np.ones((513, 1), bool), None),
        (image.astype(np.int8), None),
        (image, True),
        (image, 513),
        (image.view(type("ArraySubclass", (np.ndarray,), {})), None),
        (np.lib.stride_tricks.as_strided(image[:1], shape=(10, 8), strides=(8, 1)), None),
    )
    for value, limit in unsupported:
        with pytest.raises(NotImplementedError, match="fallback is disabled"):
            public(value, max_num_iter=limit)
