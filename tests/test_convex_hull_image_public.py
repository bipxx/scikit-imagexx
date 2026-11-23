"""Frozen planar hull public parity and compatibility controls."""

import inspect
import sys
import warnings

import numpy as np
import pytest

import skimagexx as xx
from skimagexx import _convex_hull_image
from skimagexx._compat import FallbackWarning

from fixtures.convex_hull_image_v1 import contract, controls, fixed_cases, snapshot


_CONTRACT = contract()
_CASES = {name: (image, kwargs) for name, image, kwargs in (*fixed_cases(), *controls())}
_COMPATIBILITY = {"single-no-offset", "line-no-offset", "triangle-tolerance-zero",
                  "array-zero", "shape-empty", "strided"}


@pytest.fixture
def public():
    function = xx.morphology.convex_hull_image
    assert function is _convex_hull_image.convex_hull_image
    assert str(inspect.signature(function)) == _CONTRACT["signature"]
    previous = xx.set_fallback_enabled(False)
    try:
        yield function
    finally:
        xx.set_fallback_enabled(previous)


@pytest.mark.parametrize("name", [name for name in _CASES if name not in _COMPATIBILITY])
def test_convex_hull_image_native(name, public, monkeypatch):
    from skimage import morphology as stock

    def forbidden(*args, **kwargs):
        pytest.fail("native hull called stock")

    monkeypatch.setattr(stock, "convex_hull_image", forbidden)
    image, kwargs = _CASES[name]
    expected = _CONTRACT["observations"][name]
    assert snapshot(image) == expected["input"]
    entry = xx._core.convex_hull_image_binary
    assert inspect.isbuiltin(entry) and entry.__module__ == "skimagexx._core"
    calls = []

    def profile(frame, event, arg):
        if event == "c_return" and arg is entry:
            calls.append(arg)

    previous = sys.getprofile()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sys.setprofile(profile)
        try:
            result = public(image, **kwargs)
        finally:
            sys.setprofile(previous)
    assert calls == [entry]
    assert snapshot(result) == expected["output"] and not np.shares_memory(result, image)
    assert snapshot(image) == expected["input"]


@pytest.mark.parametrize("name", sorted(_COMPATIBILITY))
def test_convex_hull_image_compatibility(name, public, monkeypatch):
    from skimage import morphology as stock

    image, kwargs = _CASES[name]
    with pytest.raises(NotImplementedError, match="fallback is disabled"):
        public(image, **kwargs)
    observed = []
    marker = object()

    def probe(value, **options):
        observed.append((value, options))
        return marker

    defaults = dict(offset_coordinates=True, tolerance=1e-10, include_borders=True)
    monkeypatch.setattr(stock, "convex_hull_image", probe)
    xx.set_fallback_enabled(True)
    with pytest.warns(FallbackWarning, match="skimage.morphology.convex_hull_image"):
        assert public(image, **kwargs) is marker
    (value, options), = observed
    assert value is image and options == (defaults | kwargs)


def test_convex_hull_image_scope(public):
    image = np.ones((6, 8), bool)
    noncanonical = np.full((6, 8), 2, np.uint8)
    unsupported = (
        (image[:, ::2], {}),
        (noncanonical, {}),
        (noncanonical.view(bool), {}),
        (image, {"include_borders": 1}),
        (image, {"tolerance": np.float64(1e-10)}),
        (np.ones((513, 1), bool), {}),
    )
    for value, kwargs in unsupported:
        with pytest.raises(NotImplementedError, match="fallback is disabled"):
            public(value, **kwargs)
