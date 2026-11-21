"""Portable golden-angle parity through the installed public backend."""
import inspect
import sys
from importlib.machinery import EXTENSION_SUFFIXES

import pytest

from order_angles_golden_ratio_contract import cases, fixture


RECIPES = {row["id"]: row for row in cases()}
ORDINARY_CASES = ("singleton-zero", "dtype-float32", "eight", "sixteen")


@pytest.fixture
def strict_backend():
    import skimagexx

    previous = skimagexx.set_fallback_enabled(False)
    try:
        yield skimagexx
    finally:
        skimagexx.set_fallback_enabled(previous)


def test_golden_installed_public_and_hot_swap(strict_backend, monkeypatch):
    import numpy as np
    import skimage
    from skimagexx import _order_angles_golden_ratio as owned

    xx = strict_backend
    original = skimage.transform.order_angles_golden_ratio
    public = xx.transform.order_angles_golden_ratio
    assert public is owned.order_angles_golden_ratio
    assert "order_angles_golden_ratio" in xx.transform.__all__
    exported = {}
    exec("from skimagexx.transform import *", exported)
    assert exported["order_angles_golden_ratio"] is public
    assert inspect.isgeneratorfunction(public)
    assert inspect.signature(public) == inspect.signature(original)
    assert owned._core is xx._core
    assert xx._core.__file__.endswith(tuple(EXTENSION_SUFFIXES))
    assert xx.get_num_threads() == 1

    names = ("_order_angles_begin", "_order_angles_resume", "_order_angles_next")
    entries = {name: getattr(xx._core, name) for name in names}
    for entry in entries.values():
        assert inspect.isbuiltin(entry) and entry.__module__ == "skimagexx._core"

    args, options = fixture(RECIPES["two-forward"], np)
    start_bytes = args[0].tobytes()

    def pipeline(backend):
        return backend.transform.order_angles_golden_ratio(*args, **options)

    expected = list(pipeline(skimage))
    returns = []

    def reject_original(*args, **kwargs):
        pytest.fail("native golden-angle ordering entered scikit-image")

    def trace_entries(frame, event, value):
        if event == "c_return":
            returns.extend(name for name, entry in entries.items() if value is entry)

    monkeypatch.setattr(skimage.transform, "order_angles_golden_ratio", reject_original)
    previous_profile = sys.getprofile()
    try:
        sys.setprofile(trace_entries)
        unopened = pipeline(xx)
        assert inspect.getgeneratorstate(unopened) == "GEN_CREATED"
        unopened.close()
        assert inspect.getgeneratorstate(unopened) == "GEN_CLOSED"
        generator = pipeline(xx)
        assert inspect.getgeneratorstate(generator) == "GEN_CREATED"
        assert returns == []
        first = next(generator)
        assert inspect.getgeneratorstate(generator) == "GEN_SUSPENDED"
        assert returns == ["_order_angles_begin"]
        second = generator.send(None)
        with pytest.raises(StopIteration):
            next(generator)
        assert inspect.getgeneratorstate(generator) == "GEN_CLOSED"
    finally:
        sys.setprofile(previous_profile)

    assert [first, second] == expected
    assert type(first) is type(second) is np.int64
    assert returns == ["_order_angles_begin", "_order_angles_resume",
                       "_order_angles_next", "_order_angles_resume"]
    assert args[0].tobytes() == start_bytes
    assert xx.get_num_threads() == 1


@pytest.mark.parametrize("case", ORDINARY_CASES)
def test_golden_installed_ordinary_parity(case, strict_backend):
    import numpy as np
    from skimage.transform import order_angles_golden_ratio as original

    args, options = fixture(RECIPES[case], np)
    before = args[0].tobytes()
    expected = list(original(*args, **options))
    actual = list(strict_backend.transform.order_angles_golden_ratio(*args, **options))
    assert actual == expected
    assert all(type(index) is np.int64 for index in actual)
    assert sorted(actual) == list(range(len(args[0])))
    assert args[0].tobytes() == before


def test_golden_installed_reads_remaining_angles_live(strict_backend):
    import numpy as np
    from skimage.transform import order_angles_golden_ratio as original

    (theta,), options = fixture(RECIPES["three"], np)
    reference_theta = theta.copy()
    expected = original(reference_theta, **options)
    actual = strict_backend.transform.order_angles_golden_ratio(theta, **options)
    assert next(actual) == next(expected) == 0
    theta[1] = reference_theta[1] = 5.0
    assert list(actual) == list(expected) == [2, 1]


def test_golden_installed_preserves_deferred_list_error(strict_backend):
    import numpy as np
    from skimage.transform import order_angles_golden_ratio as original

    args, options = fixture(RECIPES["list-lazy-fancy-index-error"], np)
    expected = original(*args, **options)
    actual = strict_backend.transform.order_angles_golden_ratio(*args, **options)
    assert next(actual) == next(expected) == 0
    with pytest.raises(TypeError) as expected_error:
        next(expected)
    with pytest.raises(TypeError) as actual_error:
        next(actual)
    assert str(actual_error.value) == str(expected_error.value)
    assert inspect.getgeneratorstate(actual) == "GEN_CLOSED"


@pytest.mark.parametrize("enabled", (True, False), ids=("fallback", "strict"))
def test_golden_installed_defers_fallback_or_refusal(enabled, monkeypatch):
    import numpy as np
    import skimagexx as xx
    from skimage.transform import order_angles_golden_ratio as original
    from skimagexx import _order_angles_golden_ratio as owned
    from skimagexx._compat import FallbackWarning

    fallback = owned.fallback
    theta = np.arange(17, dtype=np.float64)
    calls = []

    def observe_fallback(module, name, input_angles):
        assert input_angles is theta
        calls.append((module, name))
        return fallback(module, name, input_angles)

    monkeypatch.setattr(owned, "fallback", observe_fallback)
    previous = xx.set_fallback_enabled(enabled)
    try:
        generator = xx.transform.order_angles_golden_ratio(theta)
        assert inspect.getgeneratorstate(generator) == "GEN_CREATED" and calls == []
        if enabled:
            with pytest.warns(FallbackWarning, match="order_angles_golden_ratio executed"):
                first = next(generator)
            assert inspect.getgeneratorstate(generator) == "GEN_SUSPENDED"
            assert [first, *generator] == list(original(theta))
        else:
            with pytest.raises(NotImplementedError, match="fallback is disabled"):
                next(generator)
        assert calls == [("transform", "order_angles_golden_ratio")]
        assert inspect.getgeneratorstate(generator) == "GEN_CLOSED"
    finally:
        xx.set_fallback_enabled(previous)
