import inspect

import numpy as np
import pytest
from skimage.transform import PolynomialTransform as Original
from skimage import transform as original
from skimagexx import FallbackWarning, set_fallback_enabled
from skimagexx._polynomial import PolynomialTransform as Implementation
from skimagexx.transform import PolynomialTransform as Native


def test_class_contract():
    assert Native is Implementation and Native is not Original
    assert issubclass(Native, Original)
    assert set(dir(Native)) == set(dir(Original))
    assert inspect.signature(Native) == inspect.signature(Original)
    for member in dir(Original):
        baseline = getattr(Original, member)
        if isinstance(baseline, property):
            assert inspect.signature(getattr(Native, member).fget) == inspect.signature(baseline.fget)
        elif callable(baseline):
            try:
                signature = inspect.signature(baseline)
            except (TypeError, ValueError):
                continue
            assert inspect.signature(getattr(Native, member)) == signature
    assert type(Native.identity()) is Native
    with pytest.raises(NotImplementedError, match="inverse"):
        _ = Native.identity().inverse


@pytest.mark.parametrize("order", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("count", [0, 9, 1025])
def test_apply_strict(order, count):
    rng = np.random.default_rng(42)
    points = rng.uniform(-1.0, 1.0, (count, 2)).astype(np.dtype("d"))
    params = rng.normal(size=(2, (order + 1) * (order + 2) // 2))
    previous = set_fallback_enabled(False)
    try:
        actual = Native(params)(points)
        baseline = Original(params)(points)
        assert actual.shape == baseline.shape and actual.dtype == baseline.dtype
        np.testing.assert_allclose(actual, baseline, atol=2e-13, rtol=2e-13)
        np.testing.assert_array_equal(Native()(points), Original()(points))
    finally:
        set_fallback_enabled(previous)


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("weighted", [False, True])
def test_fit_strict(order, weighted):
    rng = np.random.default_rng(60)
    params = rng.normal(size=(2, (order + 1) * (order + 2) // 2))
    points = rng.uniform(-1.0, 1.0, (30, 2))
    observations = Original(params)(points) + 0.02 * rng.normal(size=points.shape)
    weights = rng.uniform(0.1, 1.0, len(points)) if weighted else None
    reference = Original.from_estimate(points, observations, order, weights)
    previous = set_fallback_enabled(False)
    try:
        fitted = Native.from_estimate(points, observations, order, weights)
        np.testing.assert_allclose(fitted.params, reference.params, atol=1e-11, rtol=1e-11)
        with pytest.warns(FutureWarning, match="estimate.*deprecated"):
            assert fitted.estimate(points, observations, order, weights)
    finally:
        set_fallback_enabled(previous)


@pytest.mark.parametrize("dtype", [np.float32, np.dtype("d", metadata={"unit": "px"})])
def test_precision_and_metadata_fallback(dtype, monkeypatch):
    rng = np.random.default_rng(903)
    points = rng.normal(size=(16, 2)).astype(dtype)
    params = rng.normal(size=(2, 6))
    monkeypatch.setattr(original, "PolynomialTransform", Native)
    previous = set_fallback_enabled(True)
    try:
        with pytest.warns(FallbackWarning):
            result = Native(params)(points)
    finally:
        set_fallback_enabled(previous)
    baseline = Original(params)(points)
    assert result.dtype.metadata == baseline.dtype.metadata
    np.testing.assert_array_equal(result, baseline)


def test_application_and_constraints_survive_upstream_method_swap(monkeypatch):
    rng = np.random.default_rng(80)
    points = rng.normal(size=(16, 2))
    target = Original(rng.normal(size=(2, 6)))(points)

    def unexpected(*args, **kwargs):
        pytest.fail("native path called a stock computation")

    monkeypatch.setattr(Original, "__call__", unexpected)
    monkeypatch.setattr(Original, "_estimate", unexpected)
    model = Native.from_estimate(points, target)
    np.testing.assert_allclose(model(points), target, atol=1e-12, rtol=1e-12)
