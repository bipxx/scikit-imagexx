import inspect

import numpy as np
import pytest
from skimage import transform as original
from skimagexx import FallbackWarning, set_fallback_enabled, transform as candidate

from skimagexx import _epipolar

NAMES = ("FundamentalMatrixTransform", "EssentialMatrixTransform")


@pytest.mark.parametrize("name", NAMES)
def test_class_contract(name):
    expected = getattr(original, name)
    actual = getattr(candidate, name)
    assert actual is getattr(_epipolar, name) and actual is not expected
    assert issubclass(actual, expected)
    assert set(dir(actual)) == set(dir(expected))
    assert inspect.signature(actual) == inspect.signature(expected)
    for member in dir(expected):
        baseline = getattr(expected, member)
        if isinstance(baseline, property):
            assert inspect.signature(getattr(actual, member).fget) == inspect.signature(baseline.fget)
        elif callable(baseline):
            try:
                signature = inspect.signature(baseline)
            except (TypeError, ValueError):
                continue
            assert inspect.signature(getattr(actual, member)) == signature
    model = actual.identity()
    assert type(model.inverse) is actual
    for cls in (expected, actual):
        with pytest.raises(AttributeError, match="matrix"):
            _ = cls.identity().dimensionality


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("dtype", [np.float64, np.dtype("d")])
@pytest.mark.parametrize("count", [0, 9, 1025])
def test_apply_and_residuals_strict(name, dtype, count):
    rng = np.random.default_rng(103)
    matrix = rng.normal(size=(3, 3)).astype(dtype)
    src, dst = rng.normal(size=(2, count, 2)).astype(dtype)
    actual = getattr(candidate, name)(matrix=matrix)
    expected = getattr(original, name)(matrix=matrix)
    previous = set_fallback_enabled(False)
    try:
        for lhs, rhs in [(actual(src), expected(src)),
                         (actual.residuals(src, dst), expected.residuals(src, dst))]:
            assert lhs.dtype == rhs.dtype and lhs.shape == rhs.shape
            np.testing.assert_allclose(lhs, rhs, atol=2e-13, rtol=2e-13)
    finally:
        set_fallback_enabled(previous)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("count", [8, 25])
@pytest.mark.parametrize("scaling", ["rms", "mrs", "raw"])
def test_fit_and_deprecated_estimate(name, count, scaling):
    rng = np.random.default_rng(873)
    inputs = rng.normal(size=(2, count, 2))
    current = getattr(candidate, name)()
    baseline = getattr(original, name)()
    current.scaling = baseline.scaling = scaling
    previous = set_fallback_enabled(False)
    try:
        with pytest.warns(FutureWarning, match="estimate.*deprecated"):
            assert current.estimate(*inputs)
        with pytest.warns(FutureWarning):
            assert baseline.estimate(*inputs)
        np.testing.assert_allclose(current.params, baseline.params, atol=5e-13, rtol=5e-13)
        fitted = type(current).from_estimate(*inputs)
        reference = type(baseline).from_estimate(*inputs)
        np.testing.assert_allclose(fitted.params, reference.params, atol=5e-13, rtol=5e-13)
    finally:
        set_fallback_enabled(previous)


@pytest.mark.parametrize("name", NAMES)
def test_outside_domain_metadata_and_failed_fit(name, monkeypatch):
    rng = np.random.default_rng(801)
    matrix = rng.normal(size=(3, 3))
    points = rng.normal(size=(12, 2)).view(np.dtype("d", metadata={"coordinates": "xy"}))
    actual = getattr(candidate, name)(matrix=matrix)
    expected = getattr(original, name)(matrix=matrix)
    monkeypatch.setattr(original, name, type(actual))
    with pytest.warns(FallbackWarning):
        result = actual(points)
    baseline = expected(points)
    np.testing.assert_array_equal(result, baseline)
    assert result.dtype.metadata == baseline.dtype.metadata
    src = np.zeros((8, 2))
    with pytest.warns(FallbackWarning):
        failed = type(actual).from_estimate(src, src)
    expected_failed = type(expected).from_estimate(src, src)
    assert type(failed) is type(expected_failed) and str(failed) == str(expected_failed)


@pytest.mark.parametrize("name", NAMES)
def test_application_and_constraints_survive_upstream_method_swap(name, monkeypatch):
    rng = np.random.default_rng(22)
    src, dst = rng.normal(size=(2, 25, 2))
    expected_cls = getattr(original, name)
    expected = expected_cls.from_estimate(src, dst)
    lines, residuals = expected(src), expected.residuals(src, dst)

    def unexpected(*args, **kwargs):
        pytest.fail("native path called a stock computation")

    for member in ("__call__", "residuals", "_setup_constraint_matrix", "_estimate"):
        monkeypatch.setattr(original.FundamentalMatrixTransform, member, unexpected)
    monkeypatch.setattr(original.EssentialMatrixTransform, "_estimate", unexpected)
    actual = getattr(candidate, name).from_estimate(src, dst)
    np.testing.assert_allclose(actual(src), lines, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(actual.residuals(src, dst), residuals, atol=1e-12, rtol=1e-12)
