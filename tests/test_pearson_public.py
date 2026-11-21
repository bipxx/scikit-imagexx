"""Parity and explicit routing for the bounded native Pearson family."""

import inspect
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest
from skimage import measure as original_measure

original = original_measure.pearson_corr_coeff

import skimagexx as xx
from skimagexx import _pearson
from skimagexx._compat import FallbackWarning

from fixtures.pearson_reference_v1 import benchmark_cases, descriptor, semantic_cases


_FROZEN = Path(__file__).with_name("fixtures") / "pearson_original_windows_v1.json"
_ADMITTED = {
    "plain-f64", "readonly", "two-positive", "two-negative", "three",
    "zero-correlation", "mask-bool", "mask-all", "mask-two", "mask-three",
    "nan-excluded",
}
_CASES = tuple(semantic_cases())
_NATIVE_CASES = tuple(case for case in _CASES if case[0] in _ADMITTED) + tuple(benchmark_cases())
_DEFERRED = tuple(case for case in _CASES if case[0] not in _ADMITTED)


@pytest.fixture
def fallback_state():
    previous = xx.set_fallback_enabled(False)
    try:
        yield
    finally:
        xx.set_fallback_enabled(previous)


@pytest.mark.parametrize("case", _NATIVE_CASES, ids=lambda case: case[0])
def test_pearson_native_pair(case, fallback_state, monkeypatch):
    name, left, right, mask, count, _ = case
    arrays = (left, right) if mask is None else (left, right, mask)
    before = [descriptor(array) for array in arrays]
    previous_policy = np.geterr(), np.geterrcall(), np.getbufsize()
    expected = original(*arrays)

    def forbidden(*args, **kwargs):
        pytest.fail("native Pearson unexpectedly entered compatibility")

    monkeypatch.setattr(_pearson, "fallback", forbidden)
    provider_calls = []
    provider = _pearson._beta_tail

    def counted(shape, x):
        provider_calls.append((shape, x))
        return provider(shape, x)

    monkeypatch.setattr(_pearson, "_beta_tail", counted)
    returns = []
    entry = xx._core.pearson_f64

    def profile(frame, event, value):
        if event == "c_return" and value is entry:
            returns.append(value)

    prior_profile = sys.getprofile()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error")
        sys.setprofile(profile)
        try:
            result = _pearson.pearson_corr_coeff(*arrays)
        finally:
            sys.setprofile(prior_profile)
    assert not caught
    assert returns == [entry] and inspect.isbuiltin(entry)
    assert type(result) is tuple and tuple(map(type, result)) == (float, float)
    if count == 2:
        assert result == expected and not provider_calls
    else:
        assert result[0] == pytest.approx(expected[0], rel=0, abs=2e-14), name
        assert result[1] == pytest.approx(expected[1], rel=2e-9, abs=0), name
        assert provider_calls == [(count * 0.5 - 1, (1 + abs(result[0])) * 0.5)]
    assert before == [descriptor(array) for array in arrays]
    assert previous_policy == (np.geterr(), np.geterrcall(), np.getbufsize())


@pytest.mark.parametrize("case", _DEFERRED, ids=lambda case: case[0])
def test_pearson_deferred_calls_refuse_strict_mode(case, fallback_state):
    _, left, right, mask, _, _ = case
    with pytest.raises(NotImplementedError, match="pearson_corr_coeff"):
        _pearson.pearson_corr_coeff(left, right, mask)


@pytest.mark.parametrize("name", ("mask-invalid-before-shape", "constant-warning-as-error",
                                 "rounded-beta-tail", "infinity-selected"))
def test_pearson_deferred_behavior_uses_original(name, fallback_state):
    xx.set_fallback_enabled(True)
    _, left, right, mask, _, action = next(case for case in _DEFERRED if case[0] == name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(action)
        warnings.simplefilter("always", FallbackWarning)
        if name == "mask-invalid-before-shape":
            with pytest.raises(ValueError) as result:
                _pearson.pearson_corr_coeff(left, right, mask)
            with pytest.raises(ValueError) as expected:
                original(left, right, mask)
            assert str(result.value) == str(expected.value)
        elif action == "error":
            row = next(row for row in json.loads(_FROZEN.read_bytes())["rows"] if row["id"] == name)
            with pytest.raises(Warning) as error:
                _pearson.pearson_corr_coeff(left, right, mask)
            assert type(error.value).__name__ == row["error"]["class"].rsplit(".", 1)[-1]
            assert str(error.value) == row["error"]["message"]
        else:
            actual = _pearson.pearson_corr_coeff(left, right, mask)
            expected = original(left, right, mask)
            np.testing.assert_equal(actual, expected)
    assert caught and type(caught[0].message) is FallbackWarning


def test_pearson_missing_provider_only_declines_larger_samples(fallback_state, monkeypatch):
    monkeypatch.setattr(_pearson, "_beta_tail", lambda shape, x: None)
    assert _pearson.pearson_corr_coeff(np.array([1., 2.]), np.array([3., 0.])) == (-1., 1.)
    _, left, right, _, _, _ = _CASES[0]
    with pytest.raises(NotImplementedError):
        _pearson.pearson_corr_coeff(left, right)


@pytest.mark.parametrize("left_values, right_values", (
    pytest.param((0., 1.), (0., -1.), id="two-exact"),
    pytest.param((-2., -1., 0., 1., 2.), (1., -2., 2., -2., 1. + 2**-30),
                 id="odd-near-zero"),
    # The orthogonal centered components give r = 33 / sqrt(33**2 + 12**2)
    # and a small positive tail for 25 samples, with room above the native floor.
    pytest.param((-1., -1., 1., 1.) * 6 + (0.,), (-45., -21., 21., 45.) * 6 + (0.,),
                 id="near-floor"),
))
def test_pearson_installed_hot_swap(left_values, right_values, fallback_state, monkeypatch):
    from types import SimpleNamespace

    from scipy import special

    public = xx.measure.pearson_corr_coeff
    assert public is _pearson.pearson_corr_coeff
    assert inspect.signature(public) == inspect.signature(original)
    entry = xx._core.pearson_f64
    assert inspect.isbuiltin(entry) and entry.__module__ == "skimagexx._core"

    def pipeline(backend, left, right):
        return backend.measure.pearson_corr_coeff(left, right)

    left, right = np.array(left_values), np.array(right_values)
    before = descriptor(left), descriptor(right)
    expected = pipeline(SimpleNamespace(measure=original_measure), left, right)

    provider = special.betaincc
    provider_calls = []

    def counted_beta_tail(shape_a, shape_b, x):
        provider_calls.append((shape_a, shape_b, x))
        assert xx.get_num_threads() == 1
        return provider(shape_a, shape_b, x)

    monkeypatch.setattr(special, "betaincc", counted_beta_tail)
    returns = []

    def profile(frame, event, value):
        if event == "c_return" and value is entry:
            returns.append(value)

    previous_workers = xx.get_num_threads()
    previous_profile = sys.getprofile()
    xx.set_num_threads(1)
    try:
        assert xx.get_num_threads() == 1
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sys.setprofile(profile)
            try:
                result = pipeline(xx, left, right)
            finally:
                sys.setprofile(previous_profile)
        assert xx.get_num_threads() == 1
    finally:
        xx.set_num_threads(previous_workers)

    assert returns == [entry]
    assert type(result) is tuple and tuple(map(type, result)) == (float, float)
    assert (descriptor(left), descriptor(right)) == before
    if left.size == 2:
        assert result == expected and not provider_calls
    else:
        assert result[0] == pytest.approx(expected[0], rel=0, abs=2e-14)
        assert result[1] == pytest.approx(expected[1], rel=2e-9, abs=0)
        shape = left.size * 0.5 - 1
        assert provider_calls == [(shape, shape, (1 + abs(result[0])) * 0.5)]
        assert all(type(value) is float for value in provider_calls[0])
