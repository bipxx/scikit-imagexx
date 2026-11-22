"""Bounded match_descriptors public hot-swap and indexed compatibility tests."""

import inspect
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest

import skimagexx as xx
from skimagexx import _match_descriptors
from skimagexx._compat import FallbackWarning

from fixtures.match_descriptors_reference_v1 import (
    assert_fixed_inputs, assert_result, contract, controls, digest, fixed_cases,
)


_CONTRACT = contract()
_CASES = {name: (left, right, kwargs) for name, left, right, kwargs in (*fixed_cases(), *controls())}
_NATIVE = [case["id"] for case in _CONTRACT["fixed_cases"]] + _CONTRACT["native_controls"]
_DEFERRED = tuple(name for name in _CASES if name not in _NATIVE)
_DESCRIPTORS = json.loads(
    (Path(__file__).with_name("fixtures") / "match_descriptors_dtypes_v1.json").read_bytes()
)["rows"]


@pytest.fixture
def public():
    public = xx.feature.match_descriptors
    assert public is _match_descriptors.match_descriptors
    assert str(inspect.signature(public)) == _CONTRACT["signature"]
    before = xx.set_fallback_enabled(False)
    workers = xx.get_num_threads()
    xx.set_num_threads(1)
    try:
        yield public
    finally:
        xx.set_num_threads(workers)
        xx.set_fallback_enabled(before)


def _call_native(public, monkeypatch, left, right, kwargs):
    from skimage import feature as original_feature

    def forbidden(*args, **kwargs):
        pytest.fail("native descriptor matching delegated to the original public function")

    monkeypatch.setattr(original_feature, "match_descriptors", forbidden)
    entry = xx._core.match_descriptors_bool
    assert inspect.isbuiltin(entry) and entry.__module__ == "skimagexx._core"
    calls = []

    def profile(frame, event, value):
        if event == "c_return" and value is entry:
            calls.append(value)

    prior_profile = sys.getprofile()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sys.setprofile(profile)
        try:
            result = public(left, right, **kwargs)
        finally:
            sys.setprofile(prior_profile)
    assert calls == [entry] and xx.get_num_threads() == 1
    return result


@pytest.mark.parametrize("name", _NATIVE)
def test_match_descriptors_installed(name, public, monkeypatch):
    left, right, kwargs = _CASES[name]
    case = _CONTRACT["observations"][name]
    assert_fixed_inputs(case, left, right)
    result = _call_native(public, monkeypatch, left, right, kwargs)
    assert_result(result, case, left, right)
    assert_fixed_inputs(case, left, right)


def _descriptor_snapshot(value):
    def state(array):
        return dict(
            type=f"{type(array).__module__}.{type(array).__name__}", shape=list(array.shape),
            strides=list(array.strides), dtype=array.dtype.str,
            metadata=None if array.dtype.metadata is None else dict(array.dtype.metadata),
            owns=array.flags.owndata, c=array.flags.c_contiguous, f=array.flags.f_contiguous,
            aligned=array.flags.aligned, writeable=array.flags.writeable, bytes_sha256=digest(array),
        )

    return state(value) | dict(
        base_type=type(value.base).__name__,
        base=state(value.base) if type(value.base) is np.ndarray else None,
    )


@pytest.mark.parametrize(
    "row", _DESCRIPTORS,
    ids=lambda row: "-".join([row["case"], row["descriptor"], *row["changed_sides"]]),
)
def test_match_descriptors_dtype_descriptors(row, public, monkeypatch):
    left, right, kwargs = _CASES[row["case"]]
    inputs = []
    for side, field, expected in zip(("left", "right"), (left, right), row["inputs"], strict=True):
        if side in row["changed_sides"]:
            plain = field.dtype
            descriptor = (plain.newbyteorder("=") if row["descriptor"] == "equivalent"
                          else np.dtype(plain, metadata=expected["metadata"]))
            assert descriptor == plain and descriptor is not plain
            field = field.view(descriptor)
        inputs.append(field)
    assert [_descriptor_snapshot(field) for field in inputs] == row["inputs"]
    result = _call_native(public, monkeypatch, *inputs, kwargs)
    assert_result(result, _CONTRACT["observations"][row["case"]], *inputs)
    assert _descriptor_snapshot(result) == row["exact_original_output"]
    assert [_descriptor_snapshot(field) for field in inputs] == row["inputs"]


@pytest.mark.parametrize("name", _DEFERRED)
def test_match_descriptors_compatibility(name, public, monkeypatch):
    from skimage import feature as original_feature

    left, right, options = _CASES[name]
    with pytest.raises(NotImplementedError, match="fallback is disabled"):
        public(left, right, **options)

    received = []
    sentinel = object()

    def original(a, b, **kwargs):
        received.append((a, b, kwargs))
        return sentinel

    monkeypatch.setattr(original_feature, "match_descriptors", original)
    xx.set_fallback_enabled(True)
    with pytest.warns(FallbackWarning, match="skimage.feature.match_descriptors"):
        assert public(left, right, **options) is sentinel
    (a, b, forwarded), = received
    assert a is left and b is right
    expected = {"metric": None, "p": 2, "max_distance": float("inf"),
                "cross_check": True, "max_ratio": 1.0}
    assert forwarded == (expected | options)


def test_match_descriptors_array_protocol_boundaries(public):
    ordinary = np.zeros((2, 4), bool)
    malformed_owner = np.lib.stride_tricks.as_strided(ordinary[:1], shape=(3, 4), strides=(4, 1))
    assert malformed_owner.flags.c_contiguous
    variants = (
        ordinary.view(np.ndarray),  # bounded native view
        ordinary.view(type("ArraySubclass", (np.ndarray,), {})),
        np.arange(8, dtype=np.uint8).reshape(2, 4).view(np.dtype(bool, metadata={"marker": True})),
        malformed_owner,
        np.zeros((1025, 4), bool),
    )
    assert public(variants[0], ordinary).tolist() == [[0, 0]]
    for variant in variants[1:]:
        with pytest.raises(NotImplementedError, match="fallback is disabled"):
            public(variant, ordinary)
    too_wide = np.zeros((2, 257), bool)
    with pytest.raises(NotImplementedError, match="fallback is disabled"):
        public(too_wide, too_wide)

    for options in ({"max_distance": np.float64(0.5)}, {"max_ratio": float("nan")},
                    {"max_distance": -1.0}, {"max_ratio": 2.0}, {"cross_check": np.bool_(True)}):
        with pytest.raises(NotImplementedError, match="fallback is disabled"):
            public(ordinary, ordinary, **options)
