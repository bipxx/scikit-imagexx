"""Frozen and live checks for bounded sparse-mask Hausdorff calls."""

import inspect
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from skimage import metrics as original

import skimagexx as xx
from skimagexx import _hausdorff
from fixtures.hausdorff_reference_v1 import benchmark_cases, capture, digest, semantic_cases


_FIXTURE = Path(__file__).with_name("fixtures") / "hausdorff_original_windows_v1.json"
_CONTRACT = _FIXTURE.parents[2] / "coverage/contracts/hausdorff-family-v1.json"
_ROWS = {row["id"]: row for row in json.loads(_FIXTURE.read_bytes())["rows"]}
_CASES = tuple(semantic_cases()) + tuple(row + (True,) for row in benchmark_cases())
_OPERATIONS = ("standard", "modified", "pair")
_DEFERRED_INPUTS = {"dimension-mismatch", "dimension-mismatch-empty", "nonzero-float",
                    "strided", "list-inputs", "scalar"}


def _route(operation, namespace):
    if operation == "pair":
        return namespace.hausdorff_pair, {}
    return namespace.hausdorff_distance, {"method": operation}


@pytest.fixture
def strict_native():
    previous = xx.set_fallback_enabled(False)
    try:
        yield
    finally:
        xx.set_fallback_enabled(previous)


def _compare(value, expected, operation):
    if operation != "modified" or "result" not in value:
        assert value == expected
        return
    assert value["warnings"] == expected["warnings"]
    actual_number = value["result"]
    expected_number = expected["result"]
    assert actual_number["type"] == expected_number["type"]
    assert math.isclose(float.fromhex(actual_number["hex"]), float.fromhex(expected_number["hex"]),
                        rel_tol=2e-14, abs_tol=0.0)


@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case[0])
def test_hausdorff_owned_cases(case, operation, strict_native):
    name, a, b, _ = case
    fn, kwargs = _route(operation, _hausdorff)
    expected_fn, _ = _route(operation, original)
    deferred = name in _DEFERRED_INPUTS or (name == "nearest-tie-tree" and operation == "pair")
    if deferred:
        with pytest.raises(NotImplementedError):
            fn(a, b, **kwargs)
        return
    entry = xx._core.hausdorff_mask
    calls = []

    def profile(frame, event, value):
        if event == "c_return" and value is entry:
            calls.append(value)

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        value = capture(fn, a, b, kwargs)
    finally:
        sys.setprofile(previous)
    assert inspect.isbuiltin(entry) and calls == [entry]
    # Normalize tuples through JSON to compare the exact frozen array descriptor.
    observed = json.loads(json.dumps(value))
    _compare(observed, _ROWS[name][operation], operation)
    _compare(value, capture(expected_fn, a, b, kwargs), operation)


def test_hausdorff_frozen_reference():
    contract = json.loads(_CONTRACT.read_bytes())
    assert contract["fixture_sha256"] == digest(_FIXTURE)
    assert contract["performance"]["case_ids"] == [row[0] for row in benchmark_cases()]
    fn = _hausdorff.hausdorff_distance
    assert capture(fn, np.array(1), np.array(0), {"method": "other"}) == _ROWS["method-unknown"]["result"]


def test_hausdorff_installed_hot_swap(strict_native):
    for operation in _OPERATIONS:
        own, kwargs = _route(operation, _hausdorff)
        native, _ = _route(operation, xx.metrics)
        reference, _ = _route(operation, original)
        assert native is own and inspect.signature(native) == inspect.signature(reference)
        assert xx._core.hausdorff_mask.__module__ == "skimagexx._core"
        _, a, b = next(benchmark_cases())
        candidate = SimpleNamespace(metrics=xx.metrics)
        backend, _ = _route(operation, candidate.metrics)
        _compare(capture(backend, a, b, kwargs), capture(reference, a, b, kwargs), operation)
