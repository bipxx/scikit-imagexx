"""Test-first moments contracts; reference-only mode never imports skimagexx.

NumPy coordinate powers and owned literal analytical transformations may be
providers. Whole upstream moment algorithms and NumPy contractions are denied
for native calls, which must complete the genuine moments_project C++ stage.
"""
import argparse
import collections
import hashlib
import inspect
import json
from pathlib import Path
import sys
import warnings

import pytest

if __package__:
    from .moments_cases import APIS, UNASSESSED, TrackedSequence, cases, error_limits, fixture, initialized_zeros
else:
    from moments_cases import APIS, UNASSESSED, TrackedSequence, cases, error_limits, fixture, initialized_zeros

CASES = cases()
SOURCE_PINS = {
    "_moments.py": "5caffe9e049832ab87e1c28a63798639f48146f219d13cf91f43a8beace9f257",
    "_moments_analytical.py": "d4eccbf2684a7eac5f293e90e2de16c2917233fd5c80a96ab8c33b8ebe4b27e4",
}


def original_runtime():
    import numpy as np
    import scipy
    import skimage
    from skimage.measure import _moments as original
    assert skimage.__version__ == "0.26.0"
    assert (np.__version__, scipy.__version__) in (("1.24.4", "1.15.3"), ("2.5.4", "1.18.1"))
    source = {}
    for name, expected in SOURCE_PINS.items():
        raw = Path(original.__file__).with_name(name).read_bytes()
        assert hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == expected
        source[name] = dict(raw_sha256=hashlib.sha256(raw).hexdigest(), canonical_lf_sha256=expected)
    assert np.geterr() == dict(divide="warn", over="warn", under="ignore", invalid="warn")
    return np, original, dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                             skimage=skimage.__version__, source=source)


@pytest.fixture(scope="module")
def reference_runtime():
    return original_runtime()


@pytest.fixture(scope="module")
def worker_results():
    # Both parametrized workers run every case. Compare the second arrival
    # with immutable words from the first; no repeated numerical call.
    return {}


def metadata(array):
    return (type(array).__module__, type(array).__name__, array.dtype.str, array.dtype.metadata,
            array.shape, array.strides, array.flags.owndata, array.flags.writeable,
            type(array.base).__module__, type(array.base).__name__)


def snapshot(value, np):
    if isinstance(value, np.ndarray):
        return metadata(value), value.tobytes(order="C")
    if isinstance(value, TrackedSequence):
        return "tracked", value.role, snapshot(value.values, np)
    if isinstance(value, dict):
        return type(value), [(key, snapshot(item, np)) for key, item in value.items()]
    if isinstance(value, (list, tuple)):
        return type(value), [snapshot(item, np) for item in value]
    return type(value), repr(value)


def observe(call, image, kwargs, events, policy, np):
    before = snapshot((image, kwargs), np)
    prior_policy, handler, buffer = np.geterr(), np.geterrcall(), np.getbufsize()
    result, error = None, None
    with np.errstate(**policy):
        selected = np.geterr()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = call(image, **kwargs)
            except Exception as exc:
                error = (type(exc).__module__, type(exc).__name__, str(exc))
        assert np.geterr() == selected and np.geterrcall() is handler and np.getbufsize() == buffer
    assert np.geterr() == prior_policy
    assert snapshot((image, kwargs), np) == before, "input or parameter mutation"
    messages = [(type(w.message).__module__, type(w.message).__name__, str(w.message)) for w in caught]
    return result, error, messages, list(events)


def reference_case(spec, np, original):
    image, kwargs, events = fixture(spec, np)
    observed = observe(getattr(original, spec["api"]), image, kwargs, events, spec["errstate"], np)
    expected, error, _, _ = observed
    if spec["outcome"] == "fp":
        # NP1's BLAS dot does not dispatch contraction FP flags; NP2's matmul
        # does. Retain the actual profile outcome rather than invent a common
        # warning/raise rule. Preceding coordinate-power errors remain visible.
        assert error is None or error[:2] == ("builtins", "FloatingPointError"), (spec["id"], error)
    else:
        assert (error is None) == (spec["outcome"] == "success"), (spec["id"], error)
    bound = None
    if error is None:
        assert type(expected) is np.ndarray and expected.dtype.kind in "fc", spec["id"]
        if spec["api"] == "centroid":
            assert expected.shape == (image.ndim,)
        else:
            assert expected.ndim == image.ndim
        forced = initialized_zeros(spec, expected, np)
        assert np.all(expected[forced] == 0), spec["id"]
        bound = error_limits(spec, image, expected, np)
        finite = np.isfinite(expected.real)
        if expected.dtype.kind == "c":
            finite = finite | np.isfinite(expected.imag)
        assert np.all(np.isfinite(bound[finite]) & (bound[finite] >= 0)), "finite reference has an invalid allowance"
    return observed, bound


def compare_result(actual, expected, bound, spec, np):
    assert metadata(actual) == metadata(expected)
    forced = initialized_zeros(spec, expected, np)
    if (expected.dtype.kind == "c" and spec["api"] == "moments_central"
            and spec["center"] is None and expected.ndim in (2, 3) and expected.shape[0] <= 4):
        # The fast analytical source casts the whole raw tensor to float64
        # and back. Every imaginary component is therefore structural +0.
        assert np.all(expected.imag == 0) and not np.any(np.signbit(expected.imag))
        assert np.all(actual.imag == 0) and not np.any(np.signbit(actual.imag))
    components = ((actual.real, expected.real), (actual.imag, expected.imag)) if expected.dtype.kind == "c" else ((actual, expected),)
    for a, e in components:
        assert np.array_equal(np.isnan(a), np.isnan(e))
        assert np.array_equal(np.isposinf(a), np.isposinf(e))
        assert np.array_equal(np.isneginf(a), np.isneginf(e))
        finite = np.isfinite(e)
        assert np.all(np.abs(a[finite] - e[finite]) <= bound[finite]), spec["id"]
        exact_zero = forced | ((bound == 0) & (e == 0))
        assert np.all(a[exact_zero] == 0)
        assert np.array_equal(np.signbit(a[exact_zero]), np.signbit(e[exact_zero]))


@pytest.mark.parametrize("api", APIS)
def test_moments_public_signature(reference_runtime, api):
    _, original, _ = reference_runtime
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    assert "measure." + api in native_functions(), "moments family remains native-pending"
    assert getattr(xx.measure, api) is not getattr(original, api)
    assert inspect.signature(getattr(xx.measure, api)) == inspect.signature(getattr(original, api))


@pytest.mark.parametrize("spec", CASES, ids=[spec["id"] for spec in CASES])
@pytest.mark.parametrize("workers", (1, 4), ids=("workers1", "workers4"))
def test_moments_owned_projection(reference_runtime, spec, workers, worker_results, monkeypatch):
    np, original, _ = reference_runtime
    expected_observation, bound = reference_case(spec, np, original)
    expected, expected_error, expected_warnings, expected_events = expected_observation
    image, kwargs, events = fixture(spec, np)
    import skimage.measure as original_public
    from skimage.measure import _moments_analytical as analytical
    import skimagexx as xx
    from skimagexx import _compat
    from skimagexx.coverage import native_functions
    assert "measure." + spec["api"] in native_functions(), "moments family remains native-pending"
    assert xx.get_num_threads() == 1
    previous_fallback = xx.set_fallback_enabled(False)
    old_profile = sys.getprofile()
    forbidden, completions = [], []

    def denied(name):
        def call(*args, **kwargs):
            forbidden.append(name)
            raise AssertionError("upstream moments/contraction use: " + name)
        return call

    def profile(frame, event, argument):
        if (event == "c_return" and getattr(argument, "__module__", None) == xx._core.__name__
                and getattr(argument, "__name__", None) == "moments_project"):
            completions.append("moments_project")

    try:
        xx.set_num_threads(workers)
        assert xx.get_num_threads() == workers and _compat._fallback_enabled is False
        with monkeypatch.context() as patch:
            for name in APIS:
                patch.setattr(original_public, name, denied("skimage.measure." + name))
                patch.setattr(original, name, denied("skimage.measure._moments." + name))
            patch.setattr(original, "moments_raw_to_central", denied("imported analytical helper"))
            for name in ("moments_raw_to_central", "_moments_raw_to_central_fast"):
                patch.setattr(analytical, name, denied("upstream analytical." + name))
            for name in ("einsum", "tensordot", "dot", "matmul"):
                patch.setattr(np, name, denied("numpy." + name))
            sys.setprofile(profile)
            try:
                actual, error, messages, actual_events = observe(getattr(xx.measure, spec["api"]),
                    image, kwargs, events, spec["errstate"], np)
            finally:
                sys.setprofile(old_profile)
        assert not forbidden
        assert error == expected_error and messages == expected_warnings and actual_events == expected_events, spec["id"]
        if error is None:
            assert len(completions) >= image.ndim, "successful moments require every owned axis projection"
            compare_result(actual, expected, bound, spec, np)
        words = (None if error else (metadata(actual), actual.tobytes(order="C")), error, messages, actual_events)
        key = spec["api"], spec["id"]
        assert worker_results.setdefault(key, words) == words, "worker1/4 result or observable differs"
        assert xx.get_num_threads() == workers and _compat._fallback_enabled is False
    finally:
        sys.setprofile(old_profile)
        xx.set_num_threads(1)
        xx.set_fallback_enabled(previous_fallback)


def reference_only():
    np, original, runtime = original_runtime()
    rows = []
    for spec in CASES:
        (result, error, messages, events), bound = reference_case(spec, np, original)
        rows.append(dict(id=spec["id"], api=spec["api"], error=error, warnings=messages, events=events,
            result_metadata=None if error else metadata(result),
            nonfinite=0 if error else int(np.count_nonzero(~np.isfinite(result))),
            initialized_zeros=0 if error else int(np.count_nonzero(initialized_zeros(spec, result, np)))))
    print(json.dumps(dict(status="REFERENCE_ONLY", native_coverage=0, specifications=len(rows),
        api_counts=collections.Counter(r["api"] for r in rows),
        source_returns=sum(r["error"] is None for r in rows), source_errors=sum(r["error"] is not None for r in rows),
        unassessed=UNASSESSED, runtime=runtime, observations=rows), indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-only", action="store_true", required=True)
    parser.parse_args()
    reference_only()
