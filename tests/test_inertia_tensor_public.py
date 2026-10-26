"""Test-first inertia tensor; reference capture precedes all native traps.

The ordinary real tensor assembly must complete one owned inertia_tensor call.
Automatic moments must reuse the existing owned per-axis projection. NumPy
coordinates, exact analytical stages and diagonal summation remain explicit
providers; exceptional/integer/complex assembly is provider-only in this packet.
"""
import argparse
import collections
import inspect
import json
from pathlib import Path
import sys

import pytest

if __package__:
    from .inertia_tensor_cases import API, STAGE, UNASSESSED, cases, error_limits, fixture
    from .test_moments_public import original_runtime, metadata
    from .test_moments_derived_public import observe, result_record
else:
    from inertia_tensor_cases import API, STAGE, UNASSESSED, cases, error_limits, fixture
    from test_moments_public import original_runtime, metadata
    from test_moments_derived_public import observe, result_record

CASES = cases()


def reference_case(spec, np, original):
    image, options = fixture(spec, np)
    captured = {}
    code = original.inertia_tensor.__code__
    prior = sys.getprofile()

    def capture(frame, event, result):
        if event == "return" and frame.f_code is code:
            captured["mu"] = frame.f_locals.get("mu")

    sys.setprofile(capture)
    try:
        observed = observe(original.inertia_tensor, image, options, spec["policy"], np)
    finally:
        sys.setprofile(prior)
    return image, options, captured.get("mu"), observed


def validate_reference(spec, image, mu, observed, np):
    result, error, messages, checks = observed
    assert all(checks.values())
    assert (error is None) == (spec["outcome"] == "return"), (spec["id"], error)
    bound = None
    if error is None:
        assert type(result) is np.ndarray and result.shape == (image.ndim, image.ndim)
        assert result.flags.owndata and result.flags.c_contiguous and result.flags.writeable
        with np.errstate(all="ignore"):
            bound = error_limits(spec, image, mu, result, np)
        finite = np.isfinite(result)
        assert np.all(np.isfinite(bound[finite]) & (bound[finite] >= 0))
    return bound


def compare(actual, expected, bound, spec, np):
    assert metadata(actual) == metadata(expected), spec["id"]
    assert actual.flags.c_contiguous == expected.flags.c_contiguous
    assert actual.flags.f_contiguous == expected.flags.f_contiguous
    if expected.dtype.kind not in "fc":
        np.testing.assert_array_equal(actual, expected)
        return
    components = ((actual.real, expected.real), (actual.imag, expected.imag)) if expected.dtype.kind == "c" else ((actual, expected),)
    for a, e in components:
        for mask in (np.isnan, np.isposinf, np.isneginf):
            np.testing.assert_array_equal(mask(a), mask(e))
        finite = np.isfinite(e)
        assert np.all(np.abs(a[finite]-e[finite]) <= bound[finite]), spec["id"]
        exact = (bound == 0) & finite
        np.testing.assert_array_equal(a[exact], e[exact])
        np.testing.assert_array_equal(np.signbit(a[exact]), np.signbit(e[exact]))
        # The source repeats the same expression for each symmetric pair.
        np.testing.assert_array_equal(a, a.T)
        np.testing.assert_array_equal(np.signbit(a), np.signbit(a.T))


@pytest.fixture(scope="module")
def reference_runtime():
    return original_runtime()


@pytest.fixture(scope="module")
def worker_results():
    return {}


def test_inertia_tensor_signature(reference_runtime):
    _, original, _ = reference_runtime
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    assert API in native_functions(), "inertia_tensor remains pending"
    assert inspect.signature(xx.measure.inertia_tensor) == inspect.signature(original.inertia_tensor)
    assert xx.measure.inertia_tensor is not original.inertia_tensor


@pytest.mark.parametrize("spec", CASES, ids=[row["id"] for row in CASES])
@pytest.mark.parametrize("workers", (1, 4))
def test_inertia_tensor_owned(reference_runtime, spec, workers, worker_results, monkeypatch, tmp_path):
    np, original, _ = reference_runtime
    image, options, reference_mu, observed = reference_case(spec, np, original)
    expected, error, messages, checks = observed
    original_row = dict(id=spec["id"], image=result_record(image), mu=result_record(reference_mu),
                        result=result_record(expected), error=error, warnings=messages, checks=checks)
    (tmp_path / "original.json").write_text(json.dumps(original_row) + "\n")
    bound = validate_reference(spec, image, reference_mu, observed, np)
    image, options = fixture(spec, np)
    import skimage.measure as public_original
    from skimage.measure import _moments_analytical as analytical
    import skimagexx as xx
    from skimagexx import _compat
    from skimagexx.coverage import native_functions
    assert API in native_functions(), "inertia_tensor remains pending"
    assert xx.get_num_threads() == 1
    previous = xx.set_fallback_enabled(False)
    prior_profile = sys.getprofile()
    completions, forbidden = [], []

    def deny(*args, **kwargs):
        forbidden.append(True)
        raise AssertionError("upstream inertia/moment or contraction forwarding")

    def trace(frame, event, arg):
        if (event == "c_return" and getattr(arg, "__module__", None) == xx._core.__name__
                and getattr(arg, "__name__", None) in (STAGE, "moments_project")):
            completions.append(arg.__name__)

    try:
        xx.set_num_threads(workers)
        with monkeypatch.context() as patch:
            for name in ("inertia_tensor", "inertia_tensor_eigvals", "moments", "moments_central", "centroid"):
                patch.setattr(public_original, name, deny)
                patch.setattr(original, name, deny)
            patch.setattr(original, "moments_raw_to_central", deny)
            for name in ("moments_raw_to_central", "_moments_raw_to_central_fast"):
                patch.setattr(analytical, name, deny)
            for name in ("einsum", "tensordot", "dot", "matmul"):
                patch.setattr(np, name, deny)
            sys.setprofile(trace)
            try:
                actual, actual_error, actual_messages, actual_checks = observe(
                    xx.measure.inertia_tensor, image, options, spec["policy"], np)
            finally:
                sys.setprofile(prior_profile)
        row = dict(id=spec["id"], workers=workers, result=result_record(actual), error=actual_error,
                   warnings=actual_messages, checks=actual_checks, actual_C_returns=completions,
                   declared_route=spec["route"], tensor_arithmetic_credit=(completions.count(STAGE) if spec["route"] == "tensor" else 0),
                   forbidden=forbidden)
        (tmp_path / "candidate.json").write_text(json.dumps(row) + "\n")
        assert not forbidden and all(actual_checks.values())
        assert actual_error == error and actual_messages == messages, spec["id"]
        if error is None:
            compare(actual, expected, bound, spec, np)
            if spec["route"] == "tensor":
                assert completions.count(STAGE) == 1, "ordinary tensor assembly must complete owned arithmetic"
            assert completions.count("moments_project") == (image.ndim if spec["mu_dtype"] is None else 0)
        words = (None if error else (metadata(actual), actual.tobytes(order="C")), actual_error, actual_messages)
        assert worker_results.setdefault(spec["id"], words) == words
        assert xx.get_num_threads() == workers and _compat._fallback_enabled is False
    finally:
        sys.setprofile(prior_profile)
        xx.set_num_threads(1)
        xx.set_fallback_enabled(previous)


def reference_only(output):
    assert not any(name == "skimagexx" or name.startswith("skimagexx.") for name in sys.modules)
    output.mkdir()
    np, original, record = original_runtime()
    (output / "runtime.json").write_text(json.dumps(record, indent=2) + "\n")
    counts = collections.Counter()
    bounds = []
    with (output / "journal.jsonl").open("x") as stream:
        for spec in CASES:
            image, options, mu, observed = reference_case(spec, np, original)
            result, error, messages, checks = observed
            row = dict(id=spec["id"], image=result_record(image), supplied_mu=result_record(options.get("mu")),
                       captured_mu=result_record(mu), result=result_record(result),
                       error=error, warnings=messages, checks=checks, declared_route=spec["route"])
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            bound = validate_reference(spec, image, mu, observed, np)
            bounds.append(dict(id=spec["id"], bound=result_record(bound)))
            counts["errors" if error else "returns"] += 1
            if error is None:
                compare(result, result, bound, spec, np)
    (output / "bounds.json").write_text(json.dumps(bounds, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(dict(status="ORIGINAL_ONLY", cases=len(CASES),
        observations=dict(counts), native_calls=0, native_coverage=0, unassessed=UNASSESSED), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-only", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference_only(args.output)
