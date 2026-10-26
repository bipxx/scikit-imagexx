"""Test-first owned window sampling contract; RED until implemented.

--reference-only evaluates the89 original specifications, never skimagexx.
Orders0/1 require real window_sample C_returns. Valid higher orders are explicit
initial strict refusals, not passes for original numerical parity. SciPy's1D
get_window and NumPy coordinate/clipping providers are allowed; whole window,
warp and ndimage interpolation forwarding are denied during native calls.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import warnings

import pytest

if __package__:
    from .window_cases import BLOCKED_INTAKE, cases, error_limits, fixture
else:
    from window_cases import BLOCKED_INTAKE, cases, error_limits, fixture

CASES = cases()
SOURCE_SHA256 = "c4eff2c7b0d21174c1216c0f85796c28d8ab99f1fa2a873b93952cffa035c1f6"


def original_runtime():
    import numpy as np
    import scipy
    import skimage
    from skimage.filters import _window as original
    assert skimage.__version__ == "0.26.0"
    assert (np.__version__, scipy.__version__) in (("1.24.4", "1.15.3"), ("2.5.4", "1.18.1"))
    raw = Path(original.__file__).read_bytes()
    assert hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == SOURCE_SHA256
    assert np.geterr() == dict(divide="warn", over="warn", under="ignore", invalid="warn")
    return np, original, dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                             skimage=skimage.__version__, original_raw_sha256=hashlib.sha256(raw).hexdigest(),
                             original_canonical_lf_sha256=SOURCE_SHA256)


@pytest.fixture(scope="module")
def reference_runtime():
    return original_runtime()


def metadata(array):
    return (type(array), array.dtype.str, array.dtype.metadata, array.shape, array.strides,
            array.flags.owndata, array.flags.writeable, type(array.base))


def snapshot(value, np):
    if isinstance(value, np.ndarray):
        return metadata(value), value.tobytes(order="C")
    if isinstance(value, dict):
        return type(value), [(key, snapshot(item, np)) for key, item in value.items()]
    if isinstance(value, (list, tuple)):
        return type(value), [snapshot(item, np) for item in value]
    return type(value), repr(value)


def observe(call, arguments, np):
    before = snapshot(arguments, np)
    policy, handler, buffer = np.geterr(), np.geterrcall(), np.getbufsize()
    result, error = None, None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = call(*arguments)
        except Exception as exc:
            error = (type(exc).__module__, type(exc).__name__, str(exc))
    assert snapshot(arguments, np) == before, "window/shape/options mutation"
    assert np.geterr() == policy and np.geterrcall() is handler and np.getbufsize() == buffer
    return result, error, [(type(w.message).__module__, type(w.message).__name__, str(w.message)) for w in caught]


def reference_case(spec, np, original):
    arguments = fixture(spec, np)
    result, error, messages = observe(original.window, arguments, np)
    assert (error is None) == (spec["outcome"] == "success"), (spec["id"], error)
    if error is None:
        assert type(result) is np.ndarray and result.shape == tuple(spec["output_shape"])
        assert result.dtype == np.dtype("float64")
        if spec["id"] in ("geometry-0", "geometry-1", "geometry-9", "boxcar-small-clip-true"):
            assert np.all(result == 1.)
        elif spec["id"] == "boxcar-small-clip-false":
            # For2x2 the radial coordinate is inside the final grid-constant
            # interpolation cell. No output is0; source clip=True therefore
            # clips the entire result back up to the positive window minimum.
            assert np.all((result > 0.) & (result < 1.))
        elif spec["id"] == "named-hamming":
            assert np.all(result == result.flat[0]) and np.all(result > 0.)
        elif spec["id"] == "hamming-small-clip-false":
            assert np.all((result > 0.) & (result < .08))
        elif spec["id"] == "boxcar-large-actual-zero":
            assert np.any(result == 0.) and np.any(result == 1.)
    return result, error, messages


def compare_result(actual, expected, np):
    assert metadata(actual) == metadata(expected)
    assert np.array_equal(np.isnan(actual), np.isnan(expected))
    assert np.array_equal(np.isposinf(actual), np.isposinf(expected))
    assert np.array_equal(np.isneginf(actual), np.isneginf(expected))
    assert np.array_equal(actual == 0, expected == 0)
    zeros = expected == 0
    assert np.array_equal(np.signbit(actual[zeros]), np.signbit(expected[zeros]))
    finite = np.isfinite(expected)
    relative, absolute = error_limits(expected, np)
    assert np.all(np.abs(actual[finite] - expected[finite]) <= relative * np.abs(expected[finite]) + absolute)


def test_window_public_signature(reference_runtime):
    _, original, _ = reference_runtime
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    assert "filters.window" in native_functions(), "window remains native-pending"
    assert xx.filters.window is not original.window
    assert inspect.signature(xx.filters.window) == inspect.signature(original.window)


@pytest.mark.parametrize("spec", CASES, ids=[spec["id"] for spec in CASES])
@pytest.mark.parametrize("workers", (1, 4), ids=("workers1", "workers4"))
def test_window_owned_sampling(reference_runtime, spec, workers, monkeypatch):
    np, original, _ = reference_runtime
    # The genuine original completes before any candidate traps or C_returns.
    expected, expected_error, expected_warnings = reference_case(spec, np, original)
    arguments = fixture(spec, np)
    import scipy.ndimage
    import skimage.filters as original_public
    import skimage.transform as original_transform
    from skimage.transform import _warps
    import skimagexx as xx
    from skimagexx import _compat
    from skimagexx.coverage import native_functions
    assert "filters.window" in native_functions(), "window remains native-pending"
    assert xx.get_num_threads() == 1
    previous_fallback = xx.set_fallback_enabled(False)
    old_profile = sys.getprofile()
    forbidden, completions = [], []

    def denied(name):
        def call(*args, **kwargs):
            forbidden.append(name)
            raise AssertionError("upstream window/warp/interpolation use: " + name)
        return call

    def profile(frame, event, argument):
        if (event == "c_return" and getattr(argument, "__module__", None) == xx._core.__name__
                and getattr(argument, "__name__", None) == "window_sample"):
            completions.append("window_sample")

    try:
        xx.set_num_threads(workers)
        assert xx.get_num_threads() == workers and _compat._fallback_enabled is False
        with monkeypatch.context() as patch:
            patch.setattr(original_public, "window", denied("skimage.filters.window"))
            patch.setattr(original, "window", denied("skimage.filters._window.window"))
            patch.setattr(original, "warp", denied("window.warp"))
            patch.setattr(original_transform, "warp", denied("skimage.transform.warp"))
            patch.setattr(_warps, "warp", denied("skimage.transform._warps.warp"))
            patch.setattr(_warps, "_clip_warp_output", denied("upstream warp clipping helper"))
            for name, module in tuple(sys.modules.items()):
                if module is None or not name.startswith("scipy.ndimage"):
                    continue
                for entry in ("map_coordinates", "geometric_transform", "spline_filter", "spline_filter1d", "zoom_shift"):
                    if callable(getattr(module, entry, None)):
                        patch.setattr(module, entry, denied(name+"."+entry))
            sys.setprofile(profile)
            try:
                actual, error, messages = observe(xx.filters.window, arguments, np)
            finally:
                sys.setprofile(old_profile)
        assert not forbidden
        if spec["route"] == "strict-refusal":
            assert expected_error is None
            assert error is not None and error[:2] == ("builtins", "NotImplementedError")
            assert not completions, "higher-order refusal receives no native sampling credit"
        else:
            assert error == expected_error and messages == expected_warnings, spec["id"]
            if error is None:
                assert completions, "successful order0/1 requires real C++ sampling"
                compare_result(actual, expected, np)
        assert xx.get_num_threads() == workers and _compat._fallback_enabled is False
    finally:
        sys.setprofile(old_profile)
        xx.set_num_threads(1)
        xx.set_fallback_enabled(previous_fallback)


def reference_only():
    np, original, runtime = original_runtime()
    rows = []
    for spec in CASES:
        result, error, messages = reference_case(spec, np, original)
        rows.append(dict(id=spec["id"], initial_route=spec["route"], error=error, warnings=messages,
            result_dtype=None if error else result.dtype.str,
            result_shape=None if error else list(result.shape),
            nonfinite=0 if error else int(np.count_nonzero(~np.isfinite(result))),
            zeros=0 if error else int(np.count_nonzero(result == 0))))
    print(json.dumps(dict(status="REFERENCE_ONLY", native_coverage=0, specifications=len(rows),
        source_returns=sum(r["error"] is None for r in rows), source_errors=sum(r["error"] is not None for r in rows),
        initial_higher_order_refusals=sum(r["initial_route"] == "strict-refusal" for r in rows),
        blocked_intake=dict(shapes=BLOCKED_INTAKE, status="NOT_CALLED_NOT_NATIVE", reason="Mixed-empty coordinate outputs require a separate safety assessment of the ndimage zero-output path"),
        runtime=runtime, observations=rows), indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-only", action="store_true", required=True)
    parser.parse_args()
    reference_only()
