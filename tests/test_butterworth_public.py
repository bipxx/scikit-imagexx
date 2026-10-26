"""Future owned Butterworth contract; intentionally RED until implemented.

Run this file with --reference-only to validate the87 original cases without
importing skimagexx. Those reference observations provide zero native credit.
Normal pytest requires the registered public implementation and owned core
execution, with upstream Butterworth/mask and NumPy/SciPy FFT calls denied.
"""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import warnings

import pytest

if __package__:
    from .butterworth_cases import cases, error_limits, fixture
else:
    from butterworth_cases import cases, error_limits, fixture

CASES = cases()
SOURCE_SHA256 = "2d4c1bf4996f6ab5612e548fe673da72c23a0b9768565256b9f4e1f7942a38e8"
FFT_ENTRIES = frozenset(("fft", "ifft", "fft2", "ifft2", "fftn", "ifftn",
    "rfft", "irfft", "rfft2", "irfft2", "rfftn", "irfftn", "hfft", "ihfft",
    "hfft2", "ihfft2", "hfftn", "ihfftn", "c2c", "c2r", "r2c",
    "c2cn", "c2rn", "r2cn", "dct", "idct", "dst", "idst", "dctn", "idctn", "dstn", "idstn"))


def original_runtime():
    import numpy as np
    import scipy
    import skimage
    from skimage.filters import _fft_based as original
    assert skimage.__version__ == "0.26.0"
    assert (np.__version__, scipy.__version__) in (("1.24.4", "1.15.3"), ("2.5.4", "1.18.1"))
    assert hashlib.sha256(Path(original.__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == SOURCE_SHA256
    assert np.geterr() == dict(divide="warn", over="warn", under="ignore", invalid="warn")
    return np, original, dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                             skimage=skimage.__version__, original_source_sha256=SOURCE_SHA256)


@pytest.fixture(scope="module")
def reference_runtime():
    return original_runtime()


def snapshot(image):
    return (image.dtype.str, image.shape, image.strides, image.flags.writeable,
            image.flags.owndata, image.tobytes(order="C"))


def observe(call, image, kwargs, np):
    before = snapshot(image)
    policy, handler, buffer = np.geterr(), np.geterrcall(), np.getbufsize()
    result, error = None, None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = call(image, **kwargs)
        except Exception as exc:
            error = (type(exc).__module__, type(exc).__name__, str(exc))
    assert snapshot(image) == before, "input mutation or metadata change"
    assert np.geterr() == policy and np.geterrcall() is handler and np.getbufsize() == buffer
    messages = [(type(w.message).__module__, type(w.message).__name__, str(w.message)) for w in caught]
    return result, error, messages


def reference_case(spec, np, original):
    image, kwargs = fixture(spec, np)
    result, error, messages = observe(original.butterworth, image, kwargs, np)
    assert (error is None) == (spec["outcome"] == "success"), (spec["id"], error)
    if error is None:
        assert type(result) is np.ndarray and result.shape == image.shape
        assert result.dtype.kind in "fc" and not np.shares_memory(result, image)
        # Independent source-derived controls constrain the Fourier convention.
        if spec["id"] in ("analytic-zero", "analytic-constant-high"):
            assert np.count_nonzero(result) == 0
        elif spec["id"] == "analytic-constant-low":
            assert np.all(result == 3)
        elif spec["id"] == "analytic-impulse-7":
            n = 7
            mask = [1 / (1 + ((k if k <= n // 2 else k - n) / (n * .2)) ** 4) for k in range(n)]
            direct = np.asarray([sum(mask[k] * math.cos(2 * math.pi * k * x / n) for k in range(n)) / n for x in range(n)])
            relative, absolute = error_limits(spec, image, kwargs, result, np)
            assert np.all(np.abs(result - direct) <= relative * np.abs(direct) + absolute)
    return image, kwargs, result, error, messages


def compare_result(actual, expected, image, spec, kwargs, np):
    assert type(actual) is type(expected) is np.ndarray
    assert actual.shape == expected.shape and actual.dtype.str == expected.dtype.str
    assert actual.dtype.metadata == expected.dtype.metadata
    assert type(actual.base) is type(expected.base)
    assert actual.strides == expected.strides
    assert actual.flags.owndata == expected.flags.owndata and actual.flags.writeable == expected.flags.writeable
    assert not np.shares_memory(actual, image)
    relative, absolute = error_limits(spec, image, kwargs, expected, np)
    pairs = [(actual.real, expected.real)]
    if expected.dtype.kind == "c":
        pairs.append((actual.imag, expected.imag))
    for got, want in pairs:
        # Exceptional masks and infinity signs are structural, never tolerated.
        assert np.array_equal(np.isnan(got), np.isnan(want))
        assert np.array_equal(np.isposinf(got), np.isposinf(want))
        assert np.array_equal(np.isneginf(got), np.isneginf(want))
        finite = np.isfinite(want)
        assert np.all(np.abs(got[finite] - want[finite]) <= relative * np.abs(want[finite]) + absolute), spec["id"]


def test_butterworth_public_signature(reference_runtime):
    _, original, _ = reference_runtime
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    assert inspect.signature(xx.filters.butterworth) == inspect.signature(original.butterworth)
    assert "filters.butterworth" in native_functions(), "Butterworth remains native-pending"


@pytest.mark.parametrize("spec", CASES, ids=[spec["id"] for spec in CASES])
@pytest.mark.parametrize("workers", (1, 4), ids=("workers1", "workers4"))
def test_butterworth_owned_fft(reference_runtime, spec, workers, monkeypatch):
    np, original, _ = reference_runtime
    # Genuine original evaluation finishes before installing any traps.
    _, _, expected, expected_error, expected_warnings = reference_case(spec, np, original)
    image, kwargs = fixture(spec, np)
    import scipy.fftpack  # Include the legacy public FFT provider in the traps.
    import skimage.filters as original_public
    import skimagexx as xx
    from skimagexx import _compat
    from skimagexx.coverage import native_functions
    assert "filters.butterworth" in native_functions(), "Butterworth remains native-pending"
    assert xx.get_num_threads() == 1
    previous_fallback = xx.set_fallback_enabled(False)
    old_profile = sys.getprofile()
    forbidden, completions = [], []

    def denied(name):
        def call(*args, **kw):
            forbidden.append(name)
            raise AssertionError("upstream Butterworth/FFT use: " + name)
        return call

    def profile(frame, event, argument):
        if event == "c_return" and getattr(argument, "__module__", None) == xx._core.__name__:
            name = getattr(argument, "__name__", "")
            if name in ("fft_r2c", "fft_c2r", "fft_c2c"):
                completions.append(name)

    try:
        xx.set_num_threads(workers)
        assert xx.get_num_threads() == workers and _compat._fallback_enabled is False
        with monkeypatch.context() as patch:
            patch.setattr(original_public, "butterworth", denied("skimage.filters.butterworth"))
            patch.setattr(original, "butterworth", denied("skimage.filters._fft_based.butterworth"))
            patch.setattr(original, "_get_nd_butterworth_filter", denied("upstream mask"))
            # The original has loaded its backend. Patch concrete public and
            # loaded private transform functions, including NumPy's ufuncs.
            for name, module in tuple(sys.modules.items()):
                if module is None or not name.startswith(("numpy.fft", "scipy.fft")):
                    continue
                for entry in tuple(vars(module)):
                    if (entry in FFT_ENTRIES or entry.startswith(("rfft_", "irfft_"))) and callable(getattr(module, entry)):
                        patch.setattr(module, entry, denied(name + "." + entry))
            sys.setprofile(profile)
            try:
                actual, error, messages = observe(xx.filters.butterworth, image, kwargs, np)
            finally:
                sys.setprofile(old_profile)
        assert not forbidden
        assert error == expected_error and messages == expected_warnings, spec["id"]
        assert xx.get_num_threads() == workers and _compat._fallback_enabled is False
        if error is None:
            assert completions, "successful filter must execute owned non-metadata C++"
            compare_result(actual, expected, image, spec, kwargs, np)
    finally:
        sys.setprofile(old_profile)
        xx.set_num_threads(1)
        xx.set_fallback_enabled(previous_fallback)


def reference_only():
    """Small original-only check, not a native test substitute or snapshot corpus."""
    np, original, runtime = original_runtime()
    rows = []
    for spec in CASES:
        _, _, result, error, messages = reference_case(spec, np, original)
        rows.append(dict(id=spec["id"], error=error, warnings=messages,
                         result_dtype=None if error else result.dtype.str,
                         result_shape=None if error else list(result.shape),
                         nonfinite_components=0 if error else int(np.count_nonzero(~np.isfinite(result)))))
    print(json.dumps(dict(status="REFERENCE_ONLY", native_coverage=0, cases=len(rows),
                         successes=sum(row["error"] is None for row in rows),
                         source_errors=sum(row["error"] is not None for row in rows),
                         runtime=runtime, observations=rows), indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-only", action="store_true", required=True)
    parser.parse_args()
    reference_only()
