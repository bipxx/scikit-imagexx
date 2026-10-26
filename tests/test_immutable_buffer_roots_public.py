"""Immutable-bytes storage contracts, frozen before the validator repair.

62 public specifications use the existing family numerical evaluators unchanged.
Each runs at workers1/4 with real FFT/projector/sampler completion required.
Ten validator-only controls have no image-parity or numerical native credit;
stale and forged descriptors are never read by NumPy or an original algorithm.
Only exact immutable bytes owners are added: mutable/foreign owners stay refused.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

import pytest

if __package__:
    from . import test_butterworth_public as butter
    from . import test_lpi_filter_public as lpi
    from . import test_moments_public as moments
    from . import test_window_public as window
else:
    import test_butterworth_public as butter
    import test_lpi_filter_public as lpi
    import test_moments_public as moments
    import test_window_public as window


APIS = ("filters.butterworth", "filters.filter_forward", "filters.filter_inverse",
        "filters.wiener", "measure.moments", "measure.moments_central", "measure.centroid")
VARIANTS = (("u8-C", "uint8", "C"), ("f32-C", "float32", "C"),
            ("f64-C", "float64", "C"), ("f32-step", "float32", "step"),
            ("f64-reverse", "float64", "reverse"), ("f32-F", "float32", "F"),
            ("f32-endian", "float32", "endian"), ("f64-unaligned", "float64", "unaligned"))
PUBLIC_CASES = tuple(dict(id=api + "-" + name, api=api, dtype=dtype, layout=layout)
                    for api in APIS for name, dtype, layout in VARIANTS) + tuple(
    dict(id="filters.window-shape-" + name, api="filters.window", dtype=dtype, layout=layout)
    for name, dtype, layout in (("u8", "uint8", "C"), ("f32", "float32", "C"),
        ("f64", "float64", "C"), ("step", "int64", "step"),
        ("reverse", "int64", "reverse"), ("endian", "float64", "endian")))
GUARDS = ("bytes", "bytes-reverse", "bytes-empty", "bytes-unaligned",
          "bytearray", "memoryview", "bytes-subclass", "foreign-owner", "stale-owner", "forged-overreach")
assert len(PUBLIC_CASES) == 62 and len(GUARDS) == 10


def original_runtime():
    np, original_butter, runtime = butter.original_runtime()
    _, original_lpi, _ = lpi.original_runtime()
    _, original_moments, _ = moments.original_runtime()
    _, original_window, _ = window.original_runtime()
    return np, dict(butter=original_butter, lpi=original_lpi,
                    moments=original_moments, window=original_window), runtime


@pytest.fixture(scope="module")
def reference_runtime():
    return original_runtime()


def immutable_input(spec, np):
    dtype = np.dtype(spec["dtype"])
    layout = spec["layout"]
    is_shape = spec["api"] == "filters.window"
    if is_shape:
        values = np.asarray([5, 99, 7, 99] if layout == "step" else [5, 7], dtype=dtype)
    else:
        shape = (5, 14) if layout == "step" else (5, 7)
        values = (np.arange(shape[0] * shape[1]).reshape(shape) % 9 + 1).astype(dtype)
        if dtype.kind == "f":
            values /= 4
    if layout == "endian":
        dtype = dtype.newbyteorder()
        values = values.astype(dtype)
    order = "F" if layout == "F" else "C"
    raw = values.tobytes(order=order)
    offset = 1 if layout == "unaligned" else 0
    owner = b"x" + raw if offset else raw
    result = np.frombuffer(owner, dtype=dtype, count=values.size, offset=offset).reshape(values.shape, order=order)
    if layout == "step":
        result = result[::2] if is_shape else result[:, ::2]
    elif layout == "reverse":
        result = result[::-1] if is_shape else result[:, ::-1]
    assert not result.flags.writeable
    return result


def owner_snapshot(array, np):
    chain = []
    current = array
    while isinstance(current, np.ndarray):
        chain.append((id(current), moments.metadata(current), current.tobytes(order="C")))
        current = current.base
    assert type(current) is bytes
    return tuple(chain), id(current), current


def make_call(spec, value, namespace, np, events):
    api = spec["api"]
    name = api.split(".")[1]
    if api == "filters.butterworth":
        kwargs = {"npad": 1} if spec["layout"] == "unaligned" else {}
        return lambda: namespace.butterworth(value, **kwargs), kwargs
    if api == "filters.window":
        options = {"order": 0 if spec["layout"] == "reverse" else 1}
        return lambda: namespace.window("hann", value, options), {}
    if api.startswith("measure."):
        return lambda: getattr(namespace, name)(value), {}

    def impulse(r, c):
        # The frozen LPI delta response isolates the storage boundary while
        # still running the source callback and all forward/inverse FFT stages.
        events.append((moments.metadata(r), r.tobytes(), moments.metadata(c), c.tobytes()))
        return ((r == 0) & (c == 0)).astype(r.dtype)

    return lambda: getattr(namespace, name)(value, impulse), {}


def observe(call, value, np):
    before = owner_snapshot(value, np)
    state = np.geterr(), np.geterrcall(), np.getbufsize()
    result, error = None, None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = call()
        except Exception as exc:
            error = type(exc).__module__, type(exc).__name__, str(exc)
    assert owner_snapshot(value, np) == before
    assert (np.geterr(), np.geterrcall(), np.getbufsize()) == state
    return result, error, [(type(w.message).__module__, type(w.message).__name__, str(w.message)) for w in caught]


def family(spec):
    if spec["api"].startswith("measure."):
        return "moments"
    return {"filters.butterworth": "butter", "filters.window": "window"}.get(spec["api"], "lpi")


def moment_spec(spec):
    return dict(id=spec["id"], api=spec["api"].split(".")[1], center=None,
                spacing=None, exact=False)


def compare_result(actual, expected, value, spec, kwargs, np):
    selected = family(spec)
    if selected == "butter":
        butter.compare_result(actual, expected, value, spec, kwargs, np)
    elif selected == "lpi":
        lpi.compare_array(actual, expected, value.shape, np, input_peak=lpi.finite_peak(value, np))
    elif selected == "window":
        window.compare_result(actual, expected, np)
    else:
        policy = moment_spec(spec)
        bound = moments.error_limits(policy, value, expected, np)
        finite = np.isfinite(expected.real)
        assert np.all(np.isfinite(bound[finite]) & (bound[finite] >= 0))
        moments.compare_result(actual, expected, bound, policy, np)
    assert not np.shares_memory(actual, value)


def reference_case(spec, np, originals):
    value = immutable_input(spec, np)
    events = []
    call, kwargs = make_call(spec, value, originals[family(spec)], np, events)
    result, error, messages = observe(call, value, np)
    assert error is None, (spec["id"], error)
    compare_result(result, result, value, spec, kwargs, np)
    return result, messages, events


@pytest.mark.parametrize("spec", PUBLIC_CASES, ids=[x["id"] for x in PUBLIC_CASES])
@pytest.mark.parametrize("workers", (1, 4), ids=("workers1", "workers4"))
def test_immutable_bytes_public(reference_runtime, spec, workers, monkeypatch):
    np, originals, _ = reference_runtime
    expected, expected_warnings, expected_events = reference_case(spec, np, originals)
    value = immutable_input(spec, np)
    import scipy.fftpack
    import skimage.filters as original_filters
    import skimage.measure as original_measure
    import skimage.transform as original_transform
    from skimage.measure import _moments_analytical
    from skimage.transform import _warps
    import skimagexx as xx
    from skimagexx import _compat
    from skimagexx.coverage import native_functions
    assert spec["api"] in native_functions() and xx.get_num_threads() == 1
    namespace = xx.measure if spec["api"].startswith("measure.") else xx.filters
    events, forbidden, completions = [], [], []
    call, kwargs = make_call(spec, value, namespace, np, events)
    previous_fallback = xx.set_fallback_enabled(False)
    old_profile = sys.getprofile()

    def denied(name):
        def fail(*args, **kw):
            forbidden.append(name)
            raise AssertionError("upstream numerical forwarding: " + name)
        return fail

    def profile(frame, event, argument):
        if event == "c_return" and getattr(argument, "__module__", None) == xx._core.__name__:
            name = getattr(argument, "__name__", "")
            if name in ("fft_r2c", "fft_c2r", "fft_c2c", "moments_project", "window_sample"):
                completions.append(name)

    try:
        xx.set_num_threads(workers)
        assert xx.get_num_threads() == workers and _compat._fallback_enabled is False
        with monkeypatch.context() as patch:
            for api in (*APIS, "filters.window"):
                module, name = api.split(".")
                patch.setattr(original_filters if module == "filters" else original_measure, name, denied(api))
            for key, names in (("butter", ("butterworth", "_get_nd_butterworth_filter")),
                               ("lpi", ("filter_forward", "filter_inverse", "wiener", "_min_limit", "_center", "_pad")),
                               ("moments", ("moments", "moments_central", "centroid", "moments_raw_to_central")),
                               ("window", ("window", "warp"))):
                for name in names:
                    patch.setattr(originals[key], name, denied(key + "." + name))
            for name in ("__call__", "_prepare"):
                patch.setattr(originals["lpi"].LPIFilter2D, name, denied("LPIFilter2D." + name))
            for name in ("moments_raw_to_central", "_moments_raw_to_central_fast"):
                patch.setattr(_moments_analytical, name, denied(name))
            patch.setattr(original_transform, "warp", denied("transform.warp"))
            patch.setattr(_warps, "warp", denied("_warps.warp"))
            for name in ("einsum", "tensordot", "dot", "matmul"):
                patch.setattr(np, name, denied("numpy." + name))
            for name, module in tuple(sys.modules.items()):
                if module is None:
                    continue
                if name.startswith(("numpy.fft", "scipy.fft")):
                    entries = butter.FFT_ENTRIES
                elif name.startswith("scipy.ndimage"):
                    entries = ("map_coordinates", "geometric_transform", "spline_filter", "spline_filter1d")
                else:
                    continue
                for entry in entries:
                    if callable(vars(module).get(entry)):
                        patch.setattr(module, entry, denied(name + "." + entry))
            sys.setprofile(profile)
            try:
                actual, error, messages = observe(call, value, np)
            finally:
                sys.setprofile(old_profile)
        assert not forbidden
        assert error is None and messages == expected_warnings and events == expected_events, (spec["id"], error)
        wanted = ({"moments_project"} if family(spec) == "moments" else
                  {"window_sample"} if family(spec) == "window" else {"fft_r2c", "fft_c2r", "fft_c2c"})
        assert wanted.intersection(completions), "a metadata-only check is not native numerical execution"
        compare_result(actual, expected, value, spec, kwargs, np)
        assert xx.get_num_threads() == workers and _compat._fallback_enabled is False
    finally:
        sys.setprofile(old_profile)
        xx.set_num_threads(1)
        xx.set_fallback_enabled(previous_fallback)


@pytest.mark.parametrize("case", GUARDS)
def test_storage_validator_metadata_only(case):
    import numpy as np
    import skimagexx as xx
    if case == "bytes-empty":
        array = np.frombuffer(b"", dtype=np.uint8)
    elif case == "bytes-unaligned":
        array = np.frombuffer(b"x" + bytes(16), dtype=np.float64, offset=1)
    elif case.startswith("bytes") and case != "bytes-subclass":
        array = np.frombuffer(bytes(range(16)), dtype=np.uint8)
        if case == "bytes-reverse":
            array = array[::-2]
    elif case == "bytearray":
        array = np.frombuffer(bytearray(16), dtype=np.uint8)
    elif case == "memoryview":
        array = np.frombuffer(memoryview(bytes(16)), dtype=np.uint8)
    elif case == "bytes-subclass":
        class DerivedBytes(bytes):
            pass
        array = np.frombuffer(DerivedBytes(16), dtype=np.uint8)
    elif case == "stale-owner":
        owner = np.arange(16, dtype=np.float64)
        array = owner.reshape(4, 4)
        owner.resize((1,), refcheck=False)
    else:
        owner = np.arange(16, dtype=np.uint8)
        array = np.lib.stride_tricks.as_strided(owner,
            shape=(16 if case == "foreign-owner" else 32,), strides=(1,), writeable=False)
    # Only descriptor metadata is inspected for invalid/stale arrays. No
    # pixel copy, arithmetic, original call or numerical core entry is used.
    before = moments.metadata(array)
    state = np.geterr(), np.geterrcall(), np.getbufsize(), xx.get_num_threads()
    if case in GUARDS[:4]:
        assert xx._core.fft_check(array) is None
    else:
        message = "FFT escapes owner allocation" if case == "stale-owner" else "FFT requires ndarray-backed storage"
        with pytest.raises(ValueError, match="^" + message + "$"):
            xx._core.fft_check(array)
    assert moments.metadata(array) == before
    assert (np.geterr(), np.geterrcall(), np.getbufsize(), xx.get_num_threads()) == state


def reference_only():
    np, originals, runtime = original_runtime()
    rows = []
    for spec in PUBLIC_CASES:
        result, messages, events = reference_case(spec, np, originals)
        rows.append(dict(id=spec["id"], dtype=result.dtype.str, shape=result.shape,
                         warnings=messages, callback_calls=len(events),
                         result_sha256=hashlib.sha256(result.tobytes()).hexdigest()))
    print(json.dumps(dict(status="REFERENCE_ONLY", native_coverage=0, runtime=runtime,
                         specifications=72, public_reference_calls=62,
                         metadata_only_controls_not_called=10, rows=rows), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-only", action="store_true", required=True)
    parser.parse_args()
    reference_only()
