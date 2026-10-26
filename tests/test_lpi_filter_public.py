"""Future owned LPI FFT contracts; intentionally RED before implementation.

--reference-only checks80 original specifications without importing skimagexx.
That mode provides no native coverage. Normal pytest requires the public owned
class/functions, exact source observables and real FFT C_returns. Callback and
duck-filter providers are explicit; neither earns native arithmetic credit.
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
    from .lpi_filter_cases import cases, error_limits, finite_peak, fixture
else:
    from lpi_filter_cases import cases, error_limits, finite_peak, fixture

CASES = cases()
SOURCE_SHA256 = "67a4965bb1003361e3dec54ecd0b54b3d47f77b647c2a21802a34076aa52398f"
FFT_ENTRIES = frozenset(("fft", "ifft", "fft2", "ifft2", "fftn", "ifftn",
    "rfft", "irfft", "rfft2", "irfft2", "rfftn", "irfftn", "hfft", "ihfft",
    "hfft2", "ihfft2", "hfftn", "ihfftn", "c2c", "c2r", "r2c", "c2cn", "c2rn", "r2cn"))


def original_runtime():
    import numpy as np
    import scipy
    import skimage
    from skimage.filters import lpi_filter as original
    assert skimage.__version__ == "0.26.0"
    assert (np.__version__, scipy.__version__) in (("1.24.4", "1.15.3"), ("2.5.4", "1.18.1"))
    assert hashlib.sha256(Path(original.__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == SOURCE_SHA256
    assert np.geterr() == dict(divide="warn", over="warn", under="ignore", invalid="warn")
    return np, original, dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                             skimage=skimage.__version__, original_source_sha256=SOURCE_SHA256)


@pytest.fixture(scope="module")
def reference_runtime():
    return original_runtime()


def metadata(array):
    return (type(array), array.dtype.str, array.dtype.metadata, array.shape,
            array.strides, array.flags.owndata, array.flags.writeable, type(array.base))


def snapshot(array):
    return metadata(array), array.tobytes(order="C")


def response_factory(spec, np, events):
    if spec["response"] == "noncallable":
        return 3

    def response(r, c, **params):
        events.append(dict(r=r.copy(), c=c.copy(), r_metadata=metadata(r), c_metadata=metadata(c),
                           kwargs=[(key, type(value).__module__, type(value).__name__, repr(value))
                                   for key, value in params.items()]))
        kind = spec["response"]
        if kind == "raise":
            raise RuntimeError("LPI callback failed deliberately")
        if kind == "scalar":
            return 1.
        if kind == "list":
            return [1.] * r.size
        if kind in ("zero", "nearzero"):
            result = np.zeros(r.shape, dtype=r.dtype)
            if kind == "nearzero":
                # Corner impulse has a constant real FFT below epsilon: no
                # near-threshold FFT rounding can select the cache mutation.
                result[0] = np.finfo(r.dtype).eps / 4
            return result
        if kind == "delta":
            return ((r == 0) & (c == 0)).astype(r.dtype)
        if kind == "box":
            return ((np.abs(r) <= 1) & (np.abs(c) <= 1)).astype(r.dtype)
        result = np.exp(-(r * r + c * c) / (2 * params["sigma"] ** 2)) * params["gain"]
        if kind == "asymmetric":
            result = result * (1 + r / 16 + c / 32)
        elif kind == "complex":
            result = result + 1j * result * (r + c) / 8
        elif kind in ("float64", "uint8"):
            result = result.astype(kind)
        elif kind == "wrong-size":
            result = result[:-1]
        return result
    return response


def observe(call, image, options, np):
    arrays = [image] + [value for value in options.values() if isinstance(value, np.ndarray)]
    before = [snapshot(array) for array in arrays]
    policy, handler, buffer = np.geterr(), np.geterrcall(), np.getbufsize()
    result, error = None, None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = call()
        except Exception as exc:
            error = (type(exc).__module__, type(exc).__name__, str(exc))
    assert [snapshot(array) for array in arrays] == before, "input/K mutation or metadata change"
    assert np.geterr() == policy and np.geterrcall() is handler and np.getbufsize() == buffer
    messages = [(type(w.message).__module__, type(w.message).__name__, str(w.message)) for w in caught]
    return dict(result=result, error=error, warnings=messages)


def run_case(namespace, spec, np, completions=None):
    events, steps = [], []
    response = response_factory(spec, np, events)
    params = dict(sigma=1.25, gain=np.float32(.75))
    scenario = spec["scenario"]
    stateful = scenario in ("cache-hit", "shape-change", "dtype-change", "inverse-forward", "wiener-forward", "params-change")
    filt = namespace.LPIFilter2D(response, **params) if stateful or scenario == "predefined" else None
    sequence = [spec["operation"]]
    if stateful:
        sequence = ["class", "class"]
        if scenario in ("inverse-forward", "wiener-forward"):
            sequence = ["class", "filter_inverse" if scenario == "inverse-forward" else "wiener", "class"]
        elif scenario in ("shape-change", "params-change"):
            sequence.append("class")
    previous_cache = None
    previous_cache_bytes = None
    for index, operation in enumerate(sequence):
        shape = spec["shape"]
        if (scenario == "shape-change" and index == 1) or (scenario == "params-change" and index == 2):
            shape = (shape[0] + 1, shape[1] + 2)
        dtype = "float64" if scenario == "dtype-change" and index else None
        image, options = fixture(spec, np, shape=shape, dtype=dtype)
        if scenario == "params-change" and index == 1:
            filt.filter_params["gain"] = np.float32(.125)
        event_start = len(events)
        completion_start = len(completions) if completions is not None else 0

        def call():
            nonlocal filt
            if operation == "class":
                if filt is None:
                    filt = namespace.LPIFilter2D(response, **params)
                return filt(image)
            if scenario == "duck":
                def duck(data):
                    events.append(dict(duck=True, input_metadata=metadata(data)))
                    return data.copy()
                return namespace.filter_forward(image, predefined_filter=duck)
            if scenario == "invalid-predefined":
                return namespace.filter_inverse(image, predefined_filter=object())
            if filt is not None:
                # Deliberately invalid alternatives must be ignored when a
                # predefined instance supplies the actual filter.
                return getattr(namespace, operation)(image, impulse_response=3,
                    filter_params=["ignored"], predefined_filter=filt, **options)
            filter_params = ["not a mapping"] if scenario == "bad-params" else params
            return getattr(namespace, operation)(image, response, filter_params, **options)

        step = observe(call, image, options, np)
        cache = None if filt is None else filt._cache
        cache_bytes = None if cache is None else cache.tobytes(order="C")
        step.update(image=image, shape=shape, events=events[event_start:],
                    cache=None if cache is None else cache.copy(),
                    cache_metadata=None if cache is None else metadata(cache),
                    cache_same=cache is not None and cache is previous_cache,
                    cache_changed=cache is previous_cache and cache_bytes != previous_cache_bytes,
                    callback_retained=filt is None or filt.impulse_response is response,
                    params=None if filt is None else [(k, type(v).__name__, repr(v)) for k, v in filt.filter_params.items()],
                    c_returns=[] if completions is None else completions[completion_start:])
        previous_cache, previous_cache_bytes = cache, cache_bytes
        steps.append(step)
    return steps


def validate_reference(spec, steps, np):
    for step in steps:
        assert (step["error"] is None) == (spec["outcome"] == "success"), (spec["id"], step["error"])
        assert step["callback_retained"]
        if step["error"] is None:
            result = step["result"]
            assert type(result) is np.ndarray and result.shape == step["shape"]
            assert not np.shares_memory(result, step["image"])
        for event in step["events"]:
            if event.get("duck"):
                continue
            # Independent enumeration locks the source's surprising even-axis
            # offset and row-major pair order, without reusing np.mgrid.
            shape = step["shape"]
            odd = tuple(n + (n % 2 == 0) for n in shape)
            row = [r + (shape[0] % 2 == 0) - (odd[0] - 1) / 2 for r in range(odd[0]) for _ in range(odd[1])]
            col = [c + (shape[1] % 2 == 0) - (odd[1] - 1) / 2 for _ in range(odd[0]) for c in range(odd[1])]
            assert np.array_equal(event["r"], row) and np.array_equal(event["c"], col)
            expected_dtype = np.dtype("float32" if step["image"].dtype.kind == "f" and step["image"].dtype.itemsize <= 4 else "float64")
            assert event["r"].dtype == event["c"].dtype == expected_dtype
    scenario = spec["scenario"]
    if scenario in ("cache-hit", "dtype-change"):
        assert [len(s["events"]) for s in steps] == [1, 0]
        assert steps[1]["cache_same"] and not steps[1]["cache_changed"]
    elif scenario == "shape-change":
        assert [len(s["events"]) for s in steps] == [1, 1, 1]
        assert not any(s["cache_same"] for s in steps)
    elif scenario == "params-change":
        assert [len(s["events"]) for s in steps] == [1, 0, 1]
        assert steps[1]["cache_same"] and not steps[1]["cache_changed"]
        assert not steps[2]["cache_same"]
    elif scenario in ("inverse-forward", "wiener-forward"):
        assert [len(s["events"]) for s in steps] == [1, 0, 0]
        assert steps[1]["cache_same"] and steps[1]["cache_changed"]
        assert steps[2]["cache_same"] and not steps[2]["cache_changed"]
        eps = np.finfo(steps[1]["cache"].real.dtype).eps
        assert np.all(steps[0]["cache"] == eps / 4) and np.all(steps[1]["cache"] == eps)


def compare_array(actual, expected, shape, np, *, input_peak=0.):
    assert metadata(actual) == metadata(expected)
    relative, absolute = error_limits(shape, expected, np, input_peak=input_peak)
    components = [(actual.real, expected.real)]
    if expected.dtype.kind == "c":
        components.append((actual.imag, expected.imag))
    for got, want in components:
        assert np.array_equal(np.isnan(got), np.isnan(want))
        assert np.array_equal(np.isposinf(got), np.isposinf(want))
        assert np.array_equal(np.isneginf(got), np.isneginf(want))
        finite = np.isfinite(want)
        assert np.all(np.abs(got[finite] - want[finite]) <= relative * np.abs(want[finite]) + absolute)


def compare_steps(actual, expected, spec, np):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        for field in ("error", "warnings", "cache_metadata", "cache_same", "cache_changed", "callback_retained", "params"):
            assert got[field] == want[field], (spec["id"], field, got[field], want[field])
        assert len(got["events"]) == len(want["events"])
        for gevent, wevent in zip(got["events"], want["events"]):
            assert gevent.keys() == wevent.keys()
            for key in gevent:
                if key in ("r", "c"):
                    assert snapshot(gevent[key]) == snapshot(wevent[key])
                else:
                    assert gevent[key] == wevent[key]
        if want["cache"] is not None:
            compare_array(got["cache"], want["cache"], want["shape"], np)
        if want["error"] is None:
            assert not np.shares_memory(got["result"], got["image"])
            compare_array(got["result"], want["result"], want["shape"], np,
                          input_peak=finite_peak(want["image"], np))
            if spec["scenario"] == "duck":
                assert not got["c_returns"], "duck-filter provider is not native FFT credit"
            else:
                assert len(got["c_returns"]) >= 2, "successful numerical step requires owned forward and inverse FFT"


def test_lpi_public_signatures(reference_runtime):
    _, original, _ = reference_runtime
    import skimagexx as xx
    from skimagexx.coverage import native_classes, native_functions
    assert "filters.LPIFilter2D" in native_classes(), "LPI class remains native-pending"
    assert xx.filters.LPIFilter2D is not original.LPIFilter2D
    for name in ("LPIFilter2D", "filter_forward", "filter_inverse", "wiener"):
        assert inspect.signature(getattr(xx.filters, name)) == inspect.signature(getattr(original, name))
        if name != "LPIFilter2D":
            assert "filters."+name in native_functions(), name+" remains native-pending"
    for name in ("__call__", "_prepare"):
        assert inspect.signature(getattr(xx.filters.LPIFilter2D, name)) == inspect.signature(getattr(original.LPIFilter2D, name))


@pytest.mark.parametrize("spec", CASES, ids=[spec["id"] for spec in CASES])
@pytest.mark.parametrize("workers", (1, 4), ids=("workers1", "workers4"))
def test_lpi_owned_fft(reference_runtime, spec, workers, monkeypatch):
    np, original, _ = reference_runtime
    expected = run_case(original, spec, np)
    validate_reference(spec, expected, np)
    import scipy.fftpack
    import skimage.filters as original_public
    import skimagexx as xx
    from skimagexx import _compat
    from skimagexx.coverage import native_classes, native_functions
    assert "filters.LPIFilter2D" in native_classes(), "LPI class remains native-pending"
    assert {"filters.filter_forward", "filters.filter_inverse", "filters.wiener"} <= set(native_functions())
    assert xx.get_num_threads() == 1
    previous_fallback = xx.set_fallback_enabled(False)
    old_profile = sys.getprofile()
    forbidden, completions = [], []

    def denied(name):
        def call(*args, **kwargs):
            forbidden.append(name)
            raise AssertionError("upstream LPI/FFT use: " + name)
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
            for name in ("__call__", "_prepare"):
                patch.setattr(original.LPIFilter2D, name, denied("LPIFilter2D."+name))
            for name in ("LPIFilter2D", "filter_forward", "filter_inverse", "wiener"):
                patch.setattr(original_public, name, denied("skimage.filters."+name))
                patch.setattr(original, name, denied("lpi_filter."+name))
            for name in ("_min_limit", "_center", "_pad"):
                patch.setattr(original, name, denied("lpi_filter."+name))
            for name, module in tuple(sys.modules.items()):
                if module is None or not name.startswith(("numpy.fft", "scipy.fft")):
                    continue
                for entry in tuple(vars(module)):
                    if (entry in FFT_ENTRIES or entry.startswith(("rfft_", "irfft_"))) and callable(getattr(module, entry)):
                        patch.setattr(module, entry, denied(name+"."+entry))
            sys.setprofile(profile)
            try:
                actual = run_case(xx.filters, spec, np, completions)
            finally:
                sys.setprofile(old_profile)
        assert not forbidden
        compare_steps(actual, expected, spec, np)
        assert xx.get_num_threads() == workers and _compat._fallback_enabled is False
    finally:
        sys.setprofile(old_profile)
        xx.set_num_threads(1)
        xx.set_fallback_enabled(previous_fallback)


def reference_only():
    np, original, runtime = original_runtime()
    rows = []
    for spec in CASES:
        steps = run_case(original, spec, np)
        validate_reference(spec, steps, np)
        rows.append(dict(id=spec["id"], steps=[dict(error=s["error"], warnings=s["warnings"],
            callback_calls=len(s["events"]), cache_same=s["cache_same"], cache_changed=s["cache_changed"],
            result_dtype=None if s["error"] else s["result"].dtype.str,
            result_shape=None if s["error"] else list(s["result"].shape),
            nonfinite_components=0 if s["error"] else int(np.count_nonzero(~np.isfinite(s["result"])))) for s in steps]))
    print(json.dumps(dict(status="REFERENCE_ONLY", native_coverage=0, cases=len(rows),
                         successful_steps=sum(s["error"] is None for r in rows for s in r["steps"]),
                         source_error_steps=sum(s["error"] is not None for r in rows for s in r["steps"]),
                         runtime=runtime, observations=rows), indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-only", action="store_true", required=True)
    parser.parse_args()
    reference_only()
