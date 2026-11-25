"""Compare repeated sample-data calls against the stock scikit-image 0.26 loaders."""

import json
import os
from os import path
from statistics import median
from time import perf_counter_ns
from types import SimpleNamespace
from unittest.mock import patch


THREAD_LIMITS = (
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def _sample(call, count):
    start = perf_counter_ns()
    for _ in range(count):
        call()
    return (perf_counter_ns() - start) / count


def _measure(reference, native):
    calls = (reference, native) if reference is not native else (reference,)
    counts = [max(1, min(20_000, int(25_000_000 / _sample(call, 16)))) for call in calls]
    samples = [[] for _ in calls]
    for batch in range(9):
        order = range(len(calls)) if batch % 2 == 0 else reversed(range(len(calls)))
        for index in order:
            samples[index].append(_sample(calls[index], counts[index]))
    if len(samples) == 1:
        # The wrapper exports the exact provider callable for this operation.
        samples *= 2
        counts *= 2
    original_ns, native_ns = map(median, samples)
    return dict(original_ns=original_ns, native_ns=native_ns,
                speedup=original_ns / native_ns, batch_calls=counts, samples_ns=samples)


def main():
    import skimage
    from skimage import data as original
    from skimage.data import _fetchers
    from skimagexx import _data_loaders_a as candidate

    assert skimage.__version__ == "0.26.0"
    assert all(os.environ.get(variable) == "1" for variable in THREAD_LIMITS)

    def forbidden_fetch(*args, **kwargs):
        raise AssertionError("benchmark attempted to fetch a remote file")

    provider = _fetchers._image_fetcher
    root = path.dirname(_fetchers.data_dir) if provider is None else str(provider.abspath)
    offline = SimpleNamespace(abspath=root, fetch=forbidden_fetch)
    remote = {"brain", "cells3d", "eagle", "human_mitosis", "kidney"}
    results = {}
    with patch.object(_fetchers, "_image_fetcher", offline), patch.object(
        _fetchers, "_ensure_cache_dir", forbidden_fetch
    ):
        for name in candidate.__all__:
            if name == "download_all":
                continue
            if name in remote and candidate._verified_local(name) is None:
                results[name] = {"status": "remote fixture unavailable"}
                continue
            reference, native = getattr(original, name), getattr(candidate, name)
            expected, actual = reference(), native()
            if isinstance(expected, str):
                assert actual == expected
            else:
                assert type(actual) is type(expected)
                assert actual.shape == expected.shape and actual.dtype == expected.dtype
                assert actual.strides == expected.strides and actual.flags.num == expected.flags.num
                assert actual.tobytes() == expected.tobytes()
            results[name] = _measure(reference, native)

    registry = {"data/" + filename: _fetchers.registry["data/" + filename]
                for name, filename in candidate._FILES.items() if name not in remote}
    distribution = _fetchers._DISTRIBUTION_DIR
    provider = SimpleNamespace(abspath=distribution, path=distribution, registry=registry,
                               fetch=forbidden_fetch)
    # The controlled directory already exists and every requested file is local.
    with patch.object(_fetchers, "_image_fetcher", provider), patch.object(
        _fetchers, "_ensure_cache_dir", lambda target_dir: None
    ):
        assert candidate.download_all() is None
        results["download_all_local_registry"] = _measure(original.download_all, candidate.download_all)

    print(json.dumps(results, indent=2))
    return 0 if all(result.get("speedup", 1.0) >= 1.0 for result in results.values()) else 1


if __name__ == "__main__":
    for variable in THREAD_LIMITS:
        os.environ[variable] = "1"
    raise SystemExit(main())
