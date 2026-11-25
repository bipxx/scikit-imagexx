"""Compare warm public calls for the seven data APIs bundled in scikit-image."""

import os
from statistics import median
from time import perf_counter_ns

for variable in (
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

import skimage
from skimage import data as stock
from skimagexx import set_fallback_enabled, set_num_threads
from skimagexx import _data_loaders_b as native


CASES = ("logo", "microaneurysms", "moon", "page", "retina", "rocket", "stereo_motorcycle")


def _elapsed(function):
    start = perf_counter_ns()
    result = function()
    elapsed = perf_counter_ns() - start
    del result
    return elapsed


def main():
    set_num_threads(1)
    set_fallback_enabled(False)
    print(f"scikit-image {skimage.__version__}; one thread; median of 31 warm calls")
    passed = True
    for name in CASES:
        original = getattr(stock, name)
        candidate = getattr(native, name)
        for _ in range(3):
            original()
            candidate()
        original_ns, native_ns = [], []
        for sample in range(31):
            if sample % 2:
                native_ns.append(_elapsed(candidate))
                original_ns.append(_elapsed(original))
            else:
                original_ns.append(_elapsed(original))
                native_ns.append(_elapsed(candidate))
        baseline, measured = median(original_ns), median(native_ns)
        ratio = baseline / measured
        passed &= ratio >= 1.0
        print(f"{name:20s} {baseline / 1e6:8.3f} ms -> {measured / 1e6:8.3f} ms  ({ratio:.2f}x)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
