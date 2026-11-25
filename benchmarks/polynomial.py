"""Fixed one-thread public polynomial call and fitting comparisons."""

import os
from statistics import median
from time import perf_counter

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from skimage.transform import PolynomialTransform as Original
import skimagexx as xx


def main():
    rng = np.random.default_rng(552)
    old_workers = xx.get_num_threads()
    xx.set_num_threads(1)
    old_fallback = xx.set_fallback_enabled(False)
    try:
        native = xx.transform.PolynomialTransform
        for order in (1, 2, 3):
            params = rng.normal(size=(2, (order + 1) * (order + 2) // 2))
            models = (Original(params), native(params))
            for count in (32, 4096, 65536):
                points = rng.uniform(-1, 1, size=(count, 2))
                _measure(f"apply order {order}, {count} points", models, (points,), max(4, 4096 // count))
            for count in (24, 256):
                src = rng.uniform(-1, 1, size=(count, 2))
                dst = models[0](src) + 0.01 * rng.normal(size=src.shape)
                _measure(f"fit order {order}, {count} points",
                         (Original.from_estimate, native.from_estimate), (src, dst, order), 16)
    finally:
        xx.set_num_threads(old_workers)
        xx.set_fallback_enabled(old_fallback)


def _measure(label, methods, args, loops):
    samples = [[], []]
    for method in methods:
        method(*args)
    for group in range(13):
        for index in ((0, 1) if group % 2 == 0 else (1, 0)):
            method = methods[index]
            start = perf_counter()
            for _ in range(loops):
                method(*args)
            samples[index].append((perf_counter() - start) * 1e6 / loops)
    baseline, native = map(median, samples)
    print(f"{label}: {baseline:.3f} -> {native:.3f} us ({baseline / native:.3f}x)")


if __name__ == "__main__":
    main()
