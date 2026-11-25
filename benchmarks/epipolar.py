"""Fixed one-thread public epipolar call and fitting comparisons."""

import os
from statistics import median
from time import perf_counter

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from skimage import transform as original
import skimagexx as xx


def main():
    rng = np.random.default_rng(291)
    matrix = rng.normal(size=(3, 3))
    old_workers = xx.get_num_threads()
    xx.set_num_threads(1)
    old_fallback = xx.set_fallback_enabled(False)
    try:
        for name in ("FundamentalMatrixTransform", "EssentialMatrixTransform"):
            base, native = (getattr(module, name) for module in (original, xx.transform))
            models = (base(matrix=matrix), native(matrix=matrix))
            for count in (32, 4096, 65536):
                src, dst = rng.normal(size=(2, count, 2))
                _measure(f"{name} apply {count}", models, (src,), max(4, 4096 // count))
                _measure(f"{name} residuals {count}",
                         [model.residuals for model in models], (src, dst), max(4, 4096 // count))
            for count in (24, 256):
                src, dst = rng.normal(size=(2, count, 2))
                _measure(f"{name} fit {count}", (base.from_estimate, native.from_estimate),
                         (src, dst), 16)
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
