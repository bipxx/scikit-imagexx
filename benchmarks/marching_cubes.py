"""One-thread public-call comparisons for volume isosurface extraction."""

import os
from statistics import median
from time import perf_counter

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
from skimage.measure import marching_cubes as original
from skimagexx import set_fallback_enabled, set_num_threads
from skimagexx.measure import marching_cubes as native


def cases():
    z, y, x = np.ogrid[-40:41, -48:49, -54:55]
    volume = (z*z + 0.8*y*y + 0.6*x*x).astype(np.float32)
    yield 'lewiner_ellipsoid', volume, {'level': 625.5}
    yield 'lorensen_masked_step2', volume, {
        'level': 625.5, 'method': 'lorensen', 'step_size': 2, 'mask': volume < 900,
    }
    yield 'lewiner_clean_spacing', volume, {
        'level': 625, 'allow_degenerate': False, 'spacing': (0.5, 1.0, 1.5),
    }
    texture = np.random.default_rng(201).standard_normal((34, 34, 34), dtype=np.float32)
    yield 'lewiner_dense_ambiguities', texture, {'level': 0}


def main():
    set_num_threads(1)
    previous = set_fallback_enabled(False)
    passed = True
    try:
        for name, volume, options in cases():
            expected = original(volume, **options)
            result = native(volume, **options)
            for actual, reference in zip(result, expected):
                np.testing.assert_array_equal(actual, reference)
            timings = [[], []]
            for trial in range(15):
                for idx in (range(2) if not trial % 2 else (1, 0)):
                    function = (original, native)[idx]
                    start = perf_counter()
                    function(volume, **options)
                    timings[idx].append(perf_counter() - start)
            upstream, candidate = map(median, timings)
            ratio = upstream / candidate
            passed &= ratio >= 1
            print(f'{name:28} {upstream*1000:.3f} ms -> {candidate*1000:.3f} ms ({ratio:.3f}x)')
    finally:
        set_fallback_enabled(previous)
    return passed


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
