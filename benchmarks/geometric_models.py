"""One-thread public-call comparisons for geometric models and RANSAC."""

import os
from functools import partial
from statistics import median
from time import perf_counter

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
from skimage import measure as original
from skimagexx import measure as native
from skimagexx import set_fallback_enabled, set_num_threads


def cases():
    rng = np.random.default_rng(942)
    angles = np.linspace(-0.2, 6.1, 8192)
    for name, args in (
        ('CircleModel', ((1., 2.), 5.)),
        ('EllipseModel', ((1., 2.), (5., 2.), 0.48)),
    ):
        upstream, candidate = getattr(original, name), getattr(native, name)
        orig_model, nat_model = upstream(*args), candidate(*args)
        points = orig_model.predict_xy(angles) + rng.normal(0, .04, (len(angles), 2))
        subset = np.ascontiguousarray(points[:768])
        yield f'{name}.from_estimate', upstream.from_estimate, candidate.from_estimate, (points,)
        yield f'{name}.residuals', orig_model.residuals, nat_model.residuals, (subset,)
        yield f'{name}.predict_xy', orig_model.predict_xy, nat_model.predict_xy, (angles,)
        if name == 'CircleModel':
            circle_points = points.copy()
            circle_points[::7] += 1.7
            yield 'ransac.circle', partial(
                original.ransac, circle_points, upstream, 4, .12, rng=223
            ), partial(
                native.ransac, circle_points, candidate, 4, .12, rng=223
            ), ()


def check_result(actual, reference):
    if isinstance(reference, np.ndarray):
        np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=4e-7)
    elif isinstance(reference, tuple):
        np.testing.assert_array_equal(actual[1], reference[1])
        check_result(actual[0], reference[0])
    else:
        np.testing.assert_allclose(actual.center, reference.center)
        if hasattr(reference, 'radius'):
            np.testing.assert_allclose(actual.radius, reference.radius)
        else:
            np.testing.assert_allclose(actual.axis_lengths, reference.axis_lengths)
            np.testing.assert_allclose(actual.theta, reference.theta)


def main():
    set_num_threads(1)
    previous = set_fallback_enabled(False)
    passed = True
    try:
        for name, upstream, candidate, args in cases():
            try:
                expected = upstream(*args)
            except TypeError as error:
                if name != 'EllipseModel.from_estimate':
                    raise
                print(f'{name:28} reference failed: {error}')
                passed = False
                continue
            check_result(candidate(*args), expected)
            timings = [[], []]
            for trial in range(13):
                for idx in (range(2) if not trial % 2 else (1, 0)):
                    function = (upstream, candidate)[idx]
                    start = perf_counter()
                    function(*args)
                    timings[idx].append(perf_counter() - start)
            before, after = map(median, timings)
            ratio = before / after
            passed &= ratio >= 1
            print(f'{name:28} {before*1000:.3f} ms -> {after*1000:.3f} ms ({ratio:.3f}x)')
    finally:
        set_fallback_enabled(previous)
    return passed


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
