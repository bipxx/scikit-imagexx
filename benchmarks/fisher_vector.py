"""One-thread public-call comparisons for diagonal Fisher vectors and GMM fits."""

import random
from statistics import median
from time import perf_counter_ns

import numpy as np
from numpy.testing import assert_allclose
from sklearn.mixture import GaussianMixture
from skimage import feature as upstream
from threadpoolctl import threadpool_limits

from skimagexx import feature, set_fallback_enabled


def fixed_cases():
    generator = np.random.default_rng(923)
    for rows, dimensions, components in [(512, 16, 4), (256, 64, 8)]:
        data = generator.normal(size=(rows, dimensions))
        gmm = GaussianMixture(n_components=components, covariance_type="diag",
                              random_state=21, max_iter=400).fit(data)
        data = generator.normal(size=(rows, dimensions))
        kwargs = {"improved": dimensions == 64}
        yield f"fisher-{rows}x{dimensions}-k{components}", "fisher_vector", (data, gmm), kwargs

    for rows, dimensions, components in [(240, 8, 3), (900, 12, 4)]:
        means = generator.normal(0., 4., (components, dimensions))
        data = means[np.arange(rows) % components] + generator.normal(0., .75, (rows, dimensions))
        kwargs = {"n_modes": components, "gm_args": {"covariance_type": "diag", "random_state": 42}}
        yield f"learn-{rows}x{dimensions}-k{components}", "learn_gmm", (data,), kwargs


def main():
    order = random.Random(4867)
    passed = True
    previous = set_fallback_enabled(False)
    try:
        with threadpool_limits(limits=1):
            for label, api, args, kwargs in fixed_cases():
                methods = getattr(upstream, api), getattr(feature, api)
                expected, result = (method(*args, **kwargs) for method in methods)
                if api == "fisher_vector":
                    assert_allclose(result, expected, rtol=2e-9, atol=2e-12)
                else:
                    for attr in ("weights_", "means_", "covariances_", "lower_bound_"):
                        assert_allclose(getattr(result, attr), getattr(expected, attr),
                                        rtol=1e-9, atol=2e-12)
                samples = [[], []]
                for _ in range(31):
                    for index in order.sample(range(2), k=2):
                        start = perf_counter_ns()
                        methods[index](*args, **kwargs)
                        samples[index].append(perf_counter_ns() - start)
                original, native = map(median, samples)
                ratio = original / native
                print(f"{label} {original / 1e3:.1f} -> {native / 1e3:.1f} us ({ratio:.3f}x)")
                passed &= ratio >= 1.
    finally:
        set_fallback_enabled(previous)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
