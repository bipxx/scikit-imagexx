"""Fixed one-thread public-call comparisons for the five restoration APIs."""

import argparse
from functools import partial
import json
import os
from statistics import median
from time import perf_counter_ns

import numpy as np
from scipy import fft
from skimage import __version__ as reference_version
from skimage import restoration as original
import skimagexx as xx
from skimagexx import restoration as native


def _attenuate(image, *, weight=0.8, channel_axis=None):
    return image * weight


def cases():
    generator = np.random.default_rng(625)
    for shape, dtype in [((96, 112), 'float32'), ((192, 224), 'float64')]:
        image = generator.random(shape).astype(dtype)
        psf = np.ones((5, 5), dtype=dtype) / 25
        suffix = f'{shape[0]}x{shape[1]}:{dtype}'
        yield f'wiener:{suffix}', 'wiener', (image, psf, 0.1), {}, 21
        params = dict(max_num_iter=24, min_num_iter=24, burnin=5)
        yield (f'unsupervised_wiener:{suffix}', 'unsupervised_wiener', (image, psf),
               dict(user_params=params, rng=782), 2)
        yield (f'denoise_invariant:{suffix}', 'denoise_invariant', (image, _attenuate),
               dict(stride=3), 14)
        sweep = {'weight': [0.3, 0.45, 0.6, 0.75, 0.9, 1.0]}
        yield (f'calibrate_denoiser:{suffix}', 'calibrate_denoiser', (image, _attenuate, sweep),
               dict(stride=3, approximate_loss=dtype == 'float32', extra_output=True), 5)
        yield (f'cycle_spin:{suffix}', 'cycle_spin', (image, _attenuate, (3, 3)),
               dict(workers=1), 12)


def _measure(function, batch):
    start = perf_counter_ns()
    for _ in range(batch):
        function()
    return (perf_counter_ns() - start) / batch


def _check_values(api, actual, expected, image):
    if api == 'calibrate_denoiser':
        actual_function, (actual_settings, actual_losses) = actual
        expected_function, (expected_settings, expected_losses) = expected
        assert actual_settings == expected_settings
        np.testing.assert_allclose(actual_losses, expected_losses, rtol=5e-7)
        actual, expected = actual_function(image), expected_function(image)
    if api == 'unsupervised_wiener':
        actual, actual_chain = actual
        expected, expected_chain = expected
        for key in ('noise', 'prior'):
            np.testing.assert_allclose(actual_chain[key], expected_chain[key], rtol=5e-6)
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', type=int, default=9)
    args = parser.parse_args()
    if args.repeat < 3:
        parser.error('--repeat must be at least 3')
    assert reference_version == '0.26.0'
    pools = ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS')
    assert all(os.environ.get(pool) == '1' for pool in pools)
    xx.set_num_threads(1)
    xx.set_fallback_enabled(False)
    rows = []
    with fft.set_workers(1):
        for case, api, positional, kwargs, batch in cases():
            ref = partial(getattr(original, api), *positional, **kwargs)
            candidate = partial(getattr(native, api), *positional, **kwargs)
            _check_values(api, candidate(), ref(), positional[0])
            candidate()
            ref()
            ref_samples, native_samples = [], []
            for repeat in range(args.repeat):
                if repeat % 2 == 0:
                    ref_samples.append(_measure(ref, batch))
                    native_samples.append(_measure(candidate, batch))
                else:
                    native_samples.append(_measure(candidate, batch))
                    ref_samples.append(_measure(ref, batch))
            reference_ns, native_ns = median(ref_samples), median(native_samples)
            rows.append(dict(case=case, original_ns=reference_ns, native_ns=native_ns,
                             ratio=reference_ns / native_ns, batch=batch,
                             original_samples_ns=ref_samples, native_samples_ns=native_samples))
    print(json.dumps(rows, indent=2))
    return int(any(row['ratio'] < 1 for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
