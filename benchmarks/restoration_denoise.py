"""Fixed one-thread public-call cases for the restoration kernels."""

import numpy as np


def fixed_public_calls():
    rng = np.random.default_rng(275861)
    gray = rng.random((128, 144), dtype=np.float32)
    rgb = rng.random((80, 96, 3), dtype=np.float64)
    volume = rng.random((14, 22, 25), dtype=np.float32)
    yield 'nl-gray-default', 'denoise_nl_means', (gray,), {}
    yield 'nl-rgb', 'denoise_nl_means', (rgb,), {
        'patch_size': 5, 'patch_distance': 4, 'h': 0.18, 'sigma': 0.02, 'channel_axis': -1,
    }
    yield 'nl-volume', 'denoise_nl_means', (volume,), {
        'patch_size': 3, 'patch_distance': 3, 'h': 0.15,
    }
    yield 'wavelet-haar-gray', 'denoise_wavelet', (gray,), {}
    yield 'wavelet-db2-rgb', 'denoise_wavelet', (rgb,), {
        'wavelet': 'db2', 'sigma': 0.18, 'method': 'VisuShrink', 'channel_axis': -1,
    }
    yield 'sigma-gray-db2', 'estimate_sigma', (gray,), {}
    yield 'sigma-rgb-db2', 'estimate_sigma', (rgb,), {'average_sigmas': True, 'channel_axis': -1}
    mask = np.zeros(gray.shape, dtype=bool)
    mask[40:51, 58:72] = True
    mask[62:79:2, 20:40:2] = True
    yield 'inpaint-defects', 'inpaint_biharmonic', (gray, mask), {}
    y, x = np.indices((132, 147), dtype=np.float64)
    phase = x * 0.17 + y * 0.19 + rng.normal(0, 0.04, x.shape)
    wrapped = (phase + np.pi) % (2 * np.pi) - np.pi
    yield 'unwrap-wrapped-2d', 'unwrap_phase', (wrapped,), {}
    z, y, x = np.indices((20, 41, 47), dtype=np.float64)
    phase = 0.4 * np.sin(x / 10) + 0.3 * np.sin(y / 10) + 0.05 * z
    yield 'unwrap-smooth-3d', 'unwrap_phase', (phase,), {}
    psf = np.outer(np.hanning(17), np.hanning(15)).astype(np.float32)
    psf /= psf.sum()
    yield 'lucy-iterative', 'richardson_lucy', (0.2 + gray / 2, psf), {'num_iter': 30}
    y, x = np.indices((61, 61), dtype=np.float32) - 30
    # A rotated blur exercises the FFT path because it is not separable.
    rotated_psf = np.exp(-((x + y / 2) ** 2 / 112 + (y - x / 2) ** 2 / 240))
    rotated_psf /= rotated_psf.sum()
    image = (0.2 + gray[:96, :96] / 2).copy()
    yield 'lucy-rotated-psf', 'richardson_lucy', (image, rotated_psf), {'num_iter': 15}
