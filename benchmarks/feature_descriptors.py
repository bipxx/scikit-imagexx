"""Fixed public descriptor calls for the single-thread paired sampler."""

from functools import partial

import numpy as np


def fixed_public_calls():
    rng = np.random.default_rng(240)
    texture = rng.integers(0, 256, (256, 283), np.uint8)
    rgb = rng.integers(0, 256, (192, 201, 3), np.uint8)
    integral = texture.cumsum(0).cumsum(1)
    signal = rng.standard_normal((320, 341))
    patch = signal[45:50, 77:82].copy()
    larger_patch = signal[45:80, 77:116].copy()
    return (
        ("hog/grayscale", (texture,), {}),
        ("hog/rgb-visualize", (rgb,), {"channel_axis": -1, "visualize": True}),
        ("daisy/default", (texture,), {}),
        ("daisy/custom", (texture,), {"step": 6, "radius": 9, "rings": 2, "histograms": 6, "normalization": "l2"}),
        ("local_binary_pattern/default", (texture, 8, 1), {}),
        ("local_binary_pattern/uniform", (texture, 16, 2, "uniform"), {}),
        ("multiblock_lbp/integral-u64", (integral, 12, 10, 17, 21), {}),
        ("match_template/spatial", (signal, patch), {}),
        ("match_template/fft-padded", (signal, larger_patch), {"pad_input": True}),
        ("multiscale_basic_features/grayscale", (texture,), {"workers": 1}),
        ("multiscale_basic_features/rgb", (rgb,), {"workers": 1, "channel_axis": -1, "num_sigma": 3}),
    )


def public_call_pairs(original, candidate):
    for name, args, kwargs in fixed_public_calls():
        api = name.split("/")[0]
        yield name, partial(getattr(original, api), *args, **kwargs), partial(getattr(candidate, api), *args, **kwargs)
