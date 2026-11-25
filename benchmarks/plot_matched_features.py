"""Fixed Matplotlib call with fresh axes state between timed samples."""

from functools import partial

import numpy as np


def fixed_public_calls():
    rng = np.random.default_rng(241)
    image0 = rng.integers(0, 256, (128, 384), np.uint8)
    image1 = rng.integers(0, 256, (256, 192), np.uint8)
    points0 = rng.uniform((0, 0), image0.shape, (64, 2))
    points1 = rng.uniform((0, 0), image1.shape, (64, 2))
    matches = np.column_stack((np.arange(64), rng.permutation(64)))
    return (
        ("plot_matched_features/matplotlib", (image0, image1),
         dict(keypoints0=points0, keypoints1=points1, matches=matches)),
    )


def public_call_pairs(original, candidate):
    from matplotlib.figure import Figure

    for name, args, kwargs in fixed_public_calls():
        left, right = Figure().subplots(), Figure().subplots()
        yield (
            name,
            partial(original.plot_matched_features, *args, ax=left, **kwargs),
            partial(candidate.plot_matched_features, *args, ax=right, **kwargs),
            left.clear, right.clear,
        )
