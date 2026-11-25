"""One-worker, allocation-inclusive polygon perimeter workloads."""

import numpy as np


def fixed_public_calls():
    api = 'draw.polygon_perimeter'
    yield 'triangle', api, ([12, 32, 16], [10, 25, 34]), {}
    yield 'filtered-diamond', api, ([-48, 64, 176, 64], [64, 176, 64, -48]), {'shape': (128, 128)}
    yield 'clipped-diamond', api, ([-48, 64, 176, 64], [64, 176, 64, -48]), {'shape': (128, 128), 'clip': True}
    yield 'long-edges', api, ([24, 5144, 3892, 365], [70, 672, 4820, 3792]), {}
    t = np.arange(96, dtype=np.float64) * (np.pi / 48)
    yield 'fractional-ring', api, (256.25 + 240 * np.sin(t), 254.5 + 175 * np.cos(t)), {}
