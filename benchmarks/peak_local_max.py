"""Fixed public calls for the native peak and subpixel corner routines."""

import numpy as np


def make_cases():
    rng = np.random.default_rng(64021)
    response = rng.uniform(size=(384, 401)).astype(np.float32)
    yield ('response-f32-distance3', 'peak_local_max', (response,),
           {'min_distance': 3, 'threshold_abs': 0.8, 'p_norm': 2})

    plateaus = np.zeros((384, 401), dtype=np.float32)
    for row in range(4, 380, 12):
        for col in range(4, 396, 12):
            plateaus[row:row + 3, col:col + 3] = (row + col) % 11 + 1
    yield ('plateau-f32-mask', 'corner_peaks', (plateaus,),
           {'indices': False, 'exclude_border': False})

    grid = ((np.arange(193)[:, None] // 16 + np.arange(195)[None, :] // 16) % 2).astype(np.float64)
    corners = np.asarray([(row, col) for row in range(16, 192, 16)
                          for col in range(16, 192, 16)], dtype=np.intp)
    yield ('checkerboard-f64-window11', 'corner_subpix', (grid, corners), {})
