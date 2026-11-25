"""Fixed public-call cases for the Dask overlap adapter."""

import numpy as np


def fixed_public_calls():
    small = np.arange(64, dtype=np.float32).reshape(8, 8)
    yield 'tiny-one-chunk', 'apply_parallel', (np.square, small), {'chunks': (8, 8)}
    image = np.arange(120 * 152, dtype=np.float32).reshape(120, 152)
    yield 'multi-chunk', 'apply_parallel', (np.square, image), {'chunks': (32, 24)}
    yield 'irregular-overlap', 'apply_parallel', (np.square, image), {
        'chunks': ((32, 48, 40), (32, 64, 56)), 'depth': (2, 1), 'mode': 'wrap',
        'dtype': image.dtype,
    }
    color = np.broadcast_to(image[..., None], (*image.shape, 3))
    yield 'channel-overlap', 'apply_parallel', (np.square, color), {
        'chunks': (40, 38), 'depth': 2, 'mode': 'symmetric', 'channel_axis': -1,
        'dtype': color.dtype,
    }
