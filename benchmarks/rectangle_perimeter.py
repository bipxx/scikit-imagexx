"""One-worker, allocation-inclusive rectangle perimeter workloads."""

import numpy as np


def fixed_public_calls():
    api = 'draw.rectangle_perimeter'
    yield 'small-end', api, ((2, 3),), {'end': (3, 4)}
    yield 'long-extent', api, ((10, 20),), {'extent': (4120, 5120)}
    yield 'filtered-extent', api, ((-20, 64),), {'extent': (350, 120), 'shape': (192, 192)}
    yield 'clipped-extent', api, ((-20, 64),), {'extent': (350, 120), 'shape': (192, 192), 'clip': True}
    yield 'negative-intp-extent', api, (np.array([700, 900], dtype=np.intp),), {
        'extent': np.array([-420, -580], dtype=np.intp), 'shape': (800, 1000),
    }
