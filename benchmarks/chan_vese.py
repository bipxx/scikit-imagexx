"""Deterministic one-worker Chan–Vese public-call comparisons."""

import numpy as np


METHOD = dict(
    workers=1, samples=7, minimum_sample_ns=100_000_000, minimum_warmup_ns=250_000_000,
    call='public grayscale segmentation with initialization, evolution and requested outputs',
)


def case_specs():
    api = 'segmentation.chan_vese'
    return (
        dict(id='chan-vese-checkerboard-uint8', api=api, shape=(128, 141), dtype='uint8',
             preset='checkerboard', iterations=32, extended=False),
        dict(id='chan-vese-disk-float32', api=api, shape=(192, 204), dtype='float32',
             preset='disk', iterations=24, extended=False),
        dict(id='chan-vese-small-disk-float64-energy', api=api, shape=(126, 159), dtype='float64',
             preset='small disk', iterations=20, extended=True),
        dict(id='chan-vese-custom-float32-energy', api=api, shape=(96, 119), dtype='float32',
             preset='custom', iterations=20, extended=True),
    )


def make_case(spec):
    height, width = spec['shape']
    row, col = np.ogrid[:height, :width]
    blob = ((row - height // 2) * 2) ** 2 + ((col - width // 2) * 3) ** 2 < (height * 3 // 5) ** 2
    image = ((row * 7 + col * 13) % 51 + blob * 150).astype(spec['dtype'])
    level_set = spec['preset']
    if level_set == 'custom':
        level_set = (0.35 - ((row / height - 0.5) ** 2 + (col / width - 0.5) ** 2)).astype('float32')
    return (image,), dict(mu=0.17, lambda1=0.85, lambda2=1.2, tol=0., dt=0.5,
                          max_num_iter=spec['iterations'], init_level_set=level_set,
                          extended_output=spec['extended'])
