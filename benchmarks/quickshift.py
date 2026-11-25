"""Fixed public quickshift segmentation calls."""


def case_specs():
    return (
        dict(id='rgb-default', shape=(112, 128, 3), dtype='uint8', options={}),
        dict(id='gray-compact-window', shape=(160, 188), dtype='float32',
             options={'convert2lab': False, 'kernel_size': 2, 'max_dist': 5, 'sigma': 0.85}),
    )


def make_case(spec):
    import numpy as np

    generator = np.random.default_rng(184652)
    sample = (generator.integers(0, 256, spec['shape'], dtype=np.uint8)
              if spec['dtype'] == 'uint8' else generator.random(spec['shape'], dtype=np.float32))
    return (sample,), spec['options']
