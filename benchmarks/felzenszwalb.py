"""Fixed public Felzenszwalb image segmentation calls."""


def case_specs():
    return (
        dict(id='rgb-default', shape=(192, 224, 3), dtype='uint8', options={}),
        dict(id='gray-unsmoothed', shape=(288, 296), dtype='float64',
             options={'sigma': 0, 'scale': 24, 'min_size': 16, 'channel_axis': None}),
    )


def make_case(spec):
    import numpy as np

    generator = np.random.default_rng(906146)
    sample = (generator.integers(0, 256, spec['shape'], dtype=np.uint8)
              if spec['dtype'] == 'uint8' else generator.random(spec['shape']))
    return (sample,), spec['options']
