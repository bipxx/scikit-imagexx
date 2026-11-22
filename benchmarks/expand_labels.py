"""Fixed one-worker, allocation-inclusive public expand_labels comparisons."""

METHOD = dict(samples=7, minimum_sample_ns=100_000_000, minimum_warmup_ns=250_000_000,
              workers=1, call='public prepared input; include result allocation and disposal')


def case_specs():
    for name, shape, dtype, density, distance, spacing in (
        ('tiny-tie', (5, 7), 'uint8', None, 2, 1),
        ('sparse-grid', (128, 192), 'uint16', 1 / 197, 4, 1),
        ('dense-grid', (128, 192), 'int32', 0.25, 3, 1),
        ('anisotropic', (160, 192), 'int32', 1 / 197, 4, (1.75, 0.625)),
        ('volume', (24, 40, 48), 'uint16', 1 / 197, 4, (2., 1., 0.625)),
        ('empty-background', (256, 320), 'uint64', 0, float('inf'), 1),
        ('negative-radius', (160, 192), 'int64', 0.05, -1, 1),
        ('line', (32768,), 'int16', 1 / 31, 12, 0.5),
    ):
        yield dict(id=name, api='segmentation.expand_labels', shape=shape, dtype=dtype,
                   density=density, distance=distance, spacing=spacing, seed=22499)


def make_case(spec):
    import numpy as np

    labels = np.zeros(spec['shape'], dtype=spec['dtype'])
    if spec['density'] is None:
        labels[1, 1] = 8
        labels[0, 2] = 3
        labels[2, 4] = 12
    elif spec['density']:
        rng = np.random.default_rng(spec['seed'])
        selected = rng.random(spec['shape']) < spec['density']
        labels[selected] = rng.integers(1, 250, size=np.count_nonzero(selected))
    options = {'distance': spec['distance'], 'spacing': spec['spacing']}
    return (labels,), options
