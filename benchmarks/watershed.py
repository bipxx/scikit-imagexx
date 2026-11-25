"""Fixed one-worker, allocation-inclusive watershed public calls."""

METHOD = dict(workers=1, samples=7, minimum_sample_ns=100_000_000,
              minimum_warmup_ns=250_000_000, call='public prepared-input call including output allocation')


def case_specs():
    for name, shape, dtype, surface, markers, connectivity, masked, compactness, line in (
        ('small-plateau', (9, 11), 'float64', 'plateau', 'explicit', 1, False, 0., False),
        ('quantized', (160, 224), 'uint8', 'random', 'explicit', 1, False, 0., False),
        ('masked-diagonal', (160, 224), 'float32', 'random', 'explicit', 2, True, 0., False),
        ('masked-compact', (160, 192), 'float64', 'random', 'explicit', 1, True, 0.125, False),
        ('plateau-line', (192, 240), 'int16', 'plateau', 'explicit', 1, False, 0., True),
        ('volume', (16, 44, 52), 'float64', 'random', 'explicit', 1, True, 0., False),
        ('count-markers', (96, 128), 'float64', 'random', 'count', 1, False, 0., False),
        ('automatic-minima', (96, 128), 'uint8', 'random', 'minima', 1, False, 0., False),
    ):
        yield dict(id=name, api='segmentation.watershed', shape=shape, dtype=dtype,
                   surface=surface, markers=markers, connectivity=connectivity, masked=masked,
                   compactness=compactness, watershed_line=line, seed=304431)


def make_case(spec):
    import numpy as np

    shape = spec['shape']
    rng = np.random.default_rng(spec['seed'])
    if spec['surface'] == 'plateau':
        image = np.zeros(shape, dtype=spec['dtype'])
    else:
        image = rng.integers(0, 97, size=shape).astype(spec['dtype'])
    if spec['markers'] == 'count':
        markers = 48
    elif spec['markers'] == 'minima':
        markers = None
    else:
        markers = np.zeros(shape, dtype=np.int32)
        grid = tuple(slice(length // 10, None, max(3, length // 7)) for length in shape)
        view = markers[grid]
        view[:] = np.arange(1, view.size + 1).reshape(view.shape)
    mask = rng.random(shape) > 0.16 if spec['masked'] else None
    options = dict(connectivity=spec['connectivity'], mask=mask, compactness=spec['compactness'],
                   watershed_line=spec['watershed_line'])
    return (image, markers), options


def make_cases():
    for spec in case_specs():
        args, kwargs = make_case(spec)
        yield spec['id'], args, kwargs
