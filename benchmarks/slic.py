"""Fixed one-worker SLIC comparisons through the public function."""

METHOD = dict(workers=1, samples=7, minimum_sample_ns=100_000_000, minimum_warmup_ns=250_000_000,
              call='public image call including normalization, optional providers and output allocation')


def case_specs():
    for name, shape, dtype, options in (
        ('grayscale-u8', (144, 224), 'uint8', dict(n_segments=100, channel_axis=None)),
        ('grayscale-slic-zero', (161, 197), 'float32',
         dict(n_segments=229, compactness=0.8, slic_zero=True, start_label=0, channel_axis=None)),
        ('rgb-default-lab', (160, 224, 3), 'uint8', {}),
        ('rgb-unsmoothed-no-connectivity', (112, 176, 3), 'float32',
         dict(n_segments=147, compactness=0.5, convert2lab=False, enforce_connectivity=False)),
        ('anisotropic-volume', (13, 47, 71), 'float64',
         dict(n_segments=83, compactness=3., sigma=0.7, spacing=(2., 0.75, 1.25), channel_axis=None)),
    ):
        yield dict(id=name, api='segmentation.slic', shape=shape, dtype=dtype, options=options,
                   seed=20039)


def make_case(spec):
    import numpy as np

    rng = np.random.default_rng(spec['seed'])
    options = dict(spec['options'])
    shape = spec['shape']
    spatial = shape if options.get('channel_axis', -1) is None else shape[:-1]
    coordinates = np.indices(spatial, sparse=True)
    cells = sum((coordinate // step) * (axis + 1)
                for axis, (coordinate, step) in enumerate(zip(coordinates, (11, 19, 17))))
    if spatial != shape:
        cells = cells[..., None] + np.arange(shape[-1]) * 3
    image = ((cells * 19 + rng.integers(0, 31, shape)) % 256).astype(spec['dtype'])
    if image.dtype.kind == 'f':
        image /= 255
    return (image,), options


def make_cases():
    for spec in case_specs():
        args, options = make_case(spec)
        yield spec['id'], args, options
