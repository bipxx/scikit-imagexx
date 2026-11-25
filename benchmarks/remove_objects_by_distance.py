"""Fixed one-worker, allocation-inclusive distance-filter public calls."""

METHOD = dict(samples=7, minimum_sample_ns=100_000_000, minimum_warmup_ns=250_000_000,
              workers=1, call='public call with prepared labels and optional priority')


def fixed_public_calls():
    import numpy as np

    api = 'morphology.remove_objects_by_distance'
    yield 'tiny-tie', api, (np.array([4, 0, 2, 0, 1], dtype=np.int16), 2), {}

    rng = np.random.default_rng(10579)
    for case, shape, stride, p_norm, radius, spacing in (
        ('sparse-objects', (256, 384), 12, 2, 15, None),
        ('crowded-objects', (192, 256), 5, 2, 4., None),
        ('anisotropic-objects', (192, 256), 6, 1, 9., (1.5, 0.75)),
        ('anisotropic-volume', (32, 48, 64), 6, np.inf, 10., (2., 1., 0.5)),
    ):
        labels = np.zeros(shape, dtype=np.int32)
        tiled_shape = tuple(extent // stride for extent in shape)
        for label, corner in enumerate(np.ndindex(*tiled_shape), 1):
            block = tuple(slice(origin * stride + 1, origin * stride + int(rng.integers(2, stride)))
                          for origin in corner)
            labels[block] = label
        options = dict(p_norm=p_norm, spacing=spacing)
        if case == 'anisotropic-objects':
            options['priority'] = rng.integers(0, 17, size=label + 1).astype(np.float64)
        yield case, api, (labels, radius), options

    labels = np.zeros((256, 512), dtype=np.uint16)
    yield 'empty-background', api, (labels, 3), {}
