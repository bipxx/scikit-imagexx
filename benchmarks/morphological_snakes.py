"""Fixed one-worker public-call comparisons for morphological snakes."""

METHOD = dict(
    workers=1, samples=7, minimum_sample_ns=100_000_000, minimum_warmup_ns=250_000_000,
    call='public preprocessing or segmentation, including initialization and result allocation',
)


def case_specs():
    for name, api, shape, dtype, kwargs in (
        ('inverse-gaussian-planar', 'inverse_gaussian_gradient', (256, 288), 'float64',
         dict(alpha=100.0, sigma=1.0)),
        ('inverse-gaussian-default', 'inverse_gaussian_gradient', (256, 288), 'float32', {}),
        ('inverse-gaussian-volume', 'inverse_gaussian_gradient', (26, 52, 62), 'float32',
         dict(alpha=46.0, sigma=1.2)),
        ('morph-chan-vese-planar', 'morphological_chan_vese', (208, 224), 'float64',
         dict(num_iter=14, smoothing=1, lambda1=1.0, lambda2=1.3)),
        ('morph-chan-vese-volume', 'morphological_chan_vese', (23, 44, 51), 'float32',
         dict(num_iter=8, init_level_set='disk', smoothing=1)),
        ('morph-geodesic-planar-balloon', 'morphological_geodesic_active_contour',
         (220, 258), 'float32', dict(num_iter=12, smoothing=1, threshold='auto', balloon=0.8)),
        ('morph-geodesic-volume-balloon', 'morphological_geodesic_active_contour',
         (22, 43, 50), 'float64', dict(num_iter=6, smoothing=1, threshold=0.45, balloon=-1)),
    ):
        yield dict(id=name, api=f'segmentation.{api}', shape=shape, dtype=dtype, kwargs=kwargs)


def make_case(spec):
    import numpy as np

    axes = np.ogrid[tuple(slice(n) for n in spec['shape'])]
    coordinates = [axis / (size - 1) - 0.5 for axis, size in zip(axes, spec['shape'])]
    radius2 = sum((i + 1) * axis ** 2 for i, axis in enumerate(coordinates))
    envelope = np.exp(-18.0 * radius2)
    if spec['api'].endswith('geodesic_active_contour'):
        image = 0.85 - 0.66 * np.exp(-400.0 * (radius2 - 0.24) ** 2)
    else:
        image = 0.1 + 0.7 * envelope + 0.05 * sum(coordinates)
    return (image.astype(spec['dtype']),), dict(spec['kwargs'])
