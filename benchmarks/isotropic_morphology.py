"""Fixed allocation-inclusive one-worker workloads for the four isotropic operations."""

METHOD = dict(samples=7, minimum_sample_ns=100_000_000, minimum_warmup_ns=250_000_000,
              workers=1, call='prepared public mask including output allocation')


def case_specs():
    for operation in ('erosion', 'dilation', 'opening', 'closing'):
        for name, shape, radius, spacing, pattern in (
            ('boundary', (11, 13), 3., None, 'dense'),
            ('anisotropic', (192, 224), 8., (0.625, 1.75), 'blocks'),
        ):
            yield dict(id=operation + ':' + name, api='morphology.isotropic_' + operation,
                       shape=shape, radius=radius, spacing=spacing, pattern=pattern, seed=16789)


def make_case(spec):
    import numpy as np

    rng = np.random.default_rng(spec['seed'])
    image = rng.random(spec['shape']) > 0.55
    if spec['pattern'] == 'blocks':
        image[9:44, 16:90] = True
        image[50:94, 21:80] = False
        image[90:135, 130:190] = True
    return (image,), dict(radius=spec['radius'], spacing=spec['spacing'])
