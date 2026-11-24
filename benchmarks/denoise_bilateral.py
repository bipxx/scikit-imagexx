"""Fixed one-worker public bilateral call recipes including NumPy preparation."""
METHOD = dict(samples=7, minimum_sample_ns=100_000_000, minimum_warmup_ns=250_000_000, workers=1)


def case_specs():
    for row in (
        ('small-gray', (32, 48), 'float32', {}),
        ('default-gray', (96, 128), 'float32', {}),
        ('wide-gray', (160, 192), 'float64', dict(win_size=15, bins=64, sigma_spatial=2.25, sigma_color=0.16, cval=0.125)),
        ('default-rgb', (96, 112, 3), 'float32', dict(channel_axis=-1)),
        ('rgb-low-bins', (128, 128, 3), 'float64',
         dict(channel_axis=-1, win_size=11, bins=257, sigma_spatial=1.5, sigma_color=0.125, cval=0.25)),
        ('wide-small', (43, 59), 'float32', dict(win_size=25, sigma_spatial=3., sigma_color=0.19)),
    ):
        name, shape, dtype, kwargs = row
        yield dict(id=name, api='restoration.denoise_bilateral', shape=shape, dtype=dtype, kwargs=kwargs)


def make_case(spec):
    import numpy as np

    rows, cols = spec['shape'][:2]
    y, x = np.indices((rows, cols))
    gray = 4 + (y * 17 + x * 13) % 19 + 12 * (x >= cols // 2) + 3 * (y % 5 == 0)
    if len(spec['shape']) == 3:
        gray = np.stack((gray, 3 + (y * 11 + x) % 26 + 12 * (y >= rows // 2),
                         5 + (y * 3 + x * 7) % 41), axis=-1)
    return (gray.astype(spec['dtype']) / 64,), spec['kwargs'].copy()
