"""Fixed one-worker, allocation-inclusive public Canny comparisons."""

METHOD = dict(samples=7, minimum_sample_ns=100_000_000, minimum_warmup_ns=250_000_000,
              workers=1, call='public prepared input; include result allocation and disposal')


def make_cases():
    import numpy as np

    rng = np.random.default_rng(66938)
    image = rng.integers(0, 255, (144, 192), dtype=np.uint8)
    yield 'absolute-uint8', 'canny', (image,), {}

    image = rng.random((192, 256), dtype=np.float32)
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    mask = ((y - 96) / 85) ** 2 + ((x - 128) / 112) ** 2 < 1
    yield 'masked-float32', 'canny', (image,), dict(sigma=1.2, mask=mask, mode='reflect')

    image = rng.random((192, 256))
    yield 'quantile-float64', 'canny', (image,), dict(
        sigma=0.8, low_threshold=0.55, high_threshold=0.82, use_quantiles=True, mode='nearest',
    )
