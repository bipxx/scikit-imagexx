"""Fixed one-worker comparisons of public scale-space blob detectors."""

METHOD = dict(workers=1, samples=7, minimum_sample_ns=100_000_000, minimum_warmup_ns=250_000_000,
              call='public detector on a prepared 2D image, including output allocation and disposal')


def make_cases():
    import numpy as np

    rows, cols = np.ogrid[:248, :312]
    image = np.zeros((248, 312), dtype=np.float64)
    for index in range(20):
        row = 24 + (index // 5) * 58 + index % 3
        col = 25 + (index % 5) * 60 + index % 4
        sigma = 3 + (index * 7 % 13) / 2
        amplitude = (42 + index) / 65
        image += amplitude * np.exp(-((rows - row) ** 2 + (cols - col) ** 2) / (2 * sigma**2))
    uint8 = np.rint(255 * image).astype(np.uint8)

    yield 'separated_spots_float64_geometric', 'blob_dog', (image,), dict(max_sigma=12, threshold=0.09)
    yield 'separated_spots_uint8_anisotropic', 'blob_log', (uint8,), dict(
        min_sigma=(2, 2.5), max_sigma=(10, 12), num_sigma=8, threshold=0.15)
    yield 'separated_spots_uint8_linear', 'blob_doh', (uint8,), dict(
        min_sigma=2, max_sigma=14, num_sigma=9, threshold=0.0015)
