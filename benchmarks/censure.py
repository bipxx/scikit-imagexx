"""Fixed public CENSURE comparisons covering each bi-level filter."""

METHOD = dict(
    workers=1,
    samples=7,
    minimum_sample_ns=100_000_000,
    minimum_warmup_ns=250_000_000,
    call='detector construction plus detect on a prepared image, including output allocation',
)


def make_cases():
    import numpy as np

    image = np.random.default_rng(4121).integers(0, 256, size=(168, 192), dtype=np.uint8)
    rows, columns = np.ogrid[:168, :192]
    for row, column, radius in [(31, 53, 4), (52, 75, 7), (81, 54, 8), (105, 128, 13), (138, 98, 6)]:
        image[(rows - row) ** 2 + (columns - column) ** 2 <= radius ** 2] //= 6
    for mode in ('DoB', 'Octagon', 'STAR'):
        yield mode.lower(), (image,), dict(mode=mode)
