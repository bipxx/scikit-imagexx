"""Frozen allocation-inclusive public Hough line comparisons."""

METHOD = dict(samples=7, minimum_sample_ns=100_000_000, minimum_warmup_ns=250_000_000,
              workers=1, call="public prepared mask and theta; include output allocation")


def benchmark_cases(np):
    m = np.zeros((160, 192), dtype=bool)
    m[38, :] = True
    m[:, 88] = True
    d = np.arange(160)
    m[d, d + 11] = True
    sy, sx = np.indices((128, 160))
    dy, dx = np.indices((80, 96))
    return (
        ("no-votes-default-64x96", np.zeros((64, 96), dtype=bool), None,
         "all-zero mask; default 180"),
        ("linework-default-160x192", m, None,
         "row 38, column 88 and diagonal x=y+11; default 180"),
        ("sparse-custom-257", ((17 * sx + 31 * sy + sx * sy) % 127 == 0).astype(np.uint8),
         np.linspace(-1.3, 1.3, 257), "(17*x+31*y+x*y)%127==0; linspace(-1.3,1.3,257)"),
        ("dense-custom-91", (5 * dx + 3 * dy) % 7 != 0, np.linspace(1., -1., 91),
         "(5*x+3*y)%7!=0; descending linspace(1,-1,91)"),
    )


def make_cases():
    import numpy as np

    for name, image, theta, recipe in benchmark_cases(np):
        yield name, (image,), {"theta": theta} if theta is not None else {}, recipe
