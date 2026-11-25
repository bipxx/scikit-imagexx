"""Fixed one-thread prepared-image public comparisons for rotate and swirl."""


def fixed_cases(np):
    """Yield case name, transform attribute, prepared image, and call options."""
    rng = np.random.default_rng(47213)
    gray32 = rng.random((513, 517), dtype=np.float32)
    gray64 = rng.random((513, 517))
    rgb = rng.integers(0, 256, size=(257, 263, 3), dtype=np.uint8)
    yield 'gray-affine-linear', 'rotate', gray32, dict(angle=19.5)
    yield 'rgb-resize-linear', 'rotate', rgb, dict(angle=32.5, resize=True, mode='edge')
    yield 'gray-affine-cubic', 'rotate', gray64, dict(angle=-28, order=3, mode='reflect')
    yield 'gray-default-linear', 'swirl', gray32, {}
    yield 'rgb-resize-linear', 'swirl', rgb, dict(strength=2.2, radius=90, output_shape=(281, 251))
    yield 'gray-custom-cubic', 'swirl', gray64, dict(strength=-1.7, radius=84, order=3, mode='wrap')
