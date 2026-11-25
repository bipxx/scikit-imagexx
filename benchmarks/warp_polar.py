"""One-worker public warp_polar benchmark against scikit-image 0.26.0."""

import argparse
import hashlib
import inspect
from pathlib import Path
import sys


def fixed_cases(np):
    rng = np.random.default_rng(605)
    yield ('linear-f32-default', rng.random((256, 384), dtype=np.float32), {})
    yield ('log-f64-gray-reflect', rng.random((512, 768)), {
        'scaling': 'log', 'radius': 338.25, 'center': (244.125, 387.25),
        'mode': 'reflect', 'output_shape': (512, 384),
    })
    yield ('linear-u8-rgb-edge', rng.integers(0, 256, size=(512, 384, 3), dtype=np.uint8), {
        'channel_axis': -1, 'center': (254.25, 193.5), 'mode': 'edge',
        'output_shape': (400, 300),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    # Importing run sets upstream worker pool limits before numerical imports.
    from run import measure
    import numpy as np
    import scipy
    import skimage
    from skimage import transform as original
    import skimagexx as xx
    from skimagexx._warp_polar import warp_polar
    from provenance import source_fingerprint
    from resize import binary_hashes, cpu_metadata, save

    assert skimage.__version__ == '0.26.0'
    assert xx.get_num_threads() == 1
    xx.set_fallback_enabled(False)
    native, stock = xx.transform.warp_polar, original.warp_polar
    assert native is warp_polar and inspect.signature(native) == inspect.signature(stock)
    assert inspect.isbuiltin(xx._core.warp_sample)
    root = Path(__file__).resolve().parents[1]
    source = source_fingerprint(root)
    results = []
    for name, image, options in fixed_cases(np):
        def reference_call():
            return stock(image, **options)

        def native_call():
            return native(image, **options)

        expected, actual = reference_call(), native_call()
        assert actual.dtype == expected.dtype
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
        del actual, expected
        results.append(dict(case=name, shape=image.shape, dtype=image.dtype.name, options=options,
                            **measure(reference_call, native_call, repeat=7, target_ms=15)))

    assert source == source_fingerprint(root)
    save({'metadata': dict(source, **cpu_metadata(), python=sys.version, numpy=np.__version__,
                           scipy=scipy.__version__, scikit_image=skimage.__version__, workers=1,
                           binary_sha256=binary_hashes(),
                           benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
          'results': results}, args.output)


if __name__ == '__main__':
    main()
