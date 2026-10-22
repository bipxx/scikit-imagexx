"""Scalar-promotion and array controls, timed against scikit-image 0.26.0."""
from run import measure
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
import skimage
from skimage import color
import skimagexx as xx
from provenance import source_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min([args.size, args.repeat] + args.threads) < 1:
        parser.error('sizes, repeats and worker limits must be positive')
    xx.set_fallback_enabled(False)
    report = {'metadata': {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'platform': platform.platform(), 'python': sys.version,
        'numpy': np.__version__, 'scikit_image': skimage.__version__,
        'avx2': xx.has_avx2(), 'reference_thread_pools': 1,
        'repeat': args.repeat, 'target_ms': 15, 'cooldown_ms': 20,
        'fallback_enabled': False, 'command': sys.argv,
        **source_fingerprint(Path(__file__).resolve().parents[1]),
        'binary_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in Path(xx.__file__).parent.glob('_core.*')},
    }, 'results': []}
    rng = np.random.default_rng(864)
    for dtype in (np.float32, np.float64):
        first, second = rng.uniform(-100, 100, (2, args.size, args.size, 3)).astype(dtype)
        for layout in ('scalar', 'array'):
            a, b = (first[0, 0], second[0, 0]) if layout == 'scalar' else (first, second)
            for name, options in [('deltaE_cie76', {}), ('deltaE_ciede94', {}),
                                  ('deltaE_ciede94', {'kL': 2, 'k1': .048, 'k2': .014}),
                                  ('deltaE_ciede94', {'k1': np.float32(.731)})]:
                old, new = getattr(color, name), getattr(xx.color, name)
                expected = old(a, b, **options)
                for threads in args.threads:
                    xx.set_num_threads(threads)
                    actual = new(a, b, **options)
                    assert actual.dtype == expected.dtype and type(actual) is type(expected)
                    assert actual.shape == expected.shape
                    tol = 2e-6 if dtype == np.float32 else 2e-13
                    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)
                    row = {'api': 'color.' + name, 'input_dtype': np.dtype(dtype).name,
                           'output_dtype': actual.dtype.name, 'layout': layout, 'threads': threads,
                           'shape': list(a.shape), 'options': {k: float(v) for k, v in options.items()},
                           'option_types': {k: type(v).__name__ for k, v in options.items()},
                           'max_abs_error': float(np.max(np.abs(actual - expected))),
                           **measure(lambda: old(a, b, **options), lambda: new(a, b, **options),
                                     args.repeat, 15)}
                    report['results'].append(row)
                    print(f'{name} {np.dtype(dtype).name} {layout} {options} t={threads}: {row["speedup"]:.3f}x', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
