"""CIEDE2000 mixed-precision parity and public-call performance."""
from run import measure
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import warnings
import numpy as np
import skimage
from skimage.color import deltaE_ciede2000 as reference
import skimagexx as xx
from provenance import source_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min([args.size, args.repeat] + args.threads) < 1:
        parser.error('size, repetitions and worker limits must be positive')
    xx.set_fallback_enabled(False)
    rng = np.random.default_rng(386)
    cases = []
    for dtype in (np.float32, np.float64):
        a, b = rng.uniform(-100, 100, (2, args.size, args.size + 1, 3)).astype(dtype)
        layouts = [('contiguous', a, b, -1),
                   ('channel-first', np.ascontiguousarray(np.moveaxis(a, -1, 0)),
                    np.ascontiguousarray(np.moveaxis(b, -1, 0)), 0),
                   ('strided', a[:, ::2], b[::-1, ::2], -1),
                   ('scalar', a[0, 0], b[0, 0], -1)]
        for label, first, second, axis in layouts:
            for options in ({}, {'kL': 2, 'kC': .5}):
                cases.append((label, first, second, {'channel_axis': axis, **options}))
        if dtype == np.float32:
            for options in ({'kC': 0}, {'kH': 0}, {'kL': np.float32(2), 'kC': np.float32(.7)}):
                cases.append(('weight-control', a, b, options))
    root = Path(__file__).resolve().parents[1]
    data = np.loadtxt(root / 'tests/data/ciede2000-sharma.txt')
    cases.append(('Sharma-boundaries', np.tile(data[:, :3].astype(np.float32), (65, 1)),
                  np.tile(data[:, 3:6].astype(np.float32), (65, 1)), {}))
    report = {'metadata': {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'numpy': np.__version__, 'scikit_image': skimage.__version__,
        'python': sys.version, 'platform': platform.platform(),
        'avx2': xx.has_avx2(), 'reference_thread_pools': 1, 'fallback_enabled': False,
        'repeat': args.repeat, 'target_ms': 15, 'cooldown_ms': 20,
        'command': sys.argv, **source_fingerprint(root),
        'binary_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in Path(xx.__file__).parent.glob('_core.*')},
    }, 'results': []}
    for label, a, b, options in cases:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            expected = reference(a, b, **options)
            for threads in args.threads:
                xx.set_num_threads(threads)
                actual = xx.color.deltaE_ciede2000(a, b, **options)
                assert type(actual) is type(expected)
                assert actual.dtype == expected.dtype and actual.shape == expected.shape
                np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
                np.testing.assert_array_equal(np.isinf(actual), np.isinf(expected))
                tol = (2e-6, 2.5e-4) if a.dtype == np.float32 else (2e-13, 2e-13)
                np.testing.assert_allclose(actual, expected, rtol=tol[0], atol=tol[1])
                finite = np.isfinite(actual) & np.isfinite(expected)
                error = np.abs(np.asarray(actual)[finite] - np.asarray(expected)[finite])
                row = {'api': 'color.deltaE_ciede2000', 'input_dtype': a.dtype.name,
                       'output_dtype': actual.dtype.name, 'shape': list(a.shape),
                       'second_shape': list(b.shape), 'layout': label, 'threads': threads,
                       'options': {k: float(v) for k, v in options.items()},
                       'option_types': {k: type(v).__name__ for k, v in options.items()},
                       'max_abs_error': float(error.max()) if error.size else 0.,
                       **measure(lambda: reference(a, b, **options),
                                 lambda: xx.color.deltaE_ciede2000(a, b, **options), args.repeat, 15)}
                report['results'].append(row)
                print(f'{a.dtype.name} {label} {options} t={threads}: {row["speedup"]:.3f}x', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
