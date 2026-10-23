"""Allocation-inclusive Frangi and ridge-Hessian against scikit-image 0.26.0."""
from run import measure
from provenance import source_fingerprint
import argparse, hashlib, json, os, platform, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import scipy, skimage
from skimage import filters as reference
import skimagexx as xx


def corpus():
    rng = np.random.default_rng(4691)
    types = (np.bool_, np.uint8, np.int8, np.uint16, np.int16, np.uint32,
             np.int32, np.uint64, np.int64, np.float16, np.float32, np.float64)
    for name in ('frangi', 'hessian'):
        for shape in ((128, 129), (9, 11, 13)):
            for dtype in types:
                if np.dtype(dtype).kind == 'f':
                    a = rng.normal(size=shape).astype(dtype)
                else:
                    raw = rng.bit_generator.random_raw(np.prod(shape)).reshape(shape)
                    a = (raw & 1).astype(dtype) if dtype is np.bool_ else raw.astype(dtype)
                ordinary = dict(sigmas=[1])
                variants = [
                    ('contiguous', a, ordinary),
                    ('strided', a[::-1, ::2], ordinary),
                    ('fortran', np.asfortranarray(a), ordinary),
                    ('tiny', a[tuple(slice(0, 3) for _ in shape)], ordinary),
                    ('broadcast', np.broadcast_to(a[(slice(0, 1),)*a.ndim], a.shape), ordinary),
                    ('zero', np.zeros_like(a), ordinary),
                    ('zero-sigma', a, dict(sigmas=[0])),
                    ('default-scales', a, {}),
                    ('mirror', a, dict(sigmas=[1], mode='mirror')),
                    ('white-multiple', a, dict(sigmas=[.7, 1.3], black_ridges=False, mode='constant', cval=-.3)),
                    ('automatic-gamma', a, dict(sigmas=[1.3, .7], gamma=None)),
                    ('typed-gamma', a, dict(sigmas=[1], gamma=np.float64(.7), beta=np.float32(.8))),
                ]
                for label, image, options in variants:
                    yield name, label, image, options


def serial_options(options):
    return {k: {'value': float(v), 'dtype': v.dtype.name} if isinstance(v, np.generic) else v
            for k, v in options.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if min(args.threads) <= 0 or args.repeat <= 0:
        parser.error('positive workers/repeats required')
    if not args.check_only and args.output is None:
        parser.error('--output required for measurements')
    assert skimage.__version__ == '0.26.0'
    xx.set_fallback_enabled(False)
    cases = list(corpus())
    specs = [dict(api=name, case=label, dtype=a.dtype.name, shape=list(a.shape),
                  strides=list(a.strides), options=serial_options(options))
             for name, label, a, options in cases]
    package = Path(xx.__file__).parent
    report = {'metadata': {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
        'cpu': platform.processor(), 'logical_cpus': os.cpu_count(), 'python': sys.version,
        'numpy': np.__version__, 'scipy': scipy.__version__, 'scikit_image': skimage.__version__,
        'avx2': xx.has_avx2(), 'fallback_enabled': False, 'reference_thread_pools': 1,
        'repeat': args.repeat, 'target_ms': 15, 'cooldown_ms': 20, 'seed': 4691, 'command': sys.argv,
        'eigen_provider_2d': 'C++ analytic symmetric eigenvalues',
        'eigen_provider_3d': 'NumPy native double eigvalsh_lo LAPACK gufunc C loop; C++ matrix packing',
        'float32_exp': 'NumPy BSD-licensed scalar FMA approximation inside SIMD response',
        'float64_exp': 'NumPy native double exp ufunc C loop; whole-field host SIMD dispatch, separate buffers',
        'scope': 'Warmed allocation-inclusive Python calls, input conversion and output destruction included',
        **source_fingerprint(Path(__file__).resolve().parents[1]),
        'case_spec_sha256': hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest(),
        'binary_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('_core.*')},
    }, 'results': []}
    try:
        for name, label, a, options in cases:
            ref = getattr(reference, name)
            native = getattr(xx.filters, name)
            expected = ref(a, **options)
            first = None
            for workers in args.threads:
                xx.set_num_threads(workers)
                actual = native(a, **options)
                assert type(actual) is type(expected) is np.ndarray
                assert actual.dtype == expected.dtype and actual.shape == expected.shape
                assert actual.strides == expected.strides
                tolerance = 2e-6 if actual.dtype == np.float32 else 2e-13
                np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
                zeros = (actual == 0) & (expected == 0)
                np.testing.assert_array_equal(np.signbit(actual[zeros]), np.signbit(expected[zeros]))
                if first is None:
                    first = actual.copy()
                else:
                    np.testing.assert_array_equal(actual, first, strict=True)
                error = float(np.max(np.abs(actual-expected)))
                if args.check_only:
                    print(f'PARITY {name} rank={a.ndim} {a.dtype} {label} workers={workers} error={error}', flush=True)
                    continue
                row = dict(api='filters.'+name, rank=a.ndim, dtype=a.dtype.name,
                           output_dtype=actual.dtype.name, case=label, shape=list(a.shape),
                           options=serial_options(options), threads=workers, max_abs_error=error,
                           **measure(lambda: ref(a, **options), lambda: native(a, **options), args.repeat, 15))
                report['results'].append(row)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_bytes((json.dumps(report, indent=2)+'\n').encode())
                print(f'{name} rank={a.ndim} {a.dtype} {label} workers={workers}: {row["speedup"]:.3f}x', flush=True)
    finally:
        xx.set_num_threads(1)
        xx.set_fallback_enabled(True)


if __name__ == '__main__':
    main()
