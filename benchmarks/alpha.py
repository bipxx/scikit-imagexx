"""Alpha compositing and grayscale expansion versus original scikit-image."""
from run import measure
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import warnings
import numpy as np
import skimage
from skimage import color as original
import skimagexx as xx
from provenance import source_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--output', type=Path, default=Path('benchmarks/alpha-parameters.json'))
    args = parser.parse_args()
    if min([args.size, args.repeat, *args.threads]) < 1:
        parser.error('size, repeats and workers must be positive')
    xx.set_fallback_enabled(False)
    root = Path(__file__).resolve().parents[1]
    fingerprint = source_fingerprint(root)
    rows = []
    rng = np.random.default_rng(114)

    def add(name, image, workload, options=None, layout='contiguous'):
        options = {} if options is None else options
        reference, native = getattr(original, name), getattr(xx.color, name)
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            expected = reference(image, **options)
        expected_notices = [(type(w.message).__name__, str(w.message)) for w in notices]
        tolerance = (2e-6 if expected.dtype == np.float32 else 2e-13) if name == 'rgba2rgb' else 0
        for workers in args.threads:
            xx.set_num_threads(workers)
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always')
                actual = native(image, **options)
            assert expected_notices == [(type(w.message).__name__, str(w.message)) for w in notices]
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True)
            timing = measure(lambda: reference(image, **options), lambda: native(image, **options), args.repeat, 15)
            finite = np.isfinite(actual) & np.isfinite(expected)
            error = float(np.max(np.abs(actual[finite].astype(np.float64)-expected[finite].astype(np.float64)), initial=0))
            rows.append({'api': 'color.'+name, 'size': args.size, 'shape': list(image.shape),
                         'dtype': str(image.dtype), 'output_dtype': str(actual.dtype), 'threads': workers,
                         'backend': 'native', 'layout': layout, 'workload': workload,
                         'parameter_repr': {key: (f'array(shape={value.shape},dtype={value.dtype})' if isinstance(value, np.ndarray) else repr(value)) for key, value in options.items()},
                         'rtol': tolerance, 'atol': tolerance, 'warnings': expected_notices,
                         'max_abs_error': error, **timing})
            print(f'{name:10} {str(image.dtype):8} {workload:24} t={workers}: {timing["speedup"]:.2f}x', flush=True)

    real = (np.bool_, np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64, np.float32, np.float64)

    def sample(dtype, channels=None):
        shape = (args.size, args.size) + (() if channels is None else (channels,))
        if dtype == np.bool_:
            return rng.random(shape) > .5
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            return rng.integers(int(info.min), int(info.max)+1, shape, dtype=dtype)
        return rng.random(shape).astype(dtype)

    for dtype in real:
        a = sample(dtype, 4)
        layouts = {'contiguous': a, 'reversed': a[::-1, ::-1],
                   'strided': np.tile(a, (2, 2, 1))[::2, ::2], 'fortran': np.asfortranarray(a),
                   'planar': np.ascontiguousarray(np.moveaxis(a, -1, 0))}
        for layout, image in layouts.items():
            add('rgba2rgb', image, layout, {'channel_axis': 0} if layout == 'planar' else {}, layout)
    for dtype in (*real, np.float16):
        a = sample(dtype)
        layouts = {'contiguous': a, 'reversed': a[::-1, ::-1],
                   'strided': np.tile(a, (2, 2))[::2, ::2], 'fortran': np.asfortranarray(a)}
        for layout, image in layouts.items():
            for name in ('gray2rgb', 'gray2rgba'):
                add(name, image, layout, layout=layout)
    for dtype in (np.uint8, np.float32, np.float64):
        a = sample(dtype, 4)
        for label, background in [('black', (0, 0, 0)), ('color', (.1, .5, .9)), ('nan', (np.nan, .3, .7))]:
            add('rgba2rgb', a, 'background-'+label, {'background': background})
        gray = sample(dtype)
        alpha = dtype(128 if dtype == np.uint8 else .5)
        for label, value in [('scalar', .5), ('typed-scalar', alpha),
                             ('array', np.full_like(gray, alpha)), ('broadcast', np.full((1, args.size), alpha, dtype=dtype))]:
            add('gray2rgba', gray, 'alpha-'+label, {'alpha': value})
    for dtype in ('>u2', '>f4', '>f8'):
        gray = sample(np.dtype(dtype).newbyteorder('=')).astype(dtype)
        for layout, image in [('contiguous', gray), ('reversed', gray[::-1, ::-1])]:
            for name in ('gray2rgb', 'gray2rgba'):
                add(name, image, 'endian-'+layout, layout=layout)
    assert fingerprint == source_fingerprint(root), 'source changed during measurements'
    metadata = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
                'cpu': platform.processor(), 'reference': skimage.__version__, 'numpy': np.__version__,
                'avx2': xx.has_avx2(), 'fallback_enabled': False, 'reference_thread_pools': 1,
                'repeat': args.repeat, **fingerprint,
                'scope': 'Allocation-inclusive warmed Python API; strict native dispatch; shape, dtype, warnings and values verified before timing'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'metadata': metadata, 'results': rows}, indent=2)+'\n', encoding='utf-8')
    lines = ['# Alpha and grayscale performance', '',
             'Original scikit-image '+skimage.__version__+'. Reference thread pools use one worker; native workers are explicit. '
             'Timings include dispatch, integer normalization, validation, layout handling and output allocation. '
             'Input construction is excluded. All samples and slower cases are retained in the JSON.', '',
             '| API | Input | Layout / workload | Workers | Original ms | C++ ms | Speedup |',
             '|---|---|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["api"]} | {row["dtype"]} | {row["workload"]} | {row["threads"]} | {row["reference_ms"]:.4f} | {row["cpp_ms"]:.4f} | {row["speedup"]:.2f}x |')
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
