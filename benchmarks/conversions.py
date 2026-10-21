"""Conversion performance on additional integer widths, with strict parity."""
from run import measure
import argparse
from datetime import datetime, timezone
from provenance import source_fingerprint
import json
from pathlib import Path
import platform
import warnings
import numpy as np
import skimage
from skimage import util as original
import skimagexx as xx

FUNCTIONS = ['img_as_float', 'img_as_float32', 'img_as_float64', 'img_as_ubyte',
             'img_as_uint', 'img_as_int', 'img_as_bool']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--threads', nargs='+', type=int, default=[1, 4])
    parser.add_argument('--dtypes', nargs='+', default=['bool', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'uint64', 'int64'])
    parser.add_argument('--force-copy', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('benchmarks/conversion-widths.json'))
    args = parser.parse_args()
    if args.size < 1 or min(args.threads) < 1:
        parser.error('size and workers must be positive')
    xx.set_fallback_enabled(False)
    results = []
    rng = np.random.default_rng(2026)
    for dtype_name in args.dtypes:
        dtype = np.dtype(dtype_name)
        if dtype.kind not in 'bui':
            parser.error('this matrix covers integer and boolean inputs')
        a = rng.integers(0, 256, args.size*args.size*dtype.itemsize, dtype=np.uint8).view(dtype).reshape(args.size,args.size)
        if dtype.kind == 'b':
            a = (a.view(np.uint8) > 127)
        for name in FUNCTIONS:
            native = getattr(xx.util, name)
            reference = getattr(original, name)
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                expected = reference(a, force_copy=args.force_copy)
                for threads in args.threads:
                    xx.set_num_threads(threads)
                    actual = native(a, force_copy=args.force_copy)
                    assert actual.dtype == expected.dtype
                    np.testing.assert_array_equal(actual, expected)
                    assert (actual is a) == (expected is a)
                    timing = measure(lambda: reference(a, force_copy=args.force_copy),
                                     lambda: native(a, force_copy=args.force_copy), 7, 15)
                    kind = 'identity' if actual is a else ('copy' if actual.dtype == a.dtype else 'native')
                    result = {'api': 'util.'+name, 'input_dtype': dtype.name, 'output_dtype': actual.dtype.name,
                              'size': args.size, 'threads': threads, 'backend': kind, **timing}
                    results.append(result)
                    print(f'{name:16} {dtype.name:7} t={threads}: {timing["speedup"]:.2f}x ({kind})',flush=True)
    root = Path(__file__).resolve().parents[1]
    fingerprint = source_fingerprint(root)
    report = {'metadata': {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
                           'cpu': platform.processor(), 'reference': skimage.__version__, 'numpy': np.__version__,
                           **fingerprint, 'force_copy': args.force_copy,
                           'fallback_enabled': False, 'avx2': xx.has_avx2(), 'reference_thread_pools': 1}, 'results': results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines = ['# Conversion performance across integer widths', '',
             'Each case requires exact values, dtype, and identity behavior against scikit-image. '
             'Seven warmed, allocation-inclusive samples; strict dispatch, single-thread reference. '
             'Identity/copy rows measure wrapper or copying costs, not a conversion-kernel speedup. '
             'All ratios below one remain visible.', '',
             '| API | Input | Output | Workers | Execution | Original ms | Current ms | Speedup |',
             '|---|---|---|---:|---|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["api"]} | {r["input_dtype"]} | {r["output_dtype"]} | {r["threads"]} | {r["backend"]} | {r["reference_ms"]:.4f} | {r["cpp_ms"]:.4f} | {r["speedup"]:.2f}x |')
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
