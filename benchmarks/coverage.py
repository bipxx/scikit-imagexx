"""Every native public function versus original scikit-image, strict dispatch."""
from run import measure
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import platform
import numpy as np
import skimage
import skimagexx as xx
from skimagexx.coverage import native_functions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', type=int, default=[256, 1024])
    parser.add_argument('--threads', nargs='+', type=int, default=[1, 4])
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--output', type=Path, default=Path('benchmarks/function-coverage.json'))
    args = parser.parse_args()
    if min(args.sizes+args.threads+[args.repeat]) < 1:
        parser.error('sizes, threads, and repeat must be positive')
    xx.set_fallback_enabled(False)
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    files = sorted(list((root/'src').glob('*'))+list((root/'include/skimagexx').glob('*'))+
                   list((root/'python/skimagexx').glob('*.py')))
    for p in files:
        digest.update(str(p.relative_to(root)).encode()); digest.update(p.read_bytes())
    results = []
    for size in args.sizes:
        for dtype in (np.uint8, np.float32, np.float64):
            rng = np.random.default_rng(2026)
            gray = rng.integers(0, 256, (size, size), dtype=np.uint8)
            rgb = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
            if dtype != np.uint8:
                gray = (gray/255).astype(dtype); rgb = (rgb/255).astype(dtype)
            for api in native_functions():
                module, name = api.rsplit('.', 1)
                if name == 'threshold_otsu' and dtype != np.uint8:
                    continue
                native = getattr(importlib.import_module('skimagexx.'+module), name)
                reference = getattr(importlib.import_module('skimage.'+module), name)
                data = rgb if name == 'rgb2gray' else gray
                expected = reference(data)
                for threads in args.threads:
                    xx.set_num_threads(threads)
                    actual = native(data)
                    tol = 2e-6 if dtype == np.float32 else 2e-13
                    assert np.asarray(actual).dtype == np.asarray(expected).dtype
                    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)
                    timing = measure(lambda: reference(data), lambda: native(data), args.repeat, 15)
                    results.append({'api': api, 'size': size, 'dtype': np.dtype(dtype).name,
                                    'threads': threads, 'backend': 'native', 'max_abs_error':
                                    float(np.max(np.abs(np.asarray(actual, dtype=float)-np.asarray(expected, dtype=float)))),
                                    **timing})
                    print(f'{api:24} {size} {np.dtype(dtype).name} t={threads}: {timing["speedup"]:.2f}x',flush=True)
    assert set(r['api'] for r in results) == set(native_functions())
    report = {'metadata': {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
                           'cpu': platform.processor(), 'reference': skimage.__version__, 'numpy': np.__version__,
                           'source_sha256': digest.hexdigest(), 'fallback_enabled': False, 'avx2': xx.has_avx2(),
                           'repeat': args.repeat, 'timing': 'allocation-inclusive, warmed Python calls; original reference pools limited to one thread'},
              'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n',encoding='utf-8')
    lines = ['# Native function performance versus scikit-image', '',
             'Every native public function is checked before timing. Fallback is disabled; '
             'forwarding to scikit-image cannot count as native performance. Reference thread pools use one thread. '
             'Numbers are medians of warmed allocation-inclusive calls on seeded synthetic images. '
             'Speedup is reference/current; below 1 means slower. This does not prove full API parity or universal performance.', '',
             '| API | Size | Dtype | Workers | Reference ms | Native ms | Speedup |', '|---|---:|---|---:|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["api"]} | {r["size"]} | {r["dtype"]} | {r["threads"]} | {r["reference_ms"]:.4f} | {r["cpp_ms"]:.4f} | {r["speedup"]:.2f}x |')
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
