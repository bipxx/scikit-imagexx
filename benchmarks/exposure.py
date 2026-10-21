"""Non-default exposure parameters; strict native execution vs scikit-image."""
from run import measure
import argparse
from datetime import datetime, timezone
from provenance import source_fingerprint
import json
from pathlib import Path
import platform
import numpy as np
import skimage
from skimage import exposure as reference
import skimagexx as xx

CASES = [
    ('adjust_gamma', {'gamma': 0.7, 'gain': 1.3}),
    ('adjust_gamma', {'gamma': 0.5, 'gain': 1.3}),
    ('adjust_gamma', {'gamma': 2, 'gain': 1.3}),
    ('adjust_log', {'gain': 0.7}),
    ('adjust_log', {'gain': 0.7, 'inv': True}),
    ('adjust_sigmoid', {'gain': 7, 'cutoff': 0.3}),
    ('adjust_sigmoid', {'gain': 7, 'cutoff': 0.7, 'inv': True}),
    ('rescale_intensity', {'in_range': (0, 0.7), 'out_range': (-1, 2)}),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path, default=Path('benchmarks/exposure-parameters.json'))
    args = parser.parse_args()
    if args.size < 1 or args.threads < 1:
        parser.error('size and threads must be positive')
    xx.set_fallback_enabled(False)
    xx.set_num_threads(args.threads)
    results = []
    for dtype in (np.uint8, np.float32, np.float64):
        a = np.random.default_rng(2026).integers(0, 256, (args.size,args.size), dtype=np.uint8)
        if dtype != np.uint8:
            a = (a/255).astype(dtype)
        for name, options in CASES:
            current = getattr(xx.exposure, name)
            original = getattr(reference, name)
            actual = current(a, **options)
            expected = original(a, **options)
            assert actual.dtype == expected.dtype
            np.testing.assert_allclose(actual, expected, rtol=2e-6 if dtype == np.float32 else 2e-13,
                                      atol=2e-6 if dtype == np.float32 else 2e-13)
            timing = measure(lambda: original(a, **options), lambda: current(a, **options), 7, 15)
            result = {'api': 'exposure.'+name, 'options': options, 'dtype': np.dtype(dtype).name,
                      'size': args.size, 'threads': args.threads, 'backend': 'native', **timing}
            results.append(result)
            print(f'{name} {options} {np.dtype(dtype).name}: {timing["speedup"]:.2f}x',flush=True)
    root = Path(__file__).resolve().parents[1]
    fingerprint = source_fingerprint(root)
    report = {'metadata': {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'cpu': platform.processor(),
                           'platform': platform.platform(), 'reference': skimage.__version__,
                           'numpy': np.__version__, 'avx2': xx.has_avx2(), **fingerprint,
                           'fallback_enabled': False, 'reference_thread_pools': 1}, 'results': results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines = ['# Exposure correction with non-default parameters', '',
             f'{args.size}² images; {args.threads} C++ worker(s). Strict native dispatch. '
             'Medians of seven warmed, allocation-inclusive batches against original scikit-image. '
             'Speedup = reference/current; below one means slower. General transcendental '
             'functions may use scalar libm even in AVX2 builds.', '',
             '| API | Options | Dtype | Original ms | Native ms | Speedup |', '|---|---|---|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["api"]} | `{r["options"]}` | {r["dtype"]} | {r["reference_ms"]:.4f} | {r["cpp_ms"]:.4f} | {r["speedup"]:.2f}x |')
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
