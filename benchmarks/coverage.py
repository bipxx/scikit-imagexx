"""Every native public function versus original scikit-image, strict dispatch."""
from run import measure
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import platform
import warnings
import numpy as np
import skimage
import skimagexx as xx
from skimagexx.coverage import native_functions, native_dtypes, execution_kind, needs_color_input, default_call, parity_tolerance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', type=int, default=[256, 1024])
    parser.add_argument('--threads', nargs='+', type=int, default=[1, 4])
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--modules', nargs='+', help='Benchmark only these registry modules; default covers all')
    parser.add_argument('--output', type=Path, default=Path('benchmarks/function-coverage.json'))
    args = parser.parse_args()
    if min(args.sizes+args.threads+[args.repeat]) < 1:
        parser.error('sizes, threads, and repeat must be positive')
    xx.set_fallback_enabled(False)
    # Binary morphology deprecation text is tested separately before timing.
    warnings.filterwarnings('ignore',message=r'`binary_.*` is deprecated since version 0.26',category=FutureWarning)
    apis = [api for api in native_functions() if args.modules is None or api.rsplit('.',1)[0] in args.modules]
    if not apis:
        parser.error('module selection matches no registered entry points')
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
            for api in apis:
                module, name = api.rsplit('.', 1)
                if np.dtype(dtype).name not in native_dtypes(api):
                    continue
                native = getattr(importlib.import_module('skimagexx.'+module), name)
                reference = getattr(importlib.import_module('skimage.'+module), name)
                data = rgb if needs_color_input(api) else gray
                call_args,options = default_call(api,data)
                expected = reference(*call_args,**options)
                for threads in args.threads:
                    xx.set_num_threads(threads)
                    actual = native(*call_args,**options)
                    rtol,atol = parity_tolerance(api,np.dtype(dtype))
                    assert np.asarray(actual).dtype == np.asarray(expected).dtype
                    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
                    timing = measure(lambda: reference(*call_args,**options), lambda: native(*call_args,**options), args.repeat, 15)
                    backend = 'identity' if actual is data else execution_kind(api)
                    results.append({'api': api, 'size': size, 'dtype': np.dtype(dtype).name,
                                    'threads': threads, 'backend': backend, 'max_abs_error':
                                    float(np.max(np.abs(np.asarray(actual, dtype=float)-np.asarray(expected, dtype=float)))),
                                    **timing})
                    print(f'{api:24} {size} {np.dtype(dtype).name} t={threads}: {timing["speedup"]:.2f}x',flush=True)
    assert set(r['api'] for r in results) == set(apis)
    report = {'metadata': {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
                           'cpu': platform.processor(), 'reference': skimage.__version__, 'numpy': np.__version__,
                           'source_sha256': digest.hexdigest(), 'fallback_enabled': False, 'avx2': xx.has_avx2(),
                           'selected_apis': apis,
                           'special_default_arguments': {'color.deltaE_cie76':'second input is first[..., ::-1]; raw Lab values',
                               'color.deltaE_ciede94':'second input is first[..., ::-1]; raw Lab; graphic-arts weights',
                               'color.deltaE_ciede2000':'second input is first[..., ::-1]; raw Lab; kL=kC=kH=1',
                               'color.deltaE_cmc':'second input is first[..., ::-1]; raw Lab; kL=kC=1',
                               'color.convert_colorspace':"fromspace='RGB', tospace='HSV'",
                               'color.separate_stains':'conv_matrix=hed_from_rgb',
                               'color.combine_stains':'conv_matrix=rgb_from_hed'},
                           'repeat': args.repeat, 'timing': 'allocation-inclusive, warmed Python calls; original reference pools limited to one thread'},
              'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n',encoding='utf-8')
    lines = ['# Native function performance versus scikit-image', '',
             'Every native public function is checked before timing. Fallback is disabled; '
             'forwarding to scikit-image cannot count as native performance. Reference thread pools use one thread. '
             'Numbers are medians of warmed allocation-inclusive calls on seeded synthetic images. '
             'Identity calls return the input unchanged; metadata calls return small values/arrays without processing pixels. '
             'Speedup is reference/current; below 1 means slower. This does not prove full API parity or universal performance.', '',
             '| API | Size | Dtype | Workers | Execution | Reference ms | Current ms | Speedup |', '|---|---:|---|---:|---|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["api"]} | {r["size"]} | {r["dtype"]} | {r["threads"]} | {r["backend"]} | {r["reference_ms"]:.4f} | {r["cpp_ms"]:.4f} | {r["speedup"]:.2f}x |')
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
