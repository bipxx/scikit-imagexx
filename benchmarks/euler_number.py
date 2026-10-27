"""Fixed twelve-case, one-thread 2D/3D Euler comparison against scikit-image."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys


def case_specs():
    return [dict(case_id=f'{len(shape)}d:'+ 'x'.join(map(str, shape))+':'+dtype,
                 rank=len(shape), shape=list(shape), dtype=dtype)
            for shape in ((256, 256), (1024, 1024), (32, 32, 32), (64, 64, 64))
            for dtype in ('uint8', 'float32', 'float64')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, choices=[1], default=1)
    parser.add_argument('--repeat', type=int, choices=[7], default=7)
    parser.add_argument('--output', type=Path, default=Path('build/ci-euler-number.json'))
    args = parser.parse_args()
    # The unchanged measure module caps reference pools before numerical imports.
    from run import measure
    from provenance import source_fingerprint
    import numpy as np
    import scipy
    import skimage
    import skimagexx as xx
    from skimagexx.coverage import euler_default_error

    assert skimage.__version__ == '0.26.0'
    reference_pool_limits = {name: os.environ[name] for name in
        ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS')}
    assert set(reference_pool_limits.values()) == {'1'}
    assert xx.get_num_threads() == 1
    xx.set_fallback_enabled(False)
    xx.set_num_threads(args.threads)
    assert xx.get_num_threads() == 1
    results = []
    for spec in case_specs():
        # Recreating this seed gives identical binary pixels for all three dtypes.
        gray = np.random.default_rng(2026).integers(0, 256, spec['shape'], dtype=np.uint8)
        image = (gray > 127).astype(spec['dtype'])
        assert list(image.shape) == spec['shape'] and image.dtype.name == spec['dtype']
        before = hashlib.sha256(image.tobytes()).hexdigest()
        expected = skimage.measure.euler_number(image)
        actual = xx.measure.euler_number(image)
        assert type(actual) is type(expected)
        assert isinstance(expected, np.integer) if spec['rank'] == 2 else type(expected) is int
        assert actual == expected
        error = euler_default_error(image, actual, expected)
        assert error == 0.0 and xx.get_num_threads() == 1
        assert hashlib.sha256(image.tobytes()).hexdigest() == before
        timing = measure(lambda: skimage.measure.euler_number(image), lambda: xx.measure.euler_number(image), args.repeat, 15)
        assert xx.get_num_threads() == 1
        assert hashlib.sha256(image.tobytes()).hexdigest() == before
        results.append(dict(**spec, api='measure.euler_number', threads=1,
            input_dtype_str=image.dtype.str, input_shape=list(image.shape), input_sha256=before,
            connectivity=None, resolved_connectivity=spec['rank'],
            result_type=[type(actual).__module__, type(actual).__name__],
            output_dtype=str(actual.dtype) if hasattr(actual, 'dtype') else None,
            result=int(actual), reference_result=int(expected), max_abs_error=error, **timing))
        print(f"{spec['case_id']}: {timing['speedup']:.3f}x", flush=True)
    assert len(results) == 12
    assert all(len(r['samples_ms']) == 2 and all(len(s) == 7 for s in r['samples_ms']) for r in results)
    root = Path(__file__).resolve().parents[1]
    fingerprint = source_fingerprint(root)
    scope = ('Fixed twelve default-connectivity 2D/3D cases; seed2026 uint8 gray values >127 '
             'are cast to the actual input dtype outside timing, with identical binary pixels per geometry. '
             'Each public call includes binary conversion, padding, private storage preparation and allocation. '
             'NumPy binary conversion/padding and source LUT selection/scalar boxing remain providers; '
             'configuration histograms and bounded exact integer LUT arithmetic are owned C++. '
             'The shared Snum<=2**53 exact checker runs before timing. '
             'This first original/current comparison is not a prior-C++ gain or full-profile qualification.')
    report = dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(), cpu=platform.processor(), python=sys.version,
        reference=skimage.__version__, numpy=np.__version__, scipy=scipy.__version__, **fingerprint,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        reference_pool_limits=reference_pool_limits, fallback_enabled=False, avx2=xx.has_avx2(),
        threads=1, repeat=7, raw_samples=168, command=sys.argv, provider_frontier=scope,
        timing='Allocation-inclusive warmed public calls; unchanged seeded-shuffle paired measure helper.'), results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    lines = ['# Euler number: one-thread comparison', '', scope, '',
             '| Case | Shape | Actual dtype | Original ms | Current ms | Original/current | Max error |',
             '|---|---|---|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['case_id']} | {r['input_shape']} | `{r['input_dtype_str']}` | {r['reference_ms']:.6f} | {r['cpp_ms']:.6f} | {r['speedup']:.9f}x | {r['max_abs_error']} |")
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
