"""Fixed twelve-case, one-thread RGB storage comparison against scikit-image."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys


def case_specs(byteorder=sys.byteorder):
    swapped = '>' if byteorder == 'little' else '<'
    types = [('native-u8', 'u1', 'control'), ('native-f32', '=f4', 'control'),
             ('native-f64', '=f8', 'control'), ('native-f16', '=f2', 'target'),
             ('swapped-f32', swapped+'f4', 'target'), ('swapped-f64', swapped+'f8', 'target')]
    return [dict(case_id=f'{size}:{name}', size=size, storage=name,
                 storage_dtype=dtype, group=group, shape=[size, size, 3])
            for size in (256, 1024) for name, dtype, group in types]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, choices=[1], default=1)
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--output', type=Path, default=Path('build/ci-rgb2yuv-storage.json'))
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('repeat must be positive')
    # run sets reference thread-pool limits before importing numerical packages.
    from run import measure
    from provenance import source_fingerprint
    import numpy as np
    import skimage
    from skimage.color import colorconv
    import skimagexx as xx

    xx.set_fallback_enabled(False)
    xx.set_num_threads(args.threads)
    results = []
    for spec in case_specs():
        # Recreate the same logical seed for each storage representation.
        base = np.random.default_rng(81 + spec['size']).integers(0, 256, spec['shape'], dtype=np.uint8)
        dtype = np.dtype(spec['storage_dtype'])
        image = base.copy() if dtype.kind == 'u' else (base/255).astype(dtype)
        before = hashlib.sha256(image.tobytes()).hexdigest()
        expected = skimage.color.rgb2yuv(image)
        actual = xx.color.rgb2yuv(image)
        assert actual.shape == expected.shape == image.shape
        assert actual.dtype == expected.dtype and actual.dtype.metadata == expected.dtype.metadata
        # Exact existing test_color_transforms.compare allowance; never fitted.
        tol = 2e-6 if actual.dtype == np.float32 else 2e-13
        absolute = tol
        if actual.dtype == np.float32:
            matrix = colorconv.yuv_from_rgb
            magnitude = np.max(np.abs(image)[np.isfinite(image)], initial=0)
            absolute += 4*np.finfo(np.float32).eps*np.max(np.sum(np.abs(matrix), axis=1))*magnitude
        np.testing.assert_allclose(actual, expected, rtol=tol, atol=absolute)
        assert not np.shares_memory(actual, image)
        error = float(np.max(np.abs(actual.astype(float)-expected.astype(float))))
        timing = measure(lambda: skimage.color.rgb2yuv(image), lambda: xx.color.rgb2yuv(image), args.repeat, 15)
        assert xx.get_num_threads() == 1
        assert hashlib.sha256(image.tobytes()).hexdigest() == before
        results.append(dict(**spec, api='color.rgb2yuv', threads=1, dtype=image.dtype.name,
                            input_dtype_str=image.dtype.str, input_isnative=image.dtype.isnative,
                            input_shape=list(image.shape), input_sha256=before,
                            output_dtype=actual.dtype.str, max_abs_error=error, **timing))
        print(f"{spec['case_id']}: {timing['speedup']:.3f}x", flush=True)
    assert len(results) == 12
    assert all(len(r['samples_ms']) == 2 and all(len(s) == args.repeat for s in r['samples_ms']) for r in results)
    root = Path(__file__).resolve().parents[1]
    fingerprint = source_fingerprint(root)
    report = dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(), cpu=platform.processor(), reference=skimage.__version__, numpy=np.__version__,
        **fingerprint,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        fallback_enabled=False, avx2=xx.has_avx2(), threads=1, repeat=args.repeat,
        raw_samples=24*args.repeat, command=sys.argv,
        scope='Fixed12 RGB images: six new storage targets and six native controls. Input creation outside timing; public conversion, allocation, math and status dispatch inside timing. First original/current comparison, not a prior-C++ gain.'), results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    lines = ['# RGB storage: one-thread comparison', '',
             '| Case | Actual dtype | Native byte order | Original ms | Current ms | Original/current |',
             '|---|---|---|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['case_id']} | `{r['input_dtype_str']}` | {r['input_isnative']} | {r['reference_ms']:.6f} | {r['cpp_ms']:.6f} | {r['speedup']:.9f}x |")
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
