"""Fixed local-mean parity/performance contract; run after the native port lands.

Every timed public call includes validation, allocation, conversion and padding.
Input construction is excluded. Fallbacks stay disabled; slower cases are retained.
"""
from run import measure
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import warnings
import numpy as np
import scipy
import skimage
from skimage import transform as original
import skimagexx as xx
from provenance import source_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=513)
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--output', type=Path, default=Path('benchmarks/downscale-parameters.json'))
    args = parser.parse_args()
    if min([args.size, args.repeat, *args.threads]) < 1:
        parser.error('size, repeat and worker counts must be positive')
    if 1 not in args.threads:
        parser.error('include one worker for serial verification')
    root = Path(__file__).resolve().parents[1]
    fingerprint = source_fingerprint(root)
    xx.set_fallback_enabled(False)
    rng = np.random.default_rng(9784)
    rows = []

    def add(image, label, factors, cval=0):
        def reference(): return original.downscale_local_mean(image, factors, cval=cval)
        def native(): return xx.transform.downscale_local_mean(image, factors, cval=cval)
        with warnings.catch_warnings(record=True) as expected_notices:
            warnings.simplefilter('always')
            expected = reference()
        warning_key = lambda notices: [(type(w.message).__name__, str(w.message)) for w in notices]
        volume = math.prod(factors)
        accumulator = np.float32 if image.dtype.kind == 'f' and image.itemsize <= 4 else np.float64
        # Ordinary reduction-order error scales with the input, including when
        # the output mean nearly cancels. Half outputs additionally round once.
        scale = max(1., abs(float(cval)), float(np.max(np.abs(image.astype(np.float64)), initial=0)))
        atol = float(8*np.finfo(accumulator).eps*scale*max(1., math.log2(volume)))
        rtol = float(2*np.finfo(expected.dtype).eps)
        for workers in args.threads:
            xx.set_num_threads(workers)
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always')
                actual = native()
            assert warning_key(notices) == warning_key(expected_notices)
            assert type(actual) is type(expected)
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            assert actual.strides == expected.strides
            assert actual.dtype.metadata == expected.dtype.metadata
            assert not np.shares_memory(actual, image)
            np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
            timing = measure(reference, native, args.repeat, 15)
            rows.append(dict(api='transform.downscale_local_mean', backend='native', dtype=str(image.dtype),
                shape=list(image.shape), output_shape=list(actual.shape), output_dtype=str(actual.dtype),
                output_dtype_metadata=None if actual.dtype.metadata is None else dict(actual.dtype.metadata),
                workload=label, factors=list(factors), cval=cval, threads=workers,
                rtol=rtol, atol=atol, max_abs_error=float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64)), initial=0)),
                **timing))
            print(f'{image.dtype} {label} workers={workers}: {timing["speedup"]:.2f}x', flush=True)

    for dtype in (np.bool_, np.uint8, np.int8, np.uint16, np.int16, np.uint32,
                  np.int32, np.uint64, np.int64, np.float16, np.float32, np.float64):
        descriptor = np.dtype(dtype)
        shape = (args.size, args.size+4)
        if descriptor.kind == 'b': image = rng.random(shape) > .5
        elif descriptor.kind in 'iu':
            limits = np.iinfo(dtype)
            image = rng.integers(max(-500, int(limits.min)), min(500, int(limits.max))+1, shape, dtype=dtype)
        else: image = rng.normal(size=shape).astype(dtype)
        for label, view, factors, cval in (
            ('factor2', image, (2, 2), 0),
            ('factor4', image, (4, 4), 0),
            ('anisotropic', image, (3, 5), 0),
            ('fortran', np.asfortranarray(image), (2, 3), 0),
            ('reversed', image[::-1, ::-1], (2, 3), 0),
            ('endian', image.astype(descriptor.newbyteorder('>')), (2, 3), 0),
            ('large-block', image, (129, 127), 0),
            ('constant-padding', image, (3, 5), 1),
            ('unit-factors', image, (1, 1), 0),
            ('single-row', image[:1], (1, 2), 0),
        ): add(view, label, factors, cval)
    assert len(rows) == 120*len(args.threads)
    assert source_fingerprint(root) == fingerprint, 'source changed during timing'
    report = dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(), cpu=platform.processor(), reference=skimage.__version__,
        numpy=np.__version__, scipy=scipy.__version__, avx2=xx.has_avx2(),
        reference_thread_pools=1, fallback_enabled=False, repeat=args.repeat,
        scope='Allocation-inclusive public raw-value means; dtype, layout, warnings and scale/depth reduction tolerances checked before timing',
        **fingerprint), results=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    lines=['# Local-block mean performance','',
        'Original scikit-image '+skimage.__version__+'. Reference pools use one thread; native worker limits are explicit. Every call includes allocation, dispatch, constant casting and required conversion. Values, shapes, dtypes, metadata, strides and warnings are checked before timing. All samples and slower cases remain in JSON.','',
        '| Input | Shape | Workload | Factors | Workers | Original ms | C++ ms | Speedup |',
        '|---|---|---|---|---:|---:|---:|---:|']
    for row in rows:
        shape='×'.join(map(str,row['shape']));factors='×'.join(map(str,row['factors']))
        lines.append(f'| {row["dtype"]} | {shape} | {row["workload"]} | {factors} | {row["threads"]} | {row["reference_ms"]:.4f} | {row["cpp_ms"]:.4f} | {row["speedup"]:.2f}x |')
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
