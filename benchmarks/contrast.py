"""Fixed allocation-inclusive low-contrast benchmark against scikit-image."""
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
import scipy
import skimage
from skimage import exposure as original
import skimagexx as xx
from provenance import source_fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=513)
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--baseline-dir', type=Path, help='Preserved initial _baseline package')
    parser.add_argument('--output', type=Path, default=Path('benchmarks/contrast-parameters.json'))
    args=parser.parse_args()
    if min([args.size, args.repeat, *args.threads])<1 or 1 not in args.threads:
        parser.error('positive size/repeat/workers and a one-worker comparison required')
    root=Path(__file__).resolve().parents[1]
    fingerprint=source_fingerprint(root)
    xx.set_fallback_enabled(False)
    baseline=None
    baseline_provenance=None
    if args.baseline_dir:
        sys.path.insert(0, str(args.baseline_dir.resolve()))
        import _baseline as baseline
        baseline.set_fallback_enabled(False)
        baseline_provenance=json.loads((args.baseline_dir/'baseline-source.json').read_text(encoding='utf-8'))
        binaries=[*Path(baseline.__file__).parent.glob('*.pyd'), *Path(baseline.__file__).parent.glob('*.so')]
        assert binaries and baseline.has_avx2()==xx.has_avx2()
        for binary in binaries:
            assert hashlib.sha256(binary.read_bytes()).hexdigest()==baseline_provenance['binary_sha256'][binary.name]
    rng=np.random.default_rng(9785)
    rows=[]

    def add(image, label, **options):
        reference=lambda: original.is_low_contrast(image, **options)
        native=lambda: xx.exposure.is_low_contrast(image, **options)
        previous=lambda: baseline.exposure.is_low_contrast(image, **options)
        with warnings.catch_warnings(record=True) as expected_notices:
            warnings.simplefilter('always')
            expected=reference()
        for workers in args.threads:
            xx.set_num_threads(workers)
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always')
                actual=native()
            assert type(actual) is type(expected)
            assert actual==expected
            key=lambda values: [(type(v.message).__name__, str(v.message)) for v in values]
            assert key(notices)==key(expected_notices)
            if baseline is not None:
                baseline.set_num_threads(workers)
                with warnings.catch_warnings(record=True) as previous_notices:
                    warnings.simplefilter('always')
                    old=previous()
                assert type(old) is type(expected) and old==expected
                assert key(previous_notices)==key(expected_notices)
            timing=measure(reference, native, args.repeat, 15,
                baseline=previous if baseline is not None else None)
            rows.append(dict(api='exposure.is_low_contrast', backend='native',
                dtype=str(image.dtype), shape=list(image.shape), workload=label,
                parameters=options, decision=bool(actual), threads=workers, **timing))
            print(f'{image.dtype} {label} workers={workers}: {timing["speedup"]:.2f}x', flush=True)

    for dtype in (np.bool_, np.uint8, np.int8, np.uint16, np.int16, np.uint32,
                  np.int32, np.uint64, np.int64, np.float16, np.float32, np.float64):
        descriptor=np.dtype(dtype)
        shape=(args.size, args.size+4)
        if descriptor.kind=='b': image=rng.random(shape)>.5
        elif descriptor.kind in 'iu':
            info=np.iinfo(dtype)
            image=rng.integers(max(-500, int(info.min)), min(500, int(info.max))+1, shape, dtype=dtype)
        else: image=rng.random(shape).astype(dtype)
        for label, view, options in [
            ('default', image, {}),
            ('full-range', image, dict(lower_percentile=0, upper_percentile=100)),
            ('reversed-percentiles', image, dict(lower_percentile=99, upper_percentile=1)),
            ('median-tail', image, dict(lower_percentile=49.25, upper_percentile=50.75)),
            ('fractional-tail', image, dict(lower_percentile=2.5, upper_percentile=97.5)),
            ('fortran', np.asfortranarray(image), {}),
            ('reversed', image[::-1, ::-1], {}),
            ('endian', image.astype(descriptor.newbyteorder('>')), {}),
            ('stepped', image[::2, 1::3], {}),
            ('broadcast', np.broadcast_to(image[:1], image.shape), {}),
        ]: add(view, label, **options)
    for dtype in (np.uint8, np.float32, np.float64):
        for channels in (3, 4):
            image=rng.random((args.size, args.size+4, channels))
            image=(image*255).astype(dtype) if dtype==np.uint8 else image.astype(dtype)
            add(image, 'rgb' if channels==3 else 'rgba')
    assert len(rows)==126*len(args.threads)
    assert source_fingerprint(root)==fingerprint, 'source changed during timing'
    package=Path(xx.__file__).parent
    report=dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(), cpu=platform.processor(), reference=skimage.__version__,
        numpy=np.__version__, scipy=scipy.__version__, avx2=xx.has_avx2(),
        reference_thread_pools=1, fallback_enabled=False, repeat=args.repeat,
        scope='Public dispatch, required color conversion, percentile selection and decision included; type, value and warnings checked before timing',
        current_binary_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*package.glob('*.pyd'), *package.glob('*.so')]}, **fingerprint), results=rows)
    if baseline_provenance is not None:
        report['metadata']['baseline']=baseline_provenance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    lines=['# Low-contrast detection performance', '',
        'Original scikit-image '+skimage.__version__+'. Reference pools use one thread; native worker limits are explicit. Public calls include all required color conversion and percentile selection. JSON retains every sample and slower case.', '',
        '| Input | Workload | Workers | Original ms | C++ ms | Speedup |',
        '|---|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["dtype"]} | {row["workload"]} | {row["threads"]} | {row["reference_ms"]:.4f} | {row["cpp_ms"]:.4f} | {row["speedup"]:.2f}x |')
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__=='__main__':
    main()
