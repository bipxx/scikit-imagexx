"""Fixed one-thread polygon comparison across work sizes and vertex counts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys


def case_specs():
    return [dict(case_id=f'{kind}:{size}:v{vertices}', kind=kind, size=size,
                 vertices=vertices, api='measure.' + name)
            for kind, name, sizes in (('points', 'points_in_poly', (4096, 65536)),
                                      ('grid', 'grid_points_in_poly', (256, 1024)))
            for size in sizes for vertices in (4, 32, 128)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, choices=[1], default=1)
    parser.add_argument('--repeat', type=int, choices=[7], default=7)
    parser.add_argument('--output', type=Path, default=Path('build/ci-polygon.json'))
    args = parser.parse_args()
    from run import measure
    from provenance import source_fingerprint
    import numpy as np
    import scipy
    import skimage
    import skimagexx as xx
    from skimagexx.coverage import polygon_default_error

    assert skimage.__version__ == '0.26.0' and xx.get_num_threads() == 1
    limits = {name: os.environ[name] for name in
              ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS')}
    assert set(limits.values()) == {'1'}
    xx.set_fallback_enabled(False)
    xx.set_num_threads(args.threads)
    rows = []
    for spec in case_specs():
        count, size = spec['vertices'], spec['size']
        angles = np.arange(count, dtype=np.float64) * (2 * np.pi / count)
        radii = np.where(np.arange(count) % 2 == 0, .4, .22)
        scale = 1. if spec['kind'] == 'points' else float(size - 1)
        vertices = scale * np.column_stack((.5 + radii * np.cos(angles),
                                            .5 + radii * np.sin(angles)))
        if spec['kind'] == 'points':
            points = np.random.default_rng(2026).random((size, 2))
            points[:4] = vertices[:4]
            call_args = (points, vertices)
            arrays = (points, vertices)
            output_shape = [size]
        else:
            call_args = ((size, size), vertices)
            arrays = (vertices,)
            output_shape = [size, size]
        before = [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]
        name = spec['api'].split('.', 1)[1]
        reference = getattr(skimage.measure, name)
        native = getattr(xx.measure, name)
        expected, actual = reference(*call_args), native(*call_args)
        error = polygon_default_error(None, actual, expected)
        assert error == 0 and list(actual.shape) == output_shape
        assert all(a.dtype == np.dtype('float64') for a in arrays)
        assert xx.get_num_threads() == 1
        timing = measure(lambda: reference(*call_args), lambda: native(*call_args), args.repeat, 17)
        assert xx.get_num_threads() == 1
        assert before == [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]
        rows.append(dict(**spec, backend='native', threads=1, dtype='float64',
            input_dtype_str=[a.dtype.str for a in arrays], input_shapes=[list(a.shape) for a in arrays],
            input_sha256=before, output_shape=output_shape, output_dtype=str(actual.dtype),
            output_owns_data=actual.flags.owndata,
            output_base_type=[type(actual.base).__module__, type(actual.base).__name__],
            output_raw_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
            binarize=True if spec['kind'] == 'grid' else None, max_abs_error=error, **timing))
        print(f"{spec['case_id']}: {timing['speedup']:.3f}x", flush=True)
    assert len(rows) == 12
    assert all([len(s) for s in r['samples_ms']] == [7, 7] for r in rows)
    scope = ('Twelve float64 workloads vary query/grid size and polygon complexity (4, 32, 128 vertices). '
             'Alternating-radius polygons and seed-2026 point clouds are prepared outside timing; '
             'the first four query points are exact vertices. Grid calls use default binarize=True. '
             'Every timed public call includes coordinate-column conversion, output allocation, '
             'ordered dual-ray classification and source memoryview/binarize adaptation. '
             'Only classifications use owned C++; coordinate/protocol/storage adaptation remains explicit. '
             'Exact metadata and raw bytes, including noncanonical bool labels, are checked before timing. '
             'This is a first original/current comparison, not a prior-C++ or full-profile claim.')
    root = Path(__file__).resolve().parents[1]
    report = dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(), cpu=platform.processor(), python=sys.version,
        reference=skimage.__version__, numpy=np.__version__, scipy=scipy.__version__,
        **source_fingerprint(root),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        threads=1, repeat=7, raw_samples=168, reference_pool_limits=limits,
        fallback_enabled=False, avx2=xx.has_avx2(), command=sys.argv,
        selected_apis=['measure.points_in_poly', 'measure.grid_points_in_poly'], provider_frontier=scope),
        results=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    lines = ['# Polygon classification: one-thread comparison', '', scope, '',
             '| Case | Original ms | Current ms | Original/current | Max error |',
             '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['case_id']} | {row['reference_ms']:.6f} | {row['cpp_ms']:.6f} | {row['speedup']:.9f}x | {row['max_abs_error']} |")
    args.output.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
