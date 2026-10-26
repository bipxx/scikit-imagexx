"""Fixed13 normalized/Hu cases on an independently verified installed product.

The common harness and fixture live outside the immutable product checkout.
Preparation is untimed; every public call includes allocation and provider work.
Importing this module and reading case_specs requires only the standard library.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]
APIS = ('measure.moments_normalized', 'measure.moments_hu')
REAL_IDS = ('u8-auto', 'f32-auto', 'central-layout-step', 'supplied-fractional-anisotropic')
COMMON_FILES = ('benchmarks/normalized_layouts.py', 'benchmarks/normalized_layouts_cases.json',
                'benchmarks/run.py', 'benchmarks/provenance.py',
                'python/skimagexx/coverage.py', 'python/skimagexx/_moments_derived_checks.py')
POOL_VARIABLES = ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS')


def saved_cases(path):
    value = json.loads(Path(path).read_bytes())
    assert value['schema_version'] == 1
    rows = value['cases']
    assert [r['id'] for r in rows] == list(REAL_IDS)
    for row in rows:
        assert row['shape'] == [4, 4] and row['dtype'] in ('<f4', '<f8')
        width = {'<f4': 4, '<f8': 8}[row['dtype']]
        assert len(row['logical_c_words_hex']) == 16
        assert all(len(word) == 2 * width and len(bytes.fromhex(word)) == width
                   for word in row['logical_c_words_hex'])
    return rows


def case_specs(saved):
    rows = [dict(case_id=f'synthetic/{api}/{dtype}/C', api=api, dtype=dtype, layout='C', input_shape=[4, 4])
            for dtype in ('uint8', 'float32', 'float64') for api in APIS
            if (api, dtype) != ('measure.moments_normalized', 'uint8')]
    rows += [dict(case_id=f'real/{row["id"]}/{layout}', api=APIS[0],
                  dtype={'<f4': 'float32', '<f8': 'float64'}[row['dtype']], layout=layout, input_shape=[4, 4])
             for row in saved for layout in ('C', 'F')]
    return rows


def prepare_cases(np, default_call, saved):
    """Keep the old five seed2026/default_call inputs; append saved C/F copies."""
    values = []
    for dtype in (np.uint8, np.float32, np.float64):
        rng = np.random.default_rng(2026)
        gray = rng.integers(0, 256, (256, 256), dtype=np.uint8)
        if dtype != np.uint8:
            gray = (gray / 255).astype(dtype)
        for api in APIS:
            if api == APIS[0] and dtype == np.uint8:
                continue
            args, options = default_call(api, gray)
            values.append((args[0], options))
    for row in saved:
        raw = bytes.fromhex(''.join(row['logical_c_words_hex']))
        value = np.frombuffer(raw, dtype=row['dtype']).reshape(4, 4)
        for layout in ('C', 'F'):
            values.append((value.copy(order=layout), {}))
    return list(zip(case_specs(saved), values))


def common_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in COMMON_FILES}


def check_output_layout(actual, expected):
    def metadata(value):
        return dict(output_dtype=value.dtype.str, output_strides=list(value.strides),
                    output_flags={name: bool(getattr(value.flags, name))
                                  for name in ('c_contiguous', 'f_contiguous', 'aligned', 'owndata', 'writeable')})
    layout = metadata(actual)
    if layout != metadata(expected):
        raise AssertionError('output dtype, strides or storage flags differ from the original')
    return layout


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--product-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, choices=[1], default=1)
    parser.add_argument('--repeat', type=int, choices=[7], default=7)
    args = parser.parse_args(argv)
    product = args.product_root.resolve()
    if ROOT == product or ROOT.is_relative_to(product):
        parser.error('the shared harness must be outside the immutable product root')
    # Existing helper sets reference pool caps before any numerical imports.
    from run import measure
    from provenance import source_fingerprint
    import numpy as np
    import skimage
    from skimage import measure as reference
    import skimagexx as xx
    from skimagexx import measure as native
    from skimagexx.coverage import default_call
    from skimagexx._moments_derived_checks import derived_default_error
    assert skimage.__version__ == '0.26.0' and np.__version__ == '1.24.4'
    assert all(os.environ.get(name) == '1' for name in POOL_VARIABLES)
    xx.set_fallback_enabled(False)
    xx.set_num_threads(1)
    assert xx.get_num_threads() == 1
    hashes = common_hashes()
    saved = saved_cases(ROOT / 'benchmarks/normalized_layouts_cases.json')
    prepared = prepare_cases(np, default_call, saved)
    assert len(prepared) == 13
    results = []
    for spec, (value, options) in prepared:
        assert value.shape == (4, 4) and value.dtype.name == spec['dtype']
        assert value.flags.owndata and (value.flags.c_contiguous if spec['layout'] == 'C' else value.flags.f_contiguous)
        name = spec['api'].split('.')[1]
        original, current = getattr(reference, name), getattr(native, name)
        raw = value.tobytes(order='C')
        assert xx.get_num_threads() == 1
        expected = original(value, **options)
        actual = current(value, **options)
        assert xx.get_num_threads() == 1
        error = derived_default_error(spec['api'], value, expected, actual)
        layout = check_output_layout(actual, expected)
        assert value.tobytes(order='C') == raw
        timing = measure(lambda: original(value, **options), lambda: current(value, **options), 7, 15)
        assert xx.get_num_threads() == 1 and value.tobytes(order='C') == raw
        results.append(dict(spec, threads=1, backend='native-partial', size=4, input_strides=list(value.strides),
                            input_c_sha256=hashlib.sha256(raw).hexdigest(), max_abs_error=error, **layout, **timing))
        print(f'{spec["case_id"]}: {timing["speedup"]:.6f}x', flush=True)
    assert hashes == common_hashes()
    report = dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
                                cpu=platform.processor(), reference=skimage.__version__, numpy=np.__version__,
                                **source_fingerprint(product), common_inputs_sha256=hashes,
                                product_root=str(product), repeat=7, workers_before=1, workers_after=xx.get_num_threads(),
                                fallback_enabled=False, avx2=xx.has_avx2(), reference_pool_caps={n: os.environ[n] for n in POOL_VARIABLES},
                                provider_frontier='Public calls may select the product-admitted batch or scalar normalization route; NumPy conversion/scalar providers and Hu wrapping remain explicit. No whole-pipeline or isolated-kernel claim.'),
                  results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
