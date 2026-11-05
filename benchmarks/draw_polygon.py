"""Source-bounded polygon raster workloads; public calls with one worker."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'coverage/contracts/draw-polygon-current-runtime-v1.json'
REFERENCE = ROOT / 'coverage/contracts/draw-polygon-v1.json'


def source_safe(spec):
    """Bound original loops/allocation; this does not predict output pixels."""
    assert spec['api'] in ('draw.polygon', 'draw.polygon2mask')
    vertices = spec['vertices']
    assert type(vertices) is list and 1 <= len(vertices) <= 64
    assert all(type(v) is list and len(v) == 2 for v in vertices)
    assert all(type(x) in (int, float) and math.isfinite(x) and -512 <= x <= 512
               and (x == 0 or abs(x) >= 2.**-128) for v in vertices for x in v)
    # Integer coordinates in this range cast exactly in every admitted dtype.
    assert all(type(x) is int for v in vertices for x in v)
    assert spec['dtype'] in ('int32', 'float32', 'float64')
    assert spec['carrier'] in ('list', 'C', 'F', 'strided')
    shape = spec['shape']
    assert shape is None or (type(shape) is list and len(shape) == 2
                            and all(type(x) is int and 0 <= x <= 512 for x in shape))
    assert spec['api'] == 'draw.polygon' or shape is not None
    columns = [[v[d] for v in vertices] for d in (0, 1)]
    first = [int(max(0, min(v))) for v in columns]
    last = [math.ceil(max(v)) for v in columns]
    if shape is not None:
        last = [min(last[d], shape[d] - 1) for d in (0, 1)]
    scan = math.prod(max(0, last[d] - first[d] + 1) for d in (0, 1))
    mask = 0 if spec['api'] == 'draw.polygon' else math.prod(shape)
    assert scan <= 512**2 and scan * len(vertices) <= 64 * 1024**2
    return dict(scan_pixels=scan, edge_tests=scan * len(vertices),
                output_pairs_upper_bound=scan, mask_pixels=mask,
                output_payload_upper_bound=16 * scan if not mask else mask,
                original_list_items_upper_bound=2 * scan)


def fixture(spec, np):
    bounds = source_safe(spec)
    buffers = []
    carrier = spec['carrier']

    def array(values):
        if carrier == 'strided':
            values = [item for value in values for item in (value, value)]
        owner = np.array(values, dtype=spec['dtype'], order='F' if carrier == 'F' else 'C')
        buffers.append(owner)
        if carrier == 'strided':
            view = owner[::2]
            buffers.append(view)
            return view
        return owner

    vertices = spec['vertices']
    shape = None if spec['shape'] is None else tuple(spec['shape'])
    if spec['api'] == 'draw.polygon':
        columns = [[v[d] for v in vertices] for d in (0, 1)]
        if carrier != 'list':
            columns = [array(v) for v in columns]
        positional, options = tuple(columns), dict(shape=shape)
    else:
        positional = (shape, vertices if carrier == 'list' else array(vertices))
        options = {}
    return positional, options, buffers, bounds


def checked_output(actual, expected, buffers, np, snapshot, bound, coordinates):
    assert type(actual) is type(expected)
    outputs = actual if coordinates else (actual,)
    expected_outputs = expected if coordinates else (expected,)
    assert type(outputs) is tuple and len(outputs) == (2 if coordinates else 1)
    for value in outputs:
        assert type(value) is np.ndarray and value.dtype.metadata is None
        assert value.dtype == np.dtype(np.intp if coordinates else np.bool_)
        assert value.ndim == (1 if coordinates else 2)
        assert value.flags.owndata and value.base is None and value.flags.writeable and value.flags.aligned
        assert all(value is not other and not np.shares_memory(value, other)
                   for other in (*buffers, *expected_outputs))
    if coordinates:
        assert outputs[0] is not outputs[1] and not np.shares_memory(*outputs)
        assert outputs[0].size <= bound['output_pairs_upper_bound']
    assert snapshot(outputs) == snapshot(expected_outputs)
    output = snapshot(outputs)
    for value in output['arrays']:
        value['raw_sha256'] = hashlib.sha256(value.pop('raw')).hexdigest()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, choices=[1], default=1)
    parser.add_argument('--repeat', type=int, choices=[7], default=7)
    parser.add_argument('--deadline-utc')
    parser.add_argument('--output', type=Path, default=Path('build/ci-draw-polygon.json'))
    args = parser.parse_args()
    deadline = args.deadline_utc or (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    assert datetime.now(timezone.utc) < datetime.fromisoformat(deadline)
    journal = args.output.with_suffix('.batches.jsonl')
    assert not args.output.exists() and not journal.exists()
    sys.path.insert(0, str(ROOT / 'tools'))
    from montage_sustained import measure
    from provenance import source_fingerprint
    from draw_coordinates import snapshot
    import numpy as np
    import scipy
    import skimage
    import skimagexx as xx
    from skimagexx import _compat
    owned = importlib.import_module('skimagexx._draw_polygon')
    public = importlib.import_module('skimage.draw')
    original = importlib.import_module('skimage.draw.draw')
    mask_module = importlib.import_module('skimage.draw._polygon2mask')
    compiled = importlib.import_module('skimage.draw._draw')
    contract_raw, reference_raw = CONTRACT.read_bytes(), REFERENCE.read_bytes()
    contract, reference = json.loads(contract_raw), json.loads(reference_raw)
    assert np.dtype(np.intp).itemsize == 8 and skimage.__version__ == '0.26.0'
    runtime = reference['original_capture'][np.__version__]['runtime']
    for module, name in ((original, 'draw'), (mask_module, 'polygon2mask')):
        assert hashlib.sha256(Path(module.__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == runtime['source_sha256'][name]
    binaries = {}
    for name, pin in runtime['binaries'].items():
        body = Path(importlib.import_module(name).__file__).read_bytes()
        binaries[name] = hashlib.sha256(body).hexdigest()
        assert binaries[name] == pin['sha256']
    assert hashlib.sha256(reference_raw).hexdigest() == contract['reference_contract_sha256']
    assert hashlib.sha256(inspect.getsource(measure).encode()).hexdigest() == contract['method_sha256']
    for name, pin in contract['source_closure'].items():
        assert hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == pin
    pools = {k: os.environ[k] for k in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS')}
    assert set(pools.values()) == {'1'} and xx.get_num_threads() == 1
    xx.set_fallback_enabled(False)
    core = Path(xx._core.__file__)
    core_sha = hashlib.sha256(core.read_bytes()).hexdigest()
    before = source_fingerprint(ROOT)
    policy, handler, buffer_size = np.geterr(), np.geterrcall(), np.getbufsize()
    rows = []
    getter = xx._core.get_num_threads
    for index, spec in enumerate(contract['cases']):
        positional, options, buffers, bound = fixture(spec, np)
        name = spec['api'].split('.')[1]
        native, upstream = getattr(xx.draw, name), getattr(public, name)
        assert native is getattr(owned, name)
        assert str(inspect.signature(native)) == str(inspect.signature(upstream)) == reference['signatures'][spec['api']]
        input_state = [(a.tobytes(order='C'), a.shape, a.strides, a.dtype, a.dtype.metadata,
                        a.ctypes.data, a.flags.writeable, a.flags.owndata, a.flags.aligned) for a in buffers]
        list_state = json.dumps((positional, options)) if spec['carrier'] == 'list' else None
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            expected = upstream(*positional, **options)
        assert not notices
        calls, accepted, snapshots = [], [], []
        entries = {getattr(xx._core, v): v for v in contract['native_entries'][spec['api']]}
        assert all(inspect.isbuiltin(v) for v in entries)
        original_codes = {original.polygon.__code__, mask_module.polygon2mask.__code__}
        original_compiled = compiled._polygon

        def denied(*a, **kw):
            raise AssertionError('native polygon raster forwarded to original algorithm')

        def profile(frame, event, value):
            assert not (event == 'call' and frame.f_code in original_codes)
            assert not (event == 'c_call' and value is original_compiled)
            if event == 'c_return' and value in entries:
                calls.append(entries[value])
            if event == 'c_return' and value is getter:
                snapshots.append('get_num_threads')
            if event == 'return' and frame.f_code is native.__code__:
                accepted.append(frame.f_locals.get('native_result') is not None)

        saved = [(public, 'polygon', public.polygon), (public, 'polygon2mask', public.polygon2mask),
                 (original, 'polygon', original.polygon), (original, '_polygon', original._polygon),
                 (mask_module, 'polygon2mask', mask_module.polygon2mask),
                 (compiled, '_polygon', compiled._polygon), (_compat, 'fallback', _compat.fallback)]
        prior_profile = sys.getprofile()
        try:
            for obj, attribute, old in saved:
                setattr(obj, attribute, denied)
            sys.setprofile(profile)
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always')
                actual = native(*positional, **options)
        finally:
            sys.setprofile(prior_profile)
            for obj, attribute, old in saved:
                setattr(obj, attribute, old)
        assert not notices and calls == list(contract['native_entries'][spec['api']])
        assert accepted == [True] and snapshots == ['get_num_threads']
        output = checked_output(actual, expected, buffers, np, snapshot, bound, name == 'polygon')
        assert input_state == [(a.tobytes(order='C'), a.shape, a.strides, a.dtype, a.dtype.metadata,
                                a.ctypes.data, a.flags.writeable, a.flags.owndata, a.flags.aligned) for a in buffers]
        if list_state is not None:
            assert json.dumps((positional, options)) == list_state
        assert np.geterr() == policy and np.geterrcall() is handler and np.getbufsize() == buffer_size
        assert xx.get_num_threads() == 1 and not _compat._fallback_enabled
        timing = measure(lambda: upstream(*positional, **options), lambda: native(*positional, **options),
                         args.repeat, 100, journal=journal, case_index=index,
                         case_id=spec['id'], pass_index=1, deadline=deadline)
        assert input_state == [(a.tobytes(order='C'), a.shape, a.strides, a.dtype, a.dtype.metadata,
                                a.ctypes.data, a.flags.writeable, a.flags.owndata, a.flags.aligned) for a in buffers]
        if list_state is not None:
            assert json.dumps((positional, options)) == list_state
        assert xx.get_num_threads() == 1 and not _compat._fallback_enabled
        assert np.geterr() == policy and np.geterrcall() is handler and np.getbufsize() == buffer_size
        rows.append(dict(case_id=spec['id'], api=spec['api'], parameters=spec, bounds=bound, threads=1,
                         backend='partial-native-raster', actual_C_returns=calls, native_accepted=True,
                         global_worker_snapshots=snapshots, output=output, error=None, warnings=[],
                         max_abs_error=0., **timing))
    assert len(rows) == 12 and sum(sum(map(len, r['samples_ms'])) for r in rows) == 168
    assert source_fingerprint(ROOT) == before and hashlib.sha256(core.read_bytes()).hexdigest() == core_sha
    assert CONTRACT.read_bytes() == contract_raw and REFERENCE.read_bytes() == reference_raw
    cpu = platform.processor()
    if Path('/proc/cpuinfo').is_file():
        cpu = next((s.partition(':')[2].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                    if s.startswith('model name')), cpu)
    report = dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(), cpu_model=cpu, logical_cpu_count=os.cpu_count(), avx2_available=bool(xx.has_avx2()),
        python=sys.version, numpy=np.__version__, scipy=scipy.__version__, reference=skimage.__version__,
        **before, core_sha256=core_sha, original_binaries_sha256=binaries, threads=1, repeat=7,
        target_ms=100, warmup_calibration_min_ms=250, raw_samples=168, reference_pool_limits=pools,
        fallback_enabled=False, deadline_utc=deadline, contract_sha256=hashlib.sha256(contract_raw).hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        method_sha256=contract['method_sha256'], scope=contract['scope']), results=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    lines = ['# Polygon raster: initial one-worker comparison', '', contract['scope'], '',
             '| Case | Original ms | C++ ms | Original/C++ |', '|---|---:|---:|---:|']
    lines += [f"| {r['case_id']} | {r['reference_ms']:.9f} | {r['cpp_ms']:.9f} | {r['speedup']:.9f}x |" for r in rows]
    args.output.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
