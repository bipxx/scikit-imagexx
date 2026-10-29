"""Twelve frozen drawing geometries; one-thread, allocation-inclusive comparison."""
import argparse
from datetime import datetime, timezone
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
CONTRACT = ROOT / 'coverage/contracts/draw-coordinates.json'
EXECUTION = ROOT / 'coverage/contracts/draw-coordinates-benchmark.json'


def case_specs():
    return json.loads(CONTRACT.read_bytes())['benchmark_cases']


def snapshot(result):
    arrays = []
    for value in result:
        base = value.base
        arrays.append(dict(dtype=value.dtype.str, dtype_metadata=value.dtype.metadata,
            shape=list(value.shape), strides=list(value.strides), own=bool(value.flags.owndata),
            writeable=bool(value.flags.writeable), aligned=bool(value.flags.aligned),
            c=bool(value.flags.c_contiguous), f=bool(value.flags.f_contiguous),
            base_type=type(base).__name__,
            base_shape=None if not hasattr(base, 'shape') else list(base.shape),
            base_strides=None if not hasattr(base, 'strides') else list(base.strides),
            raw=value.tobytes(order='C')))
    return dict(container=type(result).__name__, arrays=arrays,
                shared_bases=[[a.base is b.base for b in result] for a in result])


def oracle(spec, np):
    """Independent traversal/grid oracle; ellipse retains explicit NumPy math providers."""
    args, options = spec['args'], spec['kwargs']
    if spec['api'] == 'draw.line':
        r, c, last_r, last_c = args
        dr, dc = abs(last_r-r), abs(last_c-c)
        sr, sc = (1 if last_r > r else -1), (1 if last_c > c else -1)
        steep = dr > dc
        major, minor = (dr, dc) if steep else (dc, dr)
        error = 2*minor-major
        rows, columns = [], []
        for _ in range(major):
            rows.append(r); columns.append(c)
            if error >= 0:
                if steep: c += sc
                else: r += sr
                error -= 2*major
            if steep: r += sr
            else: c += sc
            error += 2*minor
        rows.append(last_r); columns.append(last_c)
        return [np.asarray(rows, dtype=np.intp), np.asarray(columns, dtype=np.intp)]
    if spec['api'] == 'draw.rectangle':
        start, extent = args[0], options['extent']
        shape = list(extent)
        if len(shape) > 1: shape[0], shape[1] = shape[1], shape[0]
        count = math.prod(shape)
        flat = np.arange(count, dtype=np.intp)
        results = []
        for axis, origin in enumerate(start):
            output_axis = 1-axis if axis < 2 and len(shape) > 1 else axis
            divisor = math.prod(shape[output_axis+1:])
            results.append((origin+(flat//divisor)%extent[axis]).reshape(shape))
        return results
    if spec['api'] == 'draw.disk':
        (r, c), radius = args
        rr, cc, angle = radius, radius, 0.0
    else:
        r, c, rr, cc = args
        angle = options.get('rotation', 0.0)
    angle %= np.pi
    sine, cosine = np.sin(angle), np.cos(angle)
    radii = np.array([abs(rr*cosine)+cc*sine, rr*sine+abs(cc*cosine)])
    low = np.ceil(np.array([r, c])-radii).astype(int)
    high = np.floor(np.array([r, c])+radii).astype(int)
    if options.get('shape') is not None:
        low = np.maximum(low, 0)
        high = np.minimum(high, np.asarray(options['shape'])-1)
    y = np.arange(low[0], high[0]+1, dtype=np.float64)[:, None]-r
    x = np.arange(low[1], high[1]+1, dtype=np.float64)[None, :]-c
    angle %= np.pi
    sine, cosine = np.sin(angle), np.cos(angle)
    mask = ((y*cosine+x*sine)/rr)**2+((y*sine-x*cosine)/cc)**2 < 1
    rows, columns = np.nonzero(mask)
    return [rows+low[0], columns+low[1]]


def checked_outputs(actual, expected, independent, np, bound):
    a, b = snapshot(actual), snapshot(expected)
    assert a == b and len(actual) == len(independent)
    assert sum(value.size for value in actual) <= bound
    for value, other, check in zip(actual, expected, independent):
        assert type(value) is np.ndarray and value.dtype == np.dtype(np.intp)
        assert value.dtype.metadata is None and value.shape == check.shape
        assert value.tobytes(order='C') == check.tobytes(order='C')
        assert not np.shares_memory(value, other)
    aliases = [[bool(np.shares_memory(x, y)) for y in actual] for x in actual]
    assert aliases == [[bool(np.shares_memory(x, y)) for y in expected] for x in expected]
    return dict(container=a['container'], arrays=[dict(
        **{key: value for key, value in item.items() if key != 'raw'},
        sha256=hashlib.sha256(item['raw']).hexdigest()) for item in a['arrays']],
        shared_bases=a['shared_bases'], output_aliases=aliases,
        reference_sha256=[hashlib.sha256(v.tobytes(order='C')).hexdigest() for v in expected],
        oracle_sha256=[hashlib.sha256(v.tobytes(order='C')).hexdigest() for v in independent])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, choices=[1], default=1)
    parser.add_argument('--repeat', type=int, choices=[7], default=7)
    parser.add_argument('--pass-index', type=int, choices=range(1, 9), default=1)
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--output', type=Path, default=Path('build/ci-draw-coordinates.json'))
    args = parser.parse_args()
    assert datetime.now(timezone.utc) < datetime.fromisoformat(args.deadline_utc)
    assert not args.output.exists() and not args.output.with_suffix('.batches.jsonl').exists()
    sys.path.insert(0, str(ROOT/'tools'))
    from montage_sustained import measure
    from provenance import source_fingerprint
    import numpy as np
    import scipy
    import skimage
    import skimagexx as xx
    from skimagexx import _compat
    original = importlib.import_module('skimage.draw.draw')
    compiled = importlib.import_module('skimage.draw._draw')
    public = importlib.import_module('skimage.draw')
    owned = importlib.import_module('skimagexx._draw_coordinates')
    contract_raw, execution_raw = CONTRACT.read_bytes(), EXECUTION.read_bytes()
    contract, execution = json.loads(contract_raw), json.loads(execution_raw)
    assert case_specs() == execution['cases'] and skimage.__version__ == '0.26.0'
    assert hashlib.sha256(contract_raw).hexdigest() == execution['reference_contract_sha256']
    assert hashlib.sha256(Path(original.__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == contract['source_canonical_lf_sha256']
    binary_sha = hashlib.sha256(Path(compiled.__file__).read_bytes()).hexdigest()
    assert binary_sha == contract['binary_sha256'][np.__version__]
    assert hashlib.sha256(inspect.getsource(measure).encode()).hexdigest() == execution['method_sha256']
    limits = {key: os.environ[key] for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS')}
    assert set(limits.values()) == {'1'} and xx.get_num_threads() == 1
    xx.set_fallback_enabled(False)
    core = Path(xx._core.__file__); core_sha = hashlib.sha256(core.read_bytes()).hexdigest()
    source_before = source_fingerprint(ROOT)
    policy, handler, buffer = np.geterr(), np.geterrcall(), np.getbufsize()
    rows = []
    for index, spec in enumerate(case_specs()):
        name = spec['api'].split('.')[1]
        reference, native = getattr(public, name), getattr(xx.draw, name)
        assert str(inspect.signature(reference)) == str(inspect.signature(native)) == contract['signatures'][spec['api']]
        parameters = json.loads(json.dumps(dict(args=spec['args'], kwargs=spec['kwargs'])))
        frozen_input = json.dumps(parameters, sort_keys=True)
        call_args, options = parameters['args'], parameters['kwargs']
        independent = oracle(spec, np)
        with warnings.catch_warnings(record=True) as reference_notices:
            warnings.simplefilter('always')
            expected = reference(*call_args, **options)
        entry_name = execution['native_entries'][spec['api']]
        entry = getattr(xx._core, entry_name)
        assert inspect.isbuiltin(entry) and entry.__module__ == xx._core.__name__ == 'skimagexx._core'
        code = owned.ellipse.__code__ if name == 'disk' else native.__code__
        calls, accepted = [], []
        original_codes = {getattr(original, n).__code__ for n in ('line', 'rectangle', 'disk', 'ellipse')}
        def denied(*a, **kw): raise AssertionError('admitted drawing forwarded to original algorithm')
        def profile(frame, event, value):
            if event == 'call' and frame.f_code in original_codes: denied()
            if event == 'c_return' and value is entry: calls.append(entry_name)
            if event == 'return' and frame.f_code is code:
                accepted.append(frame.f_locals.get('native_result') is not None)
        saved = [(module, n, getattr(module, n)) for module in (public, original) for n in ('line', 'rectangle', 'disk', 'ellipse')]
        saved_line, previous = compiled._line, sys.getprofile()
        try:
            for module, n, _ in saved: setattr(module, n, denied)
            compiled._line = denied
            sys.setprofile(profile)
            with warnings.catch_warnings(record=True) as native_notices:
                warnings.simplefilter('always')
                actual = native(*call_args, **options)
        finally:
            sys.setprofile(previous); compiled._line = saved_line
            for module, n, value in saved: setattr(module, n, value)
        assert calls == [entry_name] and accepted == [True]
        assert not reference_notices and not native_notices
        output = checked_outputs(actual, expected, independent, np, contract['max_output_elements'])
        assert json.dumps(parameters, sort_keys=True) == frozen_input
        assert xx.get_num_threads() == 1 and not _compat._fallback_enabled
        assert np.geterr() == policy and np.geterrcall() is handler and np.getbufsize() == buffer
        timing = measure(lambda: reference(*call_args, **options), lambda: native(*call_args, **options),
            args.repeat, 100, journal=args.output.with_suffix('.batches.jsonl'), case_index=index,
            case_id=spec['id'], pass_index=args.pass_index, deadline=args.deadline_utc)
        assert json.dumps(parameters, sort_keys=True) == frozen_input
        assert xx.get_num_threads() == 1 and not _compat._fallback_enabled
        assert np.geterr() == policy and np.geterrcall() is handler and np.getbufsize() == buffer
        rows.append(dict(case_id=spec['id'], api=spec['api'], parameters=parameters, backend='native',
            threads=1, actual_C_returns=calls, native_accepted=True, native_entry_module=entry.__module__,
            output=output, warnings=[], max_abs_error=0.0, **timing))
        print(f"{spec['id']}: {timing['speedup']:.9f}x", flush=True)
    assert len(rows) == 12 and all(list(map(len, row['samples_ms'])) == [7, 7] for row in rows)
    source_after = source_fingerprint(ROOT)
    assert source_before == source_after and hashlib.sha256(core.read_bytes()).hexdigest() == core_sha
    assert CONTRACT.read_bytes() == contract_raw and EXECUTION.read_bytes() == execution_raw
    report = dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
        cpu=platform.processor(), python=sys.version, numpy=np.__version__, scipy=scipy.__version__, reference=skimage.__version__,
        **source_after, source_before=source_before, core_path=str(core), core_sha256=core_sha,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        method_sha256=execution['method_sha256'], contract_sha256=hashlib.sha256(contract_raw).hexdigest(),
        execution_contract_sha256=hashlib.sha256(execution_raw).hexdigest(), original_source_sha256=contract['source_canonical_lf_sha256'],
        original_binary_sha256=binary_sha, threads=1, repeat=7, target_ms=100, warmup_calibration_min_ms=250,
        interbatch_sleep_ms=0, pass_index=args.pass_index, deadline_utc=args.deadline_utc, raw_samples=168,
        reference_pool_limits=limits, fallback_enabled=False, avx2=xx.has_avx2(), numpy_error_policy=policy,
        numpy_buffer_elements=buffer, command=sys.argv, selected_apis=contract['apis'], scope=execution['scope'],
        provider_frontier=execution['provider_frontier'], dtype_scope=execution['dtype_scope']), results=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    lines = ['# Drawing coordinates: one-thread comparison', '', execution['scope'], '',
             '| Case | Original ms | Current ms | Original/current |', '|---|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['case_id']} | {row['reference_ms']:.6f} | {row['cpp_ms']:.6f} | {row['speedup']:.9f}x |")
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__': main()
