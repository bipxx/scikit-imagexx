"""Twelve prospective N-D line public workloads; one worker, exact row-view parity."""
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
CONTRACT = ROOT/'coverage/contracts/line-nd-benchmark.json'
REFERENCE = ROOT/'coverage/contracts/line-nd.json'


def source_safe(spec):
    """Audit this bounded large-case packet independently of the reference512 cap."""
    assert spec['api'] == 'draw.line_nd'
    start, stop = spec['start'], spec['stop']
    assert 1 <= len(start) == len(stop) <= 8
    assert set(spec['kwargs']) == {'endpoint', 'integer'}
    assert all(type(v) is bool for v in spec['kwargs'].values())
    assert spec['dtype'] in (None, 'float16', 'float32', 'float64', 'int8')
    assert spec['layout'] in ('C', 'reverse', 'readonly-unaligned')
    assert spec['layout'] == 'C' or spec['dtype'] == 'float64'
    assert all(type(v) in (int, float) and math.isfinite(v) and abs(v) <= 32769 for v in start+stop)
    if spec['dtype'] == 'int8':
        assert start == [120] and stop == [-120]
        delta = 16  # Exact audited source subtract/abs wrap, not mathematical240.
    else:
        if spec['dtype'] in ('float16', 'float32'):
            assert all(2*v == int(2*v) and abs(v) <= (8 if spec['dtype'] == 'float16' else 8193) for v in start+stop)
        delta = max(abs(b-a) for a,b in zip(start,stop))
    points = math.ceil(delta)+int(spec['kwargs']['endpoint'])
    assert 0 <= points <= 32769 and points*len(start) <= 2_000_000
    return dict(points=points, dimensions=len(start), output_elements=points*len(start))


def fixture(spec, np):
    source_safe(spec)
    args = [spec['start'].copy(), spec['stop'].copy()]
    if spec['dtype'] is not None:
        args = [np.array(v, dtype=spec['dtype']) for v in args]
    if spec['layout'] == 'reverse':
        args = [v[::-1] for v in args]
    elif spec['layout'] == 'readonly-unaligned':
        result = []
        for value in args:
            view = np.ndarray(value.shape, dtype=value.dtype, buffer=bytearray(value.nbytes+1), offset=1)
            view[:] = value
            view.flags.writeable = False
            result.append(view)
        args = result
    return tuple(args), spec['kwargs'].copy()


def checked_outputs(actual, expected, inputs, np, snapshot, bound):
    assert type(actual) is type(expected) is tuple and len(actual) == len(expected) == len(inputs[0])
    assert snapshot(actual) == snapshot(expected)
    assert sum(a.size for a in actual) <= bound
    for i, array in enumerate(actual):
        assert type(array) is np.ndarray and not array.flags.owndata
        assert type(array.base) is np.ndarray and array.flags.writeable and array.flags.aligned
        assert all(array is not other and not np.shares_memory(array, other) for other in expected)
        assert all(array is not other for other in actual[i+1:])  # Empty identity matters too.
        assert all(array is not value and not np.shares_memory(array, value) for value in inputs if isinstance(value, np.ndarray))
    assert snapshot(tuple(a.base for a in actual)) == snapshot(tuple(a.base for a in expected))
    assert [[bool(np.shares_memory(a,b)) for b in actual] for a in actual] == [[bool(np.shares_memory(a,b)) for b in expected] for a in expected]
    output = snapshot(actual)
    for array in output['arrays']:
        array['sha256'] = hashlib.sha256(array.pop('raw')).hexdigest()
    bases = snapshot(tuple(a.base for a in actual))
    for array in bases['arrays']:
        array['sha256'] = hashlib.sha256(array.pop('raw')).hexdigest()
    return dict(**output, bases=bases, output_aliases=[[bool(np.shares_memory(a,b)) for b in actual] for a in actual])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, choices=[1], default=1)
    parser.add_argument('--repeat', type=int, choices=[7], default=7)
    parser.add_argument('--deadline-utc')
    parser.add_argument('--output', type=Path, default=Path('build/ci-line-nd.json'))
    args = parser.parse_args()
    deadline = args.deadline_utc or (datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
    assert datetime.now(timezone.utc) < datetime.fromisoformat(deadline)
    journal = args.output.with_suffix('.batches.jsonl')
    assert not args.output.exists() and not journal.exists()
    sys.path.insert(0, str(ROOT/'tools'))
    from montage_sustained import measure
    from provenance import source_fingerprint
    from draw_coordinates import snapshot
    import numpy as np
    import scipy
    import skimage
    import skimagexx as xx
    from skimagexx import _compat
    original = importlib.import_module('skimage.draw.draw_nd')
    public = importlib.import_module('skimage.draw')
    owned = importlib.import_module('skimagexx._line_nd')
    prefix = 'numpy._core' if np.__version__.startswith('2.') else 'numpy.core'
    linspace = importlib.import_module(prefix+'.function_base')
    binary = importlib.import_module(prefix+'._multiarray_umath')
    contract_raw, reference_raw = CONTRACT.read_bytes(), REFERENCE.read_bytes()
    contract, reference = json.loads(contract_raw), json.loads(reference_raw)
    assert len(contract['cases']) == 12 and len({s['id'] for s in contract['cases']}) == 12
    assert hashlib.sha256(reference_raw).hexdigest() == contract['reference_contract_sha256']
    assert hashlib.sha256(inspect.getsource(measure).encode()).hexdigest() == contract['method_sha256']
    assert skimage.__version__ == '0.26.0'
    pins = contract['source_pins']
    assert hashlib.sha256(Path(original.__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == pins['original_python_sha256']
    assert hashlib.sha256(Path(linspace.__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == pins['numpy_linspace_source_sha256'][np.__version__]
    binary_sha = hashlib.sha256(Path(binary.__file__).read_bytes()).hexdigest()
    assert binary_sha == pins['numpy_binary_sha256'][np.__version__]
    limits = {key: os.environ[key] for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS')}
    assert set(limits.values()) == {'1'} and xx.get_num_threads() == 1
    xx.set_fallback_enabled(False)
    native, upstream = xx.draw.line_nd, public.line_nd
    assert str(inspect.signature(native)) == str(inspect.signature(upstream)) == reference['signature']
    entry = getattr(xx._core, contract['native_entry'])
    assert inspect.isbuiltin(entry) and entry.__module__ == xx._core.__name__ == 'skimagexx._core'
    core = Path(xx._core.__file__)
    core_sha = hashlib.sha256(core.read_bytes()).hexdigest()
    cpu_model = platform.processor()
    cpu_info = Path('/proc/cpuinfo')
    if cpu_info.is_file():
        cpu_model = next((line.partition(':')[2].strip() for line in cpu_info.read_text().splitlines() if line.startswith('model name')), cpu_model)
    cpu_count, avx2_available = os.cpu_count(), bool(xx._core.has_avx2())
    before = source_fingerprint(ROOT)
    policy, handler, buffer = np.geterr(), np.geterrcall(), np.getbufsize()
    rows = []
    for index, spec in enumerate(contract['cases']):
        bounds = source_safe(spec)
        positional, options = fixture(spec, np)
        def input_state():
            return [snapshot((a,)) if isinstance(a,np.ndarray) else a.copy() for a in positional], options.copy()
        frozen_input = input_state()
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            expected = upstream(*positional, **options)
        assert not notices and input_state() == frozen_input
        assert np.geterr() == policy and np.geterrcall() is handler and np.getbufsize() == buffer
        calls, accepted = [], []
        def profile(frame, event, value):
            assert not (event == 'call' and frame.f_code is upstream.__code__), 'cached original algorithm called by native path'
            if event == 'c_return' and value is entry:
                calls.append(contract['native_entry'])
            if event == 'return' and frame.f_code is owned.line_nd.__code__:
                accepted.append(frame.f_locals.get(contract['native_result_local']) is not None)
        def denied(*unused_args, **unused_options):
            raise AssertionError('admitted line_nd forwarded to original algorithm')
        saved = [(module, getattr(module, 'line_nd')) for module in (public, original)]
        saved_round, previous = original._round_safe, sys.getprofile()
        try:
            for module, _ in saved:
                module.line_nd = denied
            original._round_safe = denied
            sys.setprofile(profile)
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always')
                actual = native(*positional, **options)
        finally:
            sys.setprofile(previous)
            original._round_safe = saved_round
            for module, function in saved:
                module.line_nd = function
        assert not notices and calls == [contract['native_entry']] and accepted == [True]
        output = checked_outputs(actual, expected, positional, np, snapshot, contract['max_output_elements'])
        assert all(a.size == bounds['points'] for a in actual)
        assert input_state() == frozen_input and xx.get_num_threads() == 1 and not _compat._fallback_enabled
        assert np.geterr() == policy and np.geterrcall() is handler and np.getbufsize() == buffer
        timing = measure(lambda: upstream(*positional, **options), lambda: native(*positional, **options),
            args.repeat, 100, journal=journal, case_index=index, case_id=spec['id'], pass_index=1, deadline=deadline)
        assert input_state() == frozen_input and xx.get_num_threads() == 1 and not _compat._fallback_enabled
        assert np.geterr() == policy and np.geterrcall() is handler and np.getbufsize() == buffer
        rows.append(dict(case_id=spec['id'], api=spec['api'], parameters=spec, bounds=bounds, backend='native', threads=1,
            actual_C_returns=calls, native_accepted=True, output=output, error=None, warnings=[], max_abs_error=0.0, **timing))
        print(f"{spec['id']}: {timing['speedup']:.9f}x", flush=True)
    assert sum(sum(map(len,row['samples_ms'])) for row in rows) == contract['raw_samples'] == 168
    assert source_fingerprint(ROOT) == before and hashlib.sha256(core.read_bytes()).hexdigest() == core_sha
    assert CONTRACT.read_bytes() == contract_raw and REFERENCE.read_bytes() == reference_raw
    report = dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
        python=sys.version, numpy=np.__version__, scipy=scipy.__version__, reference=skimage.__version__,
        cpu_model=cpu_model, logical_cpu_count=cpu_count, avx2_available=avx2_available,
        **before, core_sha256=core_sha, numpy_binary_sha256=binary_sha, threads=1, repeat=7, raw_samples=168, target_ms=100,
        warmup_calibration_min_ms=250, deadline_utc=deadline, reference_pool_limits=limits,
        contract_sha256=hashlib.sha256(contract_raw).hexdigest(), method_sha256=contract['method_sha256'],
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        scope=contract['scope'], dtype_scope=contract['dtype_scope'], oracle_scope=contract['oracle_scope']),results=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines = ['# N-D line: initial one-thread comparison', '', contract['scope'], '',
        '| Case | Original ms | C++ ms | Original/current |', '|---|---:|---:|---:|']
    lines.extend(f"| {r['case_id']} | {r['reference_ms']:.9f} | {r['cpp_ms']:.9f} | {r['speedup']:.9f}x |" for r in rows)
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
