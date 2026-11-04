"""Nine frozen anti-aliased line workloads, actual native ownership and one worker."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'coverage/contracts/line-aa-benchmark.json'
REFERENCE = ROOT / 'coverage/contracts/line-aa.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, choices=[1], default=1)
    parser.add_argument('--repeat', type=int, choices=[7], default=7)
    parser.add_argument('--deadline-utc')
    parser.add_argument('--output', type=Path, default=Path('build/ci-line-aa.json'))
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
    original = importlib.import_module('skimage.draw.draw')
    public = importlib.import_module('skimage.draw')
    compiled = importlib.import_module('skimage.draw._draw')
    owned = importlib.import_module('skimagexx._line_aa')
    contract_raw, reference_raw = CONTRACT.read_bytes(), REFERENCE.read_bytes()
    contract, reference = json.loads(contract_raw), json.loads(reference_raw)
    assert contract['cases'] == reference['benchmark_cases'] and len(contract['cases']) == 9
    assert hashlib.sha256(reference_raw).hexdigest() == contract['reference_contract_sha256']
    assert hashlib.sha256(inspect.getsource(measure).encode()).hexdigest() == contract['method_sha256']
    assert skimage.__version__ == '0.26.0'
    assert hashlib.sha256(Path(original.__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == reference['original_python_sha256']
    assert hashlib.sha256(Path(compiled.__file__).read_bytes()).hexdigest() == reference['original_binary_sha256'][np.__version__]
    limits = {key: os.environ[key] for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS')}
    assert set(limits.values()) == {'1'} and xx.get_num_threads() == 1
    xx.set_fallback_enabled(False)
    native, upstream = xx.draw.line_aa, public.line_aa
    assert str(inspect.signature(native)) == str(inspect.signature(upstream)) == reference['signature']
    entry = xx._core.draw_line_aa
    assert inspect.isbuiltin(entry) and entry.__module__ == xx._core.__name__ == 'skimagexx._core'
    core = Path(xx._core.__file__)
    core_sha = hashlib.sha256(core.read_bytes()).hexdigest()
    cpu_model = platform.processor()
    cpu_info = Path('/proc/cpuinfo')
    if cpu_info.is_file():
        cpu_model = next((line.partition(':')[2].strip() for line in cpu_info.read_text().splitlines()
                          if line.startswith('model name')), cpu_model)
    cpu_count, avx2_available = os.cpu_count(), bool(xx._core.has_avx2())
    before = source_fingerprint(ROOT)
    policy, handler, buffer = np.geterr(), np.geterrcall(), np.getbufsize()
    carrier = np.empty((0,), dtype=np.uint8)
    rows = []
    for index, spec in enumerate(contract['cases']):
        positional = tuple(spec['args'])
        options = dict(spec['kwargs'])
        frozen_input = json.dumps((positional, options), sort_keys=True)
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            expected = upstream(*positional, **options)
        assert not notices
        calls, accepted = [], []

        def profile(frame, event, value):
            if event == 'c_return' and value is entry:
                calls.append(contract['native_entry'])
            if event == 'return' and frame.f_code is owned.line_aa.__code__:
                accepted.append(frame.f_locals.get('native_result') is not None)

        def denied(*unused_args, **unused_options):
            raise AssertionError('admitted line-AA forwarded to original algorithm')

        saved = [(module, getattr(module, 'line_aa')) for module in (public, original)]
        saved_compiled, previous = compiled._line_aa, sys.getprofile()
        try:
            for module, _ in saved:
                module.line_aa = denied
            compiled._line_aa = denied
            sys.setprofile(profile)
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always')
                actual = native(*positional, **options)
        finally:
            sys.setprofile(previous)
            compiled._line_aa = saved_compiled
            for module, function in saved:
                module.line_aa = function
        assert not notices and calls == [contract['native_entry']] and accepted == [True]
        assert type(actual) is type(expected) is tuple and len(actual) == len(expected) == 3
        assert snapshot(actual) == snapshot(expected)
        for i, array in enumerate(actual):
            assert type(array) is np.ndarray and array.dtype == np.dtype(np.float64 if i == 2 else np.intp)
            assert array.flags.owndata and array.base is None
            assert not np.shares_memory(array, carrier)
            assert all(not np.shares_memory(array, value) for value in expected)
            assert all(not np.shares_memory(array, value) for value in actual[i+1:])
        assert sum(value.size for value in actual) <= contract['max_output_elements']
        output = snapshot(actual)
        for array in output['arrays']:
            array['sha256'] = hashlib.sha256(array.pop('raw')).hexdigest()
        assert xx.get_num_threads() == 1 and not _compat._fallback_enabled
        assert np.geterr() == policy and np.geterrcall() is handler and np.getbufsize() == buffer
        timing = measure(lambda: upstream(*positional, **options), lambda: native(*positional, **options),
            args.repeat, 100, journal=journal, case_index=index, case_id=spec['id'], pass_index=1, deadline=deadline)
        assert json.dumps((positional, options), sort_keys=True) == frozen_input
        assert xx.get_num_threads() == 1 and not _compat._fallback_enabled
        assert np.geterr() == policy and np.geterrcall() is handler and np.getbufsize() == buffer
        rows.append(dict(case_id=spec['id'], api=spec['api'], parameters=spec, backend='native', threads=1,
            actual_C_returns=calls, native_accepted=True, output=output, max_abs_error=0.0, **timing))
        print(f"{spec['id']}: {timing['speedup']:.9f}x", flush=True)
    assert sum(sum(map(len, row['samples_ms'])) for row in rows) == 126
    assert source_fingerprint(ROOT) == before and hashlib.sha256(core.read_bytes()).hexdigest() == core_sha
    assert CONTRACT.read_bytes() == contract_raw and REFERENCE.read_bytes() == reference_raw
    report = dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
        python=sys.version, numpy=np.__version__, scipy=scipy.__version__, reference=skimage.__version__,
        cpu_model=cpu_model, logical_cpu_count=cpu_count, avx2_available=avx2_available,
        **before, core_sha256=core_sha, threads=1, repeat=7, raw_samples=126, target_ms=100,
        warmup_calibration_min_ms=250, deadline_utc=deadline, reference_pool_limits=limits,
        contract_sha256=hashlib.sha256(contract_raw).hexdigest(), method_sha256=contract['method_sha256'],
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        scope=contract['scope'], dtype_scope=contract['dtype_scope'], oracle_scope=contract['oracle_scope']), results=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    lines = ['# Anti-aliased line: initial one-thread comparison', '', contract['scope'], '',
             '| Case | Original ms | C++ ms | Original/current |', '|---|---:|---:|---:|']
    lines.extend(f"| {row['case_id']} | {row['reference_ms']:.9f} | {row['cpp_ms']:.9f} | {row['speedup']:.9f}x |" for row in rows)
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
