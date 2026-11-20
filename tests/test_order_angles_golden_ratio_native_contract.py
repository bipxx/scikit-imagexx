"""Pure source/transport and actual sampler-entry refusal; science bodies0."""
import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import zlib

import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = json.loads((ROOT / 'coverage/contracts/order-angles-golden-ratio-native-v1.json').read_bytes())
BENCH = json.loads((ROOT / 'coverage/contracts/order-angles-golden-ratio-benchmark-v2.json').read_bytes())


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('name,pin', sorted({**NATIVE['source_canonical_sha256'], **NATIVE['dependency_canonical_sha256']}.items()))
def test_exact_native_and_dependency_bytes(name, pin):
    assert hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == pin


@pytest.mark.parametrize('name,row', sorted(BENCH['sources'].items()))
def test_benchmark_dependency_bytes(name, row):
    assert hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == row['current_canonical_sha256']


def test_true_lazy_frontend_and_owned_arithmetic():
    module = ast.parse((ROOT / 'python/skimagexx/_order_angles_golden_ratio.py').read_bytes())
    fn = next(n for n in module.body if isinstance(n, ast.FunctionDef))
    assert fn.name == 'order_angles_golden_ratio' and [a.arg for a in fn.args.args] == ['theta']
    assert any(isinstance(n, ast.Yield) for n in ast.walk(fn))
    calls = {ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)}
    assert not calls.intersection({'np.argsort', 'np.searchsorted', 'np.remainder', 'np.sort'})
    assert not any(isinstance(n, ast.ImportFrom) and (n.module or '').startswith('skimage.') for n in ast.walk(module))
    cpp = (ROOT / 'src/order_angles_golden_ratio.cpp').read_text()
    assert 'FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW' in cpp
    assert 'FE_INEXACT' not in cpp and 'std::fmod(a,b)' in cpp and 'std::stable_sort' in cpp


def test_complete_sdk_and_private_bridge_are_distinct():
    header = (ROOT / 'include/skimagexx/order_angles_golden_ratio.hpp').read_text()
    assert 'GoldenAngleOrder' in header and 'std::vector<std::int64_t> order_angles_golden_ratio' in header
    cmake = (ROOT / 'tests/sdk/order_angles_golden_ratio/CMakeLists.txt').read_text()
    assert 'find_package(skimagexx CONFIG REQUIRED)' in cmake and 'skimagexx::skimagexx' in cmake
    assert NATIVE['status'] == 'PRIVATE_ASSEMBLED_NOT_ADMITTED'


def test_frozen_four_cases_and_method_are_full_consumption():
    driver = load('benchmarks/order_angles_golden_ratio_native.py', '_gold_driver_cases')
    specs, factory = driver.case_specs()
    assert [s['id'] for s in specs] == ['singleton-zero', 'dtype-float32', 'eight', 'sixteen']
    assert all(factory.case_bounds(s)['n'] <= 16 for s in specs)
    assert all(value is None for value in BENCH['unbound'].values())
    assert {key: BENCH['method'][key] for key in ('threads', 'repeat', 'target_ms', 'warmup_min_ms')} == dict(threads=1, repeat=7, target_ms=100, warmup_min_ms=250)
    events = []
    def inert(theta):
        events.append(('first', theta))
        yield 2
        events.append(('second', theta))
        yield 1
    assert driver.public_call(inert, 'inert') == [2, 1]
    assert driver.public_call(inert, 'inert') == [2, 1]
    assert events == [('first', 'inert'), ('second', 'inert')] * 2


@pytest.mark.parametrize('cached', [False, True])
def test_actual_cli_unbound_before_import_and_no_path_leak(monkeypatch, cached):
    driver = load('benchmarks/order_angles_golden_ratio_native.py', '_gold_driver_cli')
    before = list(sys.path)
    if cached:
        monkeypatch.setitem(sys.modules, 'numpy', types.ModuleType('numpy'))
    monkeypatch.setattr(sys, 'argv', ['golden'])
    with pytest.raises(SystemExit, match='PROSPECTIVE_UNBOUND'):
        driver.execute()
    assert sys.path == before


@pytest.mark.parametrize('observer_kind', ['absent', 'nonstopping', 'stopping'])
def test_real_montage_entry_never_executes_body(observer_kind):
    driver = load('benchmarks/order_angles_golden_ratio_native.py', '_gold_driver_boundary')
    common, _ = driver.common_module()
    sampler = load('tools/montage_sustained.py', '_gold_real_montage')
    lines = []
    seen = []
    class Stopped(Exception):
        pass
    def observer(frame, event, value):
        if event == 'call' and frame.f_code is sampler.measure.__code__:
            seen.append('actual-entry')
            if observer_kind == 'stopping':
                raise Stopped()
    def trace(frame, event, value):
        if frame.f_code is sampler.measure.__code__ and event == 'line':
            lines.append(frame.f_lineno)
        return trace
    old_profile, old_trace = sys.getprofile(), sys.gettrace()
    try:
        sys.settrace(trace)
        # Replaces any unrelated observer with the mandatory real-entry wrapper.
        sys.setprofile(common.first_cohort_profile(sampler.measure, None if observer_kind == 'absent' else observer))
        with pytest.raises(Stopped if observer_kind == 'stopping' else AssertionError):
            sampler.measure(None, None, 7, 100, journal=None, case_index=0, case_id='inert', pass_index=0, deadline=None)
    finally:
        sys.setprofile(old_profile)
        sys.settrace(old_trace)
    assert not lines
    assert seen == ([] if observer_kind == 'absent' else ['actual-entry'])


def test_separate_timing_authority_and_c1_finally():
    driver = load('benchmarks/order_angles_golden_ratio_native.py', '_gold_driver_authority')
    common, context = driver.common_module()
    with pytest.raises(AssertionError, match='Root measurement manifest required'):
        common.load_measurement(None, {}, context, 'golden', [])
    tree = ast.parse((ROOT / 'benchmarks/order_angles_golden_ratio_native.py').read_bytes())
    assert any(isinstance(n, ast.Try) and any(isinstance(t, ast.Expr) and isinstance(t.value, ast.Call) and isinstance(t.value.func, ast.Name) and t.value.func.id == 'verify_authority' for t in n.finalbody) for n in ast.walk(tree))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'measure']
    assert len(calls) == 1 and next(k.value.value for k in calls[0].keywords if k.arg == 'pass_index') == 0
