"""Prospective source and refused-original input intake; no target or timing."""
import ast
import builtins
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import pytest
if __package__:
    from .test_draw_coordinates_public import without_draw_activity
    from .draw_coordinates_observer import scientific_profile
else:
    from test_draw_coordinates_public import without_draw_activity
    from draw_coordinates_observer import scientific_profile
ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT/'coverage/contracts/line-nd-benchmark.json').read_bytes())['cases']


def load_driver():
    spec = importlib.util.spec_from_file_location('line_nd_benchmark_probe',ROOT/'benchmarks/line_nd.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_line_nd_benchmark_source_and_independent_large_bounds():
    reference = (ROOT/'coverage/contracts/line-nd.json').read_bytes()
    contract = json.loads((ROOT/'coverage/contracts/line-nd-benchmark.json').read_bytes())
    assert contract['reference_contract_sha256'] == hashlib.sha256(reference).hexdigest()
    assert len(CASES) == len({c['id'] for c in CASES}) == 12
    assert contract['threads'] == 1 and contract['repeat'] == 7
    assert contract['raw_samples'] == 12*2*7 and contract['warmup_groups'] == 24
    assert contract['target_ms'] == 100 and contract['warmup_min_ms'] == 250
    method_spec = importlib.util.spec_from_file_location('line_nd_sampler_source_probe',ROOT/'tools/montage_sustained.py')
    method = importlib.util.module_from_spec(method_spec)
    method_spec.loader.exec_module(method)
    # Same-interpreter inspect source as actual launcher, never cross-version AST hashes.
    assert hashlib.sha256(inspect.getsource(method.measure).encode()).hexdigest() == contract['method_sha256']
    driver = load_driver()
    bounds = [driver.source_safe(spec) for spec in CASES]
    assert max(v['points'] for v in bounds) == 32769
    assert max(v['output_elements'] for v in bounds) == 65538 < contract['max_output_elements'] == 2_000_000
    assert max(v['points'] for v in bounds) > json.loads(reference)['limits']['max_points']
    assert contract['max_dimensions'] == 8 and contract['max_points'] == 32769
    assert {s['dtype'] for s in CASES} == {None,'float16','float32','float64','int8'}
    assert {s['layout'] for s in CASES} == {'C','reverse','readonly-unaligned'}
    assert any(v['points'] == 0 for v in bounds) and any(v['dimensions'] == 1 for v in bounds)


def test_line_nd_benchmark_import_refuses_scientific_modules(monkeypatch):
    original = builtins.__import__
    def checked(name,*args,**options):
        assert name.split('.')[0] not in {'numpy','scipy','skimage','skimagexx'}
        return original(name,*args,**options)
    monkeypatch.setattr(builtins,'__import__',checked)
    module = load_driver()
    assert callable(module.main) and callable(module.checked_outputs)


@pytest.mark.parametrize('spec',CASES,ids=[s['id'] for s in CASES])
def test_line_nd_benchmark_actual_input_factory_original_refused(spec):
    np = scientific_profile()
    module = load_driver()
    calls = []
    def refused(*args,**options):
        calls.append((args,options))
        raise RuntimeError('line-nd-benchmark-original-refused')
    with without_draw_activity():
        args,options = module.fixture(spec,np)
        assert options == spec['kwargs'] and len(args) == 2
        assert all(len(v) == len(spec['start']) for v in args)
        if spec['dtype'] is not None:
            assert all(type(v) is np.ndarray and v.dtype == np.dtype(spec['dtype']) for v in args)
        if spec['layout'] == 'reverse':
            assert all(v.strides[0] < 0 for v in args)
        if spec['layout'] == 'readonly-unaligned':
            assert all(not v.flags.writeable and not v.flags.aligned for v in args)
        with pytest.raises(RuntimeError,match='^line-nd-benchmark-original-refused$'):
            refused(*args,**options)
    assert len(calls) == 1


def test_line_nd_benchmark_storage_and_preflight_guard_source():
    source = (ROOT/'benchmarks/line_nd.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    helper = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'checked_outputs')
    guards = [ast.unparse(n.test) for n in ast.walk(helper) if isinstance(n,ast.Assert)]
    assert any('array is not other' in g and 'expected' in g for g in guards)
    assert any('array is not other' in g and 'actual[i + 1:]' in g for g in guards)
    assert any('snapshot(tuple((a.base for a in actual)))' in g and 'expected' in g for g in guards)
    assert any('not array.flags.owndata' in g for g in guards)
    assert "original._round_safe = denied" in source
    assert "frame.f_code is upstream.__code__" in source
    assert "calls == [contract['native_entry']] and accepted == [True]" in source
    assert "np.geterr() == policy and np.geterrcall() is handler and np.getbufsize() == buffer" in source
    assert "input_state() == frozen_input" in source
    assert "bool(xx._core.has_avx2())" in source
