"""Pure prospective source/intake tests. No NumPy, native, original or timer calls."""
import ast
import builtins
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT/'coverage/contracts/regionprops-coords-scaled-benchmark-v1.json'
CONFIG = json.loads(CONTRACT.read_bytes())


def load():
    spec = importlib.util.spec_from_file_location('coords_scaled_prospective', ROOT/'benchmarks/regionprops_coords_scaled.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coords_scaled_import_is_inert(monkeypatch):
    original = builtins.__import__
    def checked(name, *a, **kw):
        assert name.split('.')[0] not in {'numpy', 'scipy', 'skimage', 'skimagexx'}
        return original(name, *a, **kw)
    monkeypatch.setattr(builtins, '__import__', checked)
    assert len(load().case_specs()) == 6


def test_coords_scaled_prospective_main_refuses_before_imports(monkeypatch):
    module = load(); attempts = []
    original = builtins.__import__
    def checked(name, *a, **kw):
        if name.split('.')[0] in {'numpy', 'scipy', 'skimage', 'skimagexx'}:
            attempts.append(name)
            raise AssertionError(name)
        return original(name, *a, **kw)
    monkeypatch.setattr(builtins, '__import__', checked)
    if CONFIG['state'].startswith('PROSPECTIVE_'):
        with pytest.raises(AssertionError, match='prospective: genuine source/input gate'):
            module.main()
    assert attempts == []


def test_coords_scaled_frozen_cases_and_bounded_arithmetic():
    module = load(); cases = module.case_specs()
    assert [s['id'].split('/')[-1] for s in cases] == ['tiny-c-off', 'tiny-f-warm', 'medium2d-f-warm', 'medium3d-f-warm', 'sparse2d-ctor-read', 'integer3d-partial']
    assert len({s['id'] for s in cases}) == 6
    for case in cases:
        bounds = module.case_bounds(case)
        assert bounds['input_elements'] <= 24576 and bounds['max_output_elements'] <= 49152
        assert bounds['max_integer_origin_plus_index'] < 2**31 and bounds['max_absolute_scaled_plus_offset'] == 1540
    assert [s['native_partition'] for s in cases] == ['f64']*5+['integer-provider']
    assert cases[4]['pattern'] == 'sparse' and cases[4]['max_label'] == 8


@pytest.mark.parametrize('case', CONFIG['cases'], ids=lambda s:s['id'])
def test_coords_scaled_fixture_intake_stops_before_science(case, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'benchmarks'))
    class Refused:
        def __getattr__(self, name):
            raise RuntimeError('numerical-fixture-refused')
    with pytest.raises(RuntimeError, match='^numerical-fixture-refused$'):
        load().fixture(case, Refused())


def test_coords_scaled_changed_recipe_refused_before_fixture():
    module = load()
    for key, value in [('shape',[1024,1024]), ('spacing',[float('nan'),1.]), ('pattern','unknown'), ('dtype','float64'), ('native_partition','fully-native-integer')]:
        case = copy.deepcopy(module.case_specs()[0]); case[key] = value
        with pytest.raises(AssertionError):
            module.case_bounds(case)


@pytest.mark.parametrize('case', CONFIG['cases'], ids=lambda s:s['id'])
def test_coords_scaled_operation_boundaries_inert(case):
    module = load(); events = []
    class Region:
        @property
        def image(self):
            events.append('image-prime'); return object()
        @property
        def coords_scaled(self):
            events.append('coords_scaled'); return object()
    def call(*a, **kw):
        events.append('constructor'); return [Region(), Region()]
    operation = module.make_operation(call, object(), {}, case)
    if case['operation'] == 'constructor':
        assert events == []
        operation(); operation()
        assert events == ['constructor', 'coords_scaled']*2
    else:
        assert events == ['constructor']+(['image-prime'] if case['cache'] else [])
        before = list(events)
        first, second = operation(), operation()
        assert first is not second and events[len(before):] == ['coords_scaled']*2
        assert events.count('constructor') == 1


def test_coords_scaled_exact_comparison_budget_and_gates():
    c = CONFIG; protocol = c['comparison']
    assert (c['threads'], c['repeat'], c['target_ms'], c['warmup_calibration_min_ms']) == (1,7,100,250)
    assert c['raw_samples_per_process'] == 6*2*7 == 84
    assert c['warmup_groups_per_process'] == 12
    assert protocol['process_roles'] == ['prior-cpp','native','native','prior-cpp'] and protocol['pairings'] == [[0,1],[3,2]]
    assert (protocol['accepted_arm_samples'],protocol['warmup_groups']) == (336,48)
    assert (protocol['all_case_direct_max'],protocol['all_case_original_normalized_max']) == (1.05,1.05)
    assert (protocol['target_direct_max'],protocol['target_vs_original_max']) == (.90,.90)
    assert protocol['target_ids'] == [c['cases'][i]['id'] for i in (2,3)]
    assert c['baseline']['source'] == 'cf0abd3f8dba9acbc988a59a53ae32cbcccfab4b'


def test_coords_scaled_saved_semantics_unchanged_and_complete():
    saved = CONFIG['saved_semantics']
    for key in ('reference','proposal'):
        assert hashlib.sha256((ROOT/saved[key]).read_bytes().replace(b'\r\n',b'\n')).hexdigest() == saved[key+'_canonicalLF_sha256']
    recipes = json.loads((ROOT/saved['proposal']).read_bytes())['cases']
    data = json.loads((ROOT/saved['reference']).read_bytes())['original_observations']
    assert len(recipes) == 12 and saved['case_ids'] == [s['id'] for s in recipes]
    assert sum(map(len, data.values())) == saved['constructor_sequences'] == 24
    assert sum(sum(step['action'] in ('attr','item') for step in row['steps']) for rows in data.values() for row in rows) == saved['property_arrays'] == 72


def test_coords_scaled_source_and_reused_helper_pins(monkeypatch):
    for path, pin in CONFIG['source_closure'].items():
        assert hashlib.sha256((ROOT/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest() == pin
    monkeypatch.syspath_prepend(str(ROOT/'benchmarks'))
    import regionprops_state
    for name, pin in CONFIG['reused_state_method_sha256'].items():
        assert hashlib.sha256(inspect.getsource(getattr(regionprops_state,name)).encode()).hexdigest() == pin
    assert set(CONFIG['native_source_closure']) == set(CONFIG['required_native_source_paths'])
    assert not CONFIG['pending_integrated_source_paths']
    assert CONFIG['reviewed_dependencies']['active_integrated_source'] == '9558890a3f758a06c638bc778701e329c99ad0ff'
    assert CONFIG['method_sha256'] == json.loads((ROOT/'coverage/contracts/draw-coordinates-benchmark.json').read_bytes())['method_sha256']


def test_coords_scaled_actual_main_validation_precedes_sampler():
    source = (ROOT/'benchmarks/regionprops_coords_scaled.py').read_text()
    module = ast.parse(source); main = next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name == 'main')
    calls = [n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id == 'measure']
    assert len(calls) == 1 and [ast.literal_eval(n) for n in calls[0].args[2:]] == [7,100]
    assert source.index("contract['state'] == 'FROZEN_READY_FOR_ROOT_FIRST_COHORT'") < source.index('import numpy as np')
    assert source.index('sys.setprofile(previous)') < source.index('timing = measure(')
    assert "provider.coords_scaled.fget.__code__" in source and "(np, 'argwhere', np.argwhere)" in source
    assert "frame.f_locals['native_result'] is not None" in source and "frame.f_locals['workers']" in source
    assert "actual is not expected and not np.shares_memory(actual, expected)" in source
    assert "actual is not buffer and not np.shares_memory(actual, buffer)" in source
    assert 'source_fingerprint(args.source_root)' in source
    assert "sys.version == audit['runtime']['python']" in source
    assert "calls == list(STAGES[:3] if integer else STAGES)*2" in source
