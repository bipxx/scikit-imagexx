"""Stdlib-only reference migration checks; never import scientific packages."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'tests/approximate_polygon_runtime_reference.py'


@pytest.fixture
def reference(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('inert_rdp_runtime_reference', HELPER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.delenv('SKIMAGEXX_DISABLE_SIMD',raising=False)
    np = SimpleNamespace(__version__='1.24.4')
    calls = []
    rows = {name: dict(id=name, input_before={'words':'input'}, input_after={'words':'mutated'},
                      result={'shape':[3,2],'words':'result'}, error=None, warnings=[],
                      events=[{'other':.3535533905932735}], checks={'restored':True})
            for name in module.RUNTIME_REFERENCE_IDS}
    def target(name): calls.append(name)
    def observe(call, spec, numpy):
        assert numpy is np
        call(spec['id'])
        return deepcopy(rows[spec['id']])
    def prepare(numpy):
        return [dict(id=name) for name in module.RUNTIME_REFERENCE_IDS], target, observe, {'source':'inert'}
    monkeypatch.setattr(module, '_prepare', prepare)
    return module, np, calls, rows


def test_two_calls_total_cache_is_readonly_and_full_rows_are_retained(reference):
    module, np, calls, rows = reference
    first, second = module.RUNTIME_REFERENCE_IDS
    actual = module.runtime_reference(first, np)
    assert actual == rows[first]
    actual['events'][0]['other'] = 99.; actual['result']['words'] = 'changed'
    for name in (first, second, first, second):
        assert module.runtime_reference(name, np) == rows[name]
    assert calls == [first, second]
    assert all(isinstance(raw, bytes) for raw in module._CACHE.values())
    receipts = list((module.ROOT/'build/ci-artifacts').glob('*.jsonl'))
    assert len(receipts) == 1
    records = [json.loads(line) for line in receipts[0].read_text().splitlines()]
    assert [(r['stage'], r['id']) for r in records] == [('before',first),('after',first),('before',second),('after',second)]
    assert [r['observation'] for r in records if r['stage']=='after'] == [rows[first],rows[second]]
    assert '1.24.4-normal-' in receipts[0].name


@pytest.mark.parametrize('case_id', ('open-zigzag', 'unknown', None))
def test_only_the_two_explicit_protocol_ids_can_start_reference(reference, case_id):
    module, np, calls, _ = reference
    with pytest.raises(ValueError, match='two protocol IDs'): module.runtime_reference(case_id,np)
    assert not module._STARTED and not calls


def test_observer_failure_cannot_replay_either_reference(reference, monkeypatch):
    module, np, calls, _ = reference
    prepare = module._prepare
    def broken(numpy):
        specs, target, _, provenance = prepare(numpy)
        def fail(call, spec, numpy): call(spec['id']); raise LookupError('observer witness')
        return specs, target, fail, provenance
    monkeypatch.setattr(module, '_prepare', broken)
    with pytest.raises(LookupError, match='observer witness'): module.runtime_reference(module.RUNTIME_REFERENCE_IDS[0],np)
    for name in module.RUNTIME_REFERENCE_IDS:
        with pytest.raises(RuntimeError, match='no replay'): module.runtime_reference(name,np)
    assert calls == [module.RUNTIME_REFERENCE_IDS[0]] and module._CACHE is None


@pytest.mark.parametrize('failed_write,expected_calls', ((1,0),(2,1)))
def test_journal_failure_before_or_after_target_stops_replay(reference, monkeypatch, failed_write, expected_calls):
    module, np, calls, _ = reference
    class FailedJournal(io.StringIO):
        writes = 0
        def write(self, value):
            self.writes += 1
            if self.writes == failed_write: raise OSError('journal witness')
            return super().write(value)
    real_open = Path.open
    def open_path(path, *args, **kwargs):
        return FailedJournal() if path.name.startswith('approximate-polygon-runtime-reference-') else real_open(path,*args,**kwargs)
    monkeypatch.setattr(Path, 'open', open_path)
    with pytest.raises(OSError, match='journal witness'): module.runtime_reference(module.RUNTIME_REFERENCE_IDS[0],np)
    with pytest.raises(RuntimeError, match='no replay'): module.runtime_reference(module.RUNTIME_REFERENCE_IDS[1],np)
    assert len(calls) == expected_calls and module._CACHE is None


def test_cached_profile_cannot_change(reference):
    module, np, calls, _ = reference
    module.runtime_reference(module.RUNTIME_REFERENCE_IDS[0],np)
    with pytest.raises(RuntimeError, match='profile changed'):
        module.runtime_reference(module.RUNTIME_REFERENCE_IDS[0],SimpleNamespace(__version__='2.5.4'))
    assert len(calls) == 2


def load_source_helper():
    spec = importlib.util.spec_from_file_location('inert_rdp_source_guard',HELPER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_actual_source_pins_and_gold_remain_exact_without_scientific_imports():
    module = load_source_helper()
    before = {name:(ROOT/name).read_bytes() for name in module.PINS}
    module._validate_sources()
    assert before == {name:(ROOT/name).read_bytes() for name in module.PINS}
    assert module.PINS['coverage/contracts/approximate-polygon.json'] == '101bb58c8ceae38d0a5dd2d66a8cfa628d069e41e2ef4d30740bb5e5c59ee758'
    assert module.ORIGINAL_SHA == '26a27d89545d158f6b542857297995639b251ed3af0428c8c86a56c84928aabe'
    assert module._CACHE is None and not module._STARTED
    tree = ast.parse(HELPER.read_bytes())
    assert not any(isinstance(n,ast.Import) and any(a.name.startswith(('numpy','scipy','skimage')) for a in n.names) for n in tree.body)
    assert "str(inspect.signature(reference)) != '(coords, tolerance)'" in HELPER.read_text()


@pytest.mark.parametrize('changed', (0,1,2))
def test_every_frozen_source_pin_rejects_changes(tmp_path, monkeypatch, changed):
    module = load_source_helper(); names=list(module.PINS)
    for name in names:
        path=tmp_path/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes((ROOT/name).read_bytes())
    with (tmp_path/names[changed]).open('ab') as stream: stream.write(b'changed')
    monkeypatch.setattr(module,'ROOT',tmp_path)
    with pytest.raises(RuntimeError,match='source pin mismatch'): module._validate_sources()


@pytest.mark.parametrize('name', ('test_approximate_polygon_integration.py','test_approximate_polygon_pipeline.py'))
def test_consumer_inverse_preserves_full_assertions_native_ids_and_all_case_nodes(name):
    path='tests/'+name; source=(ROOT/path).read_text()
    baseline_source_sha256 = {'test_approximate_polygon_integration.py': 'd4fef85cc26ec8bfffacb2482d3ba85642171957e74821903cd750d961c69ef8', 'test_approximate_polygon_pipeline.py': 'affd7db0168941f0f1e818e77426365f10f298eb7ee009628bdb01fc870ae6c6'}
    restored='\n'.join(line for line in source.splitlines()
        if 'approximate_polygon_runtime_reference import' not in line
        and line.strip() not in ("if spec['id'] in RUNTIME_REFERENCE_IDS:","expected = runtime_reference(spec['id'], np)"))
    assert hashlib.sha256((restored+'\n').encode()).hexdigest() == baseline_source_sha256[name]
    assert source.index("expected = runtime_reference(spec['id'], np)") < source.index('with monkeypatch.context()')
    assert 'assert actual == expected' in source



def test_source_preparation_failure_cannot_be_replayed(reference,monkeypatch):
    module,np,calls,_=reference
    def fail(numpy):raise RuntimeError('source validation witness')
    monkeypatch.setattr(module,'_prepare',fail)
    with pytest.raises(RuntimeError,match='source validation witness'):
        module.runtime_reference(module.RUNTIME_REFERENCE_IDS[0],np)
    with pytest.raises(RuntimeError,match='no replay'):
        module.runtime_reference(module.RUNTIME_REFERENCE_IDS[1],np)
    assert calls==[]and module._CACHE is None


def test_receipt_names_separate_scalar_modern_process(reference,monkeypatch):
    module,np,calls,_=reference;np.__version__='2.5.4';monkeypatch.setenv('SKIMAGEXX_DISABLE_SIMD','1')
    module.runtime_reference(module.RUNTIME_REFERENCE_IDS[0],np)
    paths=list((module.ROOT/'build/ci-artifacts').glob('*.jsonl'))
    assert len(paths)==1 and '2.5.4-scalar-'in paths[0].name and len(calls)==2
