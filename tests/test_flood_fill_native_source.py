"""Pure source/authority guards: normal conftest/plugins, no science."""
import ast
import builtins
import hashlib
import json
from pathlib import Path
import runpy
import sys
import types
import pytest
from flood_fill_cost_context import historical_bytes
from flood_fill_source_context import historical_source as integration_bytes
ROOT=Path(__file__).resolve().parents[1]
CONTEXT=ROOT/'coverage/contracts/flood-fill-native-v1.json'

def test_native_source_closure_and_saved_rows():
    c=json.loads(CONTEXT.read_bytes())
    for name,pin in c['source_raw_sha256'].items():
        raw=historical_bytes(ROOT,name)if name in ('src/flood_fill.cpp','benchmarks/flood_fill_execution.py')else integration_bytes(ROOT,name)
        assert hashlib.sha256(raw).hexdigest()==pin,name
    assert hashlib.sha256((ROOT/'tests/fixtures/flood_fill_sources_v1.json').read_bytes()).hexdigest()==c['source_snapshot_sha256']
    saved=json.loads((ROOT/'tests/fixtures/flood_fill_original_v1.json').read_bytes())
    from flood_fill_contract import recipes
    specs=recipes();assert len(specs)==96
    for profile,errors in [('minimum',17),('modern',18)]:
        rows=saved['profiles'][profile]['rows']
        assert [r['id']for r in rows]==[s['id']for s in specs]
        assert sum(r['error']is not None for r in rows)==errors
        assert all(r['supplied_call_attempts']==1 and r['caller_restore']['exact32']for r in rows)

def test_complete_sdk_projection():
    from flood_fill_sdk_fixture import render
    assert (ROOT/'tests/fixtures/flood_fill_sdk_original_v1.inc').read_text()==render()
    assert render().count('check<')==79
    text=(ROOT/'tests/test_flood_fill_sdk.cpp').read_text()
    assert 'skimagexx::flood_fill('in text and 'skimagexx::flood_fill_into('in text
    assert 'find_package(skimagexx CONFIG REQUIRED)'in(ROOT/'tests/sdk/flood_fill/CMakeLists.txt').read_text()

def test_exact_existing_executor_inverse_and_method():
    c=json.loads(CONTEXT.read_bytes());raw=historical_bytes(ROOT,'benchmarks/flood_fill_execution.py')
    for fragment in c['executor_inverse']:
        a=fragment['current'].encode();assert raw.count(a)==1;raw=raw.replace(a,fragment['predecessor'].encode(),1)
    assert hashlib.sha256(raw).hexdigest()==c['executor_predecessor_raw_sha256']
    context=json.loads((ROOT/'coverage/contracts/flood-fill-benchmark-v3.json').read_bytes())
    tree=ast.parse((ROOT/'tools/montage_sustained.py').read_bytes());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='measure')
    source=(ROOT/'tools/montage_sustained.py').read_text();method=''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    assert hashlib.sha256(method.encode()).hexdigest()==context['method']['method_sha256']
    for name,pin in context['production_source_canonical_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin,name
    for name,pin in context['FENV_dependency']['source_files_raw_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==pin,name
    for name,row in context['sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==row['current_canonical_sha256'],name
    assert len(context['routes']['flood-fill']['cases'])==8 and all(v is None for v in context['unbound'].values())

@pytest.mark.parametrize('cached',[False,True])
@pytest.mark.parametrize('mode',['plan','unbound','invalid'])
def test_actual_cli_before_science(mode,cached,monkeypatch,tmp_path,capsys):
    requests=[];old=builtins.__import__
    def guarded(name,*a,**kw):
        if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):
            requests.append(name);raise AssertionError('science request before authority')
        return old(name,*a,**kw)
    with monkeypatch.context()as scope:
        scope.syspath_prepend(str(ROOT/'benchmarks'))
        if cached:
            for name in('numpy','scipy','skimage','skimagexx'):scope.setitem(sys.modules,name,types.ModuleType(name))
        scope.setattr(builtins,'__import__',guarded)
        argv=['flood_fill.py','--output',str(tmp_path/'result.json')]
        if mode=='plan':argv+=['--plan']
        if mode=='invalid':
            path=tmp_path/'binding.json';path.write_text('{}',encoding='utf8');argv+=['--root-binding',str(path)]
        scope.setattr(sys,'argv',argv)
        if mode=='plan':runpy.run_path(str(ROOT/'benchmarks/flood_fill_native.py'),run_name='__main__')
        else:
            with pytest.raises((SystemExit,KeyError,AssertionError)):runpy.run_path(str(ROOT/'benchmarks/flood_fill_native.py'),run_name='__main__')
    assert requests==[] and not(tmp_path/'result.json').exists() and not(tmp_path/'result.batches.jsonl').exists()
    if mode=='plan':assert json.loads(capsys.readouterr().out)['state']=='PROSPECTIVE_UNBOUND_NO_MEASUREMENT'

def test_first_cohort_real_entry_cannot_enter_body(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'benchmarks'))
    from flood_fill_execution import first_cohort_profile
    events=[]
    def measure():events.append('body')
    def observer(frame,event,value):events.append(event)
    previous=sys.getprofile()
    try:
        sys.setprofile(first_cohort_profile(measure,observer))
        with pytest.raises(AssertionError,match='did not stop real sampler entry'):measure()
    finally:sys.setprofile(previous)
    assert 'body'not in events and sys.getprofile()is previous

def test_frontend_native_ownership_and_no_original_forwarding():
    raw=(ROOT/'python/skimagexx/_flood_fill.py').read_text();tree=ast.parse(raw)
    imports=[ast.unparse(n)for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert imports==['import numpy as np','from . import _core']
    assert '_core.flood_mask('in raw and '_core.flood_assign('in raw and '_core.crop_copy_into('in raw
    source=(ROOT/'src/flood_fill.cpp').read_text()
    assert 'queue.push_back(next)'in source and 'out.storage[next]=1;queue.push_back(next)'in source
    binder=(ROOT/'src/bind_flood_fill.cpp').read_text()
    assert "CASE('u',4,std::uint32_t)"in binder and "CASE('u',8,std::uint64_t)"in binder
    assert 'numeric POD dtype'in binder
    assert 'get_num_threads'not in source and 'omp'not in source and 'PyObject'not in source


def test_cpp_cost_context_and_unchanged_methods_cases_providers():
    context=json.loads((ROOT/'coverage/contracts/flood-fill-native-v2.json').read_bytes())
    for name in context['changed_sources']:historical_bytes(ROOT,name)
    for name,pin in context['new_source_pins'].items():assert hashlib.sha256(integration_bytes(ROOT,name)).hexdigest()==pin,name
    for name,pin in context['unchanged_source_pins'].items():assert hashlib.sha256(integration_bytes(ROOT,name)).hexdigest()==pin,name
    old=json.loads((ROOT/'coverage/contracts/flood-fill-benchmark-v1.json').read_bytes())
    new=json.loads((ROOT/'coverage/contracts/flood-fill-benchmark-v3.json').read_bytes())
    assert old['routes']==new['routes']and old['method']==new['method']and old['apis']==new['apis']
    assert old['unbound']==new['unbound']and all(v is None for v in new['unbound'].values())
    raw=(ROOT/'src/flood_fill.cpp').read_text()
    assert 'coordinates(k,g.shape)'not in raw and 'remaining%padded'not in raw
    assert 'T v=working[next];'in raw and 'std::vector<T> working'in raw
