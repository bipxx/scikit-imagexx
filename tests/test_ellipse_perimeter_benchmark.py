"""Pure prebuild intake selectors; scientific checker negatives are installed-only."""
import ast
import builtins
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).parents[1]
CONTRACT=json.loads((ROOT/'coverage/contracts/ellipse-perimeter-fallback-runtime-v1.json').read_bytes())
CASES=CONTRACT['cases']


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_ellipse_perimeter_benchmark_source_bounds_and_pins():
    assert len(CASES)==len({s['id'] for s in CASES})==12
    assert CONTRACT['threads']==1 and CONTRACT['repeat']==7 and CONTRACT['target_ms']==100
    assert CONTRACT['warmup_min_ms']==250 and CONTRACT['raw_samples']==168 and CONTRACT['warmup_groups']==24
    assert hashlib.sha256((ROOT/'coverage/contracts/ellipse-perimeter.json').read_bytes()).hexdigest()==CONTRACT['reference_contract_sha256']
    driver=load('ellipse_perimeter_bound_probe','benchmarks/ellipse_perimeter.py')
    bounds=[driver.source_safe(s) for s in CASES]
    assert max(b['max_points'] for b in bounds)==49164==CONTRACT['max_points']
    assert max(b['max_coordinate_words'] for b in bounds)==98328==CONTRACT['max_coordinate_words']
    assert max(max(s['args'][2:]) for s in CASES)==6144
    assert any(s['args'][2:]==[0,0] for s in CASES)
    assert {s['kwargs'].get('orientation') for s in CASES if 'orientation' in s['kwargs']}=={.3,-.5}
    sampler=load('ellipse_perimeter_sampler_probe','tools/montage_sustained.py')
    assert hashlib.sha256(inspect.getsource(sampler.measure).encode()).hexdigest()==CONTRACT['method_sha256']
    for path,pin in CONTRACT['source_closure'].items():
        assert len(pin)==64 and all(c in '0123456789abcdef' for c in pin)
        if path in ('tools/montage_sustained.py','benchmarks/draw_coordinates.py'):
            assert hashlib.sha256((ROOT/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin


def test_ellipse_perimeter_benchmark_import_refuses_science(monkeypatch):
    original=builtins.__import__
    def refused(name,*args,**kwargs):
        assert name.split('.')[0] not in {'numpy','scipy','skimage','skimagexx'},name
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',refused)
    driver=load('ellipse_perimeter_import_probe','benchmarks/ellipse_perimeter.py')
    assert callable(driver.main) and callable(driver.fixture)


@pytest.mark.parametrize('spec',CASES,ids=[s['id'] for s in CASES])
def test_ellipse_perimeter_benchmark_factory_original_refused(spec):
    driver=load('ellipse_perimeter_factory_probe','benchmarks/ellipse_perimeter.py');calls=[]
    def refused(*args,**kwargs):
        calls.append((args,kwargs));raise RuntimeError('ellipse-perimeter-original-refused')
    args,kwargs=driver.fixture(spec)
    with pytest.raises(RuntimeError,match='^ellipse-perimeter-original-refused$'):refused(*args,**kwargs)
    expected=dict(spec['kwargs'])
    if 'shape' in expected:expected['shape']=tuple(expected['shape'])
    assert calls==[(tuple(spec['args']),expected)]


@pytest.mark.parametrize('fault',('negative-radius','large-radius','huge-center','float-coordinate','nonfinite-angle','unequal-rotation','short-shape','protocol-shape','negative-shape'))
def test_ellipse_perimeter_benchmark_rejects_unsafe_specs(fault):
    spec=copy.deepcopy(CASES[2])
    if fault=='negative-radius':spec['args'][2]=-1
    if fault=='large-radius':spec['args'][3]=32768
    if fault=='huge-center':spec['args'][0]=2**63
    if fault=='float-coordinate':spec['args'][0]=0.
    if fault=='nonfinite-angle':spec['kwargs']['orientation']=float('nan')
    if fault=='unequal-rotation':spec['kwargs']['orientation']=.3
    if fault=='short-shape':spec['kwargs']['shape']=[8]
    if fault=='protocol-shape':spec['kwargs']['shape']=object()
    if fault=='negative-shape':spec['kwargs']['shape']=[-1,8]
    driver=load('ellipse_perimeter_refusal_probe','benchmarks/ellipse_perimeter.py')
    with pytest.raises(AssertionError):driver.source_safe(spec)


def test_ellipse_perimeter_benchmark_guards_and_timed_boundary():
    source=(ROOT/'benchmarks/ellipse_perimeter.py').read_text();tree=ast.parse(source)
    assert 'frame.f_code is upstream.__code__' in source and "event=='c_call' and value is cached_compiled" in source
    assert "calls==['draw_ellipse_perimeter'] and accepted==[True] and snapshots==['get_num_threads']" in source
    assert 'compiled._ellipse_perimeter=denied' in source and '_compat.fallback=denied' in source
    assert source.index('bound=source_safe(spec)')<source.index('expected=upstream(')
    assert source.index('sys.setprofile(prior_profile)')<source.index('timing=measure(')
    assert 'source_fingerprint(ROOT)==before' in source and 'CONTRACT.read_bytes()==contract_raw' in source
    assert all(s in source for s in ('cpu_model=','logical_cpu_count=','avx2_available=','numpy_binary_sha256='))
    helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='checked_output')
    assertions=[ast.unparse(n.test) for n in ast.walk(helper) if isinstance(n,ast.Assert)]
    assert any('actual is not expected' in s for s in assertions)
    assert any('value is not other' in s and 'shares_memory' in s for s in assertions)
    assert any('snapshot(actual) == snapshot(expected)' in s for s in assertions)


@pytest.mark.parametrize('shape',((),(3,),(0,)))
def test_ellipse_perimeter_benchmark_checker_corruption_controls(shape):
    # Installed-stage only: never include this selector in a prebuild command.
    import numpy as np
    driver=load('ellipse_perimeter_checker_probe','benchmarks/ellipse_perimeter.py')
    snapshot=load('ellipse_perimeter_snapshot_probe','benchmarks/draw_coordinates.py').snapshot
    def pair():return tuple(np.zeros(shape,dtype=np.intp) for _ in range(2))
    actual,expected=pair(),pair();scalar=shape==()
    assert driver.checked_output(actual,expected,np,snapshot,3,scalar)['container']=='tuple'
    for invalid in (actual,(actual[0],actual[1]),(expected[0],actual[1]),(actual[0],actual[0]),(actual[0].view(),actual[1]),list(actual),(actual[0].astype(np.int32),actual[1])):
        with pytest.raises(AssertionError):driver.checked_output(invalid,actual,np,snapshot,3,scalar)
    if actual[0].size:
        changed=pair();changed[0].flat[0]=1
        with pytest.raises(AssertionError):driver.checked_output(changed,expected,np,snapshot,3,scalar)


def test_ellipse_fallback_runtime_binding_preserves_historical_contract_and_body():
    predecessor=CONTRACT['predecessor_binding']
    raw=(ROOT/predecessor['contract']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==predecessor['sha256']
    old=json.loads(raw)
    for key,value in old.items():
        if key not in ('status','frontend_source','source_closure'):assert CONTRACT[key]==value,key
    assert set(CONTRACT['source_closure'])==set(old['source_closure'])
    for path,pin in old['source_closure'].items():
        if path!='python/skimagexx/_draw_ellipse_perimeter.py':assert CONTRACT['source_closure'][path]==pin
    source=(ROOT/'python/skimagexx/_draw_ellipse_perimeter.py').read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(source).hexdigest()==CONTRACT['source_closure']['python/skimagexx/_draw_ellipse_perimeter.py']
    for fragment in predecessor['source_inverse']:
        current,previous=fragment['old'].encode(),fragment['new'].encode()
        assert source.count(current)==1;source=source.replace(current,previous,1)
    assert hashlib.sha256(source).hexdigest()==old['source_closure']['python/skimagexx/_draw_ellipse_perimeter.py']
    driver=(ROOT/'benchmarks/ellipse_perimeter.py').read_bytes().replace(b'\r\n',b'\n')
    current=b"CONTRACT=ROOT/'coverage/contracts/ellipse-perimeter-fallback-runtime-v1.json'"
    assert driver.count(current)==1
    driver=driver.replace(current,b"CONTRACT=ROOT/'coverage/contracts/ellipse-perimeter-benchmark.json'")
    assert hashlib.sha256(driver).hexdigest()==predecessor['driver_sha256']
