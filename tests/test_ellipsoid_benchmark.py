"""Prospective intake only: algorithms/timers stay refused."""
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
CONTRACT=json.loads((ROOT/'coverage/contracts/ellipsoid-benchmark.json').read_bytes())
CASES=CONTRACT['cases']


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_ellipsoid_benchmark_source_method_and_larger_domain_bound():
    assert len(CASES)==len({s['id'] for s in CASES})==12
    assert CONTRACT['reference_contract_sha256']==hashlib.sha256((ROOT/'coverage/contracts/ellipsoid.json').read_bytes()).hexdigest()
    assert CONTRACT['threads']==1 and CONTRACT['repeat']==7 and CONTRACT['target_ms']==100 and CONTRACT['warmup_min_ms']==250
    assert CONTRACT['raw_samples']==168 and CONTRACT['warmup_groups']==24
    sampler=load('ellipsoid_sampler_probe','tools/montage_sustained.py')
    assert hashlib.sha256(inspect.getsource(sampler.measure).encode()).hexdigest()==CONTRACT['method_sha256']
    driver=load('ellipsoid_benchmark_probe','benchmarks/draw_ellipsoid.py')
    bounds=[driver.source_safe(s) for s in CASES]
    assert max(b['max_output_elements'] for b in bounds)==970299<CONTRACT['max_output_elements']==1_000_000
    assert max(max(b['axis_upper_bounds']) for b in bounds)==99
    assert max(b['max_dense_grid_elements'] for b in bounds)<CONTRACT['max_dense_grid_elements']==3_000_000
    assert max(b['conservative_peak_working_bytes'] for b in bounds)<CONTRACT['max_working_allocation_bytes']
    for name,pin in CONTRACT['source_closure'].items():
        assert len(pin)==64 and all(c in '0123456789abcdef' for c in pin)
        if name in ('tools/montage_sustained.py','benchmarks/draw_coordinates.py'):
            assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin
    assert CONTRACT['source_closure']['python/skimagexx/_draw_ellipsoid.py']=='e5c80404c1d9dd2921708889513d43bec9b47e1bab5e442969abc615cb98a3b6'
    assert all(type(v) is not int or abs(v)<=2**50 for s in CASES for v in s['axes']+s['spacing'])


def test_ellipsoid_benchmark_import_refuses_science(monkeypatch):
    old=builtins.__import__
    def refused(name,*args,**kwargs):
        assert name.split('.')[0] not in {'numpy','scipy','skimage','skimagexx'},name
        return old(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',refused)
    module=load('ellipsoid_import_probe','benchmarks/draw_ellipsoid.py')
    assert callable(module.fixture) and callable(module.main)


@pytest.mark.parametrize('spec',CASES,ids=[s['id'] for s in CASES])
def test_ellipsoid_benchmark_actual_factory_original_refused(spec):
    driver=load('ellipsoid_factory_probe','benchmarks/draw_ellipsoid.py');calls=[]
    def refused(*args,**kwargs):calls.append((args,kwargs));raise RuntimeError('ellipsoid-benchmark-original-refused')
    args,kwargs=driver.fixture(spec)
    with pytest.raises(RuntimeError,match='^ellipsoid-benchmark-original-refused$'):refused(*args,**kwargs)
    assert calls==[(tuple(spec['axes']),dict(spacing=tuple(spec['spacing']),levelset=spec['levelset']))]
    assert all(type(v) in (int,float) for v in args) and type(kwargs['levelset']) is bool


@pytest.mark.parametrize('fault',('axis0','axis49','axisnan','step0','stepnan','step-int','truth-int','shape'))
def test_ellipsoid_source_safety_refuses_unaudited_cases(fault):
    spec=copy.deepcopy(CASES[0]);driver=load('ellipsoid_bound_probe','benchmarks/draw_ellipsoid.py')
    if fault=='axis0':spec['axes'][0]=0
    if fault=='axis49':spec['axes'][0]=49
    if fault=='axisnan':spec['axes'][0]=float('nan')
    if fault=='step0':spec['spacing'][0]=0.
    if fault=='stepnan':spec['spacing'][0]=float('nan')
    if fault=='step-int':spec['spacing'][0]=1
    if fault=='truth-int':spec['levelset']=1
    if fault=='shape':spec['axes'].pop()
    with pytest.raises(AssertionError):driver.source_safe(spec)


@pytest.mark.parametrize('dtype',('bool','float64'))
def test_ellipsoid_checked_output_exact_words_and_storage(dtype):
    import numpy as np
    driver=load('ellipsoid_output_probe','benchmarks/draw_ellipsoid.py')
    snapshot=load('ellipsoid_snapshot_probe','benchmarks/draw_coordinates.py').snapshot
    first=np.ones((3,3,3),dtype=dtype);second=first.copy()
    output=driver.checked_output(first,second,np,snapshot,27)
    assert output['arrays'][0]['raw_sha256']==hashlib.sha256(first.tobytes()).hexdigest()
    assert output['arrays'][0]['own'] and output['arrays'][0]['c']
    with pytest.raises(AssertionError):driver.checked_output(first,first,np,snapshot,27)
    with pytest.raises(AssertionError):driver.checked_output(first.view(),second,np,snapshot,27)
    with pytest.raises(AssertionError):driver.checked_output(first,second,np,snapshot,26)
    changed=second.copy()
    if dtype=='float64':changed[0,0,0]=np.nextafter(1.,2.)
    else:changed[0,0,0]=False
    with pytest.raises(AssertionError):driver.checked_output(first,changed,np,snapshot,27)


def test_ellipsoid_benchmark_actual_guard_order_and_metadata_source():
    source=(ROOT/'benchmarks/draw_ellipsoid.py').read_text();tree=ast.parse(source)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    assert 'frame.f_code is upstream.__code__' in source
    assert "calls==['draw_ellipsoid'] and accepted==[True] and snapshots==['get_num_threads']" in source
    assert 'original.ellipsoid=public.ellipsoid=denied' in source and '_compat.fallback=denied' in source
    assert source.index('sys.setprofile(prior_profile)')<source.index('timing=measure(')
    assert source.index('bound=source_safe(spec)')<source.index('expected=upstream(')
    assert 'cpu_model=' in source and 'logical_cpu_count=' in source and 'avx2_available=' in source
    assert "CONTRACT.read_bytes()==contract_raw and REFERENCE.read_bytes()==reference_raw" in source
    # Per-file AST semantics from this interpreter, no cross-version AST hash.
    helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='checked_output')
    asserts=[ast.unparse(n.test) for n in ast.walk(helper) if isinstance(n,ast.Assert)]
    assert any('actual is not expected' in s for s in asserts)
    assert any('actual.flags.owndata' in s and 'actual.flags.c_contiguous' in s for s in asserts)
    assert any('snapshot((actual,)) == snapshot((expected,))' in s for s in asserts)
