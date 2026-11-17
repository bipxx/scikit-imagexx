"""Pure source/safety/readiness checks; no scientific imports or target calls."""
import ast
import builtins
import hashlib
import json
from pathlib import Path
import sys
import types
import pytest
from rolling_ball_contract import recipes,source_safe,fixture,reference_runtime,SIGNATURE
ROOT=Path(__file__).resolve().parents[1]
C=ROOT/'coverage/contracts/rolling-ball-v1.json'

def test_frozen_sources_and_historical_constructor_reuse_scope():
    d=json.loads(C.read_bytes());snapshot=json.loads((ROOT/'tests/fixtures/rolling_ball_sources_v1.json').read_bytes())
    for row in snapshot['files'].values():assert hashlib.sha256(row['text'].encode()).hexdigest()==row['raw_sha256']
    for name,pin in d['source_files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==pin,name
    assert d['original_budget']==d['native_body_budget']==d['helper_body_budget']==0
    assert d['prior_full_algorithm_references_found']==0 and d['existing_constructor_scope_only']

def test_signature_decorator_order_and_no_proposed_extra_kwargs():
    s=json.loads((ROOT/'tests/fixtures/rolling_ball_sources_v1.json').read_bytes())['files'];tree=ast.parse(s['restoration/_rolling_ball.py']['text']);f=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='rolling_ball')
    assert [a.arg for a in f.args.args]==['image']
    assert [a.arg for a in f.args.kwonlyargs]==['radius','kernel','nansafe','num_threads','workers']
    assert [ast.unparse(x)for x in f.args.kw_defaults]==['100','None','False','DEPRECATED','None']
    assert len(f.decorator_list)==1 and 'deprecate_parameter'in ast.unparse(f.decorator_list[0])
    assert SIGNATURE==json.loads(C.read_bytes())['signature']
    assert all(x not in [a.arg for a in f.args.kwonlyargs]for x in('out','mask','channel_axis','in_place'))

def test_cython_geometry_nansafe_and_worker_frontier_is_literal():
    s=json.loads((ROOT/'tests/fixtures/rolling_ball_sources_v1.json').read_bytes())['files'];cy=s['restoration/_rolling_ball_cy.pyx']['text'];u=s['_shared/utils.py']['text']
    assert 'boundscheck(False)'in cy and 'num_threads=workers'in cy
    assert 'if min_value > tmp or isnan(tmp):'in cy and 'if min_value > tmp:'in cy
    assert 'return out.base.reshape(img_shape)'in cy and 'for ker_idx_last in range(kernel_last_dim)'in cy
    assert u.index('warnings.warn(',u.index('class deprecate_parameter'))<u.index('if new_value is not DEPRECATED:',u.index('class deprecate_parameter'))
    assert 'complex valued input is not supported'in u

@pytest.mark.parametrize('spec',recipes(),ids=lambda x:x['id'])
def test_geometry_storage_and_cpu_ceiling(spec):
    b=source_safe(spec);shape=spec['shape'];k=b['kernel_shape'];p=[s+2*(x//2)for s,x in zip(shape,k)]
    if spec['layout']=='stride2':
        from math import prod
        assert b['physical_elements']==prod(2*s for s in shape)
    assert b['addition_upper']<=16384 and b['named_numerical_bytes_upper']<=131072
    if b['input_elements']:
        strides=[];stride=1
        for axis in range(len(p)-1,-1,-1):strides.insert(0,stride);stride*=p[axis]
        # Bound every Cython ind2ind(output)+ind2ind(kernel-leading)+last offset;
        # no pixel arithmetic/target algorithm evaluation in this proof.
        maximum=sum((s+x-2)*st for s,x,st in zip(shape,k,strides))
        assert 0<=maximum<stride
    assert not(spec['radius']=='OMITTED'and len(shape)>1)

@pytest.mark.parametrize('cached',(False,True))
def test_actual_factory_import_request_refused_even_cached(cached,monkeypatch):
    old=builtins.__import__;requests=[]
    def guarded(name,*args,**kwargs):
        if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):requests.append(name);raise RuntimeError('ROLLING_FACTORY_REQUEST_REFUSED')
        return old(name,*args,**kwargs)
    if cached:monkeypatch.setitem(sys.modules,'numpy',types.ModuleType('numpy'))
    else:monkeypatch.delitem(sys.modules,'numpy',raising=False)
    monkeypatch.setattr(builtins,'__import__',guarded)
    with pytest.raises(RuntimeError,match='REQUEST_REFUSED'):fixture(recipes()[0])
    assert requests==['numpy']

def test_unfrozen_factory_is_refused_before_import(monkeypatch):
    old=builtins.__import__;requests=[]
    def guarded(name,*a,**kw):
        if name=='numpy':requests.append(name);raise AssertionError('UNEXPECTED_NUMPY')
        return old(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',guarded)
    with pytest.raises(AssertionError,match='RECIPE_NOT_FROZEN'):fixture(dict(shape=[1,1,1],radius=100))
    assert not requests

def test_zero_authority_and_observer_codec_exact_reuse():
    with pytest.raises(SystemExit,match='PROSPECTIVE_UNBOUND'):reference_runtime(None)
    d=json.loads(C.read_bytes())
    assert hashlib.sha256((ROOT/'tests/rolling_ball_exception_codec.py').read_bytes()).hexdigest()==d['codec_exact_raw_sha256']
    observer=(ROOT/'tests/rolling_ball_observer.py').read_text()
    for fragment in d['observer_inverse']:
        assert observer.count(fragment['current'])==1;observer=observer.replace(fragment['current'],fragment['predecessor'],1)
    assert hashlib.sha256(observer.encode()).hexdigest()==d['observer_predecessor_canonical_sha256']


@pytest.mark.parametrize('cached',(False,True))
def test_prospective_benchmark_actual_cli_refuses_before_science(cached,monkeypatch,tmp_path):
    import runpy
    requests=[];old=builtins.__import__
    def guarded(name,*a,**kw):
        if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):requests.append(name);raise AssertionError('SCIENCE_FORBIDDEN')
        return old(name,*a,**kw)
    if cached:
        for name in('numpy','scipy','skimage','skimagexx'):monkeypatch.setitem(sys.modules,name,types.ModuleType(name))
    monkeypatch.setattr(builtins,'__import__',guarded);monkeypatch.setattr(sys,'argv',['rolling_ball'])
    with pytest.raises(SystemExit,match='PROSPECTIVE_UNBOUND'):runpy.run_path(str(ROOT/'benchmarks/rolling_ball.py'),run_name='__main__')
    assert requests==[]


def test_source_plan_is_bounded_not_execution(monkeypatch,capsys):
    import runpy
    monkeypatch.setattr(sys,'argv',['rolling_ball','--plan']);runpy.run_path(str(ROOT/'benchmarks/rolling_ball.py'),run_name='__main__');d=json.loads(capsys.readouterr().out)
    assert len(d['cases'])==8 and all(s['workers']==1 and s['addition_upper']<=20000000 for s in d['cases'])
    assert all(d[x]is None for x in('source','producer','runtime','core','SDK','root_GO'))
    assert d['method']['repeat']==7 and d['method']['target_ms']==100 and d['method']['warmup_min_ms']==250
