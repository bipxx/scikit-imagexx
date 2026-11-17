"""Pure exact-source/binding/entry controls. No scientific or sampler bodies."""
import ast
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import types
import pytest
ROOT=Path(__file__).resolve().parents[1]
CONTEXT=ROOT/'coverage/contracts/finite-radon-benchmark-execution-v4.json'
DATA=json.loads(CONTEXT.read_bytes())
sys.path.insert(0,str(ROOT/'benchmarks'))
import finite_radon_execution_v1 as driver
from finite_radon_cases_v1 import case_specs,case_bounds


def test_source_pins_method_and_frozen_history():
    common=driver.common_adapter()
    for name,row in DATA['sources'].items():assert common.digest(common.source_bytes(ROOT/name))==row['current_canonical_sha256'],name
    for name,pin in DATA['production_source_canonical_sha256'].items():assert common.digest(common.source_bytes(ROOT/name))==pin,name
    for name,pin in DATA['historical_raw_sha256'].items():assert common.digest((ROOT/name).read_bytes())==pin,name
    source=(ROOT/'tools/montage_sustained.py').read_text();tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='measure')
    assert common.digest((ast.get_source_segment(source,fn)+'\n').encode())==DATA['method']['method_sha256']
    assert all(v is None for v in DATA['unbound'].values())and DATA['original_budget']==0
    assert common.CONTEXT==CONTEXT.relative_to(ROOT).as_posix()


def test_exact_case_pool_bounds_and_source_arithmetic():
    assert len(case_specs())==16 and len({s['id']for s in case_specs()})==16
    for api in ('frt2','ifrt2'):
        assert len(case_specs(api))==8
        for s in case_specs(api):
            b=case_bounds(s);n=s['n']
            assert b['input_shape']==[n+int(api=='ifrt2'),n]
            assert b['source_roll_calls']==(n-1)**2 and b['source_roll_copy_elements']==n*(n-1)**2
            assert b['addition_upper']>=n*n*(n-1)and b['total_arithmetic_upper']<=18000000
            assert b['physical_elements']<=265224 and b['peak_live_array_bytes_upper']<5000000
            assert b['source_accumulator_max']<2**32
    assert max(s['n']for s in case_specs())==257


@pytest.mark.parametrize('api',('frt2','ifrt2'))
def test_actual_plan_cli(api,monkeypatch,capsys):
    monkeypatch.setattr(sys,'argv',['finite_radon_'+api,'--plan'])
    driver.execute(api);out=json.loads(capsys.readouterr().out)
    assert out['cases']==case_specs(api)and len(out['bounds'])==8


@pytest.mark.parametrize('api',('frt2','ifrt2'))
@pytest.mark.parametrize('cached',(False,True))
def test_actual_unbound_cli_zero_import_requests(api,cached,monkeypatch):
    old=builtins.__import__;attempts=[]
    def guarded(name,*a,**kw):
        if name.split('.')[0]in ('numpy','scipy','skimage','skimagexx'):attempts.append(name);raise AssertionError('SCIENCE_REQUEST_FORBIDDEN')
        return old(name,*a,**kw)
    for name in ('numpy','skimage','skimagexx'):
        if cached:monkeypatch.setitem(sys.modules,name,types.ModuleType(name))
        else:monkeypatch.delitem(sys.modules,name,raising=False)
    monkeypatch.setattr(builtins,'__import__',guarded)
    monkeypatch.setattr(sys,'argv',['finite_radon_'+api,'--first-cohort','--pass-index','1'])
    with pytest.raises(SystemExit,match='PROSPECTIVE_UNBOUND'):driver.execute(api)
    assert attempts==[]


def synthetic(tmp_path,monkeypatch,api,fault=None):
    # Reuse the exact existing whole-product envelope constructor/validator;
    # its receipts are explicit inert test inputs, never a ready product.
    import next_util_benchmark_support as support
    monkeypatch.setattr(support,'CONTEXT',CONTEXT)
    return support.synthetic_binding(tmp_path,api,monkeypatch,fault)


@pytest.mark.parametrize('fault',('cases','core','revision','runtime','root','stager','git-root','context','route','expired','go'))
def test_real_binding_refuses_tamper_before_requests(fault,tmp_path,monkeypatch):
    p=synthetic(tmp_path,monkeypatch,'frt2',fault)
    with pytest.raises((AssertionError,KeyError)):driver.common_adapter().load_binding(p,ROOT,'frt2')


@pytest.mark.parametrize('api',('frt2','ifrt2'))
@pytest.mark.parametrize('cached',(False,True))
def test_bound_actual_cli_request_refusal(api,cached,tmp_path,monkeypatch):
    p=synthetic(tmp_path,monkeypatch,api)
    old=builtins.__import__;attempts=[]
    def guarded(name,*a,**kw):
        if name.split('.')[0]in ('numpy','scipy','skimage','skimagexx'):
            attempts.append(name);raise RuntimeError('RADON_REQUEST_REFUSED')
        return old(name,*a,**kw)
    if cached:monkeypatch.setitem(sys.modules,'numpy',types.ModuleType('numpy'))
    else:monkeypatch.delitem(sys.modules,'numpy',raising=False)
    monkeypatch.setattr(builtins,'__import__',guarded)
    monkeypatch.setattr(sys,'argv',['finite_radon_'+api,'--root-binding',str(p),'--first-cohort','--pass-index','3','--output',str(tmp_path/'absent.json')])
    with pytest.raises(RuntimeError,match='RADON_REQUEST_REFUSED'):driver.execute(api)
    assert attempts==['numpy']and not(tmp_path/'absent.json').exists()


@pytest.mark.parametrize('pass_index',range(1,9))
def test_actual_sampling_entry_order_and_mandatory_stop(pass_index,tmp_path):
    # Signature matches the real pinned sampler; only call-entry is reached.
    entered=[];seen=[]
    def measure(reference,native,repeat,target_ms,*,journal,case_index,case_id,pass_index,deadline):
        entered.append(True);raise AssertionError('INERT_SAMPLER_BODY_FORBIDDEN')
    class StopBeforeBody(Exception):pass
    def observer(frame,event,value):
        if event=='call'and frame.f_code is measure.__code__:
            seen.append(dict(frame.f_locals));raise StopBeforeBody
    cli=types.SimpleNamespace(first_cohort=True,output=tmp_path/'absent.json',pass_index=pass_index)
    old=sys.getprofile()
    try:
        sys.setprofile(observer)
        with pytest.raises(StopBeforeBody):driver.sample(driver.common_adapter(),measure,None,None,cli,case_specs()[0],2,{'deadline_utc':'BOUND_TEST_ONLY'},None,None,['frt2'],[True],{})
    finally:sys.setprofile(old)
    assert not entered and len(seen)==1 and seen[0]['pass_index']==pass_index and seen[0]['case_index']==2
    # Source-defined alternating order, never invoke actual sampler or timer.
    assert [[0,1]if(r+2+pass_index)%2==0 else[1,0]for r in range(7)][0]==([0,1]if pass_index%2==0 else[1,0])


def test_nonstopping_profiler_cannot_reach_sampler_body(tmp_path):
    entered=[]
    def measure(*a,**kw):entered.append(True)
    cli=types.SimpleNamespace(first_cohort=True,output=tmp_path/'absent.json',pass_index=1)
    old=sys.getprofile()
    try:
        sys.setprofile(lambda *a:None)
        with pytest.raises(AssertionError,match='body forbidden'):driver.sample(driver.common_adapter(),measure,None,None,cli,case_specs()[0],0,{'deadline_utc':'BOUND_TEST_ONLY'},None,None,[],[],{})
    finally:sys.setprofile(old)
    assert entered==[]


def test_real_source_order_fp_ownership_and_acceptance_interface():
    text=(ROOT/'benchmarks/finite_radon_execution_v1.py').read_text();tree=ast.parse(text)
    execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
    loop=next(n for n in execute.body if isinstance(n,ast.For))
    fp=[n for n in ast.walk(execute)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='fp_controller']
    assert len(fp)==1 and fp[0].lineno<loop.lineno
    assert text.index('common.load_binding')<text.index('    import numpy as np')
    assert text.index('else common.load_measurement')<text.index('    import numpy as np')
    sample=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='sample')
    call=next(n for n in ast.walk(sample)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='measure')
    assert ast.unparse(next(k.value for k in call.keywords if k.arg=='pass_index'))=='cli.pass_index'
    assert "returns==[api]and accepted==[True]"in text and 'native_before' in text and 'reference_FP==native_FP' in text
    assert 'restore(caller)'in text and 'not np.shares_memory(actual,expected)'in text
    binder=(ROOT/'src/bind_finite_radon.cpp').read_text()
    assert all('m.def("'+api+'"'in binder for api in ('frt2','ifrt2'))


def test_bound_missing_pass_index_refused_before_science(tmp_path,monkeypatch):
    p=synthetic(tmp_path,monkeypatch,'frt2');attempts=[];old=builtins.__import__
    def refuse(name,*a,**kw):
        if name.split('.')[0]in ('numpy','skimage','skimagexx'):attempts.append(name);raise AssertionError('FORBIDDEN')
        return old(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',refuse)
    monkeypatch.setattr(sys,'argv',['finite_radon_frt2','--root-binding',str(p),'--first-cohort'])
    with pytest.raises(AssertionError,match='Explicit pass-index'):driver.execute('frt2')
    assert not attempts


def test_target_hardware_fields_decode_exact_offsets():
    raw=bytearray(32);raw[0:2]=(0x37f).to_bytes(2,'little');raw[4:6]=(0x20).to_bytes(2,'little');raw[28:32]=(0x1fa0).to_bytes(4,'little')
    assert driver.hardware(raw)==dict(control=0x37f,status=0x20,mxcsr=0x1fa0)
    with pytest.raises(AssertionError):driver.hardware(bytes(28))


def test_bound_mutated_production_source_refuses(tmp_path,monkeypatch):
    p=synthetic(tmp_path,monkeypatch,'frt2');binding=json.loads(p.read_bytes())
    source=Path(binding['source_root'])/'src/finite_radon.cpp';source.write_bytes(source.read_bytes()+b'//changed\n')
    with pytest.raises(AssertionError,match='src/finite_radon.cpp'):driver.common_adapter().load_binding(p,ROOT,'frt2')


def test_measurement_without_authority_refuses_before_science(tmp_path,monkeypatch):
    p=synthetic(tmp_path,monkeypatch,'ifrt2');attempts=[];old=builtins.__import__
    def refuse(name,*a,**kw):
        if name.split('.')[0]in ('numpy','skimage','skimagexx'):attempts.append(name);raise AssertionError('FORBIDDEN')
        return old(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',refuse)
    monkeypatch.setattr(sys,'argv',['finite_radon_ifrt2','--root-binding',str(p),'--pass-index','2'])
    with pytest.raises(AssertionError,match='Root measurement manifest'):driver.execute('ifrt2')
    assert not attempts
