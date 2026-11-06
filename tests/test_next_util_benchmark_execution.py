"""Source/refusal controls; no scientific imports, algorithm or sampler calls."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import pytest
from next_util_benchmark_support import ROOT,CONTEXT,load_driver,bound_request_refusal

DATA=json.loads(CONTEXT.read_bytes())
ROUTES=tuple(DATA['routes'])


def test_exact_frozen_functions_context_and_sampler():
    adapter=runpy.run_path(str(ROOT/'benchmarks/next_util_execution.py'))
    for name,row in DATA['sources'].items():
        old=adapter['historical_bytes'](ROOT,name).decode();current=(ROOT/name).read_text()
        for function,pin in row['frozen_functions'].items():
            old_node=next(n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef)and n.name==function)
            node=next(n for n in ast.parse(current).body if isinstance(n,ast.FunctionDef)and n.name==function)
            assert ast.get_source_segment(old,old_node)==ast.get_source_segment(current,node)
            assert hashlib.sha256(ast.get_source_segment(current,node).encode()).hexdigest()==pin
    for name,pin in DATA['production_source_canonical_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin,name
    for name,pin in DATA['historical_contracts_raw_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==pin
    text=(ROOT/'tools/montage_sustained.py').read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='measure')
    assert hashlib.sha256((ast.get_source_segment(text,node)+'\n').encode()).hexdigest()==DATA['method']['method_sha256']
    assert DATA['method']['target_ms']==100 and DATA['method']['warmup_min_ms']==250 and DATA['method']['retain_all_losses']is True
    assert all(x is None for x in DATA['unbound'].values())
    assert len(DATA['apis'])==6 and len(DATA['routes'])==5
    assert sum(len(x['cases'])for x in DATA['routes'].values())==DATA['counts']['cases']==40


@pytest.mark.parametrize('route',ROUTES)
def test_exact_cases_bounds_and_real_native_entry_names(route):
    module=load_driver(route);rows=module.case_specs();assert rows==DATA['routes'][route]['cases']
    bound=getattr(module,'case_bounds',getattr(module,'source_safe',None));assert all(bound(r)for r in rows)
    adapter=runpy.run_path(str(ROOT/'benchmarks/next_util_execution.py'))
    binders='\n'.join((ROOT/name).read_text()for name in DATA['production_source_canonical_sha256']if name.startswith('src/bind_'))
    for row in rows:
        api=row.get('api',DATA['routes'][route]['apis'][0])
        for symbol in adapter['expected_returns'](api,row):assert '"'+symbol+'"'in binders,symbol


@pytest.mark.parametrize('route',ROUTES)
@pytest.mark.parametrize('cached',(False,True))
def test_real_cold_cached_unbound_CLI_stops_before_import(route,cached):
    path=ROOT/DATA['routes'][route]['driver']
    code="""import builtins,runpy,sys,types
old=builtins.__import__
def refuse(name,*a,**k):
 if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):raise RuntimeError('SCIENCE_REQUEST_REACHED')
 return old(name,*a,**k)
if sys.argv[2]=='True':sys.modules['numpy']=types.ModuleType('numpy')
builtins.__import__=refuse;sys.argv=[sys.argv[1]];runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result=subprocess.run([sys.executable,'-I','-c',code,str(path),str(cached)],capture_output=True,text=True)
    assert result.returncode!=0 and 'PROSPECTIVE_UNBOUND'in result.stderr
    assert 'SCIENCE_REQUEST_REACHED'not in result.stderr and 'Traceback'not in result.stderr


@pytest.mark.parametrize('route',ROUTES)
@pytest.mark.parametrize('fault',('none','none-cached','cases','core','revision','runtime','root','stager','source','git-root','context','route','expired','go'))
def test_actual_bound_metadata_refusal(tmp_path,monkeypatch,route,fault):
    bound_request_refusal(tmp_path,monkeypatch,route,fault)


@pytest.mark.parametrize('api',tuple(DATA['apis']))
def test_API_specific_first_cohort_plan_selectors(api):
    route=next(k for k,v in DATA['routes'].items()if api in v['apis']);driver=ROOT/DATA['routes'][route]['driver']
    result=subprocess.run([sys.executable,'-I',str(driver),'--plan','--first-cohort','--api',api],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    plan=json.loads(result.stdout);assert plan['state']=='PROSPECTIVE_UNBOUND_NO_MEASUREMENT'
    assert all(row.get('api',api)==api for row in plan['cases'])
    assert len(plan['cases'])==(4 if route=='views'else 8)


def test_real_sampler_interface_is_not_a_stub_or_new_engine():
    source=(ROOT/'benchmarks/next_util_execution.py').read_text();tree=ast.parse(source)
    assert not any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))and n.name=='measure'for n in ast.walk(tree))
    assert 'from montage_sustained import measure'in source
    calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='measure']
    assert len(calls)==1 and [n.value for n in calls[0].args[2:]]==[7,100]
    assert [n.arg for n in calls[0].keywords]==['journal','case_index','case_id','pass_index','deadline']
    execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
    locals_={n.id for n in ast.walk(execute)if isinstance(n,ast.Name)and isinstance(n.ctx,ast.Store)}
    assert {'spec','api','args','options','expected','actual','returns','accepted','native_FP'}<=locals_
    assert 'No target import until authenticated'in source

@pytest.mark.parametrize('route',ROUTES)
@pytest.mark.parametrize('fault',('none','producer','runtime','context','cases','scope','status'))
def test_existing_root_measurement_manifest_is_separate_authority(tmp_path,route,fault):
    adapter=runpy.run_path(str(ROOT/'benchmarks/next_util_execution.py'))
    binding=dict(prepared_source=dict(revision='a'*40,files_sha256={'x':'b'*64}),producer=dict(run=123,revision='a'*40,artifact=456,sha256='c'*64),core_sha256='d'*64,runtime_sha256='e'*64,source_revision='a'*40,genuine_root_admission_sha256='f'*64,genuine_root_stager_sha256='1'*64,base_contract_sha256=hashlib.sha256(CONTEXT.read_bytes()).hexdigest())
    specs=DATA['routes'][route]['cases'];name=Path(DATA['routes'][route]['driver']).stem
    authority=dict(schema='next-metrics-measurement-v1',status='ROOT_AUTHORIZED_ONE_INITIAL_COMPARISON',prepared_source=binding['prepared_source'],producer=binding['producer'],core_sha256=binding['core_sha256'],runtime_sha256=binding['runtime_sha256'],first_cohort=dict(status='ROOT_AUTHORIZED_GENUINE_FIRST_COHORTS',source_revision=binding['source_revision'],root_admission_sha256=binding['genuine_root_admission_sha256'],root_stager_sha256=binding['genuine_root_stager_sha256'],source_files_raw_sha256={adapter['CONTEXT']:binding['base_contract_sha256']}),families={name:dict(contract=adapter['CONTEXT'],cases=len(specs))},execution_scope=dict(workers=1,repeat=7,target_ms=100,warmup_min_ms=250,single_host_same_installed_runtime=True,no_automatic_retry=True,retain_all_losses=True))
    with pytest.raises(AssertionError,match='Root measurement manifest required'):adapter['load_measurement'](None,binding,DATA,route,specs)
    if fault=='producer':authority['producer']={**binding['producer'],'artifact':789}
    elif fault=='runtime':authority['runtime_sha256']='0'*64
    elif fault=='context':authority['first_cohort']['source_files_raw_sha256'][adapter['CONTEXT']]='0'*64
    elif fault=='cases':authority['families'][name]['cases']-=1
    elif fault=='scope':authority['execution_scope']['workers']=4
    elif fault=='status':authority['status']='UNBOUND'
    path=tmp_path/'measurement.json';path.write_text(json.dumps(authority))
    if fault=='none':assert adapter['load_measurement'](path,binding,DATA,route,specs)==hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        with pytest.raises(AssertionError):adapter['load_measurement'](path,binding,DATA,route,specs)


def test_measurement_authority_checked_before_science():
    source=(ROOT/'benchmarks/next_util_execution.py').read_text()
    assert source.index('else load_measurement(cli.measurement_go')<source.index('    import numpy as np')


@pytest.mark.parametrize('observer_kind',('absent','non-stopping','admitted-stopping'))
def test_first_cohort_real_entry_always_stops_before_sampler_body(tmp_path,observer_kind):
    # This enters only the real sampler Python CALL boundary. Trace proves no
    # sampler body line executes; callbacks, clock and journal are never used.
    adapter=runpy.run_path(str(ROOT/'benchmarks/next_util_execution.py'))
    measure=runpy.run_path(str(ROOT/'tools/montage_sustained.py'))['measure']
    prior_profile,prior_trace=sys.getprofile(),sys.gettrace();entries=[];body=[];callbacks=[]
    class Captured(Exception):pass
    def observer(frame,event,value):
        if event=='call'and frame.f_code is measure.__code__:
            entries.append(frame.f_back.f_code)
            if observer_kind=='admitted-stopping':raise Captured('external-entry-capture')
    def trace(frame,event,value):
        if frame.f_code is measure.__code__ and event=='line':body.append(frame.f_lineno)
        return trace
    profile=adapter['first_cohort_profile'](measure,None if observer_kind=='absent'else observer)
    journal=tmp_path/'never.jsonl'
    try:
        sys.settrace(trace);sys.setprofile(profile)
        with pytest.raises(Captured if observer_kind=='admitted-stopping'else AssertionError,match='external-entry-capture'if observer_kind=='admitted-stopping'else'body forbidden'):
            measure(lambda:callbacks.append('original'),lambda:callbacks.append('native'),7,100,journal=journal,case_index=0,case_id='entry-only',pass_index=0,deadline='2099-01-01T00:00:00+00:00')
    finally:sys.setprofile(prior_profile);sys.settrace(prior_trace)
    assert not body and not callbacks and not journal.exists()
    assert entries==([]if observer_kind=='absent'else[test_first_cohort_real_entry_always_stops_before_sampler_body.__code__])
    assert sys.getprofile()is prior_profile and sys.gettrace()is prior_trace


def test_F1_dependencies_exact_and_legacy_assumption_removed():
    import os
    from next_util_benchmark_support import CONTEXT
    current=json.loads(CONTEXT.read_bytes());adapter=runpy.run_path(str(ROOT/'benchmarks/next_util_execution.py'))
    donor=Path(os.environ.get('NEXT_UTIL_F1_SOURCE_ROOT',str(ROOT)))
    module=adapter['fenv_source'](donor,current)
    assert module.__file__==str(donor/'tests/util_views_fenv.py')
    assert current['FENV_dependency']['source']=='659dfd616587a73df1f9eeb5a61e201dbaf5d17e'
    assert set(current['FENV_dependency']['source_files_raw_sha256'])=={'coverage/contracts/util-views-fenv-v1.json','tests/test_util_views_fenv.py','tests/test_util_views_public.py','tests/util_views_fenv.py','tests/util_views_fenv_build.py','tests/util_views_fenv_probe.py','tests/util_views_fenv_restore.c'}
    public=(donor/'tests/test_util_views_public.py').read_bytes()
    historical=module.project_public_source(public)
    native=json.loads((ROOT/'coverage/contracts/util-view-operations-v1.json').read_bytes())
    assert hashlib.sha256(historical).hexdigest()==native['source_pins_raw_sha256']['tests/test_util_views_public.py']
    with pytest.raises(AssertionError):module.project_public_source(public+b'\n# extra-change')
    text=(ROOT/'benchmarks/next_util_execution.py').read_text();tree=ast.parse(text)
    block=ast.get_source_segment(text,next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='fp_controller'))
    assert '0c40af'not in block and 'ctypes.CDLL'not in block and 'module.backend()'in block
    assert "os.environ.get('UTIL_VIEWS_FENV_MODE')=='source-built-v1'"in block
    old=json.loads((ROOT/'coverage/contracts/next-util-benchmark-execution-v1.json').read_bytes())
    assert old['FENV_dependency']['sha256']=='0c40af724c3cbca34fc6c6c027e62ddb1635609b7575a09505d578db24466220'
    assert current['routes']==old['routes']and current['method']==old['method']and current['unbound']==old['unbound']


@pytest.mark.parametrize('fault',('mode','legacy','missing-receipt','missing-digest','bad-digest','bad-receipt','source'))
def test_F1_consumer_refuses_before_native_backend_or_target(tmp_path,monkeypatch,fault):
    import os
    adapter=runpy.run_path(str(ROOT/'benchmarks/next_util_execution.py'));context=json.loads(CONTEXT.read_bytes())
    donor=Path(os.environ.get('NEXT_UTIL_F1_SOURCE_ROOT',str(ROOT)))
    monkeypatch.setenv('UTIL_VIEWS_FENV_MODE','source-built-v1');monkeypatch.delenv('UTIL_VIEWS_FENV_HELPER',raising=False)
    receipt=tmp_path/'invalid-receipt.json';receipt.write_text('{}')
    monkeypatch.setenv('UTIL_VIEWS_FENV_RECEIPT',str(receipt));monkeypatch.setenv('UTIL_VIEWS_FENV_RECEIPT_SHA256',hashlib.sha256(receipt.read_bytes()).hexdigest())
    if fault=='mode':monkeypatch.setenv('UTIL_VIEWS_FENV_MODE','private-fixed-v1')
    elif fault=='legacy':monkeypatch.setenv('UTIL_VIEWS_FENV_HELPER','legacy-not-opened.so')
    elif fault=='missing-receipt':monkeypatch.delenv('UTIL_VIEWS_FENV_RECEIPT')
    elif fault=='missing-digest':monkeypatch.delenv('UTIL_VIEWS_FENV_RECEIPT_SHA256')
    elif fault=='bad-digest':monkeypatch.setenv('UTIL_VIEWS_FENV_RECEIPT_SHA256','0'*64)
    elif fault=='source':context={**context,'FENV_dependency':{**context['FENV_dependency'],'source_files_raw_sha256':{'tests/util_views_fenv.py':'0'*64}}}
    # Real reviewed validator processes bad metadata, but any native configure
    # request is explicitly forbidden. No helper library is read/loaded/called.
    original=adapter['fenv_source'];requests=[]
    def source(root,data):
        module=original(root,data)
        def refuse(*a,**k):requests.append('configure');raise AssertionError('NATIVE_CONFIGURE_FORBIDDEN')
        module.configure=refuse
        return module
    adapter['fp_controller'].__globals__['fenv_source']=source
    with pytest.raises((AssertionError,KeyError)):
        adapter['fp_controller'](donor,context)
    assert requests==[]



def test_FP_inner_save_restore_flags_match_frozen00f_exactly():
    # Parse both versions on this interpreter; never compare frozen AST dumps.
    data=json.loads(CONTEXT.read_bytes());text=(ROOT/'benchmarks/next_util_execution.py').read_text();old=text
    for fragment in data['00f_successor_sources']['benchmarks/next_util_execution.py']['inverse_fragments']:
        old=old.replace(fragment['current'],fragment['predecessor'],1)
    def inner(source):
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='fp_controller')
        return {n.name:ast.dump(n,include_attributes=False)for n in node.body if isinstance(n,ast.FunctionDef)}
    assert inner(text)==inner(old)
    assert set(inner(text))=={'save','restore','flags'}



def test_authenticated_FP_backend_initialized_once_before_case_loop():
    source=(ROOT/'benchmarks/next_util_execution.py').read_text();tree=ast.parse(source)
    execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
    calls=[n for n in ast.walk(execute)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='fp_controller']
    assert len(calls)==1
    loop=next(n for n in execute.body if isinstance(n,ast.For))
    assert calls[0].lineno<loop.lineno
    controller=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='fp_controller')
    backend_calls=[n for n in ast.walk(controller)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='backend']
    assert len(backend_calls)==1
