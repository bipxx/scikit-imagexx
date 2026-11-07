"""Pure source/authority/real-entry tests only; no genuine or timing claims."""
import ast
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import types
import pytest
ROOT=Path(__file__).resolve().parents[1]
CONTEXT=ROOT/'coverage/contracts/mark-boundaries-benchmark-execution-v5.json'
H=lambda b:hashlib.sha256(b).hexdigest()


def adapter():return runpy.run_path(str(ROOT/'benchmarks/mark_boundaries_execution.py'))


def synthetic(tmp_path,monkeypatch,fault='none',context_path=CONTEXT):
    """Metadata stand-in accepted only until a refused import, never a product."""
    c=json.loads(context_path.read_bytes());source=tmp_path/'source';source.mkdir();pins={}
    for name in set(c['source_files_canonical_sha256'])|set(c['FENV_dependency']['source_files_raw_sha256'])|{'CMakeLists.txt',context_path.relative_to(ROOT).as_posix()}:
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes());pins[name]=H(target.read_bytes().replace(b'\r\n',b'\n'))
    def receipt(name,value):
        p=tmp_path/(name+'.json');p.write_text(json.dumps(value));return str(p),H(p.read_bytes())
    prepared=dict(revision='a'*40,files_sha256=pins);producer=dict(run=123,revision='a'*40,artifact=456,sha256='b'*64)
    SDK={c['SDK_header']:'c'*64,'lib/libskimagexx.a':'d'*64}
    runtime,runtime_pin=receipt('runtime',dict(schema='ci-offline-measurement-v1',source=prepared,runtime=dict(python=sys.version,versions=dict(numpy='1.24.4'),packages_sha256=dict(skimagexx={'_core.so':'e'*64}),sdk_sha256=SDK)))
    admission,admission_pin=receipt('admission',dict(passed=True,root_accepted=True,profiles=4,source='a'*40,run=123,product_kind='GENUINE_INSTALLED_WHOLE_PRODUCT'))
    stage,stage_pin=receipt('stage',dict(revision='a'*40,run=123,artifact=456,artifact_sha256='b'*64))
    bundle=tmp_path/'F1';bundle.mkdir();f1=bundle/'receipt.json';f1.write_text('{"synthetic":true}')
    env=dict(UTIL_VIEWS_FENV_MODE='source-built-v1',UTIL_VIEWS_FENV_RECEIPT=str(f1),UTIL_VIEWS_FENV_RECEIPT_SHA256=H(f1.read_bytes()))
    helper,helper_pin=receipt('helper-binding',dict(env=env,files={f1.name:H(f1.read_bytes())}))
    for name,value in env.items():monkeypatch.setenv(name,value)
    for name in('UTIL_VIEWS_FENV_HELPER','GIT_DIR','GIT_WORK_TREE'):monkeypatch.delenv(name,raising=False)
    go_path=tmp_path/'go.json';output=tmp_path/'results'
    b=dict(status='ROOT_AUTHORIZED_GENUINE_FIRST_COHORTS',product_kind='GENUINE_INSTALLED_WHOLE_PRODUCT',source_root=str(source),source_revision='a'*40,git_argv=['git','-C',str(source)],prepared_source=prepared,producer=producer,interpreter=sys.executable,installed_site=str(tmp_path/'site'),numpy='1.24.4',core_path=str(tmp_path/'site/_core.so'),core_sha256='e'*64,SDK_members=SDK,runtime_path=runtime,runtime_sha256=runtime_pin,genuine_root_admission_path=admission,genuine_root_admission_sha256=admission_pin,genuine_root_stager_path=stage,genuine_root_stager_sha256=stage_pin,helper_binding_path=helper,helper_binding_sha256=helper_pin,base_contract_sha256=H(context_path.read_bytes()),root_go_path=str(go_path),routes={c['route_id']:c['route']},output_root=str(output))
    if fault=='extra-cases':b['cases']=[]
    elif fault=='extra-method':b['method']={'repeat':1}
    elif fault=='core':b['core_sha256']='0'*64
    elif fault=='runtime':b['runtime_sha256']='0'*64
    elif fault=='source':(source/'python/skimagexx/_mark_boundaries.py').write_text('changed')
    elif fault=='context':b['base_contract_sha256']='0'*64
    elif fault=='revision':b['source_revision']='0'*40
    elif fault=='producer':b['producer']={**producer,'artifact':789}
    elif fault=='git-root':b['git_argv'][2]=str(tmp_path/'wrong')
    elif fault=='root-admission':b['genuine_root_admission_sha256']='0'*64
    elif fault=='root-stager':b['genuine_root_stager_sha256']='0'*64
    elif fault=='SDK':b['SDK_members']={}
    elif fault=='route':b['routes']={}
    elif fault=='interpreter':b['interpreter']=str(tmp_path/'wrong-python')
    elif fault=='helper-digest':b['helper_binding_sha256']='0'*64
    elif fault=='helper-receipt':f1.write_text('{"changed":true}')
    elif fault=='helper-mode':monkeypatch.setenv('UTIL_VIEWS_FENV_MODE','private-fixed-v1')
    elif fault=='helper-legacy':monkeypatch.setenv('UTIL_VIEWS_FENV_HELPER','old.so')
    elif fault=='CMake':b['prepared_source']['files_sha256']['CMakeLists.txt']='0'*64
    path=tmp_path/'binding.json';path.write_text(json.dumps(b))
    go=dict(binding_sha256=H(path.read_bytes()),routes=[c['route_id']],timed_batches=0,maximum_original_topcalls=1,maximum_native_topcalls=1,helper_binding_sha256=helper_pin,deadline_utc=(datetime.now(timezone.utc)+timedelta(minutes=5)).isoformat())
    if fault=='go':go['binding_sha256']='0'*64
    elif fault=='expired':go['deadline_utc']='2000-01-01T00:00:00+00:00'
    elif fault=='go-budget':go['timed_batches']=1
    go_path.write_text(json.dumps(go))
    monkeypatch.setattr(subprocess,'check_output',lambda argv,**kw:('a'*40+'\n')if argv[-1]=='HEAD'else'dirty'if fault=='git-dirty'else'')
    return path,b,c


def test_exact_cases_method_source_and_historical_inverse():
    a=adapter();c=json.loads(CONTEXT.read_bytes());old=runpy.run_path(str(ROOT/'benchmarks/mark_boundaries.py'))
    assert c['cases']==old['case_specs']()and len(c['cases'])==8 and all(old['case_bounds'](s)for s in c['cases'])
    assert all(v is None for v in c['unbound'].values())
    for name,row in c['historical_sources'].items():assert H(a['historical_bytes'](ROOT,name))==row['predecessor_canonical_sha256']
    for name,pin in c['source_files_canonical_sha256'].items():assert H((ROOT/name).read_bytes().replace(b'\r\n',b'\n'))==pin,name
    text=(ROOT/'tools/montage_sustained.py').read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='measure')
    assert H((ast.get_source_segment(text,node)+'\n').encode())==c['method']['method_sha256']
    assert c['method']==dict(repeat=7,target_ms=100,warmup_min_ms=250,threads=1,retain_all_losses=True,method_sha256=c['method']['method_sha256'])


@pytest.mark.parametrize('cached',(False,True))
def test_cold_cached_real_UNBOUND_CLI_before_request(cached):
    driver=ROOT/'benchmarks/mark_boundaries_consumer.py'
    code="""import builtins,runpy,sys,types
old=builtins.__import__
def refuse(name,*a,**k):
 if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):raise AssertionError('TARGET_IMPORT_REACHED')
 return old(name,*a,**k)
if sys.argv[2]=='True':sys.modules['numpy']=types.ModuleType('numpy')
builtins.__import__=refuse;sys.argv=[sys.argv[1]];runpy.run_path(sys.argv[0],run_name='__main__')
"""
    p=subprocess.run([sys.executable,'-I','-c',code,str(driver),str(cached)],capture_output=True,text=True)
    assert p.returncode!=0 and 'MARK_BENCHMARK_UNBOUND'in p.stderr and 'TARGET_IMPORT_REACHED'not in p.stderr and 'Traceback'not in p.stderr


FAULTS=('none','none-cached','extra-cases','extra-method','core','runtime','source','context','revision','producer','git-root','git-dirty','root-admission','root-stager','SDK','route','interpreter','helper-digest','helper-receipt','helper-mode','helper-legacy','CMake','go','expired','go-budget')
@pytest.mark.parametrize('fault',FAULTS)
def test_bound_actual_CLI_refusal_has_zero_bodies(tmp_path,monkeypatch,fault):
    import builtins
    a=adapter();path,b,c=synthetic(tmp_path,monkeypatch,fault);attempts=[];old=builtins.__import__
    def refuse(name,*args,**kw):
        if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):attempts.append(name);raise RuntimeError('MARK_REQUEST_REFUSED')
        return old(name,*args,**kw)
    if fault=='none-cached':monkeypatch.setitem(sys.modules,'numpy',types.ModuleType('numpy'))
    monkeypatch.setattr(builtins,'__import__',refuse);output=Path(b['output_root'])/'never.json';monkeypatch.setattr(sys,'path',list(sys.path))
    monkeypatch.setattr(sys,'argv',['mark','--root-binding',str(path),'--first-cohort','--case',c['cases'][0]['id'],'--output',str(output)])
    module=runpy.run_path(str(ROOT/'benchmarks/mark_boundaries_consumer.py'))
    with pytest.raises(RuntimeError if fault.startswith('none')else AssertionError,match='MARK_REQUEST_REFUSED'if fault.startswith('none')else None):a['execute'](module)
    assert attempts==(['numpy']if fault.startswith('none')else[])
    assert not output.exists()and not output.with_suffix('.batches.jsonl').exists()
    claim=Path(b['output_root'])/'mark-first-cohort.claim.json'
    assert claim.exists()==fault.startswith('none')
    if fault.startswith('none'):
        # A changed report filename cannot replay the same root-bound cohort.
        sys.argv[-1]=str(Path(b['output_root'])/'other.json')
        with pytest.raises(FileExistsError):a['execute'](module)
        assert attempts==['numpy']


@pytest.mark.parametrize('fault',('none','missing','source','producer','core','runtime','scope','cases','cohort','cohort-binding','deadline'))
def test_separate_measurement_GO_before_bodies(tmp_path,monkeypatch,fault):
    a=adapter();path,b,c=synthetic(tmp_path,monkeypatch);pin=H(path.read_bytes());done=dict(status='PASS_REAL_CLI_STOP_BEFORE_BODY',route='mark',source=b['source_revision'],root_binding_sha256=pin,counts=dict(original=1,native=1),sampler_body_calls=0,timed_batches=0,outer_state_restored=True,poststop_source_authority_unchanged=True)
    if fault=='cohort':done['timed_batches']=1
    cohort=tmp_path/'synthetic-cohort.json';cohort.write_text(json.dumps(done))
    m=dict(schema='next-metrics-measurement-v1',status='ROOT_AUTHORIZED_ONE_INITIAL_COMPARISON',prepared_source=b['prepared_source'],producer=b['producer'],core_sha256=b['core_sha256'],runtime_sha256=b['runtime_sha256'],families={'mark_boundaries_consumer':dict(contract=a['CONTEXT'],cases=8,case_ids=[s['id']for s in c['cases']],untimed_calls_per_case=1)},execution_scope=dict(workers=1,repeat=7,target_ms=100,warmup_min_ms=250,single_host_same_installed_runtime=True,no_automatic_retry=True,retain_all_losses=True),completed_first_cohort=dict(path=str(cohort),sha256=H(cohort.read_bytes()),binding_sha256=pin),deadline_utc='2099-01-01T00:00:00+00:00')
    if fault in('source','producer','core','runtime'):m[{'source':'prepared_source','producer':'producer','core':'core_sha256','runtime':'runtime_sha256'}[fault]]={}
    elif fault=='scope':m['execution_scope']['workers']=4
    elif fault=='cases':m['families']['mark_boundaries_consumer']['cases']=7
    elif fault=='cohort-binding':m['completed_first_cohort']['binding_sha256']='0'*64
    elif fault=='deadline':m['deadline_utc']='2000-01-01T00:00:00+00:00'
    target=tmp_path/'measurement.json';target.write_text(json.dumps(m))
    if fault=='none':assert a['validate_measurement'](target,b,c,pin)[0]==H(target.read_bytes())
    else:
        with pytest.raises(AssertionError):a['validate_measurement'](None if fault=='missing'else target,b,c,pin)


@pytest.mark.parametrize('observer_kind',('absent','non-stopping','replaces-at-entry','replaces-on-any-event','admitted-stopping'))
def test_actual_sampler_entry_stops_with_replaced_profiler(tmp_path,observer_kind):
    a=adapter();c=json.loads(CONTEXT.read_bytes());common=a['support'](ROOT,c);measure=runpy.run_path(str(ROOT/'tools/montage_sustained.py'))['measure'];body=[];callbacks=[];entries=[]
    oldprofile,oldtrace=sys.getprofile(),sys.gettrace()
    class Stopped(BaseException):pass
    def observer(frame,event,value):
        if observer_kind=='replaces-on-any-event':sys.setprofile(lambda *args:None)
        if event=='call'and frame.f_code is measure.__code__:
            entries.append(frame.f_code)
            if observer_kind=='replaces-at-entry':sys.setprofile(lambda *args:None)
            if observer_kind=='admitted-stopping':raise Stopped('external-entry-stop')
    def trace(frame,event,value):
        if frame.f_code is measure.__code__ and event=='line':body.append(frame.f_lineno)
        return trace
    cli=types.SimpleNamespace(first_cohort=True,output=tmp_path/'never.json')
    try:
        sys.settrace(trace);sys.setprofile(None if observer_kind=='absent'else observer)
        with pytest.raises(Stopped if observer_kind=='admitted-stopping'else AssertionError,match='external-entry-stop'if observer_kind=='admitted-stopping'else'body forbidden'):
            a['sample_entry'](measure,lambda:callbacks.append('original'),lambda:callbacks.append('native'),cli,0,c['cases'][0],'2099-01-01T00:00:00+00:00',common)
    finally:sys.setprofile(oldprofile);sys.settrace(oldtrace)
    assert not body and not callbacks and not cli.output.with_suffix('.batches.jsonl').exists()
    assert sys.getprofile()is oldprofile and sys.gettrace()is oldtrace


def test_native_return_roles_and_actual_sampler_not_new_engine():
    a=adapter();c=json.loads(CONTEXT.read_bytes());source=(ROOT/'benchmarks/mark_boundaries_execution.py').read_text();tree=ast.parse(source)
    assert not any(isinstance(n,ast.FunctionDef)and n.name=='measure'for n in ast.walk(tree))
    assert source.index('validate_binding(cli.root_binding,')<source.index('import numpy as np')and source.index('validate_measurement(cli.measurement_go')<source.index('import numpy as np')
    calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='measure'];assert len(calls)==1 and [n.value for n in calls[0].args[2:]]==[7,100]
    assert 'common.fp_controller('in source and source.count('common.fp_controller(')==1 and 'ctypes.CDLL'not in source
    for spec in c['cases']:
        counts=a['native_requirements'](spec);assert counts['get_num_threads']==1 and counts['mark_boundaries_prepare']==counts['mark_boundaries_mask']==1
        assert counts['mark_boundaries_paint']==counts['mark_boundaries_color']==1+int(spec['outline'])
    assert c['providers']and c['genuine_first_cohort_status']=='PENDING_NO_ACTUAL_PRODUCT_BINDING'


def test_factory_descriptor_standins_only_no_array_or_algorithm_bodies():
    a=adapter();c=json.loads(CONTEXT.read_bytes())
    class Array:
        def __init__(self,n,dtype='int64',shape=None,modulus=None,layout='C'):self.n,self.dtype,self.shape,self.modulus,self.layout=n,dtype,shape,modulus,layout
        def __mod__(self,n):return Array(self.n,self.dtype,self.shape,n,self.layout)
        def reshape(self,*shape):return Array(self.n,self.dtype,shape[0]if len(shape)==1 else shape,self.modulus,self.layout)
        def astype(self,dtype):return Array(self.n,dtype,self.shape,self.modulus,self.layout)
        def __getitem__(self,key):assert key==(slice(None),slice(None,None,-1));return Array(self.n,self.dtype,self.shape,self.modulus,'reverse')
    fake=types.SimpleNamespace(int64='int64',arange=lambda n,dtype:Array(n),asfortranarray=lambda v:Array(v.n,v.dtype,v.shape,v.modulus,'F'))
    for spec in c['cases']:
        (image,labels),options=a['fixture'](spec,fake);h,w=spec['shape']
        assert image.shape==((h,w,3)if spec['rgb']else(h,w))and image.dtype==spec['dtype']and image.modulus==251 and image.layout==spec['layout']
        assert labels.shape==(h,w)and labels.modulus==7 and options==dict(mode=spec['mode'],outline_color=(0,0,1)if spec['outline']else None)
