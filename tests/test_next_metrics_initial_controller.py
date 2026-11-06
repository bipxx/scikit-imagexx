"""Inert controller checks; scientific algorithms, packages and sampler refused."""
import ast
import hashlib
import json
import os
import symtable
from pathlib import Path
import re
import subprocess
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
WORKFLOW=(ROOT/'.github/workflows/next-metrics-initial.yml').read_text()
SCRIPTS=['\n'.join(line[10:]for line in s.splitlines())for s in re.findall(r"          [^\n]*<<'PY'\n(.*?)          PY",WORKFLOW,re.S)]
CONTRACT=json.loads((ROOT/'coverage/contracts/next-metrics-measurement-v1.json').read_bytes())
PLAN=json.loads((ROOT/'coverage/contracts/next-metrics-first-cohort-v1.json').read_bytes())


def test_unbound_contract_and_frozen_scope():
    assert CONTRACT['status']=='UNBOUND_NO_DISPATCH_OR_SCIENTIFIC_CALLS'
    assert set(CONTRACT['producer'].values())=={None}
    assert all(CONTRACT[k]is None for k in ('prepared_source','core_sha256','runtime_sha256','benchmark_fingerprint','SDK_members'))
    first=CONTRACT['first_cohort']
    assert first['status']=='UNBOUND' and first['source_revision']is None and first['source_files_raw_sha256']is None
    assert all(first[k]is None for k in ('root_admission_path','root_admission_sha256','root_stager_path','root_stager_sha256'))
    for kind in ('probe','prospective'):
        assert hashlib.sha256((ROOT/first[kind+'_path']).read_bytes()).hexdigest()==first[kind+'_sha256']
    assert PLAN['source_revision']is None and PLAN['source_git_blobs']is None
    assert PLAN['budget']==dict(original_public_topcalls=4,native_public_topcalls=4,timed_batches=0)
    assert {r['api']for r in PLAN['routes'].values()}=={'metrics.adapted_rand_error','metrics.contingency_table','draw.ellipse_perimeter','segmentation.find_boundaries'}
    assert CONTRACT['families']==dict(label_pair_metrics=dict(contract='coverage/contracts/label-pair-metrics-benchmark-v1.json',cases=8),ellipse_perimeter=dict(contract='coverage/contracts/ellipse-perimeter-fallback-runtime-v1.json',cases=12),find_boundaries=dict(contract='coverage/contracts/find-boundaries-benchmark-binding-v1.json',cases=8))
    assert CONTRACT['execution_scope']['canonical_cases']==sum(f['cases']for f in CONTRACT['families'].values())==28
    assert CONTRACT['execution_scope']['canonical_raw_samples']==14*CONTRACT['execution_scope']['canonical_cases']==392
    assert CONTRACT['execution_scope']['warmup_groups']==2*CONTRACT['execution_scope']['canonical_cases']==56
    assert PLAN['routes']['boundary']['root_binding_flag']=='--root-binding'
    assert all('root_binding_flag'not in r for k,r in PLAN['routes'].items()if k!='boundary')
    for route in PLAN['routes'].values():
        assert route['kind']=='dedicated' and route['sampler']=='tools/montage_sustained.py'
        assert route['first_case'].get('api',route['api'])==route['api']
        assert route['argv'][:4]==['--threads','1','--repeat','7']


def test_workflow_reuses_exact_source_stager_and_unchanged_sample_gates():
    assert len(SCRIPTS)==4
    for script in SCRIPTS:ast.parse(script)
    assert "'.github/workflows/public-qualification.yml'"in SCRIPTS[0]
    assert "'rank-statistics-product-minimum'"in SCRIPTS[0]
    for token in ("z.extractall('stage')","Path('stage')","prepared / 'runtime.json'","product['status']==product['provenance']['status']=='staged-unaccepted'","qualification['source']==manifest['source']","stager['artifact_sha256']==os.environ['ARTIFACT_SHA256']","admission['source']==os.environ['PRODUCER_REVISION']","admission['profiles']==4","sha(prepared/'runtime.json')==frozen['runtime_sha256']"):
        assert token in SCRIPTS[1],token
    assert '--find-links stage/wheels -r stage/runtime-lock.txt'in WORKFLOW
    assert 'snapshot/tools/ci_measurement_artifacts.py verify --wheel-dir stage/wheels --artifact stage/runtime.json'in WORKFLOW
    assert '--wheel-dir "$GITHUB_WORKSPACE/stage/wheels" --artifact "$GITHUB_WORKSPACE/stage/runtime.json"'in WORKFLOW
    assert 'for family in label_pair_metrics ellipse_perimeter find_boundaries'in WORKFLOW
    assert 'binding_args=(--root-binding "$output/first-cohort-binding.json")'in WORKFLOW
    assert "prepared_source=frozen['prepared_source']"in SCRIPTS[2]
    assert "root_go_path=str(go_path)"in SCRIPTS[2]
    assert "base_contract_sha256=sha(snapshot/frozen['families']['find_boundaries']['contract'])"in SCRIPTS[2]
    assert all(token not in WORKFLOW for token in ('cmake --build','python -m pytest','pip install numpy','pip install scipy','workflow_run:'))
    baseline=(ROOT/'.github/workflows/draw-batch-initial.yml').read_text()
    last=re.findall(r"          [^\n]*<<'PY'\n(.*?)          PY",baseline,re.S)[-1]
    last='\n'.join(line[10:]for line in last.splitlines()).replace('draw-batch-measurement-v1.json','next-metrics-measurement-v1.json').replace("(('draw_polygon',12),('bezier_curve',12),('draw_set_color',8))","(('label_pair_metrics',8),('ellipse_perimeter',12),('find_boundaries',8))")
    assert SCRIPTS[-1]==last  # Full raw-sample/journal/loss validator retained verbatim otherwise.
    assert 'if: always()'in WORKFLOW and 'include-hidden-files: true'in WORKFLOW


def test_actual_workflow_auth_stops_unbound_before_network(tmp_path,monkeypatch):
    target=tmp_path/'controller/coverage/contracts';target.mkdir(parents=True)
    (target/'next-metrics-measurement-v1.json').write_text(json.dumps(CONTRACT))
    monkeypatch.chdir(tmp_path)
    for key,value in dict(PRODUCER_RUN='1',PRODUCER_REVISION='a'*40,ARTIFACT_ID='2',ARTIFACT_SHA256='b'*64).items():monkeypatch.setenv(key,value)
    def refused(*a,**k):raise RuntimeError('NETWORK_FORBIDDEN')
    monkeypatch.setattr(subprocess,'check_output',refused)
    with pytest.raises(AssertionError,match='UNBOUND'):exec(compile(SCRIPTS[0],'<actual-auth>','exec'),{})
    assert not(tmp_path/'results').exists()


def test_actual_probe_rejects_unbound_before_import_or_git(tmp_path):
    binding=tmp_path/'binding.json';binding.write_text('{"status":"UNBOUND"}')
    result=subprocess.run([sys.executable,str(ROOT/'tools/next_metrics_first_cohort_probe.py'),'--binding',str(binding),'--root-go',str(tmp_path/'absent.json'),'--route','metrics-are'],capture_output=True,text=True)
    assert result.returncode!=0 and 'UNBOUND: no scientific calls permitted'in result.stderr
    assert 'ModuleNotFoundError'not in result.stderr


# This child supplies inert Python stand-ins, never imports a scientific package,
# never runs a real benchmark driver and gives the sampler a forbidden body.
STUB_CHILD=r'''
import builtins,hashlib,importlib.abc,json,os,runpy,subprocess,sys,types
from pathlib import Path
probe,plan_path,folder,route_name,bad=sys.argv[1:];folder=Path(folder);folder.mkdir()
plan=json.loads(Path(plan_path).read_bytes());routes=plan['routes'];route=routes[route_name]
root=folder/'snapshot';site=folder/'installed';root.mkdir();site.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
core=site/'_core.so';core.write_bytes(b'INERT_STUB_NO_NATIVE_COMPONENT')
admission=folder/'admission.json';admission.write_text(json.dumps(dict(source='a'*40 if bad!='admission'else'b'*40,passed=True,root_accepted=True,profiles=4)))
np=types.ModuleType('numpy');np.__version__='1.24.4';np.ndarray=type('InertArray',(),{})
np.geterr=lambda:{'invalid':'warn'};np.geterrcall=lambda:None;np.getbufsize=lambda:8192
compat=types.ModuleType('skimagexx._compat');compat._fallback_enabled=True
xx=types.ModuleType('skimagexx');xx.__file__=str(site/'__init__.py');xx._compat=compat
workers=[1];xx.get_num_threads=lambda:workers[0];xx.set_num_threads=lambda n:workers.__setitem__(0,n)
xx.set_fallback_enabled=lambda flag:setattr(compat,'_fallback_enabled',flag)
xx._core=types.SimpleNamespace(__file__=str(core),get_num_threads=len)
for name,value in zip(dict.fromkeys(route['native_entries']),[abs,len,id,hash]):setattr(xx._core,name,value)
sk=types.ModuleType('skimage');sk.__version__='0.26.0'
namespace=route['api'].split('.')[0];name=route['api'].split('.')[1]
original=types.ModuleType('skimage.'+namespace);native_public=types.ModuleType('skimagexx.'+namespace);owned=types.ModuleType(route['owned_module'])
if route_name=='metrics-contingency':result=types.SimpleNamespace(format='csr')
else:result=(1,2,3)if namespace=='metrics'else(1,2)
if namespace=='segmentation':result=True
original.__dict__['result']=result;exec('def '+name+'(*arguments,**options):\n return result',original.__dict__)
owned.__dict__['result']=result
accept=route['acceptance_function']
exec('def '+accept+'(*arguments,**options):\n native_result=result\n return native_result',owned.__dict__)
if accept!=name:exec('def '+name+'(*arguments,**options):\n return '+accept+'(*arguments,**options)',owned.__dict__)
setattr(native_public,name,getattr(owned,name));setattr(xx,namespace,native_public)
for module in (np,compat,xx,sk,original,native_public,owned):sys.modules[module.__name__]=module
class Refuse(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in('numpy','scipy','skimage','skimagexx'):raise RuntimeError('REAL_SCIENTIFIC_IMPORT_FORBIDDEN')
sys.meta_path.insert(0,Refuse())
sampler=root/route['sampler'];sampler.parent.mkdir();sampler.write_text("def measure(*args,**kwargs):\n raise RuntimeError('SAMPLER_BODY_FORBIDDEN')\n")
driver=root/route['driver'];driver.parent.mkdir()
namespace_main={'route':route,'original':original,'native_public':native_public}
# The real probe must trace this file and this sampler, despite controller living elsewhere.
code="""import json,sys
from pathlib import Path
from montage_sustained import measure
import skimagexx as xx
route=json.loads(ROUTE)
if route.get('root_binding_flag'):
 assert sys.argv[sys.argv.index('--root-binding')+1].endswith('binding.json')
from importlib import import_module
original=import_module('skimage.'+route['api'].split('.')[0])
native_public=import_module('skimagexx.'+route['api'].split('.')[0])
def input_state(arguments):return tuple(arguments)
def output_snapshot(value):return {'storage':'inert-complete-storage'}
def snapshot(value):return output_snapshot(value)
def main():
 xx.set_fallback_enabled(False)
 options={};arguments=(1,2);positional=arguments;state=globals()['input_state'](arguments)
 if route['api'].startswith('segmentation.'):
  image=type('Image',(),{'ctypes':type('Address',(),{'data':123})()})();arguments=(image,);frozen=snapshot(image);address=image.ctypes.data
 expected=getattr(original,route['api'].split('.')[1])(*arguments,**options)
 actual=getattr(native_public,route['api'].split('.')[1])(*arguments,**options)
 spec=route['first_case'];calls=route['native_entries'];accepted=[True];snapshots=['get_num_threads'];rows=[];error=0
 LOCAL_SNAPSHOT
 if route['api'].startswith('draw.'):
  input_state=(arguments,options.copy())
 measure(None,None)
main()
""".replace('ROUTE',repr(json.dumps(route)))
code=code.replace(' LOCAL_SNAPSHOT', ' from draw_coordinates import snapshot'if namespace=='draw'else'')
standin=types.ModuleType('draw_coordinates');standin.snapshot=lambda value:{'storage':'inert-complete-storage'};sys.modules['draw_coordinates']=standin
if bad=='calls':code=code.replace("calls=route['native_entries']","calls=[]")
if bad=='input':code=code.replace('measure(None,None)',"image.ctypes.data=999;measure(None,None)"if route_name=='boundary'else'arguments=(9,9);measure(None,None)')
driver.write_text(code)
files={str(p.relative_to(root)):sha(p)for p in(driver,sampler)}
binding=dict(status='ROOT_AUTHORIZED_GENUINE_FIRST_COHORTS',source_root=str(root),source_revision='a'*40,git_argv=['INERT_GIT'],genuine_root_admission_path=str(admission),genuine_root_admission_sha256=sha(admission),product_kind='GENUINE_INSTALLED_WHOLE_PRODUCT',interpreter=sys.executable,source_files_sha256=files,installed_site=str(site),numpy='1.24.4',core_path=str(core),core_sha256=sha(core),output_root=str(folder/'results'),routes=routes)
binding_path=folder/'binding.json';binding_path.write_text(json.dumps(binding))
go=dict(binding_sha256=sha(binding_path),probe_sha256=sha(probe),routes=list(routes),timed_batches=0,maximum_original_topcalls=4,maximum_native_topcalls=4,deadline_utc='2099-01-01T00:00:00+00:00')
go_path=folder/'go.json';go_path.write_text(json.dumps(go))
subprocess.check_output=lambda argv,**kwargs:'a'*40+'\n'if argv[-1]=='HEAD'else''
for key in('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ.pop('SKIMAGEXX_DISABLE_SIMD',None)
sys.argv=[probe,'--binding',str(binding_path),'--root-go',str(go_path),'--route',route_name]
runpy.run_path(probe,run_name='__main__')
assert workers==[1] and compat._fallback_enabled is True
'''


@pytest.mark.parametrize('route',list(PLAN['routes']))
@pytest.mark.parametrize('bad',['none','admission','calls','input'])
def test_actual_probe_inert_stubs_stop_before_sampler_body(tmp_path,route,bad):
    child=tmp_path/'child.py';child.write_text(STUB_CHILD)
    folder=tmp_path/'attempt'
    result=subprocess.run([sys.executable,str(child),str(ROOT/'tools/next_metrics_first_cohort_probe.py'),str(ROOT/'coverage/contracts/next-metrics-first-cohort-v1.json'),str(folder),route,bad],capture_output=True,text=True)
    terminal=folder/'results'/route/'terminal.json'
    if route in ('boundary','ellipse'):
        driver=folder/'snapshot'/PLAN['routes'][route]['driver']
        symbol=next(t for t in symtable.symtable(driver.read_text(),str(driver),'exec').get_children()if t.get_name()=='main').lookup('snapshot')
        assert symbol.is_global() if route=='boundary' else symbol.is_local() and symbol.is_imported()
    assert 'SAMPLER_BODY_FORBIDDEN'not in result.stderr and 'REAL_SCIENTIFIC_IMPORT_FORBIDDEN'not in result.stderr
    if bad=='none':
        assert result.returncode==0,result.stderr
        value=json.loads(terminal.read_bytes());assert value['status']=='PASS_REAL_CLI_STOP_BEFORE_BODY'
        assert value['counts']==dict(original=1,native=1)
        assert value['sampler_body_calls']==value['timed_batches']==0 and value['outer_state_restored']is True
    else:
        assert result.returncode!=0 and not terminal.exists()
        if bad!='admission':
            failure=json.loads((folder/'results'/route/'failure.json').read_bytes())
            assert failure['counts']==dict(original=1,native=1) and failure['type']=='AssertionError'


def actual_interface_source(route):
    source=Path(os.environ.get('NEXT_METRICS_PRODUCER_SOURCE',str(ROOT)))
    spec=PLAN['routes'][route]
    text=(source/spec['driver']).read_text()
    tree=ast.parse(text)
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='CONTRACT'for t in n.targets))
    assert isinstance(assignment.value,ast.BinOp)and isinstance(assignment.value.op,ast.Div)
    return source,text,assignment.value.right.value


@pytest.mark.parametrize('route',['ellipse','boundary'])
def test_actual_producer_contract_and_snapshot_scope(route):
    source,text,contract_path=actual_interface_source(route)
    spec=PLAN['routes'][route];family='ellipse_perimeter'if route=='ellipse'else'find_boundaries'
    assert contract_path==spec['contract']==CONTRACT['families'][family]['contract']
    assert hashlib.sha256((source/contract_path).read_bytes()).hexdigest()==CONTRACT['benchmark_source_sha256'][contract_path]
    assert hashlib.sha256((source/spec['driver']).read_bytes()).hexdigest()==CONTRACT['benchmark_source_sha256'][spec['driver']]
    symbol=next(t for t in symtable.symtable(text,spec['driver'],'exec').get_children()if t.get_name()=='main').lookup('snapshot')
    assert symbol.is_global() if route=='boundary'else symbol.is_local()and symbol.is_imported()


@pytest.mark.parametrize('route,old_path',[('ellipse','coverage/contracts/ellipse-perimeter-benchmark.json'),('boundary','coverage/contracts/find-boundaries-benchmark-v1.json')])
@pytest.mark.parametrize('historical_mutation',[False,True])
def test_actual_report_contract_hash_uses_producer_interface(tmp_path,monkeypatch,route,old_path,historical_mutation):
    source,_,actual=actual_interface_source(route)
    family='ellipse_perimeter'if route=='ellipse'else'find_boundaries'
    frozen=json.loads(json.dumps(CONTRACT))
    if historical_mutation:frozen['families'][family]['contract']=old_path
    for path in (actual,old_path):
        target=tmp_path/'snapshot'/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((source/path).read_bytes())
    # Execute the actual collector's contract selection/hash barrier only.
    loop=next(n for n in ast.parse(SCRIPTS[-1]).body if isinstance(n,ast.For))
    select=next(i for i,n in enumerate(loop.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='contract'for t in n.targets))
    snippet='\n'.join(ast.get_source_segment(SCRIPTS[-1],n)for n in loop.body[select:select+3])
    monkeypatch.chdir(tmp_path)
    context=dict(Path=Path,hashlib=hashlib,json=json,frozen=frozen,family=family,meta=dict(contract_sha256=hashlib.sha256((source/actual).read_bytes()).hexdigest()))
    if historical_mutation:
        with pytest.raises(AssertionError):exec(compile(snippet,'<actual-contract-collector>','exec'),context)
    else:
        exec(compile(snippet,'<actual-contract-collector>','exec'),context)
        assert len(context['specs'])==frozen['families'][family]['cases']


def test_boundary_local_snapshot_lookup_mutation_is_rejected(tmp_path):
    # The old probe must fail with the faithful module-global fixture.
    original=(ROOT/'tools/next_metrics_first_cohort_probe.py').read_text()
    mutant=original.replace("frame.f_back.f_globals['snapshot']", "caller['snapshot']")
    assert mutant!=original
    probe=tmp_path/'mutant_probe.py';probe.write_text(mutant)
    child=tmp_path/'child.py';child.write_text(STUB_CHILD);folder=tmp_path/'attempt'
    result=subprocess.run([sys.executable,str(child),str(probe),str(ROOT/'coverage/contracts/next-metrics-first-cohort-v1.json'),str(folder),'boundary','none'],capture_output=True,text=True)
    assert result.returncode!=0 and 'SAMPLER_BODY_FORBIDDEN'not in result.stderr
    failure=json.loads((folder/'results/boundary/failure.json').read_bytes())
    assert failure['type']=='KeyError'and 'snapshot'in failure['error']and failure['counts']==dict(original=1,native=1)
