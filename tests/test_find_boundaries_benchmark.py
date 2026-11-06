"""Prospective benchmark source/intake guards; never run an algorithm."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]


def contract():return json.loads((ROOT/'coverage/contracts/find-boundaries-benchmark-binding-v4.json').read_bytes())


def test_find_boundaries_benchmark_source_pins_and_sampler():
    c=contract()
    for name,pin in c['source_closure'].items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin,name
    text=(ROOT/'tools/montage_sustained.py').read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='measure')
    method=''.join(text.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    assert hashlib.sha256(method.encode()).hexdigest()==c['method_sha256']
    assert c['sampler']==dict(module='tools/montage_sustained.py',function='measure',threads=1,repeat=7,target_ms=100,warmup_min_ms=250,raw_samples=112)
    assert c['state']=='PROSPECTIVE_GENUINE_PRODUCT_AND_ROOT_BINDING_REQUIRED'
    assert all(c[key] is None for key in ('shared_CMake_sha256','genuine_core_sha256','prepared_run','prepared_artifact'))


def test_find_boundaries_benchmark_bounded_case_intake():
    module=runpy.run_path(str(ROOT/'benchmarks/find_boundaries.py'))
    cases=module['case_specs']()
    assert len(cases)==len({s['id']for s in cases})==8
    assert {s['mode']for s in cases}=={'thick','inner','outer','subpixel'}
    assert {s['layout']for s in cases}=={'C','F','reverse'}
    assert {len(s['shape'])for s in cases}=={1,2,3}
    assert {s['dtype']for s in cases}=={'uint8','uint16','int32','bool'}
    assert all(module['case_bounds'](s) for s in cases)


def test_find_boundaries_benchmark_real_cli_stops_before_target_imports(tmp_path):
    output=tmp_path/'never.json'
    code="""import importlib.abc,runpy,sys
class Refuse(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in('numpy','scipy','skimage','skimagexx'):raise RuntimeError('FORBIDDEN_SCIENTIFIC_IMPORT')
sys.meta_path.insert(0,Refuse())
sys.argv=[sys.argv[1],'--output',sys.argv[2]]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result=subprocess.run([sys.executable,'-c',code,str(ROOT/'benchmarks/find_boundaries.py'),str(output)],capture_output=True,text=True)
    assert result.returncode!=0
    assert 'PROSPECTIVE: genuine input/root binding missing' in result.stderr
    assert 'FORBIDDEN_SCIENTIFIC_IMPORT' not in result.stderr
    assert not output.exists() and not output.with_suffix('.batches.jsonl').exists()


def test_boundary_external_binding_preserves_original_driver_and_contract():
    c=contract();layout=c['layout_successor'];raw=(ROOT/layout['contract']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==layout['sha256']
    previous=json.loads(raw)
    driver=runpy.run_path(str(ROOT/'benchmarks/find_boundaries.py'))
    driver['validate_source_closure'](c)
    # Exact source inversion preserves all historical driver bodies, including
    # cases, bounds, sampler, timed loops and native acceptance predicates.
    raw=(ROOT/'benchmarks/find_boundaries.py').read_bytes().replace(b'\r\n',b'\n')
    chain=[layout['driver_inverse'],previous['runtime_successor']['driver_inverse']]
    v2=json.loads((ROOT/previous['runtime_successor']['contract']).read_bytes())
    chain.extend([v2['runtime_successor']['driver_inverse'],previous['predecessor_binding']['driver_inverse']])
    old=json.loads((ROOT/previous['predecessor_binding']['contract']).read_bytes())
    for fragments in chain:
        for fragment in fragments:
            current,predecessor=fragment['current'].encode(),fragment['predecessor'].encode()
            assert current and raw.count(current)==1;raw=raw.replace(current,predecessor,1)
    assert hashlib.sha256(raw).hexdigest()==old['source_closure']['benchmarks/find_boundaries.py']
    for key in ('cases','sampler','method_sha256','first_cohort_gate','excluded'):
        assert c[key]==previous[key]==old[key]


@pytest.mark.parametrize('fault',('none','none-cached','cases','basepin','source','runtimepin','stager','GO','CMake','bound-cases','bound-method'))
def test_boundary_external_root_envelope_only_metadata(tmp_path,monkeypatch,fault):
    import importlib.util
    from datetime import datetime,timedelta,timezone
    spec=importlib.util.spec_from_file_location('boundary_binding_only',ROOT/'benchmarks/find_boundaries.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    raw_contract=driver.CONTRACT.read_bytes();base=json.loads(raw_contract)
    monkeypatch.setattr(driver,'ROOT',tmp_path)
    revision='a'*40;core='b'*64;cmake=b'canonical synthetic source\n';(tmp_path/'CMakeLists.txt').write_bytes(cmake);cmake_sha=hashlib.sha256(cmake).hexdigest()
    prepared=dict(revision=revision,files_sha256={'CMakeLists.txt':cmake_sha})
    binding=dict(status='ROOT_AUTHORIZED_GENUINE_FIRST_COHORTS',product_kind='GENUINE_INSTALLED_WHOLE_PRODUCT',source_root=str(tmp_path),source_revision=revision,prepared_source=prepared,producer=dict(run=1,revision=revision,artifact=2,sha256='c'*64),core_path=str(tmp_path/'_core.so'),core_sha256=core,shared_CMake_sha256=cmake_sha,base_contract_sha256=hashlib.sha256(raw_contract).hexdigest())
    records={'runtime':dict(source=prepared,runtime=dict(packages_sha256={'skimagexx':{'_core.so':core}})),'genuine_root_admission':dict(status='synthetic-only'),'genuine_root_stager':dict(run=1,revision=revision,artifact=2,artifact_sha256='c'*64)}
    for name,value in records.items():
        path=tmp_path/(name+'.json');raw=json.dumps(value).encode();path.write_bytes(raw);binding[name+'_path']=str(path);binding[name+'_sha256']=hashlib.sha256(raw).hexdigest()
    path=tmp_path/'binding.json';go_path=tmp_path/'GO.json';binding['root_go_path']=str(go_path)
    if fault=='cases':binding['cases']=[]
    elif fault=='basepin':binding['base_contract_sha256']='d'*64
    elif fault=='source':binding['source_revision']='d'*40
    elif fault=='runtimepin':binding['runtime_sha256']='d'*64
    elif fault=='stager':binding['producer']['artifact']=9
    elif fault=='CMake':binding['shared_CMake_sha256']='d'*64
    raw_binding=json.dumps(binding).encode();path.write_bytes(raw_binding)
    go=dict(binding_sha256=hashlib.sha256(raw_binding).hexdigest(),routes=['boundary'],timed_batches=0,deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
    if fault=='GO':go['binding_sha256']='d'*64
    go_path.write_text(json.dumps(go))
    monkeypatch.setattr(driver.subprocess,'check_output',lambda argv,**kw:revision+'\n'if argv[-1]=='HEAD'else'')
    if fault in ('none','none-cached','bound-cases','bound-method'):
        effective,actual,binding_sha=driver.load_root_binding(path,raw_contract)
        assert actual==binding and effective['state']=='FROZEN_READY_FOR_ROOT_FIRST_COHORT' and binding_sha==hashlib.sha256(raw_binding).hexdigest()
        changed={'state','shared_CMake_sha256','genuine_core_sha256','prepared_run','prepared_artifact'}
        assert {k:v for k,v in effective.items()if k not in changed}=={k:v for k,v in base.items()if k not in changed}
        assert raw_contract==driver.CONTRACT.read_bytes() and path.read_bytes()==raw_binding
        # Use real immutable closure bytes with the existing inert provenance envelope.
        for name in (*base['source_closure'],base['runtime_successor']['contract'],base['layout_successor']['contract']):
            target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/name).read_bytes())
        if fault=='bound-cases':effective['cases']=[]
        elif fault=='bound-method':effective['method_sha256']='0'*64
        if fault in ('bound-cases','bound-method'):
            with pytest.raises(AssertionError):driver.validate_source_closure(effective)
            return
        driver.validate_source_closure(effective)
        # Exercise actual main through binding + source validation; stop at the
        # first scientific import, before any target or sampler body can run.
        import builtins,types
        attempts=[]
        original_import=builtins.__import__
        def refuse_science(name,*args,**kwargs):
            if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):
                attempts.append(name);raise RuntimeError('BOUND_CLI_SCIENCE_REFUSED')
            return original_import(name,*args,**kwargs)
        monkeypatch.setattr(sys,'path',list(sys.path))
        output=tmp_path/'never-bound-output.json'
        monkeypatch.setattr(sys,'argv',[str(ROOT/'benchmarks/find_boundaries.py'),'--root-binding',str(path),'--output',str(output)])
        # Intercept the import request itself: a meta-path finder misses cached
        # modules in the scientific qualification process. Restore live modules.
        with monkeypatch.context() as import_boundary:
            if fault=='none-cached':import_boundary.setitem(sys.modules,'numpy',types.ModuleType('numpy'))
            import_boundary.setattr(builtins,'__import__',refuse_science)
            with pytest.raises(RuntimeError,match='BOUND_CLI_SCIENCE_REFUSED'):driver.main()
        assert attempts==['numpy'] and not output.exists() and not output.with_suffix('.batches.jsonl').exists()
    else:
        with pytest.raises(AssertionError):driver.load_root_binding(path,raw_contract)


@pytest.mark.parametrize('fault',('cases','method','public-pin','driver-inverse'))
def test_boundary_portability_successor_rejects_changed_context(fault):
    import copy
    driver=runpy.run_path(str(ROOT/'benchmarks/find_boundaries.py'))
    value=copy.deepcopy(contract())
    if fault=='cases':value['cases']=[]
    elif fault=='method':value['method_sha256']='0'*64
    elif fault=='public-pin':value['source_closure']['tests/test_find_boundaries_public.py']='0'*64
    else:value['layout_successor']['driver_inverse'][0]['current']='absent active driver path'
    with pytest.raises(AssertionError):driver['validate_source_closure'](value)


def test_boundary_successor_contracts_are_required_qualification_sources():
    qualifier=runpy.run_path(str(ROOT/'tools/public_qualification.py'))
    for name in ('coverage/contracts/find-boundaries-frontend-v1.json',
                 'coverage/contracts/find-boundaries-frontend-v2.json',
                 'coverage/contracts/find-boundaries-benchmark-binding-v1.json',
                 'coverage/contracts/find-boundaries-benchmark-binding-v2.json',
                 'coverage/contracts/find-boundaries-benchmark-binding-v3.json',
                 'coverage/contracts/find-boundaries-benchmark-binding-v4.json',
                 'coverage/contracts/find-boundaries-frontend-v3.json',
                 'coverage/contracts/find-boundaries-sdk-v2.json',
                 'coverage/contracts/find-boundaries-layout-source-context-v1.json'):
        assert qualifier['DRAW_SOURCE_PATHS'].count(name)==1
