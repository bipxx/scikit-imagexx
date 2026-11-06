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


def contract():return json.loads((ROOT/'coverage/contracts/find-boundaries-benchmark-binding-v1.json').read_bytes())


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
    c=contract();binding=c['predecessor_binding'];raw=(ROOT/binding['contract']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==binding['sha256']
    old=json.loads(raw)
    assert {k:v for k,v in c.items() if k not in ('source_closure','predecessor_binding')}=={k:v for k,v in old.items() if k!='source_closure'}
    assert set(c['source_closure'])==set(old['source_closure'])
    for name,pin in old['source_closure'].items():
        if name!='benchmarks/find_boundaries.py':assert c['source_closure'][name]==pin
    raw=(ROOT/'benchmarks/find_boundaries.py').read_bytes().replace(b'\r\n',b'\n')
    for fragment in binding['driver_inverse']:
        current,previous=fragment['current'].encode(),fragment['predecessor'].encode()
        assert current and raw.count(current)==1;raw=raw.replace(current,previous,1)
    assert hashlib.sha256(raw).hexdigest()==old['source_closure']['benchmarks/find_boundaries.py']


@pytest.mark.parametrize('fault',('none','cases','basepin','source','runtimepin','stager','GO','CMake'))
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
    if fault=='none':
        effective,actual,binding_sha=driver.load_root_binding(path,raw_contract)
        assert actual==binding and effective['state']=='FROZEN_READY_FOR_ROOT_FIRST_COHORT' and binding_sha==hashlib.sha256(raw_binding).hexdigest()
        changed={'state','shared_CMake_sha256','genuine_core_sha256','prepared_run','prepared_artifact'}
        assert {k:v for k,v in effective.items()if k not in changed}=={k:v for k,v in base.items()if k not in changed}
        assert raw_contract==driver.CONTRACT.read_bytes() and path.read_bytes()==raw_binding
    else:
        with pytest.raises(AssertionError):driver.load_root_binding(path,raw_contract)
