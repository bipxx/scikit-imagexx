"""Synthetic binding/source controls only; never constructs a ready product."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
CONTEXT=ROOT/'coverage/contracts/next-util-benchmark-execution-v4.json'


def load_driver(route):
    description=json.loads(CONTEXT.read_bytes())['routes'][route]
    path=ROOT/description['driver'];spec=importlib.util.spec_from_file_location('test_next_util_'+route.replace('-','_'),path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def synthetic_binding(tmp_path,route,monkeypatch,fault=None):
    context=json.loads(CONTEXT.read_bytes());root=tmp_path/'source';root.mkdir();pins={}
    (root/CONTEXT.relative_to(ROOT)).parent.mkdir(parents=True,exist_ok=True);(root/CONTEXT.relative_to(ROOT)).write_bytes(CONTEXT.read_bytes())
    for name in set(context['sources'])|set(context['production_source_canonical_sha256'])|{'CMakeLists.txt'}:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes());pins[name]=hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    donor=Path(os.environ.get('NEXT_UTIL_F1_SOURCE_ROOT',str(ROOT)))
    for name,pin in context['FENV_dependency']['source_files_raw_sha256'].items():
        payload=(donor/name).read_bytes();assert hashlib.sha256(payload).hexdigest()==pin,name
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(payload)
    H=lambda b:hashlib.sha256(b).hexdigest();revision='a'*40
    def receipt(name,value):
        p=tmp_path/(name+'.json');p.write_text(json.dumps(value));return str(p),H(p.read_bytes())
    prepared=dict(revision=revision,files_sha256=pins)
    runtime_path,runtime_sha256=receipt('runtime',dict(source=prepared,runtime=dict(packages_sha256=dict(skimagexx={'_core.so':'b'*64}))))
    admission_path,admission_sha=receipt('root-admission',dict(root_accepted=True,source=revision,run=123,product_kind='GENUINE_INSTALLED_WHOLE_PRODUCT'))
    stager_path,stager_sha=receipt('stager',dict(run=123,revision=revision,artifact=456,artifact_sha256='c'*64,status='STAGED_UNACCEPTED'))
    go_path=tmp_path/'go.json'
    binding=dict(status='ROOT_AUTHORIZED_GENUINE_FIRST_COHORTS',source_root=str(root),source_revision=revision,git_argv=['git'if sys.platform=='win32'else'/mnt/c/Program Files/Git/cmd/git.exe','-C',str(root).replace('/mnt/c/','C:/',1)],product_kind='GENUINE_INSTALLED_WHOLE_PRODUCT',interpreter=sys.executable,installed_site=str(tmp_path/'installed'),numpy='1.24.4',core_path=str(tmp_path/'installed/_core.so'),core_sha256='b'*64,prepared_source=prepared,producer=dict(run=123,revision=revision,artifact=456,sha256='c'*64),runtime_path=runtime_path,runtime_sha256=runtime_sha256,shared_CMake_sha256=pins['CMakeLists.txt'],genuine_root_admission_path=admission_path,genuine_root_admission_sha256=admission_sha,genuine_root_stager_path=stager_path,genuine_root_stager_sha256=stager_sha,base_contract_sha256=H(CONTEXT.read_bytes()),root_go_path=str(go_path))
    if fault=='cases':binding['cases']=[]
    elif fault=='core':binding['core_sha256']='d'*64
    elif fault=='revision':binding['source_revision']='d'*40
    elif fault=='runtime':binding['runtime_sha256']='d'*64
    elif fault=='root':binding['genuine_root_admission_sha256']='d'*64
    elif fault=='stager':binding['producer']['artifact']=789
    elif fault=='source':(root/'src/crop.cpp').write_text('changed')
    elif fault=='git-root':binding['git_argv'][-1]=str(tmp_path/'wrong')
    elif fault=='context':binding['base_contract_sha256']='d'*64
    path=tmp_path/'binding.json';path.write_text(json.dumps(binding))
    from datetime import datetime,timedelta,timezone
    go=dict(binding_sha256=H(path.read_bytes()),routes=[route],timed_batches=0,deadline_utc=(datetime.now(timezone.utc)+timedelta(minutes=5)).isoformat())
    if fault=='route':go['routes']=[]
    elif fault=='expired':go['deadline_utc']='2000-01-01T00:00:00+00:00'
    elif fault=='go':go['binding_sha256']='d'*64
    go_path.write_text(json.dumps(go))
    monkeypatch.setattr(subprocess,'check_output',lambda argv,**kw:revision+'\n'if argv[-1]=='HEAD'else'')
    return path


def bound_request_refusal(tmp_path,monkeypatch,route,fault='none',target='numpy',api=None):
    import builtins,types,pytest
    driver=load_driver(route);binding=synthetic_binding(tmp_path,route,monkeypatch,None if fault.startswith('none')else fault)
    output=tmp_path/'never.json';monkeypatch.setattr(sys,'path',list(sys.path))
    monkeypatch.setattr(sys,'argv',[str(driver.__file__),'--first-cohort','--root-binding',str(binding),'--output',str(output)])
    if api is not None:sys.argv+=['--api',api]
    old=builtins.__import__;attempts=[]
    def refuse(name,*args,**kwargs):
        if name.split('.')[0]in((target,)if target=='numpy'else('skimage','skimagexx')):
            attempts.append(name);raise RuntimeError('NEXT_UTIL_TARGET_REQUEST_REFUSED')
        return old(name,*args,**kwargs)
    with monkeypatch.context()as imports:
        if fault=='none-cached':imports.setitem(sys.modules,target,types.ModuleType(target))
        else:imports.delitem(sys.modules,target,raising=False)
        imports.setattr(builtins,'__import__',refuse)
        if fault.startswith('none'):
            with pytest.raises(RuntimeError,match='NEXT_UTIL_TARGET_REQUEST_REFUSED'):driver.main()
            assert attempts==[target]
        else:
            with pytest.raises(AssertionError):driver.main()
            assert not attempts
    assert builtins.__import__ is old
    assert not output.exists()and not output.with_suffix('.batches.jsonl').exists()
