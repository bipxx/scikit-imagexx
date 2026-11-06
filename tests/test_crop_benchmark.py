"""Source-only crop prospective performance guards; no algorithms."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]


def test_crop_benchmark_frozen_cases_sampler_and_source():
    contract=json.loads((ROOT/'coverage/contracts/crop-benchmark-v1.json').read_bytes())
    assert contract['state']=='PROSPECTIVE_GENUINE_PRODUCT_AND_ROOT_BINDING_REQUIRED'
    assert contract['retain_all_losses']is True
    assert contract['sampler']==dict(module='tools/montage_sustained.py',function='measure',threads=1,repeat=7,target_ms=100,warmup_min_ms=250,raw_samples=112,warmup_groups=16)
    for name,pin in contract['source_pins'].items():assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin,name
    text=(ROOT/'tools/montage_sustained.py').read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='measure')
    assert hashlib.sha256(''.join(text.splitlines(True)[node.lineno-1:node.end_lineno]).encode()).hexdigest()==contract['method_sha256']


def test_crop_benchmark_bounds_without_arrays():
    module=runpy.run_path(str(ROOT/'benchmarks/crop.py'));cases=module['case_specs']()
    assert len(cases)==len({s['id']for s in cases})==8
    assert {s['copy']for s in cases}=={False,True}and {s['layout']for s in cases}=={'C','F','reverse'}
    assert all(module['source_safe'](spec)['input_elements']<=263000 for spec in cases)
    assert sum(s['payload_provider']for s in cases)==1


def test_crop_real_cli_refuses_before_target_imports(tmp_path):
    output=tmp_path/'never.json'
    code="""import importlib.abc,runpy,sys
class Refuse(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in('numpy','scipy','skimage','skimagexx'):raise RuntimeError('SCIENCE_FORBIDDEN')
sys.meta_path.insert(0,Refuse());sys.argv=[sys.argv[1],'--output',sys.argv[2]];runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result=subprocess.run([sys.executable,'-c',code,str(ROOT/'benchmarks/crop.py'),str(output)],capture_output=True,text=True)
    assert result.returncode!=0 and 'PROSPECTIVE: genuine product/root measurement binding required'in result.stderr
    assert 'SCIENCE_FORBIDDEN'not in result.stderr and not output.exists()


def test_crop_native_boundary_has_explicit_provider_and_serial_disposition():
    frontend=(ROOT/'python/skimagexx/_crop.py').read_text()
    assert frontend.index('ar = np.array(ar, copy=False)')<frontend.index('if isinstance(crop_width, Integral)')
    assert '_core.crop_view(ar, slices)'in frontend and '_core.crop_copy_into(cropped, output)'in frontend
    assert 'np.empty_like(cropped, order=order, subok=False)'in frontend and 'np.array(cropped, order=order, copy=True)'in frontend
    assert not any((n.module or'').startswith('skimage')for n in ast.walk(ast.parse(frontend))if isinstance(n,ast.ImportFrom))
    native=(ROOT/'src/crop.cpp').read_text()
    assert 'std::memcpy'in native and 'omp 'not in native and 'avx'not in native
    assert 'CropPlan crop_view'in native and 'void crop_copy_into'in native


@pytest.mark.parametrize('fault',('none','cases'))
def test_crop_bound_cli_metadata_stops_before_science(tmp_path,monkeypatch,fault):
    import importlib.util
    import importlib.abc
    from datetime import datetime,timedelta,timezone
    spec=importlib.util.spec_from_file_location('crop_cli_metadata',ROOT/'benchmarks/crop.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    raw=driver.CONTRACT.read_bytes();contract=json.loads(raw)
    for name in contract['source_pins']:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    path=tmp_path/'coverage/contracts/crop-benchmark-v1.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    revision='a'*40;admission=tmp_path/'root.json';admission.write_text(json.dumps(dict(source=revision,product_kind='GENUINE_INSTALLED_WHOLE_PRODUCT')))
    binding=dict(status='ROOT_AUTHORIZED_CROP_PUBLIC_MEASUREMENT',source_revision=revision,core_sha256='b'*64,prepared_run=1,prepared_artifact=2,contract_sha256=hashlib.sha256(raw).hexdigest(),root_admission_path=str(admission),root_admission_sha256=hashlib.sha256(admission.read_bytes()).hexdigest(),deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
    if fault=='cases':binding['cases']=[]
    bound=tmp_path/'binding.json';bound.write_text(json.dumps(binding));output=tmp_path/'never.json'
    monkeypatch.setattr(driver,'ROOT',tmp_path);monkeypatch.setattr(driver,'CONTRACT',path)
    monkeypatch.setattr(driver.subprocess,'check_output',lambda argv,**kw:revision+'\n'if argv[-1]=='HEAD'else'')
    attempts=[]
    class Refuse(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0]in('numpy','scipy','skimage','skimagexx'):
                attempts.append(fullname);raise RuntimeError('CROP_BOUND_SCIENCE_REFUSED')
    monkeypatch.setattr(sys,'meta_path',[Refuse(),*sys.meta_path]);monkeypatch.setattr(sys,'path',list(sys.path))
    monkeypatch.setattr(sys,'argv',[str(ROOT/'benchmarks/crop.py'),'--root-binding',str(bound),'--output',str(output)])
    if fault=='none':
        with pytest.raises(RuntimeError,match='CROP_BOUND_SCIENCE_REFUSED'):driver.main()
        assert attempts==['numpy']
    else:
        with pytest.raises(AssertionError):driver.main()
        assert not attempts
    assert not output.exists()and not output.with_suffix('.batches.jsonl').exists()
