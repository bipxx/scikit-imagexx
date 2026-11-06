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
    projection=runpy.run_path(str(ROOT/'benchmarks/next_util_execution.py'))['historical_bytes']
    for name,pin in contract['source_pins'].items():
        raw=projection(ROOT,name)if name=='benchmarks/crop.py'else(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
        assert hashlib.sha256(raw).hexdigest()==pin,name
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


@pytest.mark.parametrize('fault',('none','none-cached','cases'))
def test_crop_bound_cli_metadata_stops_before_science(tmp_path,monkeypatch,fault):
    from next_util_benchmark_support import bound_request_refusal
    bound_request_refusal(tmp_path,monkeypatch,'crop',fault)
