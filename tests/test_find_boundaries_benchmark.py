"""Prospective benchmark source/intake guards; never run an algorithm."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def contract():return json.loads((ROOT/'coverage/contracts/find-boundaries-benchmark-v1.json').read_bytes())


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
