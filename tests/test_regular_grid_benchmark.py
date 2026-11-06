"""Prospective source/intake only; scientific imports refused at real CLI."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'benchmarks/regular_grid.py'
spec=importlib.util.spec_from_file_location('regular_grid_benchmark',PATH);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_actual_sampler_interface_and_intake():
    contract=json.loads(module.CONTRACT.read_bytes())
    assert contract['state']=='PROSPECTIVE_UNBOUND_NO_MEASUREMENT'
    assert all(value is None for value in contract['unbound'].values())
    path,symbol=contract['method']['sampler'].split(':');assert (ROOT/path).is_file()
    function=next(n for n in ast.parse((ROOT/path).read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name==symbol)
    assert [a.arg for a in function.args.args]==['reference','native','repeat','target_ms']
    assert [a.arg for a in function.args.kwonlyargs]==['journal','case_index','case_id','pass_index','deadline']
    assert contract['method']['threads']==1 and len(contract['cases'])==8
    assert len({row['id'] for row in contract['cases']})==8
    for row in contract['cases']:assert module.case_bounds(row)['image_elements_allocated']==0


@pytest.mark.parametrize('change',[{'shape':[]},{'shape':[2**54]},{'shape':[0,7]},{'points':0},{'carrier':'callback'}])
def test_unsafe_intake_refused(change):
    with pytest.raises(AssertionError):module.case_bounds({**module.case_specs()[0],**change})


def test_real_cli_refuses_before_science():
    script="import runpy,sys\nclass Refuse:\n def find_spec(self,name,*args):\n  if name.split('.')[0] in ('numpy','scipy','skimage','skimagexx'):raise RuntimeError('SCIENTIFIC_IMPORT_REFUSED')\nsys.meta_path.insert(0,Refuse());path=sys.argv[1];sys.argv=[path];runpy.run_path(path,run_name='__main__')"
    result=subprocess.run([sys.executable,'-I','-c',script,str(PATH)],capture_output=True,text=True)
    assert result.returncode!=0 and 'PROSPECTIVE_UNBOUND' in result.stderr and 'Traceback' not in result.stderr
