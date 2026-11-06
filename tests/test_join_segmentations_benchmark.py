"""Source/intake-only prospective benchmark controls, no scientific imports."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'benchmarks/join_segmentations.py'
spec=importlib.util.spec_from_file_location('join_benchmark',PATH);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_contract_bounds_and_sampler():
    contract=json.loads(module.CONTRACT.read_bytes())
    assert contract['state']=='PROSPECTIVE_UNBOUND_NO_MEASUREMENT'
    assert all(value is None for value in contract['unbound'].values())
    assert contract['method']['sampler']=='benchmarks/montage_sustained.py:measure'
    assert len(contract['cases'])==8 and len({s['id'] for s in contract['cases']})==8
    for row in contract['cases']:assert module.case_bounds(row)


@pytest.mark.parametrize('change',[{'dtype':'object'},{'left_count':100000},{'shape':[100000,100000]},{'right_scale':100000},{'layout':'unsafe'}])
def test_unsafe_intake_refused(change):
    row={**module.case_specs()[0],**change}
    with pytest.raises(AssertionError):module.case_bounds(row)


def test_real_cli_refuses_before_science():
    script="import runpy,sys\nclass Refuse:\n def find_spec(self,name,*args):\n  if name.split('.')[0] in ('numpy','scipy','skimage','skimagexx'):raise RuntimeError('SCIENTIFIC_IMPORT_REFUSED')\nsys.meta_path.insert(0,Refuse());path=sys.argv[1];sys.argv=[path];runpy.run_path(path,run_name='__main__')"
    completed=subprocess.run([sys.executable,'-I','-c',script,str(PATH)],capture_output=True,text=True)
    assert completed.returncode!=0 and 'PROSPECTIVE_UNBOUND' in completed.stderr
    assert 'Traceback' not in completed.stderr
