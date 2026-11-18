"""Pure current closure and saved data checks, never scientific execution."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/util-view-operations-v1.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def test_current_native_closure_and_historical_pins():
    data=json.loads(CONTRACT.read_bytes())
    assert data['original_budget']==data['timed_batches']==0
    assert data['raw_admission_sha256']=='63bb915f2a1e5b395594347560a2e2177470a6d5ce783b3e7e3d2e140d590678'
    context=json.loads((ROOT/'coverage/contracts/next-util-benchmark-execution-v3.json').read_bytes())
    projection=runpy.run_path(str(ROOT/'benchmarks/next_util_execution.py'))['historical_bytes']
    for name,pin in data['source_pins_raw_sha256'].items():
        raw=projection(ROOT,name)if name in context['sources']else(ROOT/name).read_bytes()
        if name=='tests/test_util_views_public.py'and hashlib.sha256(raw).hexdigest()!=pin:
            # Exact admitted F1 FP inverse; an arbitrary changed test cannot pass.
            fenv=runpy.run_path(str(ROOT/'benchmarks/next_util_execution.py'))['fenv_source'](ROOT,context)
            raw=fenv.project_public_source(raw)
        assert hashlib.sha256(raw).hexdigest()==pin,name
    for name,pin in data['historical_contracts'].items():assert sha(ROOT/name)==pin,name
    assert data['SDK']['complete_operations']==['view_as_blocks','view_as_windows','slice_along_axes']
    assert data['worker_disposition']=='serial metadata and existing serial POD transport; no new globals; preserve process-global default1 budget'
    leaf=(ROOT/'src/util_views.cpp').read_text()
    for name in data['SDK']['complete_operations']:assert name+'('in leaf
    assert 'crop_view(a,slices)'in leaf and 'crop_copy_into(a,slices,'in leaf
    assert '#pragma omp'not in leaf and 'std::atomic'not in leaf and 'skimage.'not in leaf
    assert 'reference_fenv_restore'in(ROOT/'tests/test_util_views_public.py').read_text()


def test_all_saved_rows_exact_recipe_source_and_profile_transition():
    data=json.loads((ROOT/'tests/fixtures/util_views_originals_v1.json').read_bytes())
    from array_views_contract import cases as arrays
    from slice_along_axes_contract import cases as slices
    count=0
    for family,cases in (('slice',slices()),('array_views',arrays())):
        for profile in ('minimum','modern'):
            cell=data['families'][family][profile];assert [r['id']for r in cell['rows']]==[s['id']for s in cases]
            assert len(cell['raw']['sha256'])==64 and cell['raw']['bytes']>0
            for row in cell['rows']:
                assert all(row['checks'].values())
                assert row['caller_restore_fenv']['saved']==row['caller_restore_fenv']['restored']
            count+=len(cases)
    assert count==548
    minimum={r['id']:r for r in data['families']['slice']['minimum']['rows']}
    modern={r['id']:r for r in data['families']['slice']['modern']['rows']}
    row='slice_along_axes/empty-array-slices'
    assert minimum[row]['error']is None and minimum[row]['warnings'][0]['type']=='DeprecationWarning'
    assert modern[row]['error']['type']=='ValueError'


def test_observers_body_unchanged_from_captured_dependency():
    # Source-only same-interpreter AST comparison; locate historical functions
    # from immutable text snapshots, never frozen interpreter-specific dumps.
    for name in ('slice_along_axes_observer','array_views_observer'):
        before=ast.parse((ROOT/('tests/fixtures/'+name+'_captured_v1.py.txt')).read_bytes())
        after=ast.parse((ROOT/('tests/'+name+'.py')).read_bytes())
        bodies=lambda tree:{n.name:ast.dump(n,include_attributes=False)for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        assert bodies(before)==bodies(after)


@pytest.mark.parametrize('name',('slice_along_axes','array_views'))
def test_benchmark_prospective_cases_and_no_science_before_authority(name):
    module=runpy.run_path(str(ROOT/'benchmarks'/ (name+'.py')))
    rows=module['case_specs']();assert len(rows)==8
    for row in rows:assert module['case_bounds'](row)['threads']==1
    code="""import builtins,runpy,sys
old=builtins.__import__;attempts=[]
def refused(name,*a,**k):
 if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):attempts.append(name);raise AssertionError('SCIENCE_IMPORT_REFUSED')
 return old(name,*a,**k)
builtins.__import__=refused;sys.argv=[sys.argv[1]]
try:runpy.run_path(sys.argv[0],run_name='__main__')
except SystemExit as e:assert e.code and not attempts
else:raise AssertionError('unbound benchmark accepted')
"""
    p=subprocess.run([sys.executable,'-I','-c',code,str(ROOT/'benchmarks'/(name+'.py'))],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
