"""Inert authority and real-entry controls; no targets or timed bodies."""
import ast
import builtins
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import types
import pytest
ROOT=Path(__file__).resolve().parents[1]
CONTEXT=ROOT/'coverage/contracts/inverse-gaussian-gradient-benchmark-execution-v4.json'


def adapter():return runpy.run_path(str(ROOT/'benchmarks/inverse_gaussian_gradient_execution.py'))


@pytest.mark.parametrize('cached',(False,True))
def test_actual_unbound_CLI_before_any_science(cached):
    code="""import builtins,runpy,sys,types
old=builtins.__import__
def refuse(n,*a,**k):
 if n.split('.')[0]in('numpy','scipy','skimage','skimagexx'):raise AssertionError('SCIENCE_REACHED')
 return old(n,*a,**k)
if sys.argv[2]=='True':sys.modules['numpy']=types.ModuleType('numpy')
builtins.__import__=refuse;sys.argv=[sys.argv[1]];runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result=subprocess.run([sys.executable,'-I','-c',code,str(ROOT/'benchmarks/inverse_gaussian_gradient_native.py'),str(cached)],capture_output=True,text=True)
    assert result.returncode!=0 and 'IGG_UNBOUND'in result.stderr and 'SCIENCE_REACHED'not in result.stderr


def test_frozen_cases_method_and_exact_source():
    context=json.loads(CONTEXT.read_bytes());driver=runpy.run_path(str(ROOT/'benchmarks/inverse_gaussian_gradient_native.py'))
    assert context['cases']==driver['case_specs']()and len(context['cases'])==8
    assert all(driver['case_bounds'](c)for c in context['cases'])
    assert all(x is None for x in context['unbound'].values())
    assert context['source_method']==json.loads((ROOT/'coverage/contracts/inverse-gaussian-gradient-v1.json').read_bytes())['benchmark_method']
    for name,pin in context['source_files_canonical_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin,name
    for c in context['cases']:assert driver['native_requirements'](c)==dict(get_num_threads=1,inverse_gaussian_magnitude=1,inverse_gaussian_finish=4)


@pytest.mark.parametrize('kind',('absent','nonstopping','replaces-any-event','replaces-entry','admitted-stop'))
def test_reused_actual_entry_guard_never_runs_sampler(tmp_path,kind):
    module=adapter();context=json.loads(CONTEXT.read_bytes())
    # Read-only admitted common source is a declared future selective dependency
    # in this old private base. Production execute never accepts this override.
    dependency=Path(os.environ.get('IGG_BENCHMARK_BOUNDARY_ROOT',str(ROOT)))
    shared=module['boundary'](ROOT,dependency);common=shared.support(ROOT,context)
    measure=runpy.run_path(str(ROOT/'tools/montage_sustained.py'))['measure']
    oldprofile,oldtrace=sys.getprofile(),sys.gettrace();body=[];calls=[]
    class Stop(BaseException):pass
    def profile(frame,event,value):
        if kind=='replaces-any-event':sys.setprofile(lambda *a:None)
        if event=='call'and frame.f_code is measure.__code__:
            if kind=='replaces-entry':sys.setprofile(lambda *a:None)
            if kind=='admitted-stop':raise Stop('authorized-stop')
    def trace(frame,event,value):
        if frame.f_code is measure.__code__ and event=='line':body.append(frame.f_lineno)
        return trace
    cli=types.SimpleNamespace(first_cohort=True,output=tmp_path/'never.json')
    try:
        sys.settrace(trace);sys.setprofile(None if kind=='absent'else profile)
        with pytest.raises(Stop if kind=='admitted-stop'else AssertionError):
            shared.sample_entry(measure,lambda:calls.append('original'),lambda:calls.append('native'),cli,0,context['cases'][0],'2099-01-01T00:00:00+00:00',common)
    finally:sys.setprofile(oldprofile);sys.settrace(oldtrace)
    assert not body and not calls and not cli.output.with_suffix('.batches.jsonl').exists()


def test_no_new_sampler_or_timed_python_instrumentation():
    source=(ROOT/'benchmarks/inverse_gaussian_gradient_execution.py').read_text();tree=ast.parse(source)
    assert not any(isinstance(n,ast.FunctionDef)and n.name=='measure'for n in ast.walk(tree))
    assert source.index('validate_binding(')<source.index('import numpy as np')
    assert 'shared.sample_entry(measure,lambda:reference(*args,**options),lambda:native(*args,**options)'in source
    assert 'source_fingerprint'in source and 'sys.path[:]=oldpath'in source


def test_actual_frontend_identity_is_igg():
    source=(ROOT/'benchmarks/inverse_gaussian_gradient_execution.py').read_text()
    assert 'reference is upstream.inverse_gaussian_gradient and native is owned.inverse_gaussian_gradient'in source
    assert 'variation_of_information'not in source
