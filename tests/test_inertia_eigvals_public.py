"""Bounded original-first inertia eigenvalues; no production implementation.

Finite real solver rows require an actual private inertia_eigvals C_return;
complex/nonfinite and validation rows explicitly carry zero solver credit.
"""
import argparse
import inspect
import json
from pathlib import Path
import sys
import pytest
if __package__:
    from .inertia_eigvals_cases import API,STAGE,UNASSESSED,cases,fixture,error_limits
    from .test_moments_public import original_runtime
    from .test_moments_derived_public import observe,result_record
else:
    from inertia_eigvals_cases import API,STAGE,UNASSESSED,cases,fixture,error_limits
    from test_moments_public import original_runtime
    from test_moments_derived_public import observe,result_record
CASES=cases()

def record(value):
    if value is None:return None
    return dict(type=type(value).__name__,items=[dict(module=type(v).__module__,type=type(v).__name__,dtype=v.dtype.str,raw=v.tobytes().hex())for v in value])

def reference_case(spec,np,original):
    image,options=fixture(spec,np);captured={};prior=sys.getprofile()
    def capture(frame,event,arg):
        if event=='return'and frame.f_code is original.inertia_tensor.__code__:captured['mu']=frame.f_locals.get('mu')
        if event=='return'and frame.f_code is original.inertia_tensor_eigvals.__code__:captured['T']=frame.f_locals.get('T')
    sys.setprofile(capture)
    try:observed=observe(original.inertia_tensor_eigvals,image,options,{},np)
    finally:sys.setprofile(prior)
    return image,options,captured,observed

def validate(spec,image,captured,observed,np):
    expected,error,messages,checks=observed
    assert all(checks.values())
    if spec['route']=='error':assert error is not None;return None
    if error is not None:
        assert spec['route']=='provider';return None
    assert type(expected)is list and len(expected)==spec['n']
    assert all(isinstance(v,np.floating)for v in expected)
    finite=np.asarray(expected)
    assert all(not np.isfinite(v)or v>=0 for v in finite)
    if np.all(np.isfinite(finite)):assert all(finite[i]>=finite[i+1]for i in range(len(finite)-1))
    with np.errstate(all='ignore'):return error_limits(spec,image,captured.get('mu'),captured['T'],expected,np)

def compare(actual,expected,bound,np):
    assert type(actual)is type(expected)is list and len(actual)==len(expected)
    for a,e in zip(actual,expected):assert type(a)is type(e)and a.dtype==e.dtype
    a,e=np.asarray(actual),np.asarray(expected)
    for mask in (np.isnan,np.isposinf,np.isneginf):np.testing.assert_array_equal(mask(a),mask(e))
    finite=np.isfinite(e);assert np.all(np.abs(a[finite].astype(np.float64)-e[finite])<=bound[finite])
    exact=finite&(bound==0);np.testing.assert_array_equal(a[exact],e[exact]);np.testing.assert_array_equal(np.signbit(a[exact]),np.signbit(e[exact]))
    assert all(not np.isfinite(v)or v>=0 for v in a)
    if np.all(finite):assert all(a[i]>=a[i+1]for i in range(len(a)-1))

@pytest.fixture(scope='module')
def reference_runtime():return original_runtime()
@pytest.fixture(scope='module')
def worker_results():return {}

def test_inertia_eigenvalues_signature(reference_runtime):
    _,original,_=reference_runtime
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    assert API in native_functions()
    assert inspect.signature(xx.measure.inertia_tensor_eigvals)==inspect.signature(original.inertia_tensor_eigvals)

@pytest.mark.parametrize('spec',CASES,ids=[r['id']for r in CASES])
@pytest.mark.parametrize('workers',(1,4))
def test_inertia_eigenvalues_owned(reference_runtime,spec,workers,worker_results,monkeypatch,tmp_path):
    np,original,_=reference_runtime
    image,options,captured,observed=reference_case(spec,np,original)
    expected,error,messages,checks=observed
    (tmp_path/'original.json').write_text(json.dumps(dict(id=spec['id'],result=record(expected),error=error,warnings=messages,checks=checks,T=result_record(captured.get('T')),mu=result_record(captured.get('mu'))))+'\n')
    bound=validate(spec,image,captured,observed,np)
    image,options=fixture(spec,np)
    import skimage.measure as upstream
    import skimagexx as xx
    from skimagexx import _compat
    from skimagexx.coverage import native_functions
    assert API in native_functions()and xx.get_num_threads()==1
    previous=xx.set_fallback_enabled(False);prior=sys.getprofile();completions=[];forbidden=[]
    def deny(*args,**kwargs):forbidden.append(True);raise AssertionError('upstream inertia/eigenvalue forwarding')
    def trace(frame,event,arg):
        if event=='c_return'and getattr(arg,'__module__',None)==xx._core.__name__ and getattr(arg,'__name__',None)in(STAGE,'inertia_tensor','moments_project'):completions.append(arg.__name__)
    try:
        xx.set_num_threads(workers)
        with monkeypatch.context()as patch:
            for name in ('inertia_tensor_eigvals','inertia_tensor','moments','moments_central'):
                patch.setattr(original,name,deny);patch.setattr(upstream,name,deny)
            if spec['route']=='solver':
                for name in ('eigvalsh','eigh','eig','eigvals'):patch.setattr(np.linalg,name,deny)
            sys.setprofile(trace)
            try:actual,actual_error,actual_messages,actual_checks=observe(xx.measure.inertia_tensor_eigvals,image,options,{},np)
            finally:sys.setprofile(prior)
        (tmp_path/'candidate.json').write_text(json.dumps(dict(id=spec['id'],workers=workers,result=record(actual),error=actual_error,warnings=actual_messages,checks=actual_checks,completions=completions,solver_credit=completions.count(STAGE)if spec['route']=='solver'else 0,forbidden=forbidden))+'\n')
        assert not forbidden and all(actual_checks.values())and actual_error==error and actual_messages==messages
        if error is None:
            compare(actual,expected,bound,np)
            if spec['route']=='solver':assert completions.count(STAGE)==1
            assert completions.count('moments_project')==(spec['n']if spec['kind']=='auto'else 0)
        words=(record(actual),actual_error,actual_messages)
        assert worker_results.setdefault(spec['id'],words)==words
        assert xx.get_num_threads()==workers and _compat._fallback_enabled is False
    finally:sys.setprofile(prior);xx.set_num_threads(1);xx.set_fallback_enabled(previous)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reference-only',action='store_true');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert args.reference_only and sys.flags.optimize==0
    assert not any(n=='skimagexx'or n.startswith('skimagexx.')for n in sys.modules)
    args.output.mkdir();np,original,runtime=original_runtime();(args.output/'runtime.json').write_text(json.dumps(runtime,indent=2))
    rows=[]
    with (args.output/'journal.jsonl').open('x')as journal:
        for spec in CASES:
            image,options,captured,observed=reference_case(spec,np,original);result,error,messages,checks=observed
            row=dict(id=spec['id'],spec=spec,input=result_record(image),T=result_record(captured.get('T')),mu=result_record(captured.get('mu')),result=record(result),error=error,warnings=messages,checks=checks)
            journal.write(json.dumps(row)+'\n');journal.flush()
            bound=validate(spec,image,captured,observed,np)
            if error is None:compare(result,result,bound,np)
            rows.append(dict(id=spec['id'],error=error,bound=None if bound is None else bound.tolist()))
    assert len(rows)==34 and not any(n=='skimagexx'or n.startswith('skimagexx.')for n in sys.modules)
    (args.output/'summary.json').write_text(json.dumps(dict(cases=len(rows),returns=sum(r['error']is None for r in rows),errors=sum(r['error']is not None for r in rows),bounds=rows),indent=2))
    print('original-only34 PASS',flush=True)
if __name__=='__main__':main()
