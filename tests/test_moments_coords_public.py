"""Original-first coordinate moments; no production implementation here."""
import argparse
import inspect
import json
from pathlib import Path
import sys
import pytest
if __package__:
    from .moments_coords_cases import APIS,STAGE,UNASSESSED,cases,fixture,error_limits
    from .test_moments_public import original_runtime,metadata
    from .test_moments_derived_public import observe,result_record
else:
    from moments_coords_cases import APIS,STAGE,UNASSESSED,cases,fixture,error_limits
    from test_moments_public import original_runtime,metadata
    from test_moments_derived_public import observe,result_record
CASES=cases()

def encode(value,np):
    if isinstance(value,np.ndarray):return result_record(value)
    if isinstance(value,np.generic):return dict(scalar_type=type(value).__name__,dtype=value.dtype.str,raw=value.tobytes().hex())
    if isinstance(value,(tuple,list)):return dict(sequence=type(value).__name__,items=[encode(x,np)for x in value])
    return value

def json_value(value):
    if isinstance(value,complex):return dict(complex_real=value.real,complex_imag=value.imag)
    raise TypeError('unhandled raw journal value: '+type(value).__name__)

def reference_case(spec,np,original):
    coords,options=fixture(spec,np);captured={};prior=sys.getprofile()
    def capture(frame,event,arg):
        if event=='return'and frame.f_code is original.moments_coords_central.__code__:
            local=frame.f_locals
            captured.update(center=local.get('center'),powers=local.get('coords')if 'calc'in local else None,calc=local.get('calc'))
    sys.setprofile(capture)
    try:observed=observe(getattr(original,spec['api']),coords,options,spec['policy'],np)
    finally:sys.setprofile(prior)
    return coords,options,captured,observed

def validate(spec,captured,observed,np):
    expected,error,messages,checks=observed;assert all(checks.values())
    assert(error is None)==(spec['outcome']=='return'),(spec['id'],error)
    if error:return None
    assert type(expected)is np.ndarray and expected.shape==(spec['order']+1,)*spec['D']
    with np.errstate(all='ignore'):bound=error_limits(spec,captured['calc'],expected,np)
    return bound

def compare(actual,expected,bound,np):
    assert metadata(actual)==metadata(expected)
    parts=((actual.real,expected.real),(actual.imag,expected.imag))if expected.dtype.kind=='c'else((actual,expected),)
    for a,e in parts:
        for mask in(np.isnan,np.isposinf,np.isneginf):np.testing.assert_array_equal(mask(a),mask(e))
        finite=np.isfinite(e);assert np.all(np.abs(a[finite]-e[finite])<=bound[finite])
        exact=finite&(bound==0);np.testing.assert_array_equal(a[exact],e[exact]);np.testing.assert_array_equal(np.signbit(a[exact]),np.signbit(e[exact]))

@pytest.fixture(scope='module')
def reference_runtime():return original_runtime()
@pytest.fixture(scope='module')
def worker_results():return {}

@pytest.mark.parametrize('api',APIS)
def test_coords_signature(reference_runtime,api):
    _,original,_=reference_runtime
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    assert 'measure.'+api in native_functions()
    assert inspect.signature(getattr(xx.measure,api))==inspect.signature(getattr(original,api))

@pytest.mark.parametrize('spec',CASES,ids=[r['id']for r in CASES])
@pytest.mark.parametrize('workers',(1,4))
def test_coords_owned(reference_runtime,spec,workers,worker_results,monkeypatch,tmp_path):
    np,original,_=reference_runtime
    coords,options,captured,observed=reference_case(spec,np,original);expected,error,messages,checks=observed
    (tmp_path/'original.json').write_text(json.dumps(dict(id=spec['id'],input=encode(coords,np),center=encode(captured.get('center'),np),powers=encode(captured.get('powers'),np),calc=encode(captured.get('calc'),np),result=encode(expected,np),error=error,warnings=messages,checks=checks))+'\n')
    bound=validate(spec,captured,observed,np)
    coords,options=fixture(spec,np)
    import skimage.measure as upstream
    import skimagexx as xx
    from skimagexx import _compat
    from skimagexx.coverage import native_functions
    assert 'measure.'+spec['api']in native_functions()and xx.get_num_threads()==1
    old=xx.set_fallback_enabled(False);prior=sys.getprofile();returns=[];forbidden=[]
    def deny(*args,**kwargs):forbidden.append(True);raise AssertionError('upstream coordinate moments or reduction forwarding')
    def profile(frame,event,arg):
        if event=='c_return'and getattr(arg,'__module__',None)==xx._core.__name__ and getattr(arg,'__name__',None)==STAGE:returns.append(STAGE)
    try:
        xx.set_num_threads(workers)
        with monkeypatch.context()as patch:
            for name in APIS:patch.setattr(upstream,name,deny);patch.setattr(original,name,deny)
            if spec['route']=='native':
                for name in ('sum','einsum','dot','tensordot','matmul'):patch.setattr(np,name,deny)
            sys.setprofile(profile)
            try:actual,actual_error,actual_messages,actual_checks=observe(getattr(xx.measure,spec['api']),coords,options,spec['policy'],np)
            finally:sys.setprofile(prior)
        (tmp_path/'candidate.json').write_text(json.dumps(dict(id=spec['id'],workers=workers,result=encode(actual,np),error=actual_error,warnings=actual_messages,checks=actual_checks,actual_C_returns=returns,owned_reduction_credit=len(returns)if spec['route']=='native'else 0,declared_route=spec['route'],forbidden=forbidden))+'\n')
        assert not forbidden and all(actual_checks.values())and actual_error==error and actual_messages==messages
        if error is None:
            compare(actual,expected,bound,np)
            if spec['route']=='native':assert len(returns)==1,'ordinary products AND reduction must be owned'
        words=(encode(actual,np),actual_error,actual_messages);assert worker_results.setdefault(spec['id'],words)==words
        assert xx.get_num_threads()==workers and _compat._fallback_enabled is False
    finally:sys.setprofile(prior);xx.set_num_threads(1);xx.set_fallback_enabled(old)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reference-only',action='store_true');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert args.reference_only and sys.flags.optimize==0
    assert not any(n=='skimagexx'or n.startswith('skimagexx.')for n in sys.modules)
    args.output.mkdir();np,original,runtime=original_runtime();(args.output/'runtime.json').write_text(json.dumps(runtime,indent=2))
    summary=[]
    with(args.output/'journal.jsonl').open('x')as journal:
        for spec in CASES:
            coords,options,captured,observed=reference_case(spec,np,original);expected,error,messages,checks=observed
            row=dict(id=spec['id'],spec=spec,input=encode(coords,np),center=encode(captured.get('center'),np),powers=encode(captured.get('powers'),np),calc=encode(captured.get('calc'),np),result=encode(expected,np),error=error,warnings=messages,checks=checks)
            journal.write(json.dumps(row,default=json_value)+'\n');journal.flush()
            bound=validate(spec,captured,observed,np)
            if error is None:compare(expected,expected,bound,np)
            summary.append(dict(id=spec['id'],error=error,result_dtype=None if expected is None else expected.dtype.str,
                calc_shape=None if captured.get('calc')is None else captured['calc'].shape,
                calc_strides=None if captured.get('calc')is None else captured['calc'].strides,
                powers_strides=None if captured.get('powers')is None else captured['powers'].strides,
                bound=None if bound is None else bound.tolist()))
    assert not any(n=='skimagexx'or n.startswith('skimagexx.')for n in sys.modules)
    (args.output/'summary.json').write_text(json.dumps(dict(cases=len(summary),returns=sum(r['error']is None for r in summary),errors=sum(r['error']is not None for r in summary),rows=summary),indent=2))
    print('original-only48 PASS',flush=True)
if __name__=='__main__':main()
