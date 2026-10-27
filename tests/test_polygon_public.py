"""Original-first exact polygon contracts; no implementation credit yet."""
import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import pytest
if __package__:
    from .polygon_cases import APIS,STAGES,UNASSESSED,cases,fixture
    from .test_moments_public import original_runtime,metadata
    from .test_moments_derived_public import observe,result_record
else:
    from polygon_cases import APIS,STAGES,UNASSESSED,cases,fixture
    from test_moments_public import original_runtime,metadata
    from test_moments_derived_public import observe,result_record
CASES=cases()
PRIMARY_SOURCE_PINS={
 'src/skimage/measure/pnpoly.py':'8e3fe4a6fc6bde58a986606a879182b2d056deaadfa9bcb0c541d1fd49d48162',
 'src/skimage/measure/_pnpoly.pyx':'ce74f71025a896a789153dfd136d412578be30b5566cccb6b8a64293d36627bc',
 'src/skimage/_shared/geometry.pyx':'889ee7d70fccf6ef3dd70054e592e87a4dc0f48aa83755b53c7c0ec95d7256c8',
 'src/skimage/_shared/geometry.pxd':'7188fa98f18e517eb226e4d8a11edbd2e35a8c9119861e132ccf515c8667959c'}


def runtime():
    np,_,record=original_runtime();original=importlib.import_module('skimage.measure.pnpoly')
    raw=Path(original.__file__).read_bytes()
    assert hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==PRIMARY_SOURCE_PINS['src/skimage/measure/pnpoly.py']
    binary=Path(importlib.import_module('skimage.measure._pnpoly').__file__)
    record.update(polygon_primary_source_pins=PRIMARY_SOURCE_PINS,
                  polygon_wrapper_raw_sha256=hashlib.sha256(raw).hexdigest(),
                  polygon_binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    return np,original,record


def encode(value,np):
    if isinstance(value,(np.ndarray,np.generic)):return result_record(value)
    if isinstance(value,dict):return {k:encode(v,np)for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [encode(v,np)for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):return value
    return repr(value)


def validate(spec,observed,np):
    result,error,_,checks=observed
    assert checks==dict(state_unchanged=True,input_unchanged=True)
    if error is not None:return
    assert type(result)is np.ndarray
    if spec['api']=='grid_points_in_poly':
        assert result.ndim==2 and result.dtype in(np.dtype('bool'),np.dtype('uint8'))
        assert np.all(result<=3)
    else:assert result.ndim==1 and result.dtype.kind in'bu'and np.all(result<=1)


def compare(actual,expected,np):
    assert metadata(actual)==metadata(expected)
    np.testing.assert_array_equal(actual,expected)
    assert actual.tobytes()==expected.tobytes()


@pytest.fixture(scope='module')
def reference_runtime():return runtime()
@pytest.fixture(scope='module')
def worker_results():return {}


@pytest.mark.parametrize('api',APIS)
def test_polygon_signature(reference_runtime,api):
    _,original,_=reference_runtime
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    assert 'measure.'+api in native_functions(),'polygon API remains upstream-only'
    assert inspect.signature(getattr(xx.measure,api))==inspect.signature(getattr(original,api))


@pytest.mark.parametrize('spec',CASES,ids=[s['id']for s in CASES])
@pytest.mark.parametrize('workers',(1,4))
def test_polygon_owned(reference_runtime,spec,workers,worker_results,monkeypatch,tmp_path):
    np,original,_=reference_runtime
    image,options,events=fixture(spec,np);input_before=encode((image,options),np)
    observed=observe(getattr(original,spec['api']),image,options,spec['policy'],np)
    expected,error,messages,checks=observed;validate(spec,observed,np)
    raw=dict(id=spec['id'],input_before=input_before,input_after=encode((image,options),np),
             result=encode(expected,np),error=error,warnings=messages,checks=checks,events=events)
    (tmp_path/'original.json').write_text(json.dumps(raw)+'\n')
    expected_events=list(events);expected_input=encode((image,options),np)
    image,options,events=fixture(spec,np)
    import skimage.measure as upstream
    import skimagexx as xx
    from skimagexx import _compat
    from skimagexx.coverage import native_functions
    assert 'measure.'+spec['api']in native_functions()and xx.get_num_threads()==1
    old=xx.set_fallback_enabled(False);prior=sys.getprofile();returns=[];forbidden=[]
    def deny(*args,**kwargs):
        forbidden.append(True);raise AssertionError('whole upstream polygon forwarding')
    def profile(frame,event,arg):
        if event=='c_return'and getattr(arg,'__module__',None)==xx._core.__name__ and getattr(arg,'__name__',None)==STAGES[spec['api']]:
            returns.append(arg.__name__)
    try:
        xx.set_num_threads(workers)
        with monkeypatch.context()as patch:
            for api in APIS:
                patch.setattr(upstream,api,deny);patch.setattr(original,api,deny)
                patch.setattr(original,'_'+api,deny)
                patch.setattr(importlib.import_module('skimage.measure._pnpoly'),'_'+api,deny)
            patch.setattr(_compat,'fallback',deny)
            sys.setprofile(profile)
            try:actual,actual_error,actual_messages,actual_checks=observe(getattr(xx.measure,spec['api']),image,options,spec['policy'],np)
            finally:sys.setprofile(prior)
        row=dict(id=spec['id'],workers=workers,input_after=encode((image,options),np),result=encode(actual,np),
                 error=actual_error,warnings=actual_messages,checks=actual_checks,events=events,
                 actual_C_returns=returns,forbidden=forbidden)
        (tmp_path/'candidate.json').write_text(json.dumps(row)+'\n')
        assert not forbidden and actual_error==error and actual_messages==messages and actual_checks==checks
        assert events==expected_events and encode((image,options),np)==expected_input
        if error is None:
            compare(actual,expected,np);assert len(returns)==1
        elif spec['pattern']=='probe-raise':assert len(returns)==1
        else:assert not returns
        state=(encode(actual,np),actual_error,actual_messages,list(events))
        if spec['id']in worker_results:assert state==worker_results[spec['id']]
        else:worker_results[spec['id']]=state
        assert xx.get_num_threads()==workers and _compat._fallback_enabled is False
    finally:
        sys.setprofile(prior);xx.set_num_threads(1);xx.set_fallback_enabled(old)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reference-only',action='store_true');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert args.reference_only and sys.flags.optimize==0
    assert not any(n=='skimagexx'or n.startswith('skimagexx.')for n in sys.modules)
    args.output.mkdir();np,original,record=runtime();(args.output/'runtime.json').write_text(json.dumps(record,indent=2)+'\n')
    rows=[]
    with(args.output/'journal.jsonl').open('x')as journal:
        for spec in CASES:
            image,options,events=fixture(spec,np);before=encode((image,options),np)
            observed=observe(getattr(original,spec['api']),image,options,spec['policy'],np)
            result,error,messages,checks=observed
            raw=dict(id=spec['id'],spec=spec,input_before=before,input_after=encode((image,options),np),
                     result=encode(result,np),error=error,warnings=messages,checks=checks,events=events)
            journal.write(json.dumps(raw)+'\n');journal.flush();validate(spec,observed,np)
            if error is None:compare(result,result,np)
            rows.append(dict(id=spec['id'],error=error,warnings=messages))
    assert not any(n=='skimagexx'or n.startswith('skimagexx.')for n in sys.modules)
    (args.output/'summary.json').write_text(json.dumps(dict(cases=len(rows),returns=sum(r['error']is None for r in rows),
        errors=sum(r['error']is not None for r in rows),native_calls=0,rows=rows),indent=2)+'\n')
    print('original-only',len(rows),'PASS',flush=True)


if __name__=='__main__':main()
