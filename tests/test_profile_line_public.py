"""Original-first exact profile sampling and reducer-protocol contracts."""
import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import pytest
if __package__:
    from .profile_line_cases import API,STAGES,UNASSESSED,cases,fixture
    from .test_moments_public import original_runtime,metadata
    from .test_moments_derived_public import observe,result_record
else:
    from profile_line_cases import API,STAGES,UNASSESSED,cases,fixture
    from test_moments_public import original_runtime,metadata
    from test_moments_derived_public import observe,result_record
CASES=cases()
PRIMARY_SOURCE_PINS={'profile.py': {'url': 'https://raw.githubusercontent.com/scikit-image/scikit-image/v0.26.0/src/skimage/measure/profile.py', 'sha256': 'fabf6c23d892b2ee7bcda14ab46734ac4e7ec755b96603ff5e5791fdb012d1dd'}, '1.15.3-ni_interpolation.c': {'url': 'https://raw.githubusercontent.com/scipy/scipy/v1.15.3/scipy/ndimage/src/ni_interpolation.c', 'sha256': '5ec01d71453bc7771a18c9f7032d08071ba89d0cd187398b5dd1049e66da9876'}, '1.15.3-ni_splines.c': {'url': 'https://raw.githubusercontent.com/scipy/scipy/v1.15.3/scipy/ndimage/src/ni_splines.c', 'sha256': '7c49045c917d827dd2c9204ffb108c244307530b0f70daa00ba85a14f1c2b65b'}, '1.15.3-_interpolation.py': {'url': 'https://raw.githubusercontent.com/scipy/scipy/v1.15.3/scipy/ndimage/_interpolation.py', 'sha256': '6656f865125b4ceadfdb575e74edbc1874d70342a1f613020d605d1af45b728e'}, '1.18.1-ni_interpolation.c': {'url': 'https://raw.githubusercontent.com/scipy/scipy/v1.18.1/scipy/ndimage/src/ni_interpolation.c', 'sha256': 'b8050f0ac53f0762a9370493dd08d5e6ef9005cdbb6a01a560304d3b895acb53'}, '1.18.1-ni_splines.c': {'url': 'https://raw.githubusercontent.com/scipy/scipy/v1.18.1/scipy/ndimage/src/ni_splines.c', 'sha256': '27a0f47f2129ce9a39e5017ecc8f02da37592e393b60cf9535f0e2b2a31f6d52'}, '1.18.1-_interpolation.py': {'url': 'https://raw.githubusercontent.com/scipy/scipy/v1.18.1/scipy/ndimage/_interpolation.py', 'sha256': '0e834f3b96ddf80ad96f30ecd3b5499fcfd3e3e049674e29a08c1c4b5dec8190'}, 'utils.py': {'url': 'https://raw.githubusercontent.com/scikit-image/scikit-image/v0.26.0/src/skimage/_shared/utils.py', 'sha256': 'b0a232570c8e990ab76cd6fe6f54a67161a691214dbf6c529656e13a799eb1f1'}}


def runtime():
    np,_,record=original_runtime();original=importlib.import_module('skimage.measure.profile')
    paths={'profile.py':Path(original.__file__),
           'utils.py':Path(importlib.import_module('skimage._shared.utils').__file__),
           record['scipy']+'-_interpolation.py':Path(importlib.import_module('scipy.ndimage._interpolation').__file__)}
    for name,path in paths.items():
        raw=path.read_bytes();assert hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==PRIMARY_SOURCE_PINS[name]['sha256'],name
    record.update(profile_primary_source_pins=PRIMARY_SOURCE_PINS,
                  actual_source_raw_sha256={name:hashlib.sha256(path.read_bytes()).hexdigest()for name,path in paths.items()},
                  sample_policy='EXACT raw pre-mutation flip snapshots and public words/metadata; no fitted tolerance')
    return np,original,record


def encode(value,np):
    if isinstance(value,(np.ndarray,np.generic)):return result_record(value)
    if isinstance(value,dict):return {k:encode(v,np)for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [encode(v,np)for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):return value
    return 'callable:'+getattr(value,'__name__',repr(value))


def captured_call(call,image,options,policy,np):
    captured={};prior=sys.getprofile();flip_code=inspect.unwrap(np.flip).__code__
    def profile(frame,event,arg):
        if event=='return'and frame.f_code is flip_code:
            # Encode immediately, never retain the live pre-reducer array.
            captured['pixels']=encode(arg,np)
    sys.setprofile(profile)
    try:observed=observe(call,image,options,policy,np)
    finally:sys.setprofile(prior)
    return captured,observed


def validate(spec,captured,observed,np):
    result,error,_,checks=observed
    assert checks==dict(state_unchanged=True,input_unchanged=True),spec['id']
    assert (error is not None)==spec['error'],(spec['id'],error)
    if not spec['error']:assert type(result)is np.ndarray
    if captured:
        info=captured['pixels']['metadata'];assert len(info[4])in(2,3)
        assert info[4][0]>0 and info[4][1]>=0
    if spec['route']=='native':assert 'pixels'in captured,spec['id']


@pytest.fixture(scope='module')
def reference_runtime():return runtime()
@pytest.fixture(scope='module')
def worker_results():return {}


def test_profile_line_signature(reference_runtime):
    _,original,_=reference_runtime
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    assert API in native_functions(),'profile_line remains upstream-only'
    assert inspect.signature(xx.measure.profile_line)==inspect.signature(original.profile_line)


@pytest.mark.parametrize('spec',CASES,ids=[s['id']for s in CASES])
@pytest.mark.parametrize('workers',(1,4))
def test_profile_line_owned(reference_runtime,spec,workers,worker_results,monkeypatch,tmp_path):
    np,original,_=reference_runtime
    image,options,events=fixture(spec,np);before=encode((image,options),np)
    captured,observed=captured_call(original.profile_line,image,options,spec['policy'],np)
    expected,error,messages,checks=observed;validate(spec,captured,observed,np)
    original_row=dict(id=spec['id'],input_before=before,input_after=encode((image,options),np),
        pre_reducer=captured,result=encode(expected,np),error=error,warnings=messages,checks=checks,events=events)
    (tmp_path/'original.json').write_text(json.dumps(original_row)+'\n')
    expected_events=list(events);expected_input=encode((image,options),np)
    image,options,events=fixture(spec,np)
    import skimage.measure as upstream
    import scipy.ndimage as ndi
    import skimagexx as xx
    from skimagexx import _compat
    from skimagexx.coverage import native_functions
    assert API in native_functions()and xx.get_num_threads()==1
    old=xx.set_fallback_enabled(False);prior=sys.getprofile();returns=[];forbidden=[]
    actual_capture={};flip_code=inspect.unwrap(np.flip).__code__
    def deny(*args,**kwargs):
        forbidden.append(True);raise AssertionError('upstream profile/coordinate/interpolation forwarding')
    def profile(frame,event,arg):
        if event=='c_return'and getattr(arg,'__module__',None)==xx._core.__name__ and getattr(arg,'__name__',None)in STAGES:
            returns.append(arg.__name__)
        if event=='return'and frame.f_code is flip_code:actual_capture['pixels']=encode(arg,np)
    try:
        xx.set_num_threads(workers)
        with monkeypatch.context()as patch:
            patch.setattr(upstream,'profile_line',deny);patch.setattr(original,'profile_line',deny)
            patch.setattr(original,'_line_profile_coordinates',deny);patch.setattr(_compat,'fallback',deny)
            if spec['route']=='native':
                patch.setattr(ndi,'map_coordinates',deny)
                patch.setattr(importlib.import_module('scipy.ndimage._interpolation'),'map_coordinates',deny)
                patch.setattr(importlib.import_module('scipy.ndimage._nd_image'),'geometric_transform',deny)
            sys.setprofile(profile)
            try:actual,actual_error,actual_messages,actual_checks=observe(xx.measure.profile_line,image,options,spec['policy'],np)
            finally:sys.setprofile(prior)
        row=dict(id=spec['id'],workers=workers,input_after=encode((image,options),np),
            pre_reducer=actual_capture,result=encode(actual,np),error=actual_error,warnings=actual_messages,
            checks=actual_checks,events=events,actual_C_returns=returns,forbidden=forbidden,
            native_sampler_credit=int(spec['route']=='native'and bool(captured)and captured['pixels']['metadata'][4][1]>0))
        (tmp_path/'candidate.json').write_text(json.dumps(row)+'\n')
        assert not forbidden and actual_error==error and actual_messages==messages and actual_checks==checks
        assert events==expected_events and encode((image,options),np)==expected_input
        assert actual_capture==captured,'sample dtype/layout/base/raw words changed before callback'
        assert encode(actual,np)==encode(expected,np),'public output metadata/raw words changed'
        planes=image.shape[2]if isinstance(image,np.ndarray)and image.ndim==3 else 1
        if spec['route']=='native':assert returns==list(STAGES)*planes
        elif spec['route']=='provider':assert len(returns)<=planes
        else:assert not returns
        state=(encode(actual,np),actual_capture,actual_error,actual_messages,list(events))
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
            captured,observed=captured_call(original.profile_line,image,options,spec['policy'],np)
            result,error,messages,checks=observed
            raw=dict(id=spec['id'],spec=spec,input_before=before,input_after=encode((image,options),np),
                pre_reducer=captured,result=encode(result,np),error=error,warnings=messages,checks=checks,events=events)
            journal.write(json.dumps(raw)+'\n');journal.flush();validate(spec,captured,observed,np)
            rows.append(dict(id=spec['id'],error=error,warnings=messages))
    assert not any(n=='skimagexx'or n.startswith('skimagexx.')for n in sys.modules)
    (args.output/'summary.json').write_text(json.dumps(dict(cases=len(rows),returns=sum(r['error']is None for r in rows),
        errors=sum(r['error']is not None for r in rows),native_calls=0,rows=rows),indent=2)+'\n')
    print('original-only',len(rows),'PASS',flush=True)


if __name__=='__main__':main()
