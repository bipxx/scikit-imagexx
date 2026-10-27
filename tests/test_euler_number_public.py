"""Original-first Euler configuration/LUT contracts; native remains RED."""
import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
import pytest
if __package__:
    from .euler_number_cases import API, STAGES, UNASSESSED, cases, fixture, exact_integer_value
    from .test_moments_public import original_runtime, metadata
    from .test_moments_derived_public import observe, result_record
else:
    from euler_number_cases import API, STAGES, UNASSESSED, cases, fixture, exact_integer_value
    from test_moments_public import original_runtime, metadata
    from test_moments_derived_public import observe, result_record
CASES=cases()
SOURCE_SHA256='b659f6b7d84b8ca5df77242ad33fa1d893d7e103e182c957cd7cb3016cd14f21'
LUT_PINS={'EULER_COEFS2D_4': '6d467219e4de1eae039e614f42094c40faa3b608da2ba45040396a0f0fa5a627', 'EULER_COEFS2D_8': 'b36cfcd3c99216ee69aea778adf13139ccbb4743107feb0c66254955de2d6c1a', 'EULER_COEFS3D_26': '034474d8920b3dc541878edca617e36c5e808f4dfc95d23d9b1ecc622ed50169'}
FILTER_PINS={'1.15.3':'e821fbd5ae150dc9fdb616ae5a213504904e5416febcde421739f3fe52769c0c',
             '1.18.1':'b90b80d80d038859966aad1b775431fec16f4d45b79150eb8ca385d769584223'}


def runtime():
    np,_,record=original_runtime()
    original=importlib.import_module('skimage.measure._regionprops_utils')
    raw=Path(original.__file__).read_bytes()
    assert hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==SOURCE_SHA256
    record['euler_source_sha256']=hashlib.sha256(raw).hexdigest()
    record['LUTs']={}
    for name,pin in LUT_PINS.items():
        value=getattr(original,name);words=[int(x)for x in value]
        assert hashlib.sha256(json.dumps(words,separators=(',',':')).encode()).hexdigest()==pin
        record['LUTs'][name]=dict(words=words,sha256=pin)
    path=Path(importlib.metadata.distribution('scipy').locate_file('scipy/ndimage/_filters.py'))
    raw=path.read_bytes();assert hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==FILTER_PINS[record['scipy']]
    record['scipy_filter_source_sha256']=hashlib.sha256(raw).hexdigest()
    return np,original,record


def encode(value,np):
    return result_record(value)if isinstance(value,(np.ndarray,np.generic))else value


def reference_case(spec,np,original):
    image,options,events=fixture(spec,np);before=encode(image,np);captured={};prior=sys.getprofile()
    def capture(frame,event,arg):
        if event=='return'and frame.f_code is original.euler_number.__code__:
            captured.update(h=frame.f_locals.get('h'),coefs=frame.f_locals.get('coefs'),prepared=frame.f_locals.get('image'))
    sys.setprofile(capture)
    try:observed=observe(original.euler_number,image,options,spec['policy'],np)
    finally:sys.setprofile(prior)
    return image,options,events,captured,observed,before


def validate(spec,captured,observed,np):
    result,error,_,checks=observed
    assert checks['state_unchanged']
    assert checks['input_unchanged']==(not spec['mutation'])
    if spec['outcome']!='source':assert (error is None)==(spec['outcome']=='return'),(spec['id'],error)
    h=captured.get('h')
    if error is not None:
        assert h is None
        return
    prepared=captured['prepared'];ndim=prepared.ndim
    assert ndim in(2,3)and type(h)is np.ndarray and h.dtype==np.dtype(np.intp)
    assert h.shape==((16,)if ndim==2 else(256,))and np.all(h>=0)
    assert sum(map(int,h))==prepared.size
    if ndim==2:assert isinstance(result,np.signedinteger)
    else:assert type(result)is int
    assert int(result)==exact_integer_value(h,captured['coefs'],ndim)


def compare(actual,expected,np):
    assert type(actual)is type(expected)and actual==expected
    if isinstance(expected,np.generic):
        assert metadata(actual)==metadata(expected)and actual.tobytes()==expected.tobytes()


@pytest.fixture(scope='module')
def reference_runtime():return runtime()
@pytest.fixture(scope='module')
def worker_results():return {}


def test_euler_signature(reference_runtime):
    _,original,_=reference_runtime
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    assert API in native_functions(),'euler_number remains upstream-only'
    assert inspect.signature(xx.measure.euler_number)==inspect.signature(original.euler_number)


@pytest.mark.parametrize('spec',CASES,ids=[r['id']for r in CASES])
@pytest.mark.parametrize('workers',(1,4))
def test_euler_owned(reference_runtime,spec,workers,worker_results,monkeypatch,tmp_path):
    np,original,_=reference_runtime
    image,options,events,captured,observed,before=reference_case(spec,np,original)
    expected,error,messages,checks=observed
    original_row=dict(id=spec['id'],input_before=before,input_after=encode(image,np),prepared=encode(captured.get('prepared'),np),
                      histogram=encode(captured.get('h'),np),coefficients=encode(captured.get('coefs'),np),
                      result=encode(expected,np),error=error,warnings=messages,checks=checks,events=events)
    (tmp_path/'original.json').write_text(json.dumps(original_row)+'\n')
    validate(spec,captured,observed,np);expected_input=encode(image,np);expected_events=list(events)
    image,options,events=fixture(spec,np)
    import skimage.measure as upstream
    import scipy.ndimage as ndi
    import skimagexx as xx
    from skimagexx import _compat
    from skimagexx.coverage import native_functions
    assert API in native_functions()and xx.get_num_threads()==1
    call=xx.measure.euler_number;prior=sys.getprofile();old=xx.set_fallback_enabled(False)
    returns=[];actual_local={};forbidden=[];events_at_counts=[]
    def deny(*args,**kwargs):
        forbidden.append(True);raise AssertionError('upstream Euler/configuration/reduction forwarding')
    def profile(frame,event,arg):
        if event=='c_return'and getattr(arg,'__module__',None)==xx._core.__name__ and getattr(arg,'__name__',None)in STAGES:
            returns.append(arg.__name__)
            if arg.__name__==STAGES[0]:events_at_counts.append(list(events))
        if event=='return'and frame.f_code is call.__code__:
            actual_local.update(h=frame.f_locals.get('h'),coefs=frame.f_locals.get('coefs'),prepared=frame.f_locals.get('image'))
    try:
        xx.set_num_threads(workers)
        with monkeypatch.context()as patch:
            patch.setattr(upstream,'euler_number',deny);patch.setattr(original,'euler_number',deny);patch.setattr(_compat,'fallback',deny)
            for name in('dot','matmul'):patch.setattr(np,name,deny)
            if error is None:
                patch.setattr(ndi,'convolve',deny);patch.setattr(importlib.import_module('scipy.ndimage._filters'),'convolve',deny)
                patch.setattr(np,'bincount',deny)
            sys.setprofile(profile)
            try:actual,actual_error,actual_messages,actual_checks=observe(call,image,options,spec['policy'],np)
            finally:sys.setprofile(prior)
        h=actual_local.get('h');coefs=actual_local.get('coefs');prepared=actual_local.get('prepared')
        raw=dict(id=spec['id'],workers=workers,input_after=encode(image,np),prepared=encode(prepared,np),histogram=encode(h,np),
                 coefficients=encode(coefs,np),result=encode(actual,np),error=actual_error,warnings=actual_messages,
                 checks=actual_checks,events=events,actual_C_returns=returns,events_at_counts=events_at_counts,
                 count_credit=returns.count(STAGES[0])if error is None and image.size else 0,
                 weighted_credit=returns.count(STAGES[1]),forbidden=forbidden)
        (tmp_path/'candidate.json').write_text(json.dumps(raw)+'\n')
        assert not forbidden and actual_error==error and actual_messages==messages and actual_checks==checks
        assert events==expected_events and encode(image,np)==expected_input
        assert encode(prepared,np)==encode(captured.get('prepared'),np)
        assert encode(coefs,np)==encode(captured.get('coefs'),np)
        if error is None:
            assert metadata(h)==metadata(captured['h']);np.testing.assert_array_equal(h,captured['h'])
            assert returns.count(STAGES[0])==returns.count(STAGES[1])==1
            assert events_at_counts==[expected_events]and returns.index(STAGES[0])<returns.index(STAGES[1])
            compare(actual,expected,np)
        else:assert not returns and h is None
        state=(encode(actual,np),encode(h,np),encode(coefs,np),actual_error,actual_messages,list(events))
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
            image,options,events,captured,observed,before=reference_case(spec,np,original);result,error,messages,checks=observed
            raw=dict(id=spec['id'],spec=spec,input_before=before,input_after=encode(image,np),prepared=encode(captured.get('prepared'),np),
                     histogram=encode(captured.get('h'),np),coefficients=encode(captured.get('coefs'),np),result=encode(result,np),
                     error=error,warnings=messages,checks=checks,events=events)
            journal.write(json.dumps(raw)+'\n');journal.flush();validate(spec,captured,observed,np)
            if error is None:compare(result,result,np)
            rows.append(dict(id=spec['id'],error=error,warnings=messages))
    assert not any(n=='skimagexx'or n.startswith('skimagexx.')for n in sys.modules)
    (args.output/'summary.json').write_text(json.dumps(dict(cases=len(rows),returns=sum(r['error']is None for r in rows),
        errors=sum(r['error']is not None for r in rows),native_calls=0,rows=rows),indent=2)+'\n')
    print('original-only',len(rows),'PASS',flush=True)


if __name__=='__main__':main()
