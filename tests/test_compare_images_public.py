"""Original-first compare_images words, layouts, stage order and protocols."""
import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import warnings

import pytest
if __package__:
    from .compare_images_cases import API, STAGES, UNASSESSED, cases, fixture
    from .test_moments_public import original_runtime
    from .test_moments_derived_public import result_record
else:
    from compare_images_cases import API, STAGES, UNASSESSED, cases, fixture
    from test_moments_public import original_runtime
    from test_moments_derived_public import result_record

CASES = cases()
SOURCE_SHA256 = '0a9bfaa68d694f6f6141a722fbc2317c045bf279b90d54a7ba8d6681d4e30622'
DTYPE_SHA256 = 'd4f99f88966e4d0dbbe80a3c0b8acb3890644cc5fbfb707ba9b5a90e9119e627'


def runtime():
    np, _, record = original_runtime()
    original = importlib.import_module('skimage.util.compare')
    sources = {}
    for module, pin in ((original, SOURCE_SHA256), (importlib.import_module('skimage.util.dtype'), DTYPE_SHA256)):
        path = Path(module.__file__); raw = path.read_bytes()
        assert hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest() == pin
        sources[module.__name__] = dict(path=str(path),raw_sha256=hashlib.sha256(raw).hexdigest(),canonical_lf_sha256=pin)
    record.update(compare_sources=sources,policy='Exact stage/public metadata and words; no numerical allowance')
    return np, original, record


def encode(value, np):
    if isinstance(value,(np.ndarray,np.generic)): return result_record(value)
    if isinstance(value,dict): return {key:encode(item,np) for key,item in value.items()}
    if isinstance(value,(tuple,list)): return dict(container=type(value).__name__,items=[encode(item,np) for item in value])
    if value is None or isinstance(value,(str,int,float,bool)): return value
    if type(value).__name__ == 'Image': return dict(protocol='Image',role=value.role,image=encode(value.image,np))
    return repr(value)


def captured_call(call,args,options,policy,np):
    before = encode((args,options),np)
    prior_policy,handler,buffer = np.geterr(),np.geterrcall(),np.getbufsize()
    function = inspect.unwrap(call)
    converted = []; prior_trace = sys.gettrace()
    def trace(frame,event,arg):
        if frame.f_code is function.__code__:
            if event == 'return':
                converted.append({name:encode(frame.f_locals[name],np) for name in ('img1','img2') if name in frame.f_locals})
            return trace
        return None
    result,error = None,None
    sys.settrace(trace)
    try:
        with np.errstate(**policy):
            selected = np.geterr()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                try: result = call(*args,**options)
                except Exception as exc: error = (type(exc).__module__,type(exc).__name__,str(exc))
            inside = np.geterr() == selected and np.geterrcall() is handler and np.getbufsize() == buffer
    finally: sys.settrace(prior_trace)
    checks = dict(state_unchanged=inside and np.geterr()==prior_policy and np.geterrcall() is handler and np.getbufsize()==buffer,
                  input_unchanged=encode((args,options),np)==before)
    messages = [(type(w.message).__module__,type(w.message).__name__,str(w.message)) for w in caught]
    return dict(converted=converted,result=encode(result,np),error=error,warnings=messages,checks=checks)


def validate(spec,row):
    assert all(row['checks'].values()),spec['id']
    assert (row['error'] is None) == (spec['outcome']=='return'),(spec['id'],row['error'])
    if spec['route']=='native':
        assert len(row['converted']) == 1 and set(row['converted'][0]) == {'img1','img2'},spec['id']


@pytest.fixture(scope='module')
def reference_runtime(): return runtime()


@pytest.fixture(scope='module')
def worker_results(): return {}


def test_compare_images_signature(reference_runtime):
    _,original,_ = reference_runtime
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    assert API in native_functions(),'compare_images remains upstream-only'
    assert inspect.signature(xx.util.compare_images) == inspect.signature(original.compare_images)
    assert xx.util.compare_images is not original.compare_images
    assert all(callable(getattr(xx._core,name,None)) for name in STAGES)


@pytest.mark.parametrize('spec',CASES,ids=[row['id'] for row in CASES])
@pytest.mark.parametrize('workers',(1,4))
def test_compare_images_owned(reference_runtime,spec,workers,worker_results,monkeypatch,tmp_path):
    np,original,_ = reference_runtime
    args,options,events = fixture(spec,np)
    before = encode((args,options),np)
    expected = captured_call(original.compare_images,args,options,spec['policy'],np)
    validate(spec,expected)
    expected.update(input_before=before,input_after=encode((args,options),np),events=list(events))
    (tmp_path/'original.json').write_text(json.dumps(expected)+'\n')
    args,options,events = fixture(spec,np)
    import skimage.util as upstream
    import skimagexx as xx
    from skimagexx import _compat
    from skimagexx.coverage import native_functions
    assert API in native_functions() and xx.get_num_threads()==1
    assert all(callable(getattr(xx._core,name,None)) for name in STAGES)
    old = xx.set_fallback_enabled(False); prior = sys.getprofile(); returns=[]; forbidden=[]; accepted=[]
    public_code = inspect.unwrap(xx.util.compare_images).__code__
    def deny(*args,**kwargs):
        forbidden.append(True); raise AssertionError('whole upstream compare_images forwarding')
    def profile(frame,event,arg):
        if event=='c_return' and getattr(arg,'__module__',None)==xx._core.__name__ and getattr(arg,'__name__',None) in STAGES:
            returns.append(arg.__name__)
        if event=='return' and frame.f_code is public_code:
            accepted.append(frame.f_locals.get('native_comparison') is not None)
    try:
        xx.set_num_threads(workers)
        with monkeypatch.context() as patch:
            patch.setattr(upstream,'compare_images',deny)
            patch.setattr(original,'compare_images',deny)
            patch.setattr(_compat,'fallback',deny)
            sys.setprofile(profile)
            try: actual = captured_call(xx.util.compare_images,args,options,spec['policy'],np)
            finally: sys.setprofile(prior)
        actual.update(input_before=before,input_after=encode((args,options),np),events=list(events))
        (tmp_path/'candidate.json').write_text(json.dumps(dict(actual,actual_C_returns=returns,native_accepted=accepted,forbidden=forbidden))+'\n')
        assert not forbidden
        assert actual == expected
        if spec['route']=='native':
            assert returns == list(STAGES) and accepted == [True]
        elif spec['route']=='intake': assert not returns
        if spec['id'] in worker_results: assert actual == worker_results[spec['id']]
        else: worker_results[spec['id']] = actual
        assert xx.get_num_threads()==workers and _compat._fallback_enabled is False
    finally:
        sys.setprofile(prior); xx.set_num_threads(1); xx.set_fallback_enabled(old)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--reference-only',action='store_true')
    parser.add_argument('--output',type=Path,required=True); args = parser.parse_args()
    assert args.reference_only and sys.flags.optimize==0
    assert not any(name=='skimagexx' or name.startswith('skimagexx.') for name in sys.modules)
    args.output.mkdir(); np,original,record = runtime()
    (args.output/'runtime.json').write_text(json.dumps(record,indent=2)+'\n')
    rows=[]
    with (args.output/'journal.jsonl').open('x') as journal:
        for spec in CASES:
            values,options,events = fixture(spec,np); before = encode((values,options),np)
            row = captured_call(original.compare_images,values,options,spec['policy'],np)
            row.update(id=spec['id'],spec=spec,input_before=before,input_after=encode((values,options),np),events=events)
            journal.write(json.dumps(row)+'\n'); journal.flush(); validate(spec,row); rows.append(row)
    assert not any(name=='skimagexx' or name.startswith('skimagexx.') for name in sys.modules)
    summary = dict(cases=len(rows),returns=sum(row['error'] is None for row in rows),errors=sum(row['error'] is not None for row in rows),native_calls=0)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('original-only',len(rows),'PASS',flush=True)


if __name__=='__main__': main()
