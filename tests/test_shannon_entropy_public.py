"""Original-first Shannon counts/base contract, no implementation here."""
import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import struct
import sys
import pytest
if __package__:
    from .shannon_entropy_cases import API,STAGES,UNASSESSED,cases,fixture,error_limit
    from .test_moments_public import original_runtime,metadata
    from .test_moments_derived_public import observe,result_record
else:
    from shannon_entropy_cases import API,STAGES,UNASSESSED,cases,fixture,error_limit
    from test_moments_public import original_runtime,metadata
    from test_moments_derived_public import observe,result_record
CASES=cases()
PINS={
 'modern':{'skimage/measure/entropy.py':'68dca961ab9aa3fbafcb08cbd437b0b5b083074d63af05fa067508028fc26e13','scipy/stats/_entropy.py':'28537a04b56b0ad9a162321d7d016ada9a10ced7e5430d614742f20e95ea4260','numpy/lib/_arraysetops_impl.py':'9f896cc0ddc2f4f94ed7cf0b28fa5dba777ec77df3c44788afcd6448d55416b4'},
 'minimum':{'skimage/measure/entropy.py':'68dca961ab9aa3fbafcb08cbd437b0b5b083074d63af05fa067508028fc26e13','scipy/stats/_entropy.py':'84c9612d5890a2cdb4298a41c2641ff5f49f99b333a0296e178b9183bc8ac537','numpy/lib/arraysetops.py':'189d91864cc826621bc321ba8772ce1533d783ada433db5c5756beeed75b56e5'}}

def runtime():
    version=importlib.metadata.version('numpy');profile='minimum'if version=='1.24.4'else'modern';source={}
    for name,pin in PINS[profile].items():
        distribution='scikit-image'if name.startswith('skimage/')else name.split('/')[0]
        path=Path(importlib.metadata.distribution(distribution).locate_file(name));raw=path.read_bytes()
        assert hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==pin,name
        source[name]=dict(raw_sha256=hashlib.sha256(raw).hexdigest(),canonical_lf_sha256=pin)
    np,_,record=original_runtime();original=importlib.import_module('skimage.measure.entropy')
    record.update(profile=profile,entropy_sources=source)
    return np,original,record

def encode(value,np):
    if value is None:return None
    if isinstance(value,np.ndarray)and value.dtype.kind=='O':
        # Only exact builtin strings occur here; never serialize object pointers.
        assert all(type(x)is str for x in value.flat)
        info=list(metadata(value));info[3]=None if info[3]is None else dict(info[3])
        return dict(metadata=info,strings=list(value.flat))
    if isinstance(value,(np.ndarray,np.generic)):return result_record(value)
    if type(value)is float:return dict(python_float_raw=struct.pack('<d',value).hex())
    return value

def reference_case(spec,np,original):
    image,options=fixture(spec,np);captured={};prior=sys.getprofile();backend=inspect.unwrap(original.scipy_entropy).__code__
    def capture(frame,event,arg):
        if frame.f_code is backend:
            if event=='c_call'and arg is math.log:captured['nats']=frame.f_locals.get('S')
            if event=='return':
                captured.update(pk=frame.f_locals.get('pk'),vec=frame.f_locals.get('vec'))
                if 'nats'not in captured:captured['nats']=frame.f_locals.get('S')
        if event=='return'and frame.f_code is original.shannon_entropy.__code__:
            captured.update(counts=frame.f_locals.get('counts'),symbols=frame.f_locals.get('_'))
    sys.setprofile(capture)
    try:observed=observe(original.shannon_entropy,image,options,spec['policy'],np)
    finally:sys.setprofile(prior)
    return image,options,captured,observed

def validate(spec,options,captured,observed,np):
    expected,error,messages,checks=observed;assert all(checks.values())
    assert(error is None)==(spec['outcome']=='return'),(spec['id'],error)
    counts=captured['counts'];assert type(counts)is np.ndarray and counts.dtype==np.dtype(np.intp)and counts.ndim==1
    assert np.all(counts>0)
    if error:return None
    assert type(expected)is np.float64
    if captured['nats']==0 and np.isfinite(expected):assert expected==0
    return error_limit(options['base'],captured['nats'],expected,np)

def compare(actual,expected,bound,np):
    assert type(actual)is type(expected)and actual.dtype==expected.dtype
    for mask in(np.isnan,np.isposinf,np.isneginf):assert bool(mask(actual))==bool(mask(expected))
    if np.isfinite(expected):
        assert abs(float(actual)-float(expected))<=bound
        if bound==0:assert actual==expected and np.signbit(actual)==np.signbit(expected)

@pytest.fixture(scope='module')
def reference_runtime():return runtime()
@pytest.fixture(scope='module')
def worker_results():return {}

def test_shannon_signature(reference_runtime):
    _,original,_=reference_runtime
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    assert API in native_functions()
    assert inspect.signature(xx.measure.shannon_entropy)==inspect.signature(original.shannon_entropy)

@pytest.mark.parametrize('spec',CASES,ids=[r['id']for r in CASES])
@pytest.mark.parametrize('workers',(1,4))
def test_shannon_owned(reference_runtime,spec,workers,worker_results,monkeypatch,tmp_path):
    np,original,_=reference_runtime
    image,options,captured,observed=reference_case(spec,np,original);expected,error,messages,checks=observed
    row=dict(id=spec['id'],input=encode(image,np),counts=encode(captured['counts'],np),symbols=encode(captured['symbols'],np),pk=encode(captured.get('pk'),np),vec=encode(captured.get('vec'),np),nats=encode(captured.get('nats'),np),result=encode(expected,np),error=error,warnings=messages,checks=checks)
    (tmp_path/'original.json').write_text(json.dumps(row)+'\n')
    bound=validate(spec,options,captured,observed,np);expected_counts=captured['counts']
    image,options=fixture(spec,np)
    import skimage.measure as upstream
    import scipy.stats as stats
    import scipy.special as special
    import skimagexx as xx
    from skimagexx import _compat
    from skimagexx.coverage import native_functions
    assert API in native_functions()and xx.get_num_threads()==1
    call=xx.measure.shannon_entropy;old=xx.set_fallback_enabled(False);prior=sys.getprofile();returns=[];actual_local={};forbidden=[]
    def deny(*args,**kwargs):forbidden.append(True);raise AssertionError('upstream Shannon/count/entropy forwarding')
    def profile(frame,event,arg):
        if event=='c_return'and getattr(arg,'__module__',None)==xx._core.__name__ and getattr(arg,'__name__',None)in STAGES:returns.append(arg.__name__)
        if event=='return'and frame.f_code is call.__code__:actual_local['counts']=frame.f_locals.get('counts')
    try:
        xx.set_num_threads(workers)
        with monkeypatch.context()as patch:
            patch.setattr(upstream,'shannon_entropy',deny);patch.setattr(original,'shannon_entropy',deny);patch.setattr(original,'unique',deny);patch.setattr(original,'scipy_entropy',deny)
            patch.setattr(stats,'entropy',deny);patch.setattr(importlib.import_module('scipy.stats._entropy'),'entropy',deny);patch.setattr(special,'entr',deny)
            if spec['unique']=='native':patch.setattr(np,'unique',deny)
            for name in('sum','log','log2','log10'):patch.setattr(np,name,deny)
            sys.setprofile(profile)
            try:actual,actual_error,actual_messages,actual_checks=observe(call,image,options,spec['policy'],np)
            finally:sys.setprofile(prior)
        actual_counts=actual_local.get('counts')
        (tmp_path/'candidate.json').write_text(json.dumps(dict(id=spec['id'],workers=workers,counts=encode(actual_counts,np),result=encode(actual,np),error=actual_error,warnings=actual_messages,checks=actual_checks,actual_C_returns=returns,unique_credit=returns.count(STAGES[0])if spec['unique']=='native'and image.size else 0,entropy_credit=returns.count(STAGES[1])if actual_counts is not None and actual_counts.size>1 else 0,forbidden=forbidden))+'\n')
        assert not forbidden and all(actual_checks.values())and actual_error==error and actual_messages==messages
        assert type(actual_counts)is np.ndarray and actual_counts.dtype==expected_counts.dtype
        np.testing.assert_array_equal(actual_counts,expected_counts)
        if spec['unique']=='native':assert returns.count(STAGES[0])==1
        # Source validates nonpositive base after unique but before normalization.
        valid_base=options['base']is None or not(options['base']<=0)
        assert returns.count(STAGES[1])==int(valid_base)
        if STAGES[0]in returns and STAGES[1]in returns:assert returns.index(STAGES[0])<returns.index(STAGES[1])
        if error is None:compare(actual,expected,bound,np)
        if spec['id']in worker_results:
            previous,previous_counts,previous_error,previous_messages=worker_results[spec['id']]
            assert actual_error==previous_error and actual_messages==previous_messages
            np.testing.assert_array_equal(actual_counts,previous_counts)
            if error is None:compare(actual,previous,0.,np)
        else:worker_results[spec['id']]=(actual,actual_counts,actual_error,actual_messages)
        assert xx.get_num_threads()==workers and _compat._fallback_enabled is False
    finally:sys.setprofile(prior);xx.set_num_threads(1);xx.set_fallback_enabled(old)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reference-only',action='store_true');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert args.reference_only and sys.flags.optimize==0 and not any(n=='skimagexx'or n.startswith('skimagexx.')for n in sys.modules)
    args.output.mkdir();np,original,record=runtime();(args.output/'runtime.json').write_text(json.dumps(record,indent=2))
    rows=[]
    with(args.output/'journal.jsonl').open('x')as journal:
        for spec in CASES:
            image,options,captured,observed=reference_case(spec,np,original);expected,error,messages,checks=observed
            row=dict(id=spec['id'],spec=spec,input=encode(image,np),counts=encode(captured.get('counts'),np),symbols=encode(captured.get('symbols'),np),pk=encode(captured.get('pk'),np),vec=encode(captured.get('vec'),np),nats=encode(captured.get('nats'),np),result=encode(expected,np),error=error,warnings=messages,checks=checks)
            journal.write(json.dumps(row)+'\n');journal.flush()
            bound=validate(spec,options,captured,observed,np)
            if error is None:compare(expected,expected,bound,np)
            rows.append(dict(id=spec['id'],error=error,bound=bound,nats=encode(captured.get('nats'),np)))
    assert not any(n=='skimagexx'or n.startswith('skimagexx.')for n in sys.modules)
    (args.output/'summary.json').write_text(json.dumps(dict(cases=len(rows),returns=sum(r['error']is None for r in rows),errors=sum(r['error']is not None for r in rows),rows=rows),indent=2))
    print('original-only30 PASS',flush=True)
if __name__=='__main__':main()
