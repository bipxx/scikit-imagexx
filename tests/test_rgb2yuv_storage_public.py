"""Original-first half/endian rgb2yuv; existing color budgets unchanged."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
import pytest
if __package__:
    from .rgb2yuv_storage_cases import API,STAGE,cases,fixture,limits
    from .test_moments_public import original_runtime,metadata
    from .test_moments_derived_public import observe,result_record
else:
    from rgb2yuv_storage_cases import API,STAGE,cases,fixture,limits
    from test_moments_public import original_runtime,metadata
    from test_moments_derived_public import observe,result_record
CASES=cases()
PINS={'skimage/color/colorconv.py':'fc5f49ac194c229325f21ddbe4c49f7df4307b9627dfbb5fcd2825133144fdfe',
      'skimage/_shared/utils.py':'b0a232570c8e990ab76cd6fe6f54a67161a691214dbf6c529656e13a799eb1f1',
      'skimage/util/dtype.py':'d4f99f88966e4d0dbbe80a3c0b8acb3890644cc5fbfb707ba9b5a90e9119e627'}
def runtime():
    distribution=importlib.metadata.distribution('scikit-image');source={}
    for name,pin in PINS.items():
        raw=Path(distribution.locate_file(name)).read_bytes()
        assert hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==pin
        source[name]=dict(raw_sha256=hashlib.sha256(raw).hexdigest(),canonical_lf_sha256=pin)
    np,_,record=original_runtime()
    from skimage import color
    from skimage.color import colorconv
    record['color_sources']=source
    return np,color,colorconv.yuv_from_rgb,record

def compare(actual,expected,image,matrix,np):
    assert metadata(actual)==metadata(expected)
    for mask in(np.isnan,np.isposinf,np.isneginf):np.testing.assert_array_equal(mask(actual),mask(expected))
    rtol,atol=limits(image,expected,matrix,np)
    np.testing.assert_allclose(actual,expected,rtol=rtol,atol=atol)
    assert not np.shares_memory(actual,image)

@pytest.fixture(scope='module')
def reference_runtime():return runtime()
def test_rgb2yuv_storage_signature(reference_runtime):
    _,original,_,_=reference_runtime
    import skimagexx as xx
    assert inspect.signature(xx.color.rgb2yuv)==inspect.signature(original.rgb2yuv)
    assert xx.rgb2yuv is xx.color.rgb2yuv

@pytest.mark.parametrize('spec',CASES,ids=[s['id']for s in CASES])
@pytest.mark.parametrize('workers',(1,4))
def test_rgb2yuv_storage_owned(reference_runtime,spec,workers,monkeypatch,tmp_path):
    np,original,matrix,_=reference_runtime;image,options=fixture(spec,np)
    expected,error,messages,checks=observe(original.rgb2yuv,image,options,spec['policy'],np)
    (tmp_path/'original.json').write_text(json.dumps(dict(id=spec['id'],input=result_record(image),result=result_record(expected),error=error,warnings=messages,checks=checks))+'\n')
    assert all(checks.values())
    if error is None:compare(expected,expected,image,matrix,np)
    image,options=fixture(spec,np)
    import skimagexx as xx
    from skimagexx import _compat
    from skimage.color import colorconv
    assert xx.get_num_threads()==1
    old=xx.set_fallback_enabled(False);prior=sys.getprofile();events=[];entries=[];forbidden=[]
    def deny(*a,**k):forbidden.append(True);raise AssertionError('upstream color forwarding')
    def profile(frame,event,arg):
        if event in('c_call','c_return')and getattr(arg,'__module__',None)==xx._core.__name__ and getattr(arg,'__name__',None)==STAGE:
            (entries if event=='c_call'else events).append(STAGE)
    try:
        xx.set_num_threads(workers)
        with monkeypatch.context()as patch:
            patch.setattr(original,'rgb2yuv',deny);patch.setattr(colorconv,'rgb2yuv',deny);patch.setattr(colorconv,'_convert',deny)
            patch.setattr(_compat,'fallback',deny)
            sys.setprofile(profile)
            try:actual,actual_error,actual_messages,actual_checks=observe(xx.color.rgb2yuv,image,options,spec['policy'],np)
            finally:sys.setprofile(prior)
        (tmp_path/'candidate.json').write_text(json.dumps(dict(id=spec['id'],workers=workers,result=result_record(actual),error=actual_error,warnings=actual_messages,checks=actual_checks,C_entries=entries,C_returns=events,forbidden=forbidden))+'\n')
        assert not forbidden and actual_error==error and actual_messages==messages and all(actual_checks.values())
        assert len(entries)==int(spec['kind']!='invalid-shape')
        if error is None:assert len(events)==1
        # A source FP error may arise inside the stage or in scalar status replay.
        assert len(events)<=len(entries)
        if error is None:compare(actual,expected,image,matrix,np)
        assert xx.get_num_threads()==workers and _compat._fallback_enabled is False
    finally:sys.setprofile(prior);xx.set_num_threads(1);xx.set_fallback_enabled(old)

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference-only',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert a.reference_only and sys.flags.optimize==0 and not any(n=='skimagexx'or n.startswith('skimagexx.')for n in sys.modules)
    a.output.mkdir();np,original,matrix,record=runtime();(a.output/'runtime.json').write_text(json.dumps(record,indent=2));returns=errors=0
    with(a.output/'journal.jsonl').open('x')as journal:
        for spec in CASES:
            image,options=fixture(spec,np);expected,error,messages,checks=observe(original.rgb2yuv,image,options,spec['policy'],np)
            journal.write(json.dumps(dict(id=spec['id'],spec=spec,input=result_record(image),result=result_record(expected),error=error,warnings=messages,checks=checks,limits=None if error else list(limits(image,expected,matrix,np))))+'\n');journal.flush()
            assert all(checks.values())
            if error:errors+=1
            else:compare(expected,expected,image,matrix,np);returns+=1
    assert not any(n=='skimagexx'or n.startswith('skimagexx.')for n in sys.modules)
    (a.output/'summary.json').write_text(json.dumps(dict(cases=len(CASES),returns=returns,errors=errors,candidate_calls=0)))
    print('original-only',len(CASES),'PASS',returns,errors,flush=True)
if __name__=='__main__':main()
