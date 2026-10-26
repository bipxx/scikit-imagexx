"""Small ordinary batch/fallback parity packet; frozen family budgets reused."""
import inspect
import itertools
import json
import sys
import pytest
if __package__:
    from .moments_derived_cases import error_limits
    from .test_moments_derived_public import runtime,observe,compare,result_record
else:
    from moments_derived_cases import error_limits
    from test_moments_derived_public import runtime,observe,compare,result_record

KINDS=("accepted","zero-mass","negative-mass","large-mass","small-mass",
       "tiny-entry","huge-entry","nan-entry")
CASES=[dict(id=dtype+"-"+kind,dtype=dtype,kind=kind)for dtype in ("float32","float64")for kind in KINDS]


@pytest.fixture(scope="module")
def reference_runtime():return runtime()


def fixture(spec,np):
    image=np.arange(1,17,dtype=spec["dtype"]).reshape(4,4).copy()
    image[0,0]=4;image[0,1]=image[1,0]=0
    kind=spec["kind"]
    if kind=="zero-mass":image[0,0]=0
    elif kind=="negative-mass":image[0,0]=-4
    elif kind=="large-mass":image[0,0]=1e7
    elif kind=="small-mass":image[0,0]=1e-7
    elif kind=="tiny-entry":image[3,3]=1e-13
    elif kind=="huge-entry":image[3,3]=1e13
    elif kind=="nan-entry":image[3,3]=np.nan
    assert type(image)is np.ndarray and image.flags.owndata and image.flags.c_contiguous
    return image


@pytest.mark.parametrize("spec",CASES,ids=[s["id"]for s in CASES])
def test_default_batch_and_numeric_fallback(reference_runtime,spec,monkeypatch,tmp_path):
    np,original,_=reference_runtime
    frozen=dict(api="moments_normalized",exact=False,order=3,spacing=None)
    image=fixture(spec,np)
    expected,error,messages,checks=observe(original.moments_normalized,image,{}, {},np)
    # Retain the genuine original observation before any candidate call/assert.
    original_row=dict(id=spec["id"],phase="original",input=result_record(image),result=result_record(expected),error=error,warnings=messages,checks=checks)
    (tmp_path/"original.json").write_text(json.dumps(original_row)+"\n")
    assert error is None and all(checks.values())
    with np.errstate(all="ignore"):bound=error_limits(frozen,image,expected,np)
    image=fixture(spec,np)
    import skimage.measure as public_original
    import skimagexx as xx
    from skimagexx import _moments_derived as implementation,_compat
    assert xx.get_num_threads()==1 and inspect.isbuiltin(xx._core.moments_normalize)
    old_policy=xx.set_fallback_enabled(False);old_profile=sys.getprofile()
    returns=dict(batch_probe=0,scalar_divide=0);batch={};forbidden=[]
    def deny(*args,**kwargs):forbidden.append(True);raise AssertionError("upstream normalization forwarding")
    def trace(frame,event,arg):
        if event=="c_return"and getattr(arg,"__module__",None)==xx._core.__name__ and getattr(arg,"__name__",None)=="moments_normalize":
            if frame.f_code is implementation.moments_normalized.__code__:returns["batch_probe"]+=1
            elif frame.f_code is implementation._divide.__code__:returns["scalar_divide"]+=1
            else:raise AssertionError("unclassified normalization return")
        if event=="return"and frame.f_code is implementation.moments_normalized.__code__:
            probe=frame.f_locals.get("batch")
            batch.update(accepted=probe is not None,returned_probe=arg is probe,storage_checked=frame.f_locals.get("storage_checked"))
    try:
        with monkeypatch.context()as patch:
            patch.setattr(public_original,"moments_normalized",deny);patch.setattr(original,"moments_normalized",deny)
            sys.setprofile(trace)
            try:actual,actual_error,actual_messages,actual_checks=observe(xx.measure.moments_normalized,image,{}, {},np)
            finally:sys.setprofile(old_profile)
        row=dict(id=spec["id"],phase="candidate",result=result_record(actual),error=actual_error,warnings=actual_messages,checks=actual_checks,batch=batch,actual_C_returns=returns,forbidden=forbidden)
        (tmp_path/"candidate.json").write_text(json.dumps(row)+"\n")
        assert not forbidden and actual_error==error and actual_messages==messages and all(actual_checks.values())
        compare(actual,expected,bound,frozen,np)
        accepted=spec["kind"]=="accepted"
        assert batch==dict(accepted=accepted,returned_probe=accepted,storage_checked=True)
        assert returns["batch_probe"]==1
        assert returns["scalar_divide"]==(0 if accepted else sum(sum(p)>1 for p in itertools.product(range(4),repeat=2)))
        assert xx.get_num_threads()==1 and _compat._fallback_enabled is False
    finally:sys.setprofile(old_profile);xx.set_fallback_enabled(old_policy)
