"""Test-first normalized/Hu contracts; no production registration here."""
import argparse
import base64
import csv
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
import warnings
import zlib

import pytest
if __package__:
    from .moments_derived_cases import APIS, STAGES, UNASSESSED, cases, error_limits, fixture
    from .test_moments_public import original_runtime, metadata, snapshot
else:
    from moments_derived_cases import APIS, STAGES, UNASSESSED, cases, error_limits, fixture
    from test_moments_public import original_runtime, metadata, snapshot

CASES=cases()
CYTHON_SOURCE_SHA256="0e55b3a0470b715315128bbfa8d4799571363dd866e42667bb5fdbdb15c700f8"


def runtime():
    np,original,record=original_runtime()
    binary=Path(original._moments_cy.__file__)
    distribution=importlib.metadata.distribution("scikit-image")
    relative=binary.relative_to(Path(distribution.locate_file("")))
    entries=list(csv.reader(distribution.read_text("RECORD").splitlines()))
    entry=next(row for row in entries if row[0]==relative.as_posix())
    algorithm,encoded=entry[1].split("=",1);assert algorithm=="sha256"
    raw=binary.read_bytes();digest=hashlib.sha256(raw).digest()
    assert base64.urlsafe_b64encode(digest).decode().rstrip("=")==encoded and len(raw)==int(entry[2])
    record.update(cython_binary=dict(path=str(binary),sha256=digest.hex(),record=entry),
                  audited_cython_source_sha256=CYTHON_SOURCE_SHA256)
    return np,original,record


def observe(call,image,options,policy,np):
    before=snapshot((image,options),np)
    prior,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize()
    result,error=None,None
    with np.errstate(**policy):
        selected=np.geterr()
        with warnings.catch_warnings(record=True)as caught:
            warnings.simplefilter("always")
            try:result=call(image,**options)
            except Exception as exc:error=(type(exc).__module__,type(exc).__name__,str(exc))
        inside=np.geterr()==selected and np.geterrcall()is handler and np.getbufsize()==buffer
    checks=dict(state_unchanged=inside and np.geterr()==prior and np.geterrcall()is handler and np.getbufsize()==buffer,
                input_unchanged=snapshot((image,options),np)==before)
    messages=[(type(w.message).__module__,type(w.message).__name__,str(w.message))for w in caught]
    return result,error,messages,checks


def result_record(value):
    if value is None:return None
    info=list(metadata(value));info[3]=None if info[3] is None else dict(info[3])
    return dict(metadata=info,raw_c_zlib_base64=base64.b64encode(zlib.compress(value.tobytes(order="C"))).decode())


def compare(actual,expected,bound,spec,np):
    assert metadata(actual)==metadata(expected)
    if expected.dtype.kind not in "fc":np.testing.assert_array_equal(actual,expected);return
    components=((actual.real,expected.real),(actual.imag,expected.imag))if expected.dtype.kind=="c"else((actual,expected),)
    for a,e in components:
        for mask in (np.isnan,np.isposinf,np.isneginf):np.testing.assert_array_equal(mask(a),mask(e))
        finite=np.isfinite(e);assert np.all(np.abs(a[finite]-e[finite])<=bound[finite]),spec["id"]
        exact=(bound==0)&finite
        np.testing.assert_array_equal(a[exact],e[exact]);np.testing.assert_array_equal(np.signbit(a[exact]),np.signbit(e[exact]))
    if spec["api"]=="moments_normalized":
        visited=np.zeros(expected.shape,dtype=bool)
        for powers in __import__('itertools').product(range(spec["order"]+1),repeat=expected.ndim):
            visited[powers]=True
            if sum(powers)<2:assert np.isnan(actual[powers]) and np.isnan(expected[powers])
        np.testing.assert_array_equal(actual[~visited],expected[~visited])
        np.testing.assert_array_equal(np.signbit(actual[~visited]),np.signbit(expected[~visited]))


@pytest.fixture(scope="module")
def reference_runtime():return runtime()


@pytest.fixture(scope="module")
def worker_results():return {}


@pytest.mark.parametrize("api",APIS)
def test_derived_moments_signature(reference_runtime,api):
    _,original,_=reference_runtime
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    assert "measure."+api in native_functions(),"derived moments remain pending"
    assert inspect.signature(getattr(xx.measure,api))==inspect.signature(getattr(original,api))
    assert getattr(xx.measure,api)is not getattr(original,api)


@pytest.mark.parametrize("spec",CASES,ids=[s["id"]for s in CASES])
@pytest.mark.parametrize("workers",(1,4))
def test_derived_moments_owned(reference_runtime,spec,workers,worker_results,monkeypatch):
    np,original,_=reference_runtime
    image,options,_=fixture(spec,np)
    expected,expected_error,expected_warnings,expected_checks=observe(getattr(original,spec["api"]),image,options,spec["policy"],np)
    assert all(expected_checks.values())
    with np.errstate(all="ignore"):bound=None if expected_error else error_limits(spec,image,expected,np)
    image,options,_=fixture(spec,np)
    import skimage.measure as public_original
    import skimagexx as xx
    from skimagexx import _compat
    from skimagexx.coverage import native_functions
    assert "measure."+spec["api"]in native_functions(),"derived moments remain pending"
    assert xx.get_num_threads()==1
    previous=xx.set_fallback_enabled(False);old_profile=sys.getprofile();completions=[];forbidden=[]
    def deny(*args,**kwargs):forbidden.append(True);raise AssertionError("upstream derived moment forwarding")
    def profile(frame,event,arg):
        if event=="c_return"and getattr(arg,"__module__",None)==xx._core.__name__ and getattr(arg,"__name__",None)==STAGES[spec["api"]]:completions.append(STAGES[spec["api"]])
    try:
        xx.set_num_threads(workers)
        with monkeypatch.context()as patch:
            for name in APIS:patch.setattr(public_original,name,deny);patch.setattr(original,name,deny)
            patch.setattr(original._moments_cy,"moments_hu",deny)
            sys.setprofile(profile)
            try:actual,error,messages,checks=observe(getattr(xx.measure,spec["api"]),image,options,spec["policy"],np)
            finally:sys.setprofile(old_profile)
        assert not forbidden and all(checks.values())
        assert error==expected_error and messages==expected_warnings,spec["id"]
        if error is None:
            if spec["api"]=="moments_hu"or spec["order"]>=2:assert completions,"successful arithmetic must complete owned stage"
            compare(actual,expected,bound,spec,np)
        words=(None if error else(metadata(actual),actual.tobytes(order="C")),error,messages)
        assert worker_results.setdefault(spec["id"],words)==words
        assert xx.get_num_threads()==workers and _compat._fallback_enabled is False
    finally:sys.setprofile(old_profile);xx.set_num_threads(1);xx.set_fallback_enabled(previous)


def test_derived_intake_metadata():
    assert len(CASES)==76 and len({s["id"]for s in CASES})==76
    assert all(s["shape"]!=(3,3)for s in CASES if s["api"]=="moments_hu")
    assert UNASSESSED and all(n>=0 for s in CASES for n in s["shape"])


def reference_only(output):
    assert not any(n=="skimagexx"or n.startswith("skimagexx.")for n in sys.modules)
    output.mkdir();np,original,record=runtime();(output/"runtime.json").write_text(json.dumps(record,indent=2)+"\n")
    returns=0;errors=0
    with(output/"journal.jsonl").open("x")as stream:
        for spec in CASES:
            image,options,_=fixture(spec,np)
            with(output/"phase.json").open("w")as phase:json.dump(dict(id=spec["id"],phase="before-original"),phase)
            result,error,messages,checks=observe(getattr(original,spec["api"]),image,options,spec["policy"],np)
            row=dict(id=spec["id"],api=spec["api"],input=result_record(image),result=result_record(result),error=error,warnings=messages,checks=checks)
            stream.write(json.dumps(row,allow_nan=False)+"\n");stream.flush()
            assert all(checks.values())
            if error:errors+=1
            else:
                returns+=1
                with np.errstate(all="ignore"):bound=error_limits(spec,image,result,np)
                assert np.all(np.isfinite(bound[np.isfinite(result)]))and np.all(bound>=0)
    (output/"summary.json").write_text(json.dumps(dict(status="ORIGINAL_ONLY",cases=76,returns=returns,errors=errors,native_coverage=0,unassessed=UNASSESSED),indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--reference-only",action="store_true",required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();reference_only(args.output)
