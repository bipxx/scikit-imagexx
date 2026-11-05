"""Actual shipping defaults against saved unit ellipsoid words, no originals."""
import base64
import inspect
import json
from pathlib import Path
import sys
import zlib
import pytest

API='draw.ellipsoid'
DTYPES=('uint8','float32','float64')
ROOT=Path(__file__).parents[1]

def test_ellipsoid_saved_sdk_oracle_metadata_without_science():
    import hashlib
    import struct
    raw=(ROOT/'tests/cmake_consumer/ellipsoid_source_oracle.bin').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='8c9967242b2bb5ddfb66eeb551ad8a9367c0fc446a12efff16a98d48a39290ec'
    assert struct.unpack_from('<I',raw)[0]==19
    assert '8c9967242b2bb5ddfb66eeb551ad8a9367c0fc446a12efff16a98d48a39290ec' in (ROOT/'tests/cmake_consumer/ellipsoid_source_oracle.cmake').read_text()

def saved_array(np,levelset=False):
    matrix=json.loads((ROOT/'coverage/contracts/ellipsoid.json').read_bytes())
    row=next(r for r in matrix['original_observations'][np.__version__]
             if r['id']==f'ellipsoid/unit-levelset-{levelset}')
    data=row['result'];result=np.empty(data['shape'],dtype=data['dtype'])
    result.flat[:]=np.frombuffer(zlib.decompress(base64.b64decode(data['raw_zlib_base64'])),dtype=data['dtype'])
    return result

def test_ellipsoid_registered_public_default_scope():
    import skimagexx as xx
    from skimagexx._draw_ellipsoid import ellipsoid
    from skimagexx.coverage import native_functions,native_dtypes,execution_kind,parity_tolerance
    assert API in native_functions() and native_dtypes(API)==DTYPES and execution_kind(API)=='native'
    assert xx.draw.ellipsoid is ellipsoid and 'ellipsoid' in xx.draw.__all__
    assert str(inspect.signature(ellipsoid))=='(a, b, c, spacing=(1.0, 1.0, 1.0), levelset=False)'
    with pytest.raises(ValueError,match='exact array words'):parity_tolerance(API,'float64')

@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('workers',(1,4))
def test_ellipsoid_actual_default_saved_words(dtype,workers,monkeypatch):
    import numpy as np
    import skimagexx as xx
    import skimage.draw.draw3d as original
    from skimagexx._draw_ellipsoid import ellipsoid
    from skimagexx.coverage import default_call,assert_default_parity
    carrier=np.arange(64).reshape(8,8)[:,::-1].astype(dtype)
    carrier=carrier.view(np.dtype(dtype,metadata={'role':'unused descriptor carrier'}))
    before=carrier.tobytes(),carrier.shape,carrier.strides,carrier.dtype,carrier.ctypes.data
    expected=saved_array(np);returns=[];accepted=[];snapshots=[]
    monkeypatch.setattr(original,'ellipsoid',lambda *a,**k:pytest.fail('original forbidden'))
    entry,getter=xx._core.draw_ellipsoid,xx._core.get_num_threads
    def snapshot():value=getter();snapshots.append(value);return value
    def profile(frame,event,arg):
        if event=='c_return' and arg is entry:returns.append(True)
        if event=='return' and frame.f_code is ellipsoid.__code__:accepted.append(frame.f_locals.get('native_result') is not None)
    old=xx.get_num_threads();xx.set_num_threads(workers);previous=sys.getprofile()
    try:
        monkeypatch.setattr(xx._core,'get_num_threads',snapshot)
        args,options=default_call(API,carrier);assert args==(1,1,1) and options=={}
        sys.setprofile(profile)
        try:actual=xx.draw.ellipsoid(*args,**options)
        finally:sys.setprofile(previous)
        assert inspect.isbuiltin(entry) and returns==accepted==[True] and snapshots==[workers]
        assert_default_parity(API,dtype,actual,expected,image=carrier)
        assert actual is not expected and not np.shares_memory(actual,carrier)
    finally:xx.set_num_threads(old)
    assert before==(carrier.tobytes(),carrier.shape,carrier.strides,carrier.dtype,carrier.ctypes.data)

@pytest.mark.parametrize('levelset',(False,True))
@pytest.mark.parametrize('checker',('exact','shared'))
@pytest.mark.parametrize('fault',('word','dtype','view','layout','missing','container'))
def test_ellipsoid_checker_rejects_corruption(levelset,checker,fault):
    import numpy as np
    from skimagexx.coverage import ellipsoid_default_error,assert_default_parity
    expected=saved_array(np,levelset);actual=expected.copy();carrier=np.zeros((8,8))
    if fault=='word':
        if levelset:actual.flat[0]=np.nextafter(actual.flat[0],np.inf)
        else:actual.flat[0]=not actual.flat[0]
    elif fault=='dtype':actual=actual.astype('float32' if levelset else 'uint8')
    elif fault=='view':actual=actual.view()
    elif fault=='layout':actual=np.array(actual,order='F')
    elif fault=='missing':actual=actual[:-1].copy()
    else:actual=[actual]
    with pytest.raises(AssertionError):
        if checker=='exact':ellipsoid_default_error(carrier,actual,expected)
        else:assert_default_parity(API,'uint8',actual,expected,image=carrier)
