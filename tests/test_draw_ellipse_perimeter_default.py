"""Shipping ellipse defaults from saved original words; no original algorithm calls."""
import base64,inspect,json,sys,zlib
from pathlib import Path
import pytest
ROOT=Path(__file__).parents[1]
API='draw.ellipse_perimeter'
DTYPES=('uint8','float32','float64')
def saved(np,case='ellipse-perimeter/unequal'):
    matrix=json.loads((ROOT/'coverage/contracts/ellipse-perimeter.json').read_bytes())
    row=next(r for r in matrix['original_observations'][np.__version__] if r['id']==case)
    out=[]
    for value in row['result']['items']:
        a=np.empty(value['shape'],dtype=value['dtype'])
        a.flat[:]=np.frombuffer(zlib.decompress(base64.b64decode(value['raw_zlib_base64'])),dtype=value['dtype'])
        out.append(a)
    return tuple(out)
def test_ellipse_perimeter_saved_sdk_oracle_metadata_without_science():
    import hashlib,struct
    raw=(ROOT/'tests/cmake_consumer/ellipse_perimeter_fixtures.bin').read_bytes()
    pin='079da2ae19560e01b5e098a6a12121046479392ced66d6572ff648b2448f5772'
    assert hashlib.sha256(raw).hexdigest()==pin and struct.unpack_from('<I',raw)[0]==23
    assert pin in (ROOT/'tests/cmake_consumer/ellipse_perimeter_fixtures.cmake').read_text()
def test_ellipse_perimeter_registered_default():
    import skimagexx as xx
    from skimagexx._draw_ellipse_perimeter import ellipse_perimeter
    from skimagexx.coverage import native_functions,native_dtypes,parity_tolerance
    assert API in native_functions() and native_dtypes(API)==DTYPES
    assert xx.draw.ellipse_perimeter is ellipse_perimeter and 'ellipse_perimeter' in xx.draw.__all__
    namespace={};exec('from skimagexx.draw import *',namespace)
    assert namespace['ellipse_perimeter'] is ellipse_perimeter
    assert str(inspect.signature(ellipse_perimeter))=='(r, c, r_radius, c_radius, orientation=0, shape=None)'
    with pytest.raises(ValueError,match='exact coordinate'):parity_tolerance(API,'float64')
@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('workers',(1,4))
def test_ellipse_perimeter_actual_default_saved_words(dtype,workers,monkeypatch):
    import numpy as np,skimagexx as xx
    import skimage.draw.draw as original
    from skimagexx.coverage import default_call,assert_default_parity
    carrier=np.arange(64).reshape(8,8)[:,::-1].astype(dtype)
    carrier=carrier.view(np.dtype(dtype,metadata={'role':'descriptor only'}));before=carrier.tobytes(),carrier.dtype,carrier.strides
    expected=saved(np);entry,getter=xx._core.draw_ellipse_perimeter,xx._core.get_num_threads
    call=xx.draw.ellipse_perimeter;returns=[];accepted=[];snapshots=[]
    def refused(*a,**k):raise AssertionError('original forbidden')
    monkeypatch.setattr(original,'_ellipse_perimeter',refused)
    def snapshot():v=getter();snapshots.append(v);return v
    def profile(frame,event,arg):
        if event=='c_return' and arg is entry:returns.append(True)
        if event=='return' and frame.f_code is call.__code__:accepted.append(frame.f_locals.get('native_result') is not None)
    old=xx.get_num_threads();xx.set_num_threads(workers);previous=sys.getprofile()
    try:
        monkeypatch.setattr(xx._core,'get_num_threads',snapshot)
        args,kwargs=default_call(API,carrier);assert args==(0,0,2,4) and kwargs=={}
        sys.setprofile(profile)
        try:actual=call(*args,**kwargs)
        finally:sys.setprofile(previous)
        assert inspect.isbuiltin(entry) and returns==accepted==[True] and snapshots==[workers]
        assert_default_parity(API,dtype,actual,expected,image=carrier)
    finally:xx.set_num_threads(old)
    assert before==(carrier.tobytes(),carrier.dtype,carrier.strides)
@pytest.mark.parametrize('case',('ellipse-perimeter/unequal','ellipse-perimeter/point','ellipse-perimeter/clip-away'))
@pytest.mark.parametrize('fault',('word','dtype','view','missing','container','same-object','cross-object','pair-alias'))
def test_ellipse_perimeter_exact_checker_rejects_corruption(case,fault):
    import numpy as np
    from skimagexx.coverage import assert_default_parity
    expected=saved(np,case);actual=tuple(v.copy() for v in expected);carrier=np.zeros((8,8))
    assert_default_parity(API,'uint8',actual,expected,image=carrier)
    if fault=='word':
        if not actual[0].size:return
        actual[0].flat[0]+=1
    elif fault=='dtype':actual=(actual[0].astype(np.int32),actual[1])
    elif fault=='view':actual=(actual[0].view(),actual[1])
    elif fault=='missing':actual=(np.empty((1,),np.intp),actual[1])
    elif fault=='container':actual=list(actual)
    elif fault=='same-object':actual=expected
    elif fault=='cross-object':actual=(expected[0],actual[1])
    else:actual=(actual[0],actual[0])
    with pytest.raises(AssertionError):assert_default_parity(API,'uint8',actual,expected,image=carrier)
