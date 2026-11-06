"""In-place default verification uses complete independent mutated images."""
import base64
import inspect
import json
from pathlib import Path
import sys
import zlib
import pytest

ROOT=Path(__file__).parents[1]
DTYPES=('float32','float64')


def saved(np,dtype):
    data=json.loads((ROOT/'coverage/contracts/draw-set-color.json').read_bytes())
    key='float32_original_observations' if dtype=='float32' else 'observations'
    row=next(r for r in data[key][np.__version__] if r['id']==('f32-gray' if dtype=='float32' else 'gray'))
    values=[]
    for phase in ('before','after'):
        value=row[phase]['items'][0]['items'][0]
        result=np.empty(value['shape'],dtype=value['dtype'])
        result.flat[:]=np.frombuffer(zlib.decompress(base64.b64decode(value['raw_zlib_base64'])),dtype=value['dtype'])
        values.append(result)
    return values


def test_set_color_default_saved_contract_without_science():
    data=json.loads((ROOT/'coverage/contracts/draw-set-color.json').read_bytes())
    for version in ('1.24.4','2.5.4'):
        for key,name in (('observations','gray'),('float32_original_observations','f32-gray')):
            row=next(r for r in data[key][version] if r['id']==name)
            assert row['error'] is None
            assert row['before']['items'][0]['items'][0]['raw_sha256']!=row['after']['items'][0]['items'][0]['raw_sha256']
    text=(ROOT/'python/skimagexx/coverage.py').read_text()
    assert 'assert result is None' in text and 'def set_color_default_result(' in text


@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('workers',(1,4))
def test_set_color_actual_shipping_default(dtype,workers,monkeypatch):
    import numpy as np
    import skimagexx as xx
    from skimagexx import _draw_set_color as owned
    from skimagexx.coverage import default_call,native_dtypes,native_functions,set_color_default_result,assert_default_parity,parity_tolerance
    assert 'draw.set_color' in native_functions() and native_dtypes('draw.set_color')==DTYPES
    assert xx.draw.set_color is owned.set_color and str(inspect.signature(owned.set_color))=='(image, coords, color, alpha=1)'
    namespace={};exec('from skimagexx.draw import *',namespace);assert namespace['set_color'] is owned.set_color
    carrier=np.arange(64,dtype=dtype).reshape(8,8);before=carrier.tobytes()
    saved_before,expected=saved(np,dtype);args,options=default_call('draw.set_color',carrier)
    assert args[0].tobytes()==saved_before.tobytes() and args[0].shape==saved_before.shape
    entry=xx._core.draw_set_color;getter=xx._core.get_num_threads;calls=[];snapshots=[];accepted=[]
    def denied(*a,**k):raise AssertionError('original/provider forbidden for saved default')
    monkeypatch.setattr(owned,'fallback',denied)
    def trace(frame,event,value):
        if event=='c_return' and value is entry:calls.append(True)
        if event=='c_return' and value is getter:snapshots.append(True)
        if event=='return' and frame.f_code is owned.set_color.__code__:accepted.append(frame.f_locals.get('native_result'))
    old=xx.get_num_threads();fallback=xx.set_fallback_enabled(False);profile=sys.getprofile();xx.set_num_threads(workers)
    try:
        sys.setprofile(trace)
        try:actual=set_color_default_result(xx.draw.set_color,args,options)
        finally:sys.setprofile(profile)
        assert calls==snapshots==accepted==[True]
        assert_default_parity('draw.set_color',dtype,actual,expected,image=carrier)
        with pytest.raises(ValueError,match='mutated images'):parity_tolerance('draw.set_color',dtype)
    finally:xx.set_num_threads(old);xx.set_fallback_enabled(fallback)
    assert carrier.tobytes()==before and args[0].tobytes()==saved_before.tobytes()


@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('fault',('none','self','view','word','dtype','shape','layout','input-alias','unmutated'))
def test_set_color_default_checker_rejects_corruption(dtype,fault):
    import numpy as np
    from skimagexx.coverage import assert_default_parity
    before,expected=saved(np,dtype);actual=expected.copy();carrier=np.zeros((8,8),dtype=dtype)
    if fault=='none':actual=expected=None
    elif fault=='self':actual=expected
    elif fault=='view':actual=actual.view()
    elif fault=='word':actual.flat[0]+=1
    elif fault=='dtype':actual=actual.astype(np.float64 if dtype=='float32' else np.float32)
    elif fault=='shape':actual=actual.reshape(-1)
    elif fault=='layout':actual=np.asfortranarray(actual)
    elif fault=='input-alias':carrier=actual
    elif fault=='unmutated':actual=before
    with pytest.raises(AssertionError):assert_default_parity('draw.set_color',dtype,actual,expected,image=carrier)


def test_set_color_observation_requires_none_return():
    import numpy as np
    from skimagexx.coverage import set_color_default_result
    image=np.zeros((2,2));before=image.copy()
    with pytest.raises(AssertionError):set_color_default_result(lambda *a,**k:np.zeros((2,2)),(image,),{})
    assert np.array_equal(image,before)
