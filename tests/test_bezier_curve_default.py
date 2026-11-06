"""Shipping defaults decoded from once-only original observations."""
import base64
import inspect
import json
from pathlib import Path
import sys
import zlib
import pytest

ROOT=Path(__file__).parents[1]
DTYPES=('uint8','float32','float64')


def saved(np):
    data=json.loads((ROOT/'coverage/contracts/draw-bezier-v1.json').read_bytes())
    row=next(r for r in data['original_observations'][np.__version__] if r['id']=='bezier/monotone')
    values=[]
    for value in row['result']['items']:
        result=np.empty(value['shape'],dtype=value['dtype'])
        result.flat[:]=np.frombuffer(zlib.decompress(base64.b64decode(value['raw_zlib_base64'])),dtype=value['dtype'])
        values.append(result)
    return tuple(values)


def test_bezier_default_saved_contract_without_science():
    data=json.loads((ROOT/'coverage/contracts/draw-bezier-v1.json').read_bytes())
    for version in ('1.24.4','2.5.4'):
        row=next(r for r in data['original_observations'][version] if r['id']=='bezier/monotone')
        assert row['error'] is None
        assert [v['shape'] for v in row['result']['items']]==[[8],[8]]
    assert "if api=='draw.bezier_curve':return (0,0,1,4,6,6,1.),{}" in (ROOT/'python/skimagexx/coverage.py').read_text()


@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('workers',(1,4))
def test_bezier_actual_shipping_default(dtype,workers,monkeypatch):
    import numpy as np
    import skimagexx as xx
    from skimagexx import _bezier_curve as owned
    from skimagexx.coverage import default_call,native_dtypes,native_functions,assert_default_parity
    assert 'draw.bezier_curve' in native_functions() and native_dtypes('draw.bezier_curve')==DTYPES
    assert xx.draw.bezier_curve is owned.bezier_curve
    namespace={};exec('from skimagexx.draw import *',namespace);assert namespace['bezier_curve'] is owned.bezier_curve
    assert str(inspect.signature(owned.bezier_curve))=='(r0, c0, r1, c1, r2, c2, weight, shape=None)'
    carrier=np.arange(64,dtype=dtype).reshape(8,8);before=carrier.tobytes()
    expected=saved(np);entry=xx._core.draw_bezier_curve;getter=xx._core.get_num_threads
    calls=[];snapshots=[];accepted=[]
    def trace(frame,event,value):
        if event=='c_return' and value is entry:calls.append(True)
        if event=='c_return' and value is getter:snapshots.append(True)
        if event=='return' and frame.f_code is owned.bezier_curve.__code__:accepted.append(frame.f_locals.get('native_result') is not None)
    old=xx.get_num_threads();fallback=xx.set_fallback_enabled(False);profile=sys.getprofile();xx.set_num_threads(workers)
    try:
        args,options=default_call('draw.bezier_curve',carrier)
        sys.setprofile(trace)
        try:actual=xx.draw.bezier_curve(*args,**options)
        finally:sys.setprofile(profile)
        assert calls==snapshots==accepted==[True]
        assert_default_parity('draw.bezier_curve',dtype,actual,expected,image=carrier)
    finally:xx.set_num_threads(old);xx.set_fallback_enabled(fallback)
    assert carrier.tobytes()==before


@pytest.mark.parametrize('fault',('self','view','word','dtype','container','missing','pair-alias'))
def test_bezier_default_checker_rejects_corruption(fault):
    import numpy as np
    from skimagexx.coverage import assert_default_parity
    expected=saved(np);actual=[v.copy() for v in expected];carrier=np.zeros((8,8))
    if fault=='self':actual[0]=expected[0]
    elif fault=='view':actual[0]=actual[0].view()
    elif fault=='word':actual[0][0]+=1
    elif fault=='dtype':actual[0]=actual[0].astype(np.int32)
    elif fault=='missing':actual[0]=actual[0][:-1]
    elif fault=='pair-alias':actual[1]=actual[0]
    with pytest.raises(AssertionError):assert_default_parity('draw.bezier_curve','uint8',actual if fault=='container' else tuple(actual),expected,image=carrier)
