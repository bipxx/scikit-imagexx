"""Shipping triangle defaults from saved words; originals are never called here."""
import base64,inspect,json,sys,zlib
from pathlib import Path
import pytest
ROOT=Path(__file__).parents[1]
APIS=('draw.polygon','draw.polygon2mask')
DTYPES=('uint8','float32','float64')


def saved(np,api,case=None):
    data=json.loads((ROOT/'coverage/contracts/draw-polygon-v1.json').read_bytes())
    row=next(r for r in data['original_observations'][np.__version__]if r['id']==(case or api+'/triangle-list'))
    assert row['error']is None
    def array(value):
        result=np.empty(value['shape'],dtype=value['dtype'])
        result.flat[:]=np.frombuffer(zlib.decompress(base64.b64decode(value['raw_zlib_base64'])),dtype=value['dtype'])
        return result
    return tuple(array(v)for v in row['result']['items'])if api=='draw.polygon'else array(row['result'])


def test_polygon_defaults_source_and_saved_contract_without_science():
    data=json.loads((ROOT/'coverage/contracts/draw-polygon-v1.json').read_bytes())
    assert data['original_calls']==48 and data['native_calls']==0
    for version in ('1.24.4','2.5.4'):
        for api in APIS:
            row=next(r for r in data['original_observations'][version]if r['id']==api+'/triangle-list')
            assert row['error']is None and all(row['checks'].values())
    coverage=(ROOT/'python/skimagexx/coverage.py').read_text()
    assert "if api=='draw.polygon':return ([1,1,4],[1,5,2]),{}"in coverage
    assert "if api=='draw.polygon2mask':return ((6,7),[[1,1],[1,5],[4,2]]),{}"in coverage
    assert '_DrawPolygonPlan'not in(ROOT/'python/skimagexx/draw.py').read_text()


@pytest.mark.parametrize('api',APIS)
def test_polygon_registered_default(api):
    import skimagexx as xx
    from skimagexx import _draw_polygon as owned
    from skimagexx.coverage import native_functions,native_dtypes,parity_tolerance
    name=api.split('.')[1];call=getattr(owned,name)
    assert api in native_functions()and native_dtypes(api)==DTYPES
    assert getattr(xx.draw,name)is call and name in xx.draw.__all__
    namespace={};exec('from skimagexx.draw import *',namespace);assert namespace[name]is call
    assert str(inspect.signature(call))=={'draw.polygon':'(r, c, shape=None)','draw.polygon2mask':'(image_shape, polygon)'}[api]
    with pytest.raises(ValueError,match='exact coordinate/bool'):parity_tolerance(api,'float64')


@pytest.mark.parametrize('api',APIS)
@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('workers',(1,4))
def test_polygon_actual_default_saved_words(api,dtype,workers,monkeypatch):
    import importlib
    import numpy as np,skimagexx as xx
    from skimagexx import _draw_polygon as owned
    from skimagexx.coverage import default_call,assert_default_parity
    carrier=np.arange(64).reshape(8,8)[:,::-1].astype(dtype)
    carrier=carrier.view(np.dtype(dtype,metadata={'role':'descriptor only'}));before=carrier.tobytes(),carrier.dtype,carrier.strides
    expected=saved(np,api);call=getattr(xx.draw,api.split('.')[1]);getter=xx._core.get_num_threads
    entries={xx._core.draw_polygon_prepare:'prepare',getattr(xx._core,'draw_polygon_coords_into'if api=='draw.polygon'else'draw_polygon_mask_into'):'fill'}
    returns=[];accepted=[];snapshots=[]
    def denied(*a,**kw):raise AssertionError('original polygon algorithm forbidden')
    for module,name in [('skimage.draw','polygon'),('skimage.draw','polygon2mask'),('skimage.draw.draw','polygon'),('skimage.draw._draw','_polygon'),('skimage.draw._polygon2mask','polygon2mask')]:monkeypatch.setattr(importlib.import_module(module),name,denied)
    monkeypatch.setattr(owned,'fallback',denied)
    def snapshot():value=getter();snapshots.append(value);return value
    def trace(frame,event,arg):
        if event=='c_return'and arg in entries:returns.append(entries[arg])
        if event=='return'and frame.f_code is call.__code__:accepted.append(frame.f_locals.get('native_result')is not None)
    previous=sys.getprofile();budget=xx.get_num_threads();old_fallback=xx.set_fallback_enabled(False);xx.set_num_threads(workers)
    try:
        monkeypatch.setattr(xx._core,'get_num_threads',snapshot)
        args,options=default_call(api,carrier);assert options=={}
        sys.setprofile(trace)
        try:actual=call(*args,**options)
        finally:sys.setprofile(previous)
        assert all(inspect.isbuiltin(e)for e in entries)and returns==['prepare','fill']and accepted==[True]and snapshots==[workers]
        assert_default_parity(api,dtype,actual,expected,image=carrier)
    finally:xx.set_num_threads(budget);xx.set_fallback_enabled(old_fallback)
    assert before==(carrier.tobytes(),carrier.dtype,carrier.strides)


@pytest.mark.parametrize('api,empty,fault',[(api,empty,fault)for api in APIS for empty in (False,True)for fault in ('dtype','view','same-object','metadata','missing','container','pair-alias','word')if not(api=='draw.polygon2mask'and fault in ('pair-alias','container'))and not(empty and fault=='word')])
def test_polygon_exact_default_checker_rejects_corruption(api,empty,fault):
    import numpy as np
    from skimagexx.coverage import assert_default_parity
    case=api+('/fully-negative'if api=='draw.polygon'else'/zero-shape')if empty else None
    expected=saved(np,api,case);values=list(expected)if api=='draw.polygon'else[expected]
    actual=[v.copy()for v in values];carrier=np.zeros((8,8))
    wrap=lambda result:tuple(result)if api=='draw.polygon'else result[0]
    assert_default_parity(api,'uint8',wrap(actual),expected,image=carrier)
    if fault=='dtype':actual[0]=actual[0].astype(np.int32 if api=='draw.polygon'else np.uint8)
    elif fault=='view':actual[0]=actual[0].view()
    elif fault=='same-object':actual[0]=values[0]
    elif fault=='metadata':actual[0]=actual[0].view(np.dtype(actual[0].dtype,metadata={'wrong':'tag'}))
    elif fault=='missing':actual[0]=actual[0][:-1]if not empty else actual[0].reshape((0,1)if api=='draw.polygon'else(0,))
    elif fault=='container':
        with pytest.raises(AssertionError):assert_default_parity(api,'uint8',actual,expected,image=carrier)
        return
    elif fault=='pair-alias':actual[1]=actual[0]
    elif fault=='word':actual[0].flat[0]=actual[0].flat[0]+1 if api=='draw.polygon'else not actual[0].flat[0]
    with pytest.raises(AssertionError):assert_default_parity(api,'uint8',wrap(actual),expected,image=carrier)
