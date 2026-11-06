"""Actual polygon builtin component, immutable saved words; no new originals."""
import importlib
import inspect
import json
from pathlib import Path
import sys
import pytest
from draw_polygon_contract import cases,APIS
from draw_polygon_observer import observe
ROOT=Path(__file__).parents[1]
MATRIX=json.loads((ROOT/'coverage/contracts/draw-polygon-v1.json').read_bytes())

@pytest.mark.parametrize('spec',cases(),ids=lambda s:s['id'])
@pytest.mark.parametrize('workers',[1,4])
def test_draw_polygon_saved_public_pipeline(spec,workers,monkeypatch):
    import numpy as np,skimagexx as xx
    from skimagexx import _draw_polygon as owned
    assert xx.draw.polygon is owned.polygon and xx.draw.polygon2mask is owned.polygon2mask
    expected=next(r for r in MATRIX['original_observations'][np.__version__]if r['id']==spec['id'])
    actual_calls=[];accepted=[];budgets=[]
    prepare=xx._core.draw_polygon_prepare
    coords=xx._core.draw_polygon_coords_into;mask=xx._core.draw_polygon_mask_into
    assert all(inspect.isbuiltin(f)and f.__module__==xx._core.__name__ for f in (prepare,coords,mask))
    entries={prepare:'prepare',coords:'coords',mask:'mask'};getter=xx._core.get_num_threads
    def snapshot():value=getter();budgets.append(value);return value
    def profile(frame,event,value):
        if event=='c_return'and value in entries:actual_calls.append(entries[value])
        if event=='return'and frame.f_globals.get('__name__')==owned.__name__ and 'native_result'in frame.f_locals:
            accepted.append(frame.f_locals['native_result']is not None)
    def denied(*a,**k):raise AssertionError('admitted polygon route called original numerical provider')
    modules=[(importlib.import_module('skimage.draw'),'polygon'),(importlib.import_module('skimage.draw'),'polygon2mask'),(importlib.import_module('skimage.draw.draw'),'polygon'),(importlib.import_module('skimage.draw._draw'),'_polygon'),(importlib.import_module('skimage.draw._polygon2mask'),'polygon2mask')]
    old_budget=getter();old_fallback=xx.set_fallback_enabled(False);prior=sys.getprofile();xx.set_num_threads(workers)
    try:
        with monkeypatch.context()as patch:
            for module,name in modules:patch.setattr(module,name,denied)
            patch.setattr(xx._core,'get_num_threads',snapshot);sys.setprofile(profile)
            try:actual=observe(getattr(owned,spec['api'].split('.')[-1]),spec,np)
            finally:sys.setprofile(prior)
        assert {k:v for k,v in expected.items()if k!='call_accounting'}==actual
        if expected['error']is None:
            assert actual_calls==['prepare','coords'if spec['api']==APIS[0]else'mask']
            assert accepted==[True]and budgets==[workers]
        elif spec['id'].endswith('/negative-shape'):
            assert actual_calls==['prepare']and budgets==[workers]
        else:assert actual_calls==[]and budgets==[]
        assert getter()==workers
    finally:sys.setprofile(prior);xx.set_num_threads(old_budget);xx.set_fallback_enabled(old_fallback)


def test_draw_polygon_public_signatures_and_aliases():
    import skimagexx as xx
    from skimagexx import _draw_polygon as owned
    assert str(inspect.signature(owned.polygon))=='(r, c, shape=None)'
    assert str(inspect.signature(owned.polygon2mask))=='(image_shape, polygon)'
    assert xx.draw.polygon is owned.polygon and xx.draw.polygon2mask is owned.polygon2mask
    assert xx.draw.__all__.count('polygon')==xx.draw.__all__.count('polygon2mask')==1


def test_draw_polygon_plan_owns_raster_after_input_lifetime_and_private_guards():
    import numpy as np,skimagexx as xx
    core=xx._core;r=np.array([0.,0.,2.]);c=np.array([0.,2.,0.]);plan=core.draw_polygon_prepare(r,c,(0,0),(2,2),1)
    assert plan is not None and plan.count==6
    with pytest.raises(AttributeError):plan.count=1
    r[:]=c[:]=100;del r,c
    rr=np.empty(plan.count,np.intp);cc=np.empty(plan.count,np.intp)
    assert core.draw_polygon_coords_into(plan,rr,cc,1)is True
    assert rr.tolist()==[0,0,0,1,1,2]and cc.tolist()==[0,1,2,0,1,0]
    with pytest.raises(ValueError,match='independent'):core.draw_polygon_coords_into(plan,rr,rr,1)
    with pytest.raises(ValueError,match='fresh owning'):core.draw_polygon_coords_into(plan,rr[:],cc,1)
    mask=np.full((3,3),True);assert core.draw_polygon_mask_into(plan,mask,1)is True
    assert mask.tolist()==[[True,True,True],[True,True,False],[True,False,False]]
    with pytest.raises(ValueError):core.draw_polygon_mask_into(plan,mask[:,::-1],1)

@pytest.mark.parametrize('api',APIS)
def test_draw_polygon_protocol_provider_boundary_once(api,monkeypatch):
    import numpy as np,skimagexx as xx
    from skimagexx import _draw_polygon as owned
    from skimagexx._compat import FallbackWarning
    monkeypatch.setattr(importlib.import_module('skimagexx._compat'),'_fallback_enabled',True)
    events=[];token=object()
    class ArrayProvider:
        def __init__(self,value,name):self.value=value;self.name=name
        def __array__(self,dtype=None,copy=None):events.append(self.name);return np.asarray(self.value,dtype=dtype)
    if api==APIS[0]:
        module=importlib.import_module('skimage.draw._draw')
        def provider(r,c,shape):events.append('provider');assert r.shape==c.shape==(3,);return token
        monkeypatch.setattr(module,'_polygon',provider)
        monkeypatch.setattr(importlib.import_module('skimage.draw'),'polygon',provider)
        with pytest.warns(FallbackWarning):
            assert owned.polygon(ArrayProvider([0,0,2],'r'),ArrayProvider([0,2,0],'c'))is token
        assert events==['r','c','provider']
    else:
        module=importlib.import_module('skimage.draw._polygon2mask')
        def provider(shape,polygon):events.append('provider');assert polygon.shape==(3,2);return token
        monkeypatch.setattr(module,'polygon2mask',provider)
        monkeypatch.setattr(importlib.import_module('skimage.draw'),'polygon2mask',provider)
        with pytest.warns(FallbackWarning):
            assert owned.polygon2mask((3,3),ArrayProvider([[0,0],[0,2],[2,0]],'polygon'))is token
        assert events==['polygon','provider']

@pytest.mark.parametrize('kind',['unequal','short-tuple','short-array','mask-short-array'])
def test_draw_polygon_unsafe_boundaries_never_forward_original(kind,monkeypatch):
    import numpy as np
    from skimagexx import _draw_polygon as owned
    def denied(*a,**k):raise AssertionError('unsafe original forwarded')
    monkeypatch.setattr(importlib.import_module('skimage.draw._draw'),'_polygon',denied)
    monkeypatch.setattr(importlib.import_module('skimage.draw._polygon2mask'),'polygon2mask',denied)
    with pytest.raises(NotImplementedError):
        if kind=='unequal':owned.polygon([0,1],[0,1,2])
        elif kind=='short-tuple':owned.polygon([0,1],[0,1],())
        elif kind=='short-array':owned.polygon([0,1],[0,1],np.array([2]))
        else:owned.polygon2mask(np.array([2]),[[0,0],[0,1],[1,0]])


def test_draw_polygon_readonly_view_fails_before_column_cast_or_worker_lookup(monkeypatch):
    import numpy as np
    from skimagexx import _draw_polygon as owned
    polygon=np.asfortranarray([[0.,0.],[0.,2.],[2.,0.]]);polygon.flags.writeable=False
    actual=np.ascontiguousarray;calls=[]
    def cast(a,*args,**kwargs):calls.append(a);return actual(a,*args,**kwargs)
    def denied():raise AssertionError('readonly error reached native getter')
    monkeypatch.setattr(np,'ascontiguousarray',cast);monkeypatch.setattr(owned._core,'get_num_threads',denied)
    with pytest.raises(ValueError,match='^buffer source array is read-only$'):owned.polygon2mask((3,3),polygon)
    assert len(calls)==1 and not calls[0].flags.writeable


def test_draw_polygon_declined_plan_preserves_original_normalized_dtype(monkeypatch):
    import numpy as np,skimagexx as xx
    from skimagexx import _draw_polygon as owned
    value=2**53+1
    rows=np.array([value,value,value+2],dtype=np.uint64)
    columns=np.array([0,2,0],dtype=np.uint64)
    before=(rows.tobytes(),columns.tobytes());token=object();calls=[]
    def provider(r,c,shape):
        calls.append('provider')
        assert r is rows and c is columns and shape is None
        assert r.dtype==c.dtype==np.dtype(np.uint64)
        assert (r.tobytes(),c.tobytes())==before
        assert int(r[0])==value
        return token
    monkeypatch.setattr(importlib.import_module('skimage.draw._draw'),'_polygon',provider)
    monkeypatch.setattr(importlib.import_module('skimage.draw'),'polygon',provider)
    monkeypatch.setattr(importlib.import_module('skimagexx._compat'),'_fallback_enabled',True)
    from skimagexx._compat import FallbackWarning
    entry=xx._core.draw_polygon_prepare;returned=[];prior=sys.getprofile()
    def trace(frame,event,value):
        if event=='c_return'and value is entry:returned.append(True)
    try:
        sys.setprofile(trace)
        with pytest.warns(FallbackWarning):
            actual=owned.polygon(rows,columns)
    finally:sys.setprofile(prior)
    assert actual is token and calls==['provider']and returned==[True]
    assert (rows.tobytes(),columns.tobytes())==before


@pytest.mark.parametrize('api',APIS)
def test_draw_polygon_provider_obeys_disabled_fallback(api,monkeypatch):
    import numpy as np
    from skimagexx import _draw_polygon as owned
    calls=[]
    class ArrayProvider:
        def __array__(self,dtype=None,copy=None):
            calls.append('convert')
            return np.asarray([[0,0],[0,2],[2,0]] if api==APIS[1] else [0,0,2],dtype=dtype)
    def denied(*args,**kwargs):raise AssertionError('strict provider entered original/native/getter')
    for module,name in [('skimage.draw','polygon'),('skimage.draw','polygon2mask'),('skimage.draw.draw','polygon'),('skimage.draw._draw','_polygon'),('skimage.draw._polygon2mask','polygon2mask')]:
        monkeypatch.setattr(importlib.import_module(module),name,denied)
    for name in ('get_num_threads','draw_polygon_prepare','draw_polygon_coords_into','draw_polygon_mask_into'):
        monkeypatch.setattr(owned._core,name,denied)
    monkeypatch.setattr(importlib.import_module('skimagexx._compat'),'_fallback_enabled',False)
    with pytest.raises(NotImplementedError,match='fallback is disabled'):
        if api==APIS[0]:owned.polygon(ArrayProvider(),[0,2,0])
        else:owned.polygon2mask((3,3),ArrayProvider())
    assert calls==['convert']


@pytest.mark.parametrize('api',['draw.polygon','draw.polygon2mask'])
@pytest.mark.parametrize('workers',[1,4])
def test_draw_polygon_tagged_public_pipeline(api,workers,monkeypatch):
    from test_draw_polygon_dtype_tag_reference import tag_fixture
    observer=importlib.import_module('draw_polygon_observer')
    matrix=json.loads((ROOT/'coverage/contracts/draw-polygon-dtype-tag-v1.json').read_bytes())
    monkeypatch.setattr(observer,'fixture',tag_fixture)
    monkeypatch.setattr(sys.modules[__name__],'MATRIX',matrix)
    spec=next(s for s in matrix['cases']if s['api']==api)
    test_draw_polygon_saved_public_pipeline(spec,workers,monkeypatch)

@pytest.mark.parametrize('kind',['float32','swapped-float64','rank2'])
def test_draw_polygon_private_prepare_rejects_incompatible_storage(kind):
    import numpy as np,skimagexx as xx
    rows=np.array([0,0,2],dtype=np.float64);columns=np.array([0,2,0],dtype=np.float64)
    if kind=='float32':rows=rows.astype(np.float32)
    elif kind=='swapped-float64':rows=rows.astype(np.dtype(np.float64).newbyteorder('S'))
    else:rows=rows.reshape(1,3)
    before=(rows.tobytes(),columns.tobytes())
    assert xx._core.draw_polygon_prepare(rows,columns,(0,0),(2,2),1)is None
    assert before==(rows.tobytes(),columns.tobytes())
