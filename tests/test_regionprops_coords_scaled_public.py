"""Saved coords_scaled rows and source-ordered native/provider boundaries."""
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import types
import pytest
from test_regionprops_coords_scaled_reference import CASES,observe

ROOT=Path(__file__).parents[1]
REFERENCE=json.loads((ROOT/'coverage/contracts/regionprops-coords-scaled-reference-v1.json').read_bytes())
STAGES=('local','origin_into','multiply_f64','add_f64_into')


@pytest.mark.parametrize('spec',CASES,ids=lambda value:value['id'])
@pytest.mark.parametrize('workers',(1,4))
def test_coords_scaled_saved_rows_and_actual_four_stages(spec,workers,monkeypatch):
    import numpy as np,skimagexx as xx
    from skimagexx import _regionprops_state as owned
    from skimagexx import _regionprops_state_class as cls
    expected=next(row for row in REFERENCE['original_observations'][np.__version__]if row['id']==spec['id'])
    entries={getattr(xx._core,'regionprops_state_coords_scaled_'+name):name for name in STAGES}
    assert all(inspect.isbuiltin(call)for call in entries)
    returns=[];accepted=[];snapshots=[];events=[]
    getter=xx._core.get_num_threads;image=cls.RegionProperties.image.fget
    def snapshot():
        value=getter();snapshots.append(value);events.append('workers');return value
    def image_once(self):events.append('image');return image(self)
    def denied(*args,**options):raise AssertionError('owned coords_scaled used original algorithm/provider')
    code=cls.RegionProperties.coords_scaled.fget.__code__
    def profile(frame,event,value):
        if event=='c_return'and value in entries:returns.append(entries[value])
        if event=='return'and frame.f_code is code:accepted.append(frame.f_locals.get('native_result')is not None)
    previous=sys.getprofile();budget=getter();fallback=xx.set_fallback_enabled(False);xx.set_num_threads(workers)
    try:
        original=importlib.import_module('skimage.measure._regionprops')
        monkeypatch.setattr(original,'regionprops',denied)
        monkeypatch.setattr(importlib.import_module('skimage.measure'),'regionprops',denied)
        monkeypatch.setattr(importlib.import_module('scipy.ndimage'),'find_objects',denied)
        monkeypatch.setattr(owned,'_provider_regionprops',denied)
        monkeypatch.setattr(cls._Provider,'coords_scaled',property(denied))
        monkeypatch.setattr(np,'argwhere',denied)
        monkeypatch.setattr(cls.RegionProperties,'image',property(image_once))
        monkeypatch.setattr(xx._core,'get_num_threads',snapshot)
        sys.setprofile(profile)
        try:actual=observe(owned.regionprops,spec,np)
        finally:sys.setprofile(previous)
        output=Path(os.environ.get('SKIMAGEXX_COORDS_SCALED_ROWS_DIR',ROOT/'build/regionprops-coords-scaled-rows'))
        output.mkdir(parents=True,exist_ok=True)
        filename=f"{np.__version__}-{int(bool(xx._core.has_avx2()))}-{workers}-{spec['id'].split('/')[-1]}.json"
        (output/filename).write_bytes((json.dumps(dict(row=actual,C_returns=returns,full_native=accepted,worker_snapshots=snapshots,events=events),indent=2)+'\n').encode())
        assert set(actual)==set(expected)-{'call_accounting'}
        for key in actual:
            if key!='region_type':assert actual[key]==expected[key],(spec['id'],key)
        assert actual['region_type']==['skimagexx._regionprops_state','RegionProperties']
        assert expected['region_type']==['skimage.measure._regionprops','RegionProperties']
        assert issubclass(cls.RegionProperties,cls._Provider)and cls.RegionProperties is not cls._Provider
        integer=spec['spacing_kind']=='tuple-int'
        assert accepted==[not integer]*3 and snapshots==[workers]*3
        assert returns==list(STAGES if not integer else STAGES[:3])*3
        assert events.count('image')==3+sum(a=='prime'for a,v in spec['sequence'])
        assert all(events[i-1]=='image'for i,value in enumerate(events)if value=='workers')
    finally:
        sys.setprofile(previous);xx.set_num_threads(budget);xx.set_fallback_enabled(fallback)


def test_coords_scaled_provider_argwhere_owner_is_not_modified(monkeypatch):
    import numpy as np
    from skimagexx import _core
    from skimagexx._regionprops_state_class import RegionProperties
    held=np.array([[0,0],[1,1]],dtype=np.intp,order='F');before=held.tobytes();calls=[]
    obj=types.SimpleNamespace(image=np.ones((2,2),bool),slice=(slice(2,4),slice(3,5)),_ndim=2,_spacing=np.ones(2),_offset=np.zeros(2,dtype=np.intp))
    monkeypatch.setattr(_core,'regionprops_state_coords_scaled_local',lambda *args:None)
    def provider(mask):calls.append(mask);return held
    def denied(*args):raise AssertionError('provider-owned argwhere array reached inplace origin')
    monkeypatch.setattr(np,'argwhere',provider)
    monkeypatch.setattr(_core,'regionprops_state_coords_scaled_origin_into',denied)
    actual=RegionProperties.coords_scaled.fget(obj)
    assert calls==[obj.image]and held.flags.owndata and held.tobytes()==before
    assert actual.tolist()==[[2.,3.],[3.,4.]]and not np.shares_memory(actual,held)


def test_coords_scaled_provider_multiply_owner_is_not_modified(monkeypatch):
    import numpy as np
    from skimagexx import _core
    from skimagexx._regionprops_state_class import RegionProperties
    held=np.array([[8.,9.]],order='F');before=held.tobytes();calls=[]
    class Spacing:
        def __array_ufunc__(self,ufunc,method,*args,**options):
            assert ufunc is np.multiply and method=='__call__';calls.append(ufunc);return held
    obj=types.SimpleNamespace(image=np.ones((1,1),bool),slice=(slice(0,1),slice(0,1)),_ndim=2,_spacing=Spacing(),_offset=np.array([4,5],dtype=np.intp))
    def denied(*args):raise AssertionError('provider-owned multiply array reached inplace add')
    monkeypatch.setattr(_core,'regionprops_state_coords_scaled_add_f64_into',denied)
    actual=RegionProperties.coords_scaled.fget(obj)
    assert calls==[np.multiply]and held.flags.owndata and held.tobytes()==before
    assert actual.tolist()==[[12.,14.]]and not np.shares_memory(actual,held)


def test_coords_scaled_offset_is_read_after_eventful_multiply():
    import numpy as np
    from skimagexx._regionprops_state_class import RegionProperties
    events=[]
    class Object:
        image=np.ones((1,1),bool);slice=(slice(0,1),slice(0,1));_ndim=2
        _spacing=np.array([np.inf,1.]);_offset=np.zeros(2,dtype=np.intp)
        def __getattribute__(self,name):
            if name=='_offset':events.append('offset')
            return object.__getattribute__(self,name)
    obj=Object()
    def handler(message,flags):
        events.append((message,flags));obj._offset=np.array([4,7],dtype=np.intp)
    with np.errstate(all='ignore',invalid='call',call=handler):actual=RegionProperties.coords_scaled.fget(obj)
    assert events==[('invalid value',8),'offset']and np.isnan(actual[0,0])and actual[0,1]==7


@pytest.mark.parametrize('bad',('negative','max','min','outside'))
def test_coords_scaled_private_origin_refuses_before_writes(bad):
    import numpy as np
    from skimagexx import _core
    value={'negative':-1,'max':np.iinfo(np.intp).max,'min':np.iinfo(np.intp).min,'outside':2}[bad]
    coordinates=np.array([[value,0]],dtype=np.intp,order='F');before=coordinates.tobytes()
    assert _core.regionprops_state_coords_scaled_origin_into(coordinates,np.array([1,1],dtype=np.intp),(2,2),1)is None
    assert coordinates.tobytes()==before


@pytest.mark.parametrize('kind',('metadata','opposite-endian','float32-subnormal','nonfinite','protocol'))
def test_coords_scaled_private_coefficient_refusal_keeps_owned_array(kind):
    import numpy as np
    from skimagexx import _core
    values=np.array([[1.,2.],[3.,4.]],order='F');before=values.tobytes()
    if kind=='metadata':offset=np.array([1.,2.],dtype=np.dtype('f8',metadata={'tag':'unassessed'}))
    elif kind=='opposite-endian':offset=np.array([1.,2.],dtype=np.dtype('f8').newbyteorder('S'))
    elif kind=='float32-subnormal':offset=np.array([1,0],dtype=np.uint32).view(np.float32)
    elif kind=='nonfinite':offset=np.array([np.inf,0.])
    else:
        class Protocol:
            def __array__(self,*args,**kwargs):raise AssertionError('private boundary replayed conversion')
        offset=Protocol()
    assert _core.regionprops_state_coords_scaled_add_f64_into(values,offset,1)is None
    assert values.tobytes()==before


def test_coords_scaled_private_stage_lifetime_empty_layout_and_alias_refusal():
    import gc,weakref
    import numpy as np
    from skimagexx import _core
    mask=np.array([[1,0],[0,1]],bool,order='F');owner=weakref.ref(mask)
    indices,shape=_core.regionprops_state_coords_scaled_local(mask,1);del mask;gc.collect()
    assert owner()is None and shape==(2,2)and indices.tolist()==[[0,0],[1,1]]
    original=indices.tobytes();alias=indices[:,0]
    assert _core.regionprops_state_coords_scaled_origin_into(indices,alias,shape,1)is None
    assert indices.tobytes()==original
    empty,shape=_core.regionprops_state_coords_scaled_local(np.zeros((2,3),bool),1)
    assert empty.shape==(0,2)and empty.strides==(0,0)and empty.base is None and empty.flags.owndata
    assert empty.flags.c_contiguous and empty.flags.f_contiguous
    result=_core.regionprops_state_coords_scaled_multiply_f64(empty,np.ones(2),1)
    assert result.shape==(0,2)and result.strides==(0,0)and result.base is None and result.flags.owndata
