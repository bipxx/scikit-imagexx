"""Actual builtin scalar stages, immutable returned-region observations."""
import importlib
import inspect
import json
from pathlib import Path
import sys
import pytest
from regionprops_bbox_extent_contract import cases
from regionprops_bbox_extent_observer import observe
ROOT=Path(__file__).parents[1]
MATRIX=json.loads((ROOT/'coverage/contracts/regionprops-bbox-extent.json').read_bytes())
CASES=cases()


@pytest.mark.parametrize('spec',CASES,ids=[s['id'] for s in CASES])
@pytest.mark.parametrize('workers',(1,4))
def test_bbox_extent_saved_public_rows(spec,workers,monkeypatch):
    import numpy as np,skimagexx as xx
    from skimagexx import _regionprops_state as owned
    from skimagexx import _regionprops_state_class as cls
    assert xx.measure.regionprops is owned.regionprops
    expected=next(r for r in MATRIX['original_observations'][np.__version__] if r['id']==spec['id'])
    entries={getattr(xx._core,'regionprops_state_'+n):n for n in ('prepare','population','scaled_count_f64','extent_f64')}
    assert all(inspect.isbuiltin(f) for f in entries)
    returns=[];accepted=[]
    def profile(frame,event,value):
        if event=='c_return' and value in entries:returns.append(entries[value])
        if event=='return' and frame.f_globals.get('__name__')==cls.__name__ and 'native_result' in frame.f_locals:
            accepted.append(frame.f_locals['native_result'] is not None)
    def denied(*a,**k):raise AssertionError('admitted scalar route called original numerical provider')
    original=importlib.import_module('skimage.measure._regionprops');public=importlib.import_module('skimage.measure');ndi=importlib.import_module('scipy.ndimage')
    old_workers,old_fallback,prior=xx.get_num_threads(),xx.set_fallback_enabled(False),sys.getprofile();xx.set_num_threads(workers)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(original,'regionprops',denied);patch.setattr(public,'regionprops',denied);patch.setattr(ndi,'find_objects',denied);patch.setattr(owned,'_provider_regionprops',denied)
            for name in ('area','area_bbox','extent'):patch.setattr(owned._Provider,name,property(denied))
            patch.setattr(np,'sum',denied)
            sys.setprofile(profile)
            try:actual=observe(owned.regionprops,spec,np)
            finally:sys.setprofile(prior)
        for key in expected:
            if key!='region_type':assert actual[key]==expected[key],(spec['id'],key)
        if expected['constructor_error'] is None:
            assert expected['region_type']==['skimage.measure._regionprops','RegionProperties']
            assert actual['region_type']==['skimagexx._regionprops_state','RegionProperties']
            assert owned.RegionProperties is not owned._Provider and issubclass(owned.RegionProperties,owned._Provider)
            assert 'scaled_count_f64' in returns and 'extent_f64' in returns and accepted and all(accepted)
        else:assert actual['region_type'] is None and not accepted
        assert returns.count('prepare')==1 and xx.get_num_threads()==workers
    finally:sys.setprofile(prior);xx.set_num_threads(old_workers);xx.set_fallback_enabled(old_fallback)


def test_bbox_extent_area_cache_uses_current_mask_not_cached_num_pixels():
    import numpy as np
    from skimagexx import _regionprops_state as owned
    region=owned.regionprops(np.array([[0,1,1],[0,1,0]],np.int16))[0]
    mask=region.image;assert region.num_pixels==3
    mask[:]=False
    assert type(region.area) is np.float64 and region.area==0
    assert region.area_bbox==4 and region.extent==0
    mask[:]=True
    assert region.area==0 and region.area_bbox==4 and region.num_pixels==3


def test_bbox_extent_dependency_order_and_callback_after_visible_operands(monkeypatch):
    import numpy as np
    from skimagexx import _regionprops_state as owned
    region=owned.regionprops(np.array([[1]],np.int16),cache=False)[0];events=[]
    masks=[np.ones((2,2),bool),np.ones((0,2),bool)]
    def image(self):events.append('image');return masks.pop(0)
    monkeypatch.setattr(owned.RegionProperties,'image',property(image))
    def handler(message,flags):events.append((message,flags))
    with np.errstate(all='call',call=handler):value=region.extent
    assert value==np.inf and type(value) is np.float64
    assert events==['image','image',('divide by zero',1)] and not masks


def test_bbox_extent_provider_frontiers_reuse_operands_and_exact_types(monkeypatch):
    import numpy as np
    from skimagexx import _regionprops_state as owned
    from skimagexx import _regionprops_state_class as cls
    assert owned._core.regionprops_state_scaled_count_f64(3,2.) is None
    assert owned._core.regionprops_state_extent_f64(np.float32(1),np.float64(2)) is None
    region=owned.regionprops(np.array([[1]],np.int16),cache=False)[0];events=[]
    def image(self):events.append('image');return np.ones((2,2),bool)
    monkeypatch.setattr(owned.RegionProperties,'image',property(image));region._pixel_area=2.
    assert type(region.area) is np.float64 and type(region.area_bbox) is float
    assert events==['image','image']
    def forbidden(*a):raise AssertionError('print/log native policy credit')
    monkeypatch.setattr(owned._core,'regionprops_state_extent_f64',forbidden)
    with np.errstate(all='ignore',invalid='print'):
        assert cls._extent(np.float64(1),np.float64(2))==np.float64(.5)


def test_bbox_extent_native_boxing_status_and_f64_frontier():
    import numpy as np
    from skimagexx import _core
    value,status=_core.regionprops_state_extent_f64(np.float64(0),np.float64(0))
    assert type(value) is np.float64 and value.tobytes().hex()=='000000000000f8ff' and status==1
    value,status=_core.regionprops_state_scaled_count_f64(3,np.float64(.5))
    assert type(value) is np.float64 and value==1.5 and status==0
    assert _core.regionprops_state_scaled_count_f64(2**53+1,np.float64(1)) is None
    assert _core.regionprops_state_extent_f64(np.float64(-1),np.float64(2)) is None
