"""Actual bounded component; original rows/types stay immutable and distinct."""
import gc
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import weakref
import pytest
if __package__:
    from .regionprops_state_contract import cases
    from .regionprops_state_observer import observe
else:
    from regionprops_state_contract import cases
    from regionprops_state_observer import observe
ROOT=Path(__file__).parents[1]
MATRIX=ROOT/'coverage/contracts/regionprops-state-v1.json'
CASES=cases()


@pytest.fixture(scope='module')
def expected_rows():
    matrix=json.loads(MATRIX.read_bytes())
    return {p:{r['id']:r for r in rows}for p,rows in matrix['original_observations'].items()}


def test_regionprops_state_explicit_type_and_scope_policy():
    from skimagexx import _regionprops_state as owned
    from skimage.measure._regionprops import RegionProperties as provider
    policy=json.loads((ROOT/'coverage/contracts/regionprops-state-frontend-v1.json').read_bytes())['type_policy']
    assert owned.RegionProperties is not provider and issubclass(owned.RegionProperties,provider)
    assert [provider.__module__,provider.__qualname__]==policy['original_qualified_type']
    assert [owned.RegionProperties.__module__,owned.RegionProperties.__qualname__]==policy['backend_qualified_type']
    assert str(inspect.signature(owned.regionprops))==str(inspect.signature(importlib.import_module('skimage.measure._regionprops').regionprops))
    assert all(inspect.isbuiltin(getattr(owned._core,'regionprops_state_'+name)) for name in ('prepare','image','population','coords'))


@pytest.mark.parametrize('spec',CASES,ids=[s['id']for s in CASES])
@pytest.mark.parametrize('workers',(1,4))
def test_regionprops_state_saved_rows_exact_except_declared_distinct_class(spec,workers,monkeypatch,expected_rows):
    import numpy as np
    import skimagexx as xx
    from skimagexx import _regionprops_state as owned
    expected=expected_rows[np.__version__][spec['id']]
    entries={getattr(owned._core,'regionprops_state_'+name):name for name in ('prepare','image','population','coords')}
    returns=[];witnesses=[]
    codes={owned.RegionProperties.image.fget.__wrapped__.__code__,owned.RegionProperties.coords.fget.__code__,owned.RegionProperties.num_pixels.fget.__wrapped__.__code__,owned.regionprops.__code__}
    def profiler(frame,event,value):
        if event=='c_return'and value in entries:returns.append(entries[value])
        if event=='return'and frame.f_code in codes and 'native_result'in frame.f_locals:witnesses.append(frame.f_locals['native_result']is not None)
    def denied(*args,**kwargs):raise AssertionError('admitted region state invoked an original constructor/image/count/coords algorithm')
    old=xx.get_num_threads();old_profile=sys.getprofile();xx.set_num_threads(workers)
    module=importlib.import_module('skimage.measure._regionprops');public=importlib.import_module('skimage.measure');ndi=importlib.import_module('scipy.ndimage')
    try:
        with monkeypatch.context()as patch:
            patch.setattr(module,'regionprops',denied);patch.setattr(public,'regionprops',denied);patch.setattr(ndi,'find_objects',denied);patch.setattr(owned,'_provider_regionprops',denied)
            for name in ('image','num_pixels','coords'):patch.setattr(owned._Provider,name,property(denied))
            sys.setprofile(profiler)
            try:actual=observe(owned.regionprops,spec,np)
            finally:sys.setprofile(old_profile)
        # Raw rows are retained as observed. No golden/type canonicalization.
        for key in expected:
            if key!='region_types':assert actual[key]==expected[key],(spec['id'],key)
        assert expected['region_types']==[['skimage.measure._regionprops','RegionProperties']for _ in actual['regions']]
        assert actual['region_types']==[['skimagexx._regionprops_state','RegionProperties']for _ in actual['regions']]
        if spec['rejection']is None or spec['rejection']=='spacing':assert returns.count('prepare')==1 and any(witnesses)
        else:assert not returns
        if actual['regions']:assert set(returns)=={'prepare','image','population','coords'}and all(witnesses)
        folder=Path(os.environ['SKIMAGEXX_REGIONPROPS_ROWS_DIR']);folder.mkdir(parents=True,exist_ok=True)
        (folder/(spec['id'].split('/')[-1]+'-workers'+str(workers)+'.json')).write_text(json.dumps(actual,indent=2,allow_nan=False)+'\n')
    finally:sys.setprofile(old_profile);xx.set_num_threads(old)


def test_regionprops_state_original_owner_and_cached_mask_lifetimes():
    import numpy as np
    from skimagexx import _regionprops_state as owned
    image=np.array([[0,1,1],[0,1,0]],dtype=np.int16);owner=weakref.ref(image)
    regions=owned.regionprops(image);region=regions[0];mask=region.image
    del image,regions;gc.collect();assert owner()is not None
    assert region.num_pixels==3
    mask[mask]=False
    assert region.coords.shape==(0,2)and region.num_pixels==3
    # Cached scalar is intentionally stale; it is provider-compatible cache behavior.
    del region,mask;gc.collect();assert owner()is None


def test_regionprops_state_changed_original_descriptor_uses_explicit_property_tail(monkeypatch):
    import numpy as np
    from skimagexx import _regionprops_state as owned
    image=np.array([[0,1,1],[0,1,0]],dtype=np.int16);region=owned.regionprops(image,cache=False)[0]
    image.shape=(3,2)
    expected=image[region.slice]==region.label
    assert np.array_equal(region.image,expected)
    # Stale native geometry is never dereferenced after this ordinary metadata change.
    prepared=region._native_state
    out=np.empty_like(expected,dtype=bool,order='K')
    assert owned._core.regionprops_state_image(prepared,0,image,out)is None
