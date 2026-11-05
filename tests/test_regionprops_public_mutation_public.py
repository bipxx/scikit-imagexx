"""Public mutation correction; original golds and failed diagnostics immutable."""
import ast
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import pytest
if __package__:
    from .regionprops_public_mutation_contract import cases
    from .regionprops_public_mutation_observer import observe
else:
    from regionprops_public_mutation_contract import cases
    from regionprops_public_mutation_observer import observe
ROOT = Path(__file__).parents[1]
CASES = cases()


@pytest.fixture(scope='module')
def reference():
    return json.loads((ROOT/'coverage/contracts/regionprops-public-mutation.json').read_bytes())


def test_public_mutation_geometry_guard_uses_identity_and_real_public_export():
    import numpy as np
    import skimagexx as xx
    from skimagexx import _regionprops_state as owned
    assert xx.measure.regionprops is owned.regionprops
    assert str(inspect.signature(owned.regionprops)) == '(label_image, intensity_image=None, cache=True, *, extra_properties=None, spacing=None, offset=None)'
    source = (ROOT/'python/skimagexx/_regionprops_state_class.py').read_text(encoding='utf-8')
    cls = next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name == 'RegionProperties')
    helper = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == '_current_native_geometry')
    assert all(isinstance(op,(ast.Is,ast.IsNot)) for n in ast.walk(helper) if isinstance(n,ast.Compare) for op in n.ops)
    region = xx.measure.regionprops(np.array([[0,1,1],[0,1,0]],dtype=np.int64),cache=False)[0]
    label, slices = region.label, region.slice
    prepared = region._current_native_geometry()
    assert prepared is not None and region._native_initial_label is label and region._native_initial_slice is slices
    region.label = 2
    assert region._current_native_geometry() is None
    region.label = label
    assert region._current_native_geometry() is prepared
    region.slice = tuple(slice(s.start,s.stop,s.step) for s in slices)
    assert region.slice == slices and region.slice is not slices
    assert region._current_native_geometry() is None
    region.slice = slices
    assert region._current_native_geometry() is prepared


@pytest.mark.parametrize('spec',CASES,ids=[s['id'] for s in CASES])
@pytest.mark.parametrize('workers',(1,4))
def test_regionprops_public_mutation_saved_pipeline(spec,workers,reference,monkeypatch):
    import numpy as np
    import skimagexx as xx
    from skimagexx import _regionprops_state as owned
    assert xx.measure.regionprops is owned.regionprops
    expected = next(r for r in reference['original_observations'][np.__version__] if r['id'] == spec['id'])
    entries = {getattr(owned._core,'regionprops_state_'+name):name for name in ('prepare','image','population','coords')}
    assert all(inspect.isbuiltin(entry) for entry in entries)
    returns,geometry = [],[]
    tails = dict(image=0,coords=0)
    old_image,old_coords = owned._Provider.image,owned._Provider.coords
    def image_tail(region):
        tails['image'] += 1
        return old_image.fget(region)
    def coords_tail(region):
        tails['coords'] += 1
        return old_coords.fget(region)
    def denied(*args,**options):
        raise AssertionError('mutation path called original constructor/find_objects')
    def profile(frame,event,value):
        if event == 'c_return' and value in entries:
            returns.append(entries[value])
        if event == 'return' and frame.f_code is owned.RegionProperties._current_native_geometry.__code__:
            geometry.append(value is not None)
    old_workers,old_fallback,old_profile = xx.get_num_threads(),xx.set_fallback_enabled(False),sys.getprofile()
    xx.set_num_threads(workers)
    original = importlib.import_module('skimage.measure._regionprops')
    try:
        with monkeypatch.context() as patch:
            patch.setattr(original,'regionprops',denied)
            patch.setattr(importlib.import_module('skimage.measure'),'regionprops',denied)
            patch.setattr(importlib.import_module('scipy.ndimage'),'find_objects',denied)
            patch.setattr(owned,'_provider_regionprops',denied)
            patch.setattr(owned._Provider,'image',property(image_tail))
            patch.setattr(owned._Provider,'coords',property(coords_tail))
            sys.setprofile(profile)
            try:
                actual = observe(xx.measure.regionprops,spec,np)
            finally:
                sys.setprofile(old_profile)
        for key in expected:
            if key != 'region_type':
                assert actual[key] == expected[key],(spec['id'],key)
        assert expected['region_type'] == ['skimage.measure._regionprops','RegionProperties']
        assert actual['region_type'] == ['skimagexx._regionprops_state','RegionProperties']
        assert returns.count('prepare') == 1 and 'population' in returns
        assert returns.count('coords') == int(spec['prime'])
        assert tails['coords'] == 2 and False in geometry
        assert (tails['image'] == 0) is (spec['cache'] and spec['prime'])
        assert xx.get_num_threads() == workers
        folder = Path(os.environ.get('SKIMAGEXX_REGION_PUBLIC_MUTATION_ROWS_DIR',ROOT/'build/region-public-mutation-rows'))
        folder.mkdir(parents=True,exist_ok=True)
        row = dict(actual=actual,native_C_returns=returns,geometry_native_accepted=geometry,
                   original_property_tails=tails,workers=workers,scope='Changedgeometry usesexplicitproviderproperties, notnativegeometrycredit')
        with (folder/(spec['id'].split('/')[-1]+'-workers'+str(workers)+'.json')).open('w',encoding='utf-8') as output:
            json.dump(row,output,indent=2,allow_nan=False)
            output.write('\n')
    finally:
        sys.setprofile(old_profile)
        xx.set_num_threads(old_workers)
        xx.set_fallback_enabled(old_fallback)
        assert xx.get_num_threads() == old_workers
