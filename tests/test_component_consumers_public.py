"""Actual public component consumers; frozen assertions remain unchanged.

Sparse count safety checks are definition-level native evidence, separately
named; they never allocate an impractical original bincount/dense map.
"""
import importlib
import importlib.util
import inspect
import os
from pathlib import Path
import pickle
import subprocess
import sys
import warnings
import numpy as np
import pytest
import skimage
from skimage import morphology,segmentation
import skimagexx as xx
from skimagexx import _components as labels,_component_consumers as implementation
from skimagexx.coverage import native_functions,native_classes,class_metadata_providers

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('public_consumer_frozen_contract',ROOT/'tests/test_component_consumer_contracts.py')
contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
configuration=contract.configuration
api=contract.api
function=contract.function
relabel_function=contract.relabel_function
for _name in vars(contract):
    if _name.startswith('test_'):globals()[_name]=getattr(contract,_name)
del _name


@pytest.fixture(autouse=True)
def actual_public_gate():
    assert set(contract.APIS)<=set(native_functions())
    assert implementation._core is labels._core is xx._core
    assert xx._core.component_consumers_interface_version==1
    assert not any(k.startswith('SKIMAGEXX_') and any(v in k for v in ('CANDIDATE','BACKEND','MODULE_DIR')) for k in os.environ)
    for name in contract.APIS:
        module,attribute=name.split('.');candidate=getattr(getattr(xx,module),attribute)
        assert candidate is not contract.reference(name)
        assert candidate.__wrapped__ is getattr(implementation,attribute)
        assert attribute in getattr(xx,module).__all__
    if os.environ.get('SKIMAGEXX_DISABLE_SIMD'):assert not xx.has_avx2()


def pipeline(backend):
    image=np.zeros((11,13),bool);image[2:7,3:8]=True;image[4,5]=False
    image[:2,:2]=True;image[8,10]=True
    objects=backend.morphology.remove_small_objects(image,max_size=1)
    filled=backend.morphology.remove_small_holes(objects,max_size=1)
    cleared=backend.segmentation.clear_border(filled)
    components=backend.measure.label(cleared)
    return backend.segmentation.relabel_sequential(components,offset=3)


def test_component_pipeline_can_swap_backend():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');expected=pipeline(skimage);actual=pipeline(xx)
        np.testing.assert_array_equal(actual[0],expected[0])
        for got,want in zip(actual[1:],expected[1:]):
            np.testing.assert_array_equal(got.in_values,want.in_values)
            np.testing.assert_array_equal(got.out_values,want.out_values)
        np.testing.assert_array_equal(actual[2][actual[0]],expected[2][expected[0]])


def test_public_import_aliases_resolve_to_same_native_functions():
    misc=importlib.import_module('skimagexx.morphology.misc')
    assert misc.remove_small_objects is xx.morphology.remove_small_objects
    assert misc.remove_small_holes is xx.morphology.remove_small_holes
    from skimagexx.segmentation import clear_border,relabel_sequential
    assert clear_border is xx.segmentation.clear_border
    assert relabel_sequential is xx.segmentation.relabel_sequential
    image=np.array([0,2,2,0,3],np.int16)
    np.testing.assert_array_equal(misc.remove_small_objects(image,max_size=1),[0,2,2,0,0])


def test_public_consumers_and_returned_maps_never_forward_algorithms(monkeypatch):
    def reject(*args,**kwargs):raise AssertionError('upstream image algorithm or ArrayMap constructor called')
    for namespace,names in [(morphology,['remove_small_objects','remove_small_holes','label']),
                            (segmentation,['clear_border','relabel_sequential']),
                            (skimage.measure,['label'])]:
        for name in names:monkeypatch.setattr(namespace,name,reject)
    from skimage.util._map_array import ArrayMap as OriginalArrayMap
    monkeypatch.setattr(OriginalArrayMap,'__init__',reject)
    import scipy.ndimage
    monkeypatch.setattr(scipy.ndimage,'label',reject)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');result,forward,inverse=pipeline(xx)
        assert type(forward) is type(inverse) is implementation.ArrayMap
        np.testing.assert_array_equal(forward[inverse[result]],result)
    assert implementation._label is xx.morphology.label is xx.measure.label is labels.label


def test_returned_map_pickle_and_cross_aliases_are_preserved():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        expected=segmentation.relabel_sequential(np.array([0,3,8,11],np.int16),offset=2)
        actual=xx.segmentation.relabel_sequential(np.array([0,3,8,11],np.int16),offset=2)
        wanted=pickle.loads(pickle.dumps(expected));restored=pickle.loads(pickle.dumps(actual))
        for got,want in zip(restored[1:],wanted[1:]):
            assert type(got) is implementation.ArrayMap
            assert contract.describe(got,actual[0],{})==contract.describe(want,expected[0],{})
        assert restored[1].in_values is restored[2].out_values
        assert restored[1].out_values is restored[2].in_values
        np.testing.assert_array_equal(restored[2][restored[0]],[0,3,8,11])
        restored[1][3]=7
        assert actual[1][3]==2 and restored[2][2]==3


def test_returned_arraymap_is_not_a_public_class_registration():
    assert all('ArrayMap' not in name for name in (*native_classes(),*class_metadata_providers()))
    assert len(native_classes())==4 and class_metadata_providers()==('util.FailedEstimationAccessError',)
    assert 'ArrayMap' not in xx.segmentation.__all__ and 'ArrayMap' not in xx.util.__all__


@pytest.mark.parametrize('api',contract.APIS[:2])
@pytest.mark.parametrize('dtype',[np.complex64,np.complex128,object,'U3','V4',np.longdouble])
@pytest.mark.parametrize('deprecated',[False,True])
def test_defined_morphology_dtype_rejection_stays_native(api,dtype,deprecated):
    # Defined error/deprecation order does not need an image-kernel provider.
    # This does not claim object/complex pixel processing or ArrayMap support.
    candidate=getattr(xx.morphology,api.split('.')[1])
    old='min_size' if api.endswith('objects') else 'area_threshold'
    options={old:2} if deprecated else {}
    contract.check(candidate,api,lambda:(np.zeros((3,4),dtype),(),dict(options)))


def test_native_sparse_count_safety_without_impractical_reference_allocation(monkeypatch):
    monkeypatch.setattr(morphology,'remove_small_objects',lambda *a,**k:(_ for _ in ()).throw(AssertionError('unsafe reference allocation')))
    image=np.array([0,10**12,10**12,3],np.int64);before=image.copy()
    result=xx.morphology.remove_small_objects(image,max_size=1)
    np.testing.assert_array_equal(result,[0,10**12,10**12,0]);np.testing.assert_array_equal(image,before)
    counts=xx._core.consumer_count(xx._core.consumer_indices(image))
    np.testing.assert_array_equal(counts.keys,[0,3,10**12]);np.testing.assert_array_equal(counts.sizes,[1,1,2])


@pytest.mark.parametrize('dtype',[np.int64,np.uint64])
def test_sparse_relabel_lookup_reference_is_safe_without_dense_maps(dtype):
    image=np.array([0,10**12,10**12,5,np.iinfo(dtype).max],dtype)
    contract.check(xx.segmentation.relabel_sequential,'segmentation.relabel_sequential',lambda:(image.copy(),(),{}))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');output,forward,inverse=xx.segmentation.relabel_sequential(image)
        np.testing.assert_array_equal(forward[image],output);np.testing.assert_array_equal(inverse[output],image)


@pytest.mark.parametrize('name',contract.APIS)
def test_ndarray_subclass_uses_explicit_fallback(name):
    class TaggedArray(np.ndarray):pass
    module,attribute=name.split('.');candidate=getattr(getattr(xx,module),attribute)
    image=contract.image_for(np.int16).view(TaggedArray)
    with pytest.raises(NotImplementedError,match='fallback is disabled'):candidate(image)
    old=xx.set_fallback_enabled(True)
    try:
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always');actual=candidate(image)
        assert any(isinstance(w.message,xx.FallbackWarning) for w in notices)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');expected=contract.reference(name)(image)
        assert contract.describe(actual,image,{})==contract.describe(expected,image,{})
    finally:xx.set_fallback_enabled(old)


def test_positional_custom_mask_obeys_protocol_fallback():
    class TaggedArray(np.ndarray):pass
    image=contract.image_for(np.int16);mask=np.ones(image.shape,bool).view(TaggedArray)
    with pytest.raises(NotImplementedError,match='fallback is disabled'):
        xx.segmentation.clear_border(image,0,0,mask)
    old=xx.set_fallback_enabled(True)
    try:
        with pytest.warns(xx.FallbackWarning):actual=xx.segmentation.clear_border(image,0,0,mask)
        np.testing.assert_array_equal(actual,segmentation.clear_border(image,0,0,mask))
    finally:xx.set_fallback_enabled(old)


def test_installed_public_default_workers_and_identity():
    code='import skimagexx as x;from skimagexx import _component_consumers as c,_components as l;assert x.get_num_threads()==1;assert x._core is c._core is l._core;assert x._core.component_consumers_interface_version==1'
    env=os.environ.copy();env['PYTHONPATH']=str(Path(xx.__file__).resolve().parent.parent)
    result=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
