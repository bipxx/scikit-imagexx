"""Private consumer adapter proof; no public registration or forwarding claim."""
import importlib.util
import os
from pathlib import Path
import sys
import warnings
import numpy as np
import pytest
import skimagexx as xx

ROOT=Path(__file__).parents[1]
if os.environ.get('SKIMAGEXX_COMPONENT_CONSUMERS_MODULE_DIR'):
    sys.path.insert(0,os.environ['SKIMAGEXX_COMPONENT_CONSUMERS_MODULE_DIR'])
    import _component_consumers_core as core
else:
    from skimagexx import _core as core
assert getattr(core,'component_consumers_interface_version',None)==1,'consumer interface1 required'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


components=load('skimagexx._components_consumer_adapter',ROOT/'python/skimagexx/_components.py');components._core=core
candidate=load('skimagexx._component_consumers_adapter',ROOT/'python/skimagexx/_component_consumers.py');candidate._core=core;candidate._label=components.label
contract=load('_consumer_contract_adapter',ROOT/'tests/test_component_consumer_contracts.py')
for name in vars(contract):
    if name.startswith('test_'):globals()[name]=getattr(contract,name)


@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    previous=xx.set_fallback_enabled(False);core.set_num_threads(request.param)
    assert core.get_num_threads()==request.param
    yield
    core.set_num_threads(1);xx.set_fallback_enabled(previous)


@pytest.fixture(params=contract.APIS)
def api(request):return request.param


@pytest.fixture
def function(api):return getattr(candidate,api.split('.')[1])


@pytest.fixture
def relabel_function():return candidate.relabel_sequential


def test_sparse_count_filter_without_impractical_reference_allocation():
    # Native definition-level safety proof only: original bincount would try
    # to allocate through10**12, so it is deliberately never called here.
    large=10**12;image=np.array([0,large,large,3],np.int64)
    actual=candidate.remove_small_objects(image,max_size=1)
    np.testing.assert_array_equal(actual,[0,large,large,0])
    np.testing.assert_array_equal(image,[0,large,large,3])
    counts=core.consumer_count(core.consumer_indices(image))
    np.testing.assert_array_equal(counts.keys,[0,3,large])
    np.testing.assert_array_equal(counts.sizes,[1,1,2])


@pytest.mark.parametrize('dtype',[np.int64,np.uint64])
def test_sparse_relabel_and_lookup_safe_reference(dtype):
    # relabel_sequential uses sparse maps; no forward dense materialization.
    image=np.array([0,10**12,10**12,5,np.iinfo(dtype).max],dtype)
    contract.check(candidate.relabel_sequential,'segmentation.relabel_sequential',lambda:(image.copy(),(),{}))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');labels,forward,inverse=candidate.relabel_sequential(image)
        np.testing.assert_array_equal(forward[image],labels)
        np.testing.assert_array_equal(inverse[labels],image)


def test_large_unsigned_holes_reference_is_boolean_and_safe():
    image=np.array([[0,np.iinfo(np.uint64).max],[10**12,0]],np.uint64)
    contract.check(candidate.remove_small_holes,'morphology.remove_small_holes',
                   lambda:(image.copy(),(),dict(max_size=1)))


@pytest.mark.parametrize('api',contract.APIS[:2])
@pytest.mark.parametrize('connectivity',[np.nan,np.inf,-np.inf,None,0,-1])
def test_consumer_nonfinite_and_clamped_connectivity(function,api,connectivity):
    contract.check(function,api,lambda:(np.array([[1,0,1],[0,1,0]],bool),(),dict(max_size=1,connectivity=connectivity)))


@pytest.mark.parametrize('api',contract.APIS[:2])
@pytest.mark.parametrize('dtype',[np.bool_,np.int16])
@pytest.mark.parametrize('threshold',[None,'bad',np.nan])
def test_empty_threshold_dispatch_matches_runtime(function,api,dtype,threshold):
    contract.check(function,api,lambda:(np.empty((0,3),dtype),(),dict(max_size=threshold)))
