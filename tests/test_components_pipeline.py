"""Private version1 native pipeline proof; this does not register public APIs.

Set SKIMAGEXX_COMPONENTS_MODULE_DIR to a separately built standalone module.
All frozen L1 contract assertions are reused unchanged against this candidate.
"""
import importlib.util
import os
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).parents[1]
if os.environ.get('SKIMAGEXX_COMPONENTS_MODULE_DIR'):
    sys.path.insert(0,os.environ['SKIMAGEXX_COMPONENTS_MODULE_DIR'])
    import _components_core as core
else:
    from skimagexx import _core as core
assert getattr(core,'components_interface_version',None)==1,'version1 component backend required'
spec=importlib.util.spec_from_file_location('skimagexx._components_private',ROOT/'python/skimagexx/_components.py')
candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate);candidate._core=core
spec=importlib.util.spec_from_file_location('_components_contracts',ROOT/'tests/test_connected_components_contracts.py')
contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
for name in vars(contract):
    if name.startswith('test_') and name!='test_unregistered_label_rejects_strict_fallback':globals()[name]=getattr(contract,name)


@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    core.set_num_threads(request.param)
    yield
    core.set_num_threads(1)


@pytest.fixture(params=contract.APIS)
def api(request):return request.param


@pytest.fixture
def label_function():return candidate.label


@pytest.mark.parametrize('dtype',contract.TYPES[1:])
@pytest.mark.parametrize('layout',['C','endian','reverse','unaligned'])
def test_native_preparation(dtype,layout):
    image=np.resize(np.array([0,1,-1,2,3,127],np.int64),(7,13)).astype(dtype)
    if layout=='endian':image=image.astype(image.dtype.newbyteorder('>'))
    elif layout=='reverse':image=image[::-1,::-1]
    elif layout=='unaligned':
        buffer=np.empty(image.nbytes+1,np.uint8);unaligned=np.ndarray(image.shape,image.dtype,buffer,offset=1);unaligned[:]=image;image=unaligned
    before=image.tobytes();actual,invalid=core.components_prepare_index(image)
    assert not invalid and actual.dtype==np.dtype(np.intp)
    assert actual.flags.c_contiguous and actual.flags.owndata and actual.flags.writeable
    assert not np.shares_memory(image,actual) and image.tobytes()==before
    np.testing.assert_array_equal(actual,image.astype(np.intp))


@pytest.mark.parametrize('connectivity',[1,2,3])
@pytest.mark.parametrize('shape',[(37,73,101),(1,1,300001),(300001,1,1),(257,1025)])
@pytest.mark.parametrize('boolean',[False,True])
def test_parallel_boundaries_random_and_singleton_axes(shape,connectivity,boolean):
    if connectivity>len(shape):return
    rng=np.random.default_rng(191);image=rng.integers(0,4,shape,dtype=np.int16)
    if boolean:image=image.astype(bool)
    contract.check(candidate.label,'measure.label',image,connectivity=connectivity,return_num=True,oracle=False)


def test_all_half_words_conversion():
    image=np.arange(65536,dtype=np.uint16).view(np.float16)
    with np.errstate(invalid='ignore'):expected=image.astype(np.intp)
    actual,invalid=core.components_prepare_index(image)
    assert invalid
    np.testing.assert_array_equal(actual,expected)


def test_prepared_stage_guards():
    with pytest.raises(TypeError):core.components_prepare_index(np.ones(2,bool))
    with pytest.raises(TypeError):core.components_prepare_bool(np.ones(2,np.uint8))
    with pytest.raises(ValueError):core.components_label(np.arange(4,dtype=np.intp)[::2],0,1,False,np.dtype(np.intp))
    with pytest.raises(ValueError):core.components_label(np.ones(2,np.uint8),2,1,True,np.dtype(np.int32))


@pytest.mark.parametrize('boolean',[False,True])
@pytest.mark.parametrize('background',[2**63,2**64-1,-2**63-1,np.uint64(2**64-1),np.array([1]),np.array(1),np.array([1,2]),1+0j])
def test_platform_background_metadata(boolean,background):
    image=np.ones((2,3),bool if boolean else np.int16)
    contract.check(candidate.label,'measure.label',image,background=background,oracle=False)


@pytest.mark.parametrize('boolean',[False,True])
@pytest.mark.parametrize('connectivity',[np.array([1]),np.array(1),np.array([1.5]),np.array([1,2]),np.float64(1.5)])
def test_runtime_connectivity_coercion(boolean,connectivity):
    image=np.ones((2,3),bool if boolean else np.int16)
    contract.check(candidate.label,'measure.label',image,connectivity=connectivity,oracle=False)


@pytest.mark.parametrize('shape',[(3,1),(1,3),(3,1,1),(1,3,1),(1,1,3),(2,3,1),(2,1,3),(1,2,3)])
@pytest.mark.parametrize('boolean',[False,True])
def test_singleton_axis_storage_and_traversal(shape,boolean):
    image=contract.sample(bool if boolean else np.int16,shape)
    before=image.tobytes()
    expected,number=contract.original('measure.label')(image,return_num=True)
    actual,count=candidate.label(image,return_num=True)
    # Upstream can return a non-C view after moving singleton axes back.
    # Compare its exact storage contract rather than assuming C everywhere.
    assert actual.dtype==expected.dtype and actual.dtype.metadata==expected.dtype.metadata
    assert actual.shape==expected.shape and actual.strides==expected.strides
    for flag in ['c_contiguous','f_contiguous','owndata','writeable','aligned']:
        assert getattr(actual.flags,flag)==getattr(expected.flags,flag)
    assert type(count) is type(number) is int and count==number
    assert not np.shares_memory(actual,image) and image.tobytes()==before
    np.testing.assert_array_equal(actual,expected)
