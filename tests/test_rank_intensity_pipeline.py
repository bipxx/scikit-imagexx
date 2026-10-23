"""Private R3 actual-binding parity; this adapter is not a public export.

The unchanged production normalization shell feeds IDs9..12 into the candidate
binding. No image operation is delegated to the original implementation.
"""
import importlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from skimage.filters import rank as reference


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


contracts=load('rank_intensity_contract_provider',Path(__file__).with_name('test_rank_intensity_contracts.py'))
if os.environ.get('SKIMAGEXX_RANK_CANDIDATE'):
    shell=load('skimagexx._rank_intensity_candidate',os.environ['SKIMAGEXX_RANK_CANDIDATE'])
    shell._core=importlib.import_module(os.environ['SKIMAGEXX_RANK_BACKEND'])
else:
    shell=importlib.import_module('skimagexx._rank')


def private_function(operation,name):
    def operation_function(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):
        if not (shell._core.rank_histogram_capabilities & (1<<operation)):
            raise NotImplementedError('rank-histogram-v3 backend required')
        return shell._apply(operation,image,footprint,out,mask,shift_x,shift_y,shift_z)
    operation_function.__name__=operation_function.__qualname__=name
    return operation_function


native=SimpleNamespace(**{name:private_function(9+i,name) for i,name in enumerate(contracts.NAMES)})


@pytest.fixture(autouse=True,params=[1,4])
def workers(request):
    previous=shell._core.get_num_threads();shell._core.set_num_threads(request.param)
    yield
    shell._core.set_num_threads(previous)


@pytest.mark.parametrize('case',contracts.CASES,ids=[c[0] for c in contracts.CASES])
def test_frozen_original_and_private_binding(case):
    key,name,parameters=case
    assert contracts.observe(getattr(native,name),parameters)==contracts.observe(getattr(reference,name),parameters),key


# Execute the independent definitions against the candidate as well as the
# original, retaining the sealed definition bodies unchanged.
@pytest.mark.parametrize('case',range(1))
def test_private_capability_and_rejected_ids(case):
    assert shell._core.rank_histogram_capabilities==8191
    for operation in (-1,13,31,32):
        with pytest.raises(ValueError,match='invalid rank operation'):
            shell._core.rank_run(np.ones((3,5),np.uint8),np.ones((3,3),np.uint8),None,np.zeros((3,5,1),np.uint8),[0,0],256,operation)


@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('ndim',[2,3])
@pytest.mark.parametrize('dtype',[np.uint8,np.uint16])
@pytest.mark.parametrize('output',[None,np.float32,np.float64])
@pytest.mark.parametrize('pattern',['ties','asymmetric','zero-window','center-below','center-above'])
def test_definition_against_private(name,ndim,dtype,output,pattern,monkeypatch):
    monkeypatch.setattr(contracts,'reference',native)
    contracts.test_independent_histogram_intensity_definition(name,ndim,dtype,output,pattern)


@pytest.mark.parametrize('ndim',[2,3])
@pytest.mark.parametrize('output',[np.float32,np.float64])
@pytest.mark.parametrize('direction',['below','above'])
def test_signed_wide_integer_and_float_stores(ndim,output,direction,monkeypatch):
    monkeypatch.setattr(contracts,'reference',native)
    contracts.test_autolevel_signed_wide_quotient_is_observable_in_float_output(ndim,output,direction)


@pytest.mark.parametrize('name',contracts.NAMES)
def test_no_original_image_algorithm_forwarding(name,monkeypatch):
    image=np.arange(35,dtype=np.uint8).reshape(5,7);fp=np.ones((3,3),np.uint8)
    expected=getattr(reference,name)(image,fp)
    def forbidden(*args,**kwargs):raise AssertionError('upstream algorithm forwarding')
    for operation in contracts.NAMES:monkeypatch.setattr(reference,operation,forbidden)
    np.testing.assert_array_equal(getattr(native,name)(image,fp),expected)


@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('dtype',[np.uint8,np.uint16])
@pytest.mark.parametrize('ndim',[2,3])
def test_large_worker_identity(name,dtype,ndim):
    shape=(513,519) if ndim==2 else (3,299,301)
    image=((np.arange(int(np.prod(shape)))*37)%71).astype(dtype).reshape(shape)
    fp=np.ones((3,5) if ndim==2 else (3,3,5),np.uint8);fp.flat[::4]=0
    mask=(np.arange(image.size).reshape(shape)%4!=0).astype(np.uint8)
    expected=getattr(reference,name)(image,fp,mask=mask)
    one=getattr(native,name)(image,fp,mask=mask)
    previous=shell._core.get_num_threads()
    try:
        shell._core.set_num_threads(1);serial=getattr(native,name)(image,fp,mask=mask)
        shell._core.set_num_threads(4);parallel=getattr(native,name)(image,fp,mask=mask)
    finally:shell._core.set_num_threads(previous)
    assert one.tobytes()==serial.tobytes()==parallel.tobytes()==expected.tobytes()


@pytest.mark.parametrize('operation',range(9,13))
def test_aliases_explicitly_rejected_by_private_bridge(operation):
    image=np.arange(35,dtype=np.uint8).reshape(5,7)
    with pytest.raises(ValueError,match='overlapping buffers'):
        shell._core.rank_run(image,np.ones((3,3),np.uint8),None,image.reshape(5,7,1),[0,0],256,operation)


@pytest.mark.parametrize('operation',range(9,13))
def test_footprint_alias_and_masked_invalid_center_are_safe(operation):
    image=np.full((3,3),255,np.uint8);footprint=np.ones((3,3),np.uint8)
    with pytest.raises(ValueError,match='overlapping buffers'):
        shell._core.rank_run(image,footprint,None,footprint.reshape(3,3,1),[0,0],256,operation)
    with pytest.raises(ValueError,match='histogram bounds'):
        shell._core.rank_run(image,footprint,np.zeros_like(image),np.empty((3,3,1),np.uint8),[0,0],4,operation)
