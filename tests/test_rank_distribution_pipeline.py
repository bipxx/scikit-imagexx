"""Strict private v2 parity; no public exposure or forwarding evidence."""
import importlib
import importlib.util
import os
from pathlib import Path
import numpy as np
import pytest
from skimage.filters import rank as reference

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

contracts=load('rank_distribution_contract_provider',Path(__file__).with_name('test_rank_distribution_contracts.py'))
if os.environ.get('SKIMAGEXX_RANK_CANDIDATE'):
    native=load('skimagexx._rank_distribution_candidate',os.environ['SKIMAGEXX_RANK_CANDIDATE'])
    native._core=importlib.import_module(os.environ['SKIMAGEXX_RANK_BACKEND'])
else:native=importlib.import_module('skimagexx._rank')

@pytest.fixture(autouse=True,params=[1,4])
def workers(request):
    previous=native._core.get_num_threads();native._core.set_num_threads(request.param)
    yield
    native._core.set_num_threads(previous)

@pytest.mark.parametrize('case',contracts.CASES,ids=[c[0] for c in contracts.CASES])
def test_exact_frozen_safe_contract(case):
    key,name,parameters=case
    expected=contracts.observe(getattr(reference,name),parameters)
    actual=contracts.observe(getattr(native,name),parameters)
    assert actual==expected,key

@pytest.mark.parametrize('name',contracts.NAMES)
def test_mixed_v1_backend_is_not_advertised_as_v2(name,monkeypatch):
    class OldCore:pass
    monkeypatch.setattr(native,'_core',OldCore())
    with pytest.raises(NotImplementedError,match='rank-histogram-v2'):
        getattr(native,name)(np.ones((3,5),np.uint8),np.ones((3,3),np.uint8))

@pytest.mark.parametrize('operation',[-1,9,31,32])
def test_capabilities_and_invalid_operation_admission(operation):
    assert native._core.rank_histogram_capabilities==0x1ff
    with pytest.raises(ValueError,match='invalid rank operation'):
        native._core.rank_run(np.ones((3,5),np.uint8),np.ones((3,3),np.uint8),None,np.zeros((3,5,1),np.uint8),[0,0],256,operation)


@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('dtype',[np.uint8,np.uint16])
@pytest.mark.parametrize('ndim',[2,3])
@pytest.mark.parametrize('masked',[False,True])
def test_parallel_full_image_exact_worker_bits(name,dtype,ndim,masked):
    shape=(513,519) if ndim==2 else (3,299,301)
    image=((np.arange(int(np.prod(shape))) * 37)%31).astype(dtype).reshape(shape)
    footprint=np.ones((3,5) if ndim==2 else (3,3,5),np.uint8);footprint.flat[::4]=0
    options={'mask':(np.arange(image.size).reshape(shape)%4!=0).astype(np.uint8)} if masked else {}
    expected=getattr(reference,name)(image,footprint,**options)
    actual=getattr(native,name)(image,footprint,**options)
    assert actual.dtype==expected.dtype and actual.strides==expected.strides
    assert actual.tobytes()==expected.tobytes()
    previous=native._core.get_num_threads()
    try:
        native._core.set_num_threads(1);one=getattr(native,name)(image,footprint,**options)
        native._core.set_num_threads(4);four=getattr(native,name)(image,footprint,**options)
    finally:native._core.set_num_threads(previous)
    assert one.tobytes()==four.tobytes()==expected.tobytes()


@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('dtype',[np.uint8,np.uint16,np.float32,np.float64,np.bool_])
@pytest.mark.parametrize('ndim',[2,3])
def test_full_u16_range_output_dtype_raw_bytes(name,dtype,ndim):
    shape=(5,7) if ndim==2 else (3,4,5)
    image=np.resize(np.array([0,65535,1,65534,32768],np.uint16),int(np.prod(shape))).reshape(shape)
    footprint=np.ones((3,)*ndim,np.uint8)
    expected_out=np.full(shape,99,dtype);actual_out=expected_out.copy()
    expected=getattr(reference,name)(image,footprint,out=expected_out)
    actual=getattr(native,name)(image,footprint,out=actual_out)
    assert actual.dtype==expected.dtype and actual.strides==expected.strides
    assert actual_out.tobytes()==expected_out.tobytes()==actual.tobytes()


policy=load('rank_distribution_policy_helpers',Path(__file__).with_name('test_rank_policies.py'))


@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('stage',['image','mask','footprint'])
@pytest.mark.parametrize('error_policy',['warn','raise','ignore','call','log','print'])
@pytest.mark.parametrize('signal',[False,True])
def test_inherited_fp_provider_diagnostics(name,dtype,stage,error_policy,signal,capfd):
    assert policy.observe(native,name,dtype,stage,error_policy,'none',signal,capfd)==policy.observe(reference,name,dtype,stage,error_policy,'none',signal,capfd)


@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('stage',['image','mask','footprint'])
@pytest.mark.parametrize('error_policy',['call','log'])
@pytest.mark.parametrize('action',['fill','shape','dtype','ignore','handler','reentrant'])
def test_inherited_stateful_mutations(name,dtype,stage,error_policy,action,capfd):
    assert policy.observe(native,name,dtype,stage,error_policy,action,True,capfd)==policy.observe(reference,name,dtype,stage,error_policy,action,True,capfd)


@pytest.mark.parametrize('name',contracts.NAMES)
def test_no_upstream_image_function_forwarding(name,monkeypatch):
    image=np.arange(35,dtype=np.uint8).reshape(5,7);fp=np.ones((3,3),np.uint8)
    expected=getattr(reference,name)(image,fp)
    def forbidden(*args,**kwargs):raise AssertionError('upstream image algorithm forwarding')
    for operation in contracts.NAMES:monkeypatch.setattr(reference,operation,forbidden)
    actual=getattr(native,name)(image,fp)
    assert actual.tobytes()==expected.tobytes()
