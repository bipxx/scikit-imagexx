"""Private R4 normalization/actual binding proof; never public admission.

Only the existing owned rank conversion foundation and small shape/index
providers are reused. No upstream rank algorithm/preprocessor is called.
"""
import importlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import warnings
import numpy as np
import pytest
from skimage.filters import rank as reference


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


if 'SKIMAGEXX_RANK_STATISTICS_BACKEND' not in os.environ and importlib.util.find_spec('_rank_statistics_core') is None:
    pytest.skip('Private compiled fixture _rank_statistics_core unavailable; supplementary private tests are not public parity',allow_module_level=True)

contracts=load('rank_statistics_private_contracts',Path(__file__).with_name('test_rank_statistics_contracts.py'))
core=importlib.import_module(os.environ.get('SKIMAGEXX_RANK_STATISTICS_BACKEND','_rank_statistics_core'))
shell=load('skimagexx._rank_statistics_private_shell',Path(__file__).parents[1]/'python/skimagexx/_rank.py')


class PrivateScalarAdmission:
    """Metadata adapter only: admitted IDs call the actual new C++ binding."""
    def __getattr__(self,name):return getattr(core,name)
    def rank_run(self,image,footprint,mask,out,shifts,bins,operation):
        assert 13<=operation<=15
        return core.rank_statistics_run(image,footprint,mask,out,shifts,bins,operation)


shell._core=PrivateScalarAdmission()


def scalar_function(operation,name):
    def function(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):
        if operation==14:
            ndim=np.asanyarray(image).ndim
            if shell._is_sequence(footprint):raise ValueError('footprint sequences are not currently supported by rank filters')
            if ndim not in (2,3):raise ValueError(f'`image` must have 2 or 3 dimensions, got {ndim}.')
            footprint=core.rank_noise_footprint(footprint,[shift_x,shift_y] if ndim==2 else [shift_x,shift_y,shift_z])
        return shell._apply(operation,image,footprint,out,mask,shift_x,shift_y,shift_z)
    function.__name__=function.__qualname__=name
    return function


def windowed_histogram(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,n_bins=None):
    if n_bins is None:n_bins=int(image.max())+1
    value=np.asanyarray(image)
    if value.size==0:raise ValueError('The parameter `image` cannot be an empty array')
    if value.ndim!=2:raise ValueError('The parameter `image` must be a 2-dimensional array')
    dtype=image.dtype
    if dtype==bool:raise ValueError('dtype cannot be bool.')
    if dtype not in (np.uint8,np.uint16):
        warnings.warn(f'Possible precision loss converting image of type {dtype} to uint8 as required by rank filters. Convert manually using skimage.util.img_as_ubyte to silence this warning.',UserWarning,stacklevel=2)
        image=shell._ubyte(image)
    if shell._is_sequence(footprint):raise ValueError('footprint sequences are not currently supported by rank filters')
    footprint=np.ascontiguousarray(shell._positive(footprint))
    if footprint.ndim!=image.ndim:raise ValueError('Image dimensions and neighborhood dimensionsdo not match')
    image=np.ascontiguousarray(image)
    if mask is not None:mask=np.ascontiguousarray(shell._ubyte(mask))
    if image is out:raise NotImplementedError('Cannot perform rank operation in place.')
    if out is None:out=np.empty(image.shape+(n_bins,),np.float64)
    elif len(out.shape)==2:out=out.reshape(out.shape+(n_bins,))
    bins=256 if image.dtype in (np.uint8,np.int8) else core.rank_bins(image)
    if bins>1024:warnings.warn(f'Bad rank filter performance is expected due to a large number of bins ({bins}), equivalent to an approximate bitdepth of {np.log2(bins):.1f}.',UserWarning,stacklevel=2)
    for name,value in zip(('shift_x','shift_y'),(shift_x,shift_y)):
        if np.dtype(type(value))==bool:warnings.warn(f'Paramter `{name}` is boolean and will be interpreted as int. This is not officially supported, use int instead.',UserWarning,stacklevel=2)
    if out.ndim!=3 or not (out.dtype.kind in 'ub' and out.dtype.itemsize in (1,2) or out.dtype.kind=='f' and out.dtype.itemsize in (4,8)):raise TypeError('Function call with ambiguous argument types')
    core.rank_windowed_histogram_run(image,footprint,mask,out,[shift_x,shift_y],bins)
    return out


native=SimpleNamespace(**{name:scalar_function(13+i,name) for i,name in enumerate(contracts.SCALAR)},windowed_histogram=windowed_histogram)


@pytest.fixture(autouse=True,params=[1,4])
def workers(request):
    previous=core.get_num_threads();core.set_num_threads(request.param)
    yield
    core.set_num_threads(previous)


@pytest.mark.parametrize('case',contracts.CASES,ids=[c[0] for c in contracts.CASES])
def test_frozen_original_and_actual_private_binding(case):
    key,name,parameters=case
    assert contracts.observe(getattr(native,name),name,parameters)==contracts.observe(getattr(reference,name),name,parameters),key


def test_separate_interfaces_preserve_old_admission():
    assert core.rank_histogram_capabilities==8191
    assert core.rank_alias_capabilities==480
    assert core.rank_statistics_interface=='rank-statistics-v1'
    assert core.rank_statistics_capabilities==0xe000
    assert core.rank_vector_interface=='rank-vector-v1'
    assert core.rank_vector_capabilities==0x10000
    for op in (-1,13,16,31,32):
        with pytest.raises(ValueError,match='invalid rank operation'):core.rank_run(np.ones((2,3),np.uint8),np.ones((1,1),np.uint8),None,np.zeros((2,3,1),np.uint8),[0,0],256,op)


@pytest.mark.parametrize('name',contracts.NAMES)
def test_no_original_algorithm_forwarding(name,monkeypatch):
    image=np.arange(35,dtype=np.uint8).reshape(5,7)%11;fp=np.ones((3,3),np.uint8)
    expected=getattr(reference,name)(image,fp)
    def forbidden(*a,**k):raise AssertionError('upstream algorithm forwarding')
    for operation in contracts.NAMES:monkeypatch.setattr(reference,operation,forbidden)
    np.testing.assert_array_equal(getattr(native,name)(image,fp),expected)


@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('dtype',[np.uint8,np.uint16])
def test_large_worker_identity_and_original(name,dtype):
    image=((np.arange(513*519)*37)%71).astype(dtype).reshape(513,519)
    fp=np.ones((3,5),np.uint8);fp.flat[::4]=0
    mask=(np.arange(image.size).reshape(image.shape)%4!=0).astype(np.uint8)
    options=dict(mask=mask,out=np.empty(image.shape,np.float64)) if name=='subtract_mean' else dict(mask=mask)
    if name=='windowed_histogram':options['n_bins']=7
    # A supplied output is reused on every call: retain each completed value
    # before the next call can overwrite it, including the reference result.
    expected=getattr(reference,name)(image,fp,**options).copy()
    a=getattr(native,name)(image,fp,**options).copy()
    core.set_num_threads(1);b=getattr(native,name)(image,fp,**options).copy()
    np.testing.assert_array_equal(a,expected);np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('depth',[1,2,3,4,5,7,8,9,15,16,17,255,256])
@pytest.mark.parametrize('dtype',[np.uint8,np.uint16,np.float32,np.float64])
def test_vector_tail_reciprocal_order(depth,dtype):
    image=np.full((7,7),2,np.uint8);fp=np.ones((7,7),np.uint8)
    left=np.full(image.shape+(depth,),99,dtype);right=left.copy()
    result=windowed_histogram(image,fp,out=left)
    expected=reference.windowed_histogram(image,fp,out=right)
    assert result is left
    np.testing.assert_array_equal(result,expected)


@pytest.mark.parametrize('mode',['bad-bin','alias','zero-depth','deep','strides'])
def test_new_safety_guards_before_write(mode):
    image=np.arange(15,dtype=np.uint8).reshape(3,5);fp=np.ones((3,3),np.uint8);out=np.full((3,5,4),99,np.float64);bins=256
    if mode=='bad-bin':bins=4
    elif mode=='alias':out=image.reshape(3,5,1)
    elif mode=='zero-depth':out=np.empty((3,5,0),np.float64)
    elif mode=='deep':out=np.full((3,5,257),99,np.float64)
    elif mode=='strides':out=out[...,::2]
    before=out.copy();source=image.copy()
    with pytest.raises(ValueError):core.rank_windowed_histogram_run(image,fp,None,out,[0,0],bins)
    np.testing.assert_array_equal(out,before);np.testing.assert_array_equal(image,source)


def test_debug_assertion_build_when_required():
    if os.environ.get('SKIMAGEXX_REQUIRE_GIL_ASSERTIONS')=='1':assert core.gil_assertions is True


@pytest.mark.parametrize('name',contracts.SCALAR)
@pytest.mark.parametrize('ndim',(2,3))
@pytest.mark.parametrize('dtype',('u1','u2'))
@pytest.mark.parametrize('kind',('dense','even','disconnected','zero','masked','shift'))
@pytest.mark.parametrize('out_type',('same','f4','f8'))
def test_independent_arithmetic_replay(name,ndim,dtype,kind,out_type,monkeypatch):
    monkeypatch.setattr(contracts,'reference',native)
    contracts.test_independent_arithmetic_and_ordered_replay(name,ndim,dtype,kind,out_type)


@pytest.mark.parametrize('dtype',('u1','u2'))
@pytest.mark.parametrize('kind',('dense','even','zero','masked'))
@pytest.mark.parametrize('depth',(4,32))
@pytest.mark.parametrize('out_type',('u1','u2','f4','f8'))
def test_independent_histogram_replay(dtype,kind,depth,out_type,monkeypatch):
    monkeypatch.setattr(contracts,'reference',native)
    contracts.test_histogram_independent_counts_and_exact_reciprocal_replay(dtype,kind,depth,out_type)


@pytest.mark.parametrize('name',contracts.SCALAR)
@pytest.mark.parametrize('dtype',[np.uint8,np.uint16])
def test_large_3d_mask_shift_and_worker_bits(name,dtype):
    image=((np.arange(3*299*301)*17)%31).astype(dtype).reshape(3,299,301)
    fp=np.ones((3,5,3),np.uint8);fp.flat[::3]=0
    mask=(np.arange(image.size).reshape(image.shape)%5!=0).astype(np.uint8)
    options=dict(mask=mask,shift_x=-1,shift_y=1,shift_z=-1)
    if name=='subtract_mean':options['out']=np.empty(image.shape,np.float64)
    expected=getattr(reference,name)(image,fp,**options).copy()
    core.set_num_threads(1);left=getattr(native,name)(image,fp,**options).copy()
    core.set_num_threads(4);right=getattr(native,name)(image,fp,**options).copy()
    assert left.tobytes()==expected.tobytes()==right.tobytes()


def test_subtract_undefined_unsigned_domain_rejected_prewrite():
    image=np.array([[0,70]],np.uint16);fp=np.array([[0,0,1]],np.uint8)
    out=np.full((1,2,1),99,np.uint16)
    with pytest.raises(ValueError,match='unsigned conversion outside defined coverage'):
        core.rank_statistics_run(image,fp,None,out,[0,0],71,15)
    assert out.tobytes()==np.full_like(out,99).tobytes()
