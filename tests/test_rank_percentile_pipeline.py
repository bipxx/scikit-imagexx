"""Private percentile normalization/actual C++ binding proof, not public API."""
import importlib
import importlib.util
import json
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


if 'SKIMAGEXX_RANK_PERCENTILE_BACKEND' not in os.environ and importlib.util.find_spec('_rank_percentile_core') is None:
    pytest.skip('Private compiled fixture _rank_percentile_core unavailable; supplementary private tests are not public parity',allow_module_level=True)

contracts=load('rank_percentile_private_contracts',Path(__file__).with_name('test_rank_percentile_contracts.py'))
core=importlib.import_module(os.environ.get('SKIMAGEXX_RANK_PERCENTILE_BACKEND','_rank_percentile_core'))
shell=load('skimagexx._rank_percentile_private_shell',Path(__file__).parents[1]/'python/skimagexx/_rank.py');shell._core=core


def check_image(image):
    value=np.asanyarray(image)
    if value.size==0:raise ValueError('The parameter `image` cannot be an empty array')
    if value.ndim!=2:raise ValueError('The parameter `image` must be a 2-dimensional array')


def apply(operation,image,footprint,out,mask,shift_x,shift_y,p0,p1):
    # Both original wrapper and common preprocessing check dimensionality.
    # Only own native conversion foundations and structural metadata are used.
    check_image(image);check_image(image)
    dtype=image.dtype
    if dtype==bool:raise ValueError('dtype cannot be bool.')
    if dtype not in (np.uint8,np.uint16):
        warnings.warn(f'Possible precision loss converting image of type {dtype} to uint8 as required by rank filters. Convert manually using skimage.util.img_as_ubyte to silence this warning.',UserWarning,stacklevel=3)
        image=shell._ubyte(image)
    if shell._is_sequence(footprint):raise ValueError('footprint sequences are not currently supported by rank filters')
    footprint=np.ascontiguousarray(shell._positive(footprint))
    if footprint.ndim!=image.ndim:raise ValueError('Image dimensions and neighborhood dimensionsdo not match')
    image=np.ascontiguousarray(image)
    if mask is not None:mask=np.ascontiguousarray(shell._ubyte(mask))
    if image is out:raise NotImplementedError('Cannot perform rank operation in place.')
    if out is None:out=np.empty(image.shape+(1,),image.dtype)
    elif len(out.shape)==2:out=out.reshape(out.shape+(1,))
    bins=256 if image.dtype in (np.uint8,np.int8) else core.rank_bins(image)
    if bins>1024:warnings.warn(f'Bad rank filter performance is expected due to a large number of bins ({bins}), equivalent to an approximate bitdepth of {np.log2(bins):.1f}.',UserWarning,stacklevel=3)
    for name,value in zip(('shift_x','shift_y'),(shift_x,shift_y)):
        if np.dtype(type(value))==bool:warnings.warn(f'Paramter `{name}` is boolean and will be interpreted as int. This is not officially supported, use int instead.',UserWarning,stacklevel=3)
    if out.ndim!=3 or not (out.dtype.kind in 'ub' and out.dtype.itemsize in (1,2) or out.dtype.kind=='f' and out.dtype.itemsize in (4,8)):raise TypeError('Function call with ambiguous argument types')
    core.rank_percentile_run(image,footprint,mask,out,[shift_x,shift_y],bins,operation,p0,p1)
    # In contrast to the ordinary scalar shell, original percentile reshapes;
    # a multi-channel destination can fail here after first-channel writes.
    return out.reshape(out.shape[:2])


def function(operation,name):
    if name in contracts.SINGLE:
        def call(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,p0=0):return apply(operation,image,footprint,out,mask,shift_x,shift_y,p0,0.)
    else:
        def call(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,p0=0,p1=1):return apply(operation,image,footprint,out,mask,shift_x,shift_y,p0,p1)
    call.__name__=call.__qualname__=name
    return call


native=SimpleNamespace(**{name:function(17+i,name) for i,name in enumerate(contracts.NAMES)})


@pytest.fixture(autouse=True,params=(1,4))
def workers(request):
    previous=core.get_num_threads()
    if request.param==4 and os.environ.get('SKIMAGEXX_EXPECT_OPENMP_OFF')=='1':
        with pytest.raises(ValueError,match='this build has no OpenMP support'):core.set_num_threads(4)
        assert core.get_num_threads()==previous==1
        pytest.skip('OpenMP-off rejects worker4; parallel parity is not admitted')
    core.set_num_threads(request.param)
    yield
    core.set_num_threads(previous)


@pytest.fixture(scope='module')
def frozen():return json.loads(contracts.MATRIX.read_text())


@pytest.mark.parametrize('case',contracts.CASES,ids=[c[0] for c in contracts.CASES])
def test_frozen_original_and_actual_private_binding(case,frozen):
    key,name,parameters=case
    assert contracts.observe(getattr(native,name),name,parameters)==frozen['observations'][frozen['snapshots'][contracts.profile()][key]],key


@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('failure',('none','p0','p1','bad-output','bad-shift','bad-footprint'))
def test_exact_parameter_conversion_order(name,failure,frozen,monkeypatch):
    monkeypatch.setattr(contracts,'reference',native)
    assert contracts.parameter_observe(name,failure)==frozen['parameter_events'][contracts.profile()][name+'/'+failure]


@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('dtype',('u1','u2'))
@pytest.mark.parametrize('kind',('dense','even','disconnected','zero','masked','shift'))
@pytest.mark.parametrize('interval',((0.,1.),(.25,.75),(.5,.5),(1.,1.),(0.,0.)))
@pytest.mark.parametrize('output',('same','f4','f8'))
def test_independent_bin_and_cast_replay(name,dtype,kind,interval,output,monkeypatch):
    monkeypatch.setattr(contracts,'reference',native)
    contracts.test_independent_bin_scan_replay(name,dtype,kind,interval,output)


@pytest.mark.parametrize('dtype',('u1','u2'))
@pytest.mark.parametrize('kind',('dense','even','masked','empty'))
@pytest.mark.parametrize('q',(contracts.Fraction(0),contracts.Fraction(1,8),contracts.Fraction(1,4),contracts.Fraction(1,2),contracts.Fraction(3,4),contracts.Fraction(7,8),contracts.Fraction(1)))
def test_independent_sorted_order_statistics(dtype,kind,q,monkeypatch):
    monkeypatch.setattr(contracts,'reference',native)
    contracts.test_sorted_order_statistic_oracle_without_histogram_scan(dtype,kind,q)


def test_separate_capabilities_and_old_admission():
    assert (core.rank_histogram_capabilities,core.rank_alias_capabilities,core.rank_statistics_capabilities,core.rank_vector_capabilities)==(8191,480,0xe000,0x10000)
    assert core.rank_percentile_interface=='rank-percentile-v1' and core.rank_percentile_capabilities==0x3fe0000
    for op in (17,25,26,31,32):
        with pytest.raises(ValueError,match='invalid rank operation'):core.rank_run(np.ones((2,3),np.uint8),np.ones((1,1),np.uint8),None,np.zeros((2,3,1),np.uint8),[0,0],256,op)


@pytest.mark.parametrize('name',contracts.NAMES)
def test_no_original_image_algorithm_forwarding(name,monkeypatch):
    image=np.arange(35,dtype=np.uint8).reshape(5,7);fp=np.ones((3,3),np.uint8)
    expected=getattr(reference,name)(image,fp)
    def forbidden(*a,**k):raise AssertionError('upstream percentile forwarding')
    for target in (reference,contracts.percentile_module):
        for operation in contracts.NAMES:monkeypatch.setattr(target,operation,forbidden)
    assert getattr(native,name)(image,fp).tobytes()==expected.tobytes()


@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('dtype',(np.uint8,np.uint16))
def test_parallel_large_exact_bits_and_original(name,dtype):
    image=((np.arange(513*519)*37)%71).astype(dtype).reshape(513,519)
    fp=np.ones((3,5),np.uint8);fp.flat[::4]=0;mask=(np.arange(image.size).reshape(image.shape)%4!=0).astype(np.uint8)
    options=dict(mask=mask,p0=.25,out=np.empty(image.shape,np.float64))
    if name not in contracts.SINGLE:options['p1']=.75
    expected=getattr(reference,name)(image,fp,**options).copy()
    core.set_num_threads(1);left=getattr(native,name)(image,fp,**options).copy()
    if os.environ.get('SKIMAGEXX_EXPECT_OPENMP_OFF')=='1':
        with pytest.raises(ValueError,match='this build has no OpenMP support'):core.set_num_threads(4)
        assert core.get_num_threads()==1 and expected.tobytes()==left.tobytes()
        return
    core.set_num_threads(4);right=getattr(native,name)(image,fp,**options).copy()
    assert expected.tobytes()==left.tobytes()==right.tobytes()


@pytest.mark.parametrize('p0,p1',[(float('nan'),1.),(0.,float('inf')),(-.1,1.),(.8,.2),(0.,1.1)])
def test_unsafe_percentile_domain_rejected_without_original_call_or_writes(p0,p1):
    image=np.arange(15,dtype=np.uint8).reshape(3,5);out=np.full((3,5,1),99,np.uint8)
    with pytest.raises(ValueError,match='parameters outside defined coverage'):core.rank_percentile_run(image,np.ones((3,3),np.uint8),None,out,[0,0],256,17,p0,p1)
    assert np.all(out==99)


def test_undefined_unsigned_subtraction_rejected_prewrite():
    image=np.full((3,5),65535,np.uint16);out=np.full((3,5,1),99,np.uint8)
    with pytest.raises(ValueError,match='unsigned conversion outside defined coverage'):core.rank_percentile_run(image,np.ones((3,3),np.uint8),None,out,[0,0],65536,22,0.,1.)
    assert np.all(out==99)


def test_debug_assertions_when_required():
    if os.environ.get('SKIMAGEXX_REQUIRE_GIL_ASSERTIONS')=='1':assert core.gil_assertions is True


@pytest.mark.parametrize('workers',[1],indirect=True)
def test_configured_worker_policy():
    assert core.get_num_threads()==1
    if os.environ.get('SKIMAGEXX_EXPECT_OPENMP_OFF')=='1':
        with pytest.raises(ValueError,match='this build has no OpenMP support'):core.set_num_threads(4)
        assert core.get_num_threads()==1
    else:
        core.set_num_threads(4);assert core.get_num_threads()==4
        core.set_num_threads(1)
