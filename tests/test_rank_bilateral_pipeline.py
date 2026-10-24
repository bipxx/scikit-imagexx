"""Private bilateral source shell and real compiled entry, not public admission."""
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


contracts=load('rank_bilateral_private_contracts',Path(__file__).with_name('test_rank_bilateral_contracts.py'))
core=importlib.import_module(os.environ.get('SKIMAGEXX_RANK_BILATERAL_BACKEND','_rank_bilateral_core'))
shell=load('skimagexx._rank_bilateral_private_shell',Path(__file__).parents[1]/'python/skimagexx/_rank.py');shell._core=core


def check_image(image):
    value=np.asanyarray(image)
    if value.size==0:raise ValueError('The parameter `image` cannot be an empty array')
    if value.ndim!=2:raise ValueError('The parameter `image` must be a 2-dimensional array')


def apply(operation,image,footprint,out,mask,shift_x,shift_y,s0,s1):
    # Own metadata/preprocessing shell around existing native providers. No
    # upstream rank function or original preprocessing implementation is used.
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
    core.rank_bilateral_run(image,footprint,mask,out,[shift_x,shift_y],bins,operation,s0,s1)
    return out.reshape(out.shape[:2])


def function(operation,name):
    def call(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,s0=10,s1=10):
        return apply(operation,image,footprint,out,mask,shift_x,shift_y,s0,s1)
    call.__name__=call.__qualname__=name
    return call


native=SimpleNamespace(**{name:function(26+i,name) for i,name in enumerate(contracts.NAMES)})


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
@pytest.mark.parametrize('kind',('u1','f4'))
@pytest.mark.parametrize('scenario',contracts.SCENARIOS)
def test_exact_provider_mutation_stages(name,kind,scenario,frozen,monkeypatch):
    monkeypatch.setattr(contracts,'reference',native)
    assert contracts.provider_observe(name,kind,scenario)==frozen['provider_events'][contracts.profile()][name+'/'+kind+'/'+scenario]


@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('dtype',('u1','u2'))
@pytest.mark.parametrize('kind',('dense','even','disconnected','zero','masked','shift'))
@pytest.mark.parametrize('radius',((0,0),(1,1),(2,5),(5,2),(-1,5),(300,300),(65536,65536)))
@pytest.mark.parametrize('output',('same','f4','f8'))
def test_independent_integer_neighborhood(name,dtype,kind,radius,output,monkeypatch):
    monkeypatch.setattr(contracts,'reference',native)
    contracts.test_independent_integer_neighborhood_oracle(name,dtype,kind,radius,output)


@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('dtype',('u1','u2'))
@pytest.mark.parametrize('output',('u1','u2','f4','f8'))
def test_full_capacity_wrap(name,dtype,output,monkeypatch):
    monkeypatch.setattr(contracts,'reference',native)
    contracts.test_full_capacity_counts_and_integer_wrap(name,dtype,output)


@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('dtype',('u1','u2'))
@pytest.mark.parametrize('edge',('upper-max','lower-min','upper-min-safe','lower-max-safe'))
def test_signed_size_edges(name,dtype,edge,monkeypatch):
    monkeypatch.setattr(contracts,'reference',native)
    contracts.test_signed_size_safe_predicate_edges_against_python_integers(name,dtype,edge)


def test_separate_capabilities_and_old_admission():
    assert (core.rank_histogram_capabilities,core.rank_alias_capabilities,core.rank_statistics_capabilities,core.rank_vector_capabilities,core.rank_percentile_capabilities)==(8191,480,0xe000,0x10000,0x3fe0000)
    assert core.rank_bilateral_interface=='rank-bilateral-v1' and core.rank_bilateral_capabilities==0x1c000000
    if os.environ.get('SKIMAGEXX_REQUIRE_GIL_ASSERTIONS')=='1':assert core.gil_assertions is True
    for op in (25,29,31,-1):
        with pytest.raises(ValueError,match='invalid rank bilateral operation'):core.rank_bilateral_run(np.ones((2,3),np.uint8),np.ones((1,1),np.uint8),None,np.zeros((2,3,1),np.uint8),[0,0],256,op,10,10)
    for op in (26,27,28):
        with pytest.raises(ValueError,match='invalid rank operation'):core.rank_run(np.ones((2,3),np.uint8),np.ones((1,1),np.uint8),None,np.zeros((2,3,1),np.uint8),[0,0],256,op)


@pytest.mark.parametrize('name',contracts.NAMES)
def test_no_original_image_algorithm_forwarding(name,monkeypatch):
    image=np.arange(35,dtype=np.uint8).reshape(5,7);fp=np.ones((3,3),np.uint8)
    expected=getattr(reference,name)(image,fp)
    def forbidden(*a,**k):raise AssertionError('upstream bilateral forwarding')
    for target in (reference,contracts.bilateral_module):
        for operation in contracts.NAMES:monkeypatch.setattr(target,operation,forbidden)
    assert getattr(native,name)(image,fp).tobytes()==expected.tobytes()


@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('dtype',(np.uint8,np.uint16))
def test_parallel_large_exact_bits_and_original(name,dtype):
    image=((np.arange(513*519)*37)%71).astype(dtype).reshape(513,519)
    fp=np.ones((3,5),np.uint8);fp.flat[::4]=0;mask=(np.arange(image.size).reshape(image.shape)%4!=0).astype(np.uint8)
    options=dict(mask=mask,s0=17,s1=5,out=np.empty(image.shape,np.float64))
    expected=getattr(reference,name)(image,fp,**options).copy()
    first=getattr(native,name)(image,fp,**options).copy()
    assert first.tobytes()==expected.tobytes()
    if os.environ.get('SKIMAGEXX_EXPECT_OPENMP_OFF')!='1':
        previous=core.get_num_threads();core.set_num_threads(4 if previous==1 else 1)
        try:assert getattr(native,name)(image,fp,**options).tobytes()==first.tobytes()
        finally:core.set_num_threads(previous)


@pytest.mark.parametrize('operation',(26,27,28))
@pytest.mark.parametrize('dtype',('u1','u2'))
@pytest.mark.parametrize('kind',('image','mask'))
@pytest.mark.parametrize('offset',(0,1))
def test_independent_signed_feedback_replay(operation,dtype,kind,offset):
    shape=(5,7);n=np.prod(shape);wide=dtype=='u2';bins=65536 if wide else 256
    owner=((np.arange(n+1)*37+11)%bins).astype(dtype if kind=='image' else 'u1')
    if kind=='image':image=owner[:n].reshape(shape);mask=(np.arange(n).reshape(shape)%3!=0).astype('u1')
    else:image=((np.arange(n)*17)%bins).astype(dtype).reshape(shape);mask=owner[:n].reshape(shape)
    out=owner[offset:offset+n].reshape(shape);fp=np.array([[1,0,1],[1,1,0],[0,1,1]],np.uint8)
    initial=owner.copy();saved_image=image.copy();saved_mask=mask.copy()
    # Independent coordinate-set replay: updates sample the CURRENT backing
    # arrays at each move, before the current output write. Signed bins and
    # population are retained, rather than rebuilding a fresh neighborhood.
    counts={};population=0;previous=set()
    def neighborhood(y,x):
        return {(y+fy-1,x+fx-1) for fy,fx in np.ndindex(fp.shape) if fp[fy,fx] and 0<=y+fy-1<shape[0] and 0<=x+fx-1<shape[1]}
    for y in range(shape[0]):
        for x in range(shape[1]) if y%2==0 else range(shape[1]-1,-1,-1):
            current=neighborhood(y,x)
            for points,sign in ((current-previous,1),(previous-current,-1)):
                for yy,xx in points:
                    if mask[yy,xx]:
                        value=int(image[yy,xx]);counts[value]=counts.get(value,0)+sign;population+=sign
            g=int(image[y,x]);selected=[(i,c) for i,c in counts.items() if g>i-31 and g<i+9]
            count=sum(c for _,c in selected);total=sum(i*c for i,c in selected)
            value=0 if not population else count if operation==27 else 0 if not count else total if operation==28 else (abs(total)//abs(count))*(-1 if (total<0)!=(count<0) else 1)
            out[y,x]=value%(65536 if out.dtype.itemsize==2 else 256)
            previous=current
    expected=owner.copy();owner[:]=initial
    if kind=='image':mask[:]=saved_mask
    else:image[:]=saved_image
    core.rank_bilateral_run(image,fp,mask,out.reshape(shape+(1,)),[0,0],bins,operation,31,9)
    assert owner.tobytes()==expected.tobytes()
