"""Raw Half identity finish: copy, bounds, payloads and optional workers.

This direct helper contract uses NumPy's copy/reduction/clip as its reference.
The wrapper's warning ordering is covered by test_resize_native_pipeline.py.
"""
import warnings
import numpy as np
import pytest
import skimagexx as xx


WORDS=np.array([0x0000,0x8000,0x0001,0x8001,0x03ff,0x83ff,0x0400,0x8400,
                0x3c00,0xbc00,0x7bff,0xfbff,0x7c00,0xfc00,
                0x7c01,0x7d55,0x7e00,0x7fff,0xfc01,0xfd55,0xfe00,0xffff],np.uint16)


@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    previous=xx.set_fallback_enabled(False);xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous);xx.set_num_threads(1)


def reference(image,clip):
    out=np.array(image,dtype=np.float16,copy=True,order='C',subok=False)
    if not clip:return out,None,None,False,False,False
    if not image.size:
        # Use the exact runtime error for the input minimum, before maximum.
        np.min(image)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        warnings.simplefilter('ignore',DeprecationWarning)
        nan=bool(np.isnan(image).any());valid=bool((~np.isnan(image)).any())
        low,high=(np.nanmin(image),np.nanmax(image)) if nan else (np.min(image),np.max(image))
        lower,upper=np.asarray(low),np.asarray(high)
        np.clip(out,lower,upper,out=out)
    positive=bool(((image==0)&~np.signbit(image)).any())
    negative=bool(((image==0)&np.signbit(image)).any())
    runtime_order=bool(not valid or (positive and negative and (low==0 or high==0)))
    return out,lower,upper,nan,valid,runtime_order


def check(image,clip,policy):
    before=image.tobytes();metadata=(image.dtype,image.shape,image.strides,image.flags.writeable,image.ctypes.data)
    expected=reference(image,clip)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        actual=xx._core.resize_half_identity(image,clip,policy)
    # Raw helper work is quiet. The public wrapper reproduces warnings after
    # sampling succeeds, preserving skimage's validation/warning precedence.
    assert not notices
    assert type(actual) is tuple and len(actual)==6
    out,low,high,nan,valid,runtime_order=actual
    wanted=expected[0]
    assert type(out) is np.ndarray
    assert out.dtype==np.dtype(np.float16) and out.dtype.metadata is None and out.dtype.isnative
    assert out.shape==wanted.shape and out.strides==wanted.strides
    assert out.flags.c_contiguous and out.flags.aligned and out.flags.writeable and out.flags.owndata
    assert not np.shares_memory(out,image)
    assert out.tobytes()==wanted.tobytes()
    if not clip:
        assert low is None and high is None
        assert nan is False and valid is False and runtime_order is False
    else:
        for bound,expected_bound in zip((low,high),expected[1:3]):
            assert type(bound) is np.ndarray and bound.shape==()
            assert bound.dtype==expected_bound.dtype and bound.dtype.metadata is None
            assert bound.tobytes()==expected_bound.tobytes()
            assert not np.shares_memory(bound,image)
        assert all(type(flag) is bool for flag in (nan,valid,runtime_order))
        assert (nan,valid,runtime_order)==expected[3:]
    assert image.tobytes()==before
    assert (image.dtype,image.shape,image.strides,image.flags.writeable,image.ctypes.data)==metadata
    return out


@pytest.mark.parametrize('clip',[False,True])
@pytest.mark.parametrize('policy',[0,1])
@pytest.mark.parametrize('size',[65536,5*65536+3])
def test_all_raw_half_encodings(clip,policy,size):
    # Includes every subnormal, both signed zeros, infinities, and all positive
    # and negative signaling/quiet NaN payloads, without any float conversion.
    words=np.resize(np.arange(65536,dtype=np.uint16),size)
    check(words.view(np.float16),clip,policy)


@pytest.mark.parametrize('clip',[False,True])
@pytest.mark.parametrize('policy',[0,1])
@pytest.mark.parametrize('size',[1,2,3,4,7,8,15,16,17,31,32,33,63,64,65,1031])
def test_copy_and_range_vector_tails(clip,policy,size):
    check(np.resize(WORDS,size).view(np.float16),clip,policy)


@pytest.mark.parametrize('clip',[False,True])
@pytest.mark.parametrize('policy',[0,1])
@pytest.mark.parametrize('layout',['c','fortran','reverse','step','broadcast','readonly','unaligned','endian','endian_reverse','tagged'])
def test_layouts_and_metadata(clip,policy,layout):
    image=np.resize(WORDS,(17,19)).view(np.float16)
    if layout=='fortran':image=np.asfortranarray(image)
    elif layout=='reverse':image=image[::-1,::-1]
    elif layout=='step':image=np.resize(WORDS,(34,38)).view(np.float16)[::2,::2]
    elif layout=='broadcast':image=np.broadcast_to(image[:1],image.shape)
    elif layout=='readonly':image.flags.writeable=False
    elif layout=='unaligned':
        source=image;image=np.ndarray(image.shape,np.float16,buffer=bytearray(image.nbytes+1),offset=1);image[...]=source
    elif layout in ('endian','endian_reverse'):
        image=image.astype('>f2')
        if layout=='endian_reverse':image=image[::-1,::-1]
    elif layout=='tagged':image=image.view(np.dtype(np.float16,metadata={'unit':'raw'}))
    check(image,clip,policy)


@pytest.mark.parametrize('policy',[0,1])
@pytest.mark.parametrize('layout',['fortran','reverse','endian_reverse','unaligned'])
def test_parallel_normalization_and_range(policy,layout):
    image=np.resize(np.arange(65536,dtype=np.uint16),(257,1025)).view(np.float16)
    if layout=='fortran':image=np.asfortranarray(image)
    elif layout=='reverse':image=image[::-1,::-1]
    elif layout=='endian_reverse':image=image.astype('>f2')[::-1,::-1]
    else:
        source=image;image=np.ndarray(image.shape,np.float16,buffer=bytearray(image.nbytes+1),offset=1);image[...]=source
    check(image,True,policy)


@pytest.mark.parametrize('clip',[False,True])
@pytest.mark.parametrize('policy',[0,1])
@pytest.mark.parametrize('pattern',[[0x7c01],[0xfc01],[0x7fff],[0xffff],
    [0x7c01,0x7e00,0xfc01,0xfe00],[0x7fff,0xfc01,0x7c01,0xffff],
    [0x0000],[0x8000],[0x0000,0x8000],[0x8000,0x0000],
    [0x7c01,0x0000,0x8000],[0x0000,0x8000,0x3c00],[0xbc00,0x0000,0x8000]])
def test_nan_payloads_zero_ties_and_boundary_flags(clip,policy,pattern):
    check(np.resize(np.asarray(pattern,np.uint16),1031).view(np.float16),clip,policy)


@pytest.mark.parametrize('policy',[0,1])
@pytest.mark.parametrize('pattern',[[0x7c01,0x7d55,0xfe00,0xffff],[0x0000,0x8000]])
def test_parallel_runtime_order_provider(policy,pattern):
    check(np.resize(np.asarray(pattern,np.uint16),(257,1025)).view(np.float16),True,policy)


@pytest.mark.parametrize('policy',[0,1])
@pytest.mark.parametrize('pattern',[[0x7c01,0x7fff,0xfc01,0xffff],
                                  [0x0000,0x8000],[0x7c01,0x0000,0x8000]])
@pytest.mark.parametrize('layout',['c','fortran','reverse','step','broadcast','readonly','unaligned','endian','endian_reverse','tagged'])
def test_runtime_order_provider_layouts(policy,pattern,layout):
    words=np.asarray(pattern,np.uint16);image=np.resize(words,(17,19)).view(np.float16)
    if layout=='fortran':image=np.asfortranarray(image)
    elif layout=='reverse':image=image[::-1,::-1]
    elif layout=='step':image=np.resize(words,(34,38)).view(np.float16)[::2,::2]
    elif layout=='broadcast':image=np.broadcast_to(image[:1],image.shape)
    elif layout=='readonly':image.flags.writeable=False
    elif layout=='unaligned':
        source=image;image=np.ndarray(image.shape,np.float16,buffer=bytearray(image.nbytes+1),offset=1);image[...]=source
    elif layout in ('endian','endian_reverse'):
        image=image.astype('>f2')
        if layout=='endian_reverse':image=image[::-1,::-1]
    elif layout=='tagged':image=image.view(np.dtype(np.float16,metadata={'unit':'raw'}))
    check(image,True,policy)


@pytest.mark.parametrize('word',[0x0000,0x8000,0x0001,0x7bff,0x7c00,0x7c01,0xfe01])
@pytest.mark.parametrize('clip',[False,True])
@pytest.mark.parametrize('policy',[0,1])
def test_scalar_payloads(word,clip,policy):
    check(np.asarray(word,np.uint16).view(np.float16),clip,policy)


@pytest.mark.parametrize('shape',[(0,),(0,7),(3,0,5)])
@pytest.mark.parametrize('policy',[0,1])
def test_empty_copy_and_minimum_error(shape,policy):
    image=np.empty(shape,np.float16)
    check(image,False,policy)
    with pytest.raises(ValueError,match='zero-size array to reduction operation minimum which has no identity'):
        xx._core.resize_half_identity(image,True,policy)


@pytest.mark.parametrize('dtype',[np.bool_,np.uint16,np.float32,np.float64,np.complex64])
def test_input_dtype_guard(dtype):
    with pytest.raises(TypeError):xx._core.resize_half_identity(np.ones((3,5),dtype),True,1)


@pytest.mark.parametrize('policy',[-1,2,99])
def test_policy_guard(policy):
    with pytest.raises(ValueError,match='policy'):
        xx._core.resize_half_identity(np.ones((3,5),np.float16),True,policy)


def test_no_python_image_operations(monkeypatch):
    images=[np.arange(65536,dtype=np.uint16).view(np.float16),
            np.resize(np.array([0x0000,0x8000,0x7c01],np.uint16),1031).view(np.float16),
            np.resize(np.array([0x7c01,0xffff],np.uint16),1031).view(np.float16)]
    expected=[reference(image,True) for image in images]
    def forbidden(*args,**kwargs):raise AssertionError('Python image operation called')
    for name in ['min','max','nanmin','nanmax','minimum','maximum','fmin','fmax','clip','copyto']:
        monkeypatch.setattr(np,name,forbidden)
    for image,wanted in zip(images,expected):
        out,low,high,nan,valid,runtime_order=xx._core.resize_half_identity(image,True,1)
        assert out.tobytes()==wanted[0].tobytes()
        assert low.tobytes()==wanted[1].tobytes() and high.tobytes()==wanted[2].tobytes()
        assert (nan,valid,runtime_order)==wanted[3:]
