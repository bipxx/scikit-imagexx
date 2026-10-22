"""Histogram matching: per-call contract and reference parity with fallback off."""
import inspect
import warnings
import numpy as np
import pytest
from skimage.exposure import match_histograms as reference
import skimagexx as xx

TYPES=[np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,
       np.uint64,np.int64,np.float16,np.float32,np.float64]

@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    previous=xx.set_fallback_enabled(False);xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous);xx.set_num_threads(1)

def compare(a,b,**options):
    snapshots=[x.copy() for x in (a,b) if isinstance(x,np.ndarray)]
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:expected=reference(a,b,**options)
        except Exception as error:
            with pytest.raises(type(error)) as actual:xx.exposure.match_histograms(a,b,**options)
            assert str(actual.value)==str(error)
            return
    with warnings.catch_warnings(record=True) as actual_notices:
        warnings.simplefilter('always');actual=xx.exposure.match_histograms(a,b,**options)
    assert [(type(w.message),str(w.message)) for w in actual_notices]==[(type(w.message),str(w.message)) for w in notices]
    assert type(actual) is type(expected)
    assert actual.shape==expected.shape and actual.dtype==expected.dtype
    assert actual.dtype.metadata==expected.dtype.metadata
    np.testing.assert_allclose(actual,expected,rtol=0,atol=0,equal_nan=True)
    assert not np.shares_memory(actual,a) and not np.shares_memory(actual,b)
    for value,snapshot in zip((x for x in (a,b) if isinstance(x,np.ndarray)),snapshots):np.testing.assert_array_equal(value,snapshot)

@pytest.mark.parametrize('source',TYPES)
@pytest.mark.parametrize('template',TYPES)
@pytest.mark.parametrize('axis',[None,-1,0])
def test_dtype_pairs(source,template,axis):
    rng=np.random.default_rng(613)
    a=rng.integers(0,11,(7,9,3)).astype(source)
    b=rng.integers(0,11,(5,13,3)).astype(template)
    if source in (np.float16,np.float32,np.float64):a/=source(7)
    if template in (np.float16,np.float32,np.float64):b/=template(7)
    if axis==0:a,b=np.moveaxis(a,-1,0),np.moveaxis(b,-1,0)
    compare(a,b,channel_axis=axis)

@pytest.mark.parametrize('dtype',[np.uint8,np.int16,np.int64,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('layout',['C','F','reverse','step','unaligned','broadcast','readonly','endian','tagged'])
@pytest.mark.parametrize('axis',[None,-1])
def test_layouts_and_descriptors(dtype,layout,axis):
    a=(np.arange(9*11*3).reshape(9,11,3)%13).astype(dtype)
    b=(np.arange(5*7*3).reshape(5,7,3)%17).astype(dtype)
    if layout=='F':a,b=np.asfortranarray(a),np.asfortranarray(b)
    elif layout=='reverse':a,b=a[::-1,:,::-1],b[:,::-1,::-1]
    elif layout=='step':a,b=a[:,::2],b[::2]
    elif layout=='unaligned':
        raw=np.empty(a.nbytes+1,np.uint8);unaligned=np.ndarray(a.shape,a.dtype,raw,1);unaligned[:]=a;a=unaligned
    elif layout=='broadcast':a,b=np.broadcast_to(a[:1,:1],a.shape),np.broadcast_to(b[:1,:1],b.shape)
    elif layout=='readonly':a.flags.writeable=False;b.flags.writeable=False
    elif layout=='endian':a,b=a.astype(a.dtype.newbyteorder('>')),b.astype(b.dtype.newbyteorder('>'))
    elif layout=='tagged':a=a.view(np.dtype(a.dtype,metadata={'units':'histogram','source':1}))
    compare(a,b,channel_axis=axis)

@pytest.mark.parametrize('dtype',[np.bool_,np.uint8,np.int16,np.float32,np.float64])
@pytest.mark.parametrize('shape,other,axis',[((),(),None),((0,),(0,),None),((0,),(3,),None),
    ((3,),(0,),None),((3,0),(5,0),-1),((0,3),(5,3),-1),((3,),(2,3),None),
    ((2,3),(2,4),-1),((2,3),(4,3),2),((2,3),(4,3),()),((2,3),(4,3),(0,1)),
    ((2,3),(4,3),[-1]),((2,3),(4,3),(-1,))])
def test_empty_scalar_and_errors(dtype,shape,other,axis):
    compare(np.ones(shape,dtype),np.ones(other,dtype),channel_axis=axis)

@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('axis',[None,-1])
def test_nonfinite_and_signed_zero(dtype,axis):
    a=np.array([0.,-0.,1.,-1.,np.inf,-np.inf,np.nan,np.nan],dtype)
    b=np.array([-np.inf,0.,-0.,0.,1.,np.inf,np.nan],dtype)
    if axis is not None:a,b=np.tile(a[:,None],(1,3)),np.tile(b[:,None],(1,3))
    compare(a,b,channel_axis=axis)

@pytest.mark.parametrize('dtype',[np.int8,np.int16,np.int32,np.int64,np.uint8,np.uint16,np.uint32])
def test_integer_channel_casts(dtype):
    a=np.array([[0,1,2,3],[3,1,2,0]],dtype)
    for values in ([0,1000,2048,4095],[-1000,0,1000,100000],[0,1,2147483648,4294967295]):
        if np.dtype(dtype).kind=='u' and max(values)>100000:continue
        b=np.tile(np.array(values,np.int64)[:,None],(1,4))
        compare(a,b,channel_axis=-1)

def test_public_signature_and_keyword_axes():
    assert inspect.signature(xx.exposure.match_histograms)==inspect.signature(reference)
    a=np.arange(24,dtype=np.float32).reshape(2,4,3)
    b=np.arange(24,dtype=np.float32).reshape(2,4,3)[::-1]
    compare(a,b,channel_axis=np.int64(-1))
    # Positional-axis behavior is also checked above; upstream's decorator only
    # moves positional images when a non-final channel axis is supplied.
    expected=reference(image=a,reference=b,channel_axis=0)
    actual=xx.exposure.match_histograms(image=a,reference=b,channel_axis=0)
    np.testing.assert_array_equal(actual,expected)

@pytest.mark.parametrize('dtype',[np.uint8,np.int8,np.uint16,np.int16,np.float32,np.float64])
@pytest.mark.parametrize('length',range(18))
def test_simd_packet_boundaries(dtype,length):
    a=(np.arange(length*3).reshape(length,3)%13).astype(dtype)
    b=(np.arange((length+3)*3).reshape(length+3,3)%17).astype(dtype)
    compare(a,b,channel_axis=-1)

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('groups',[256,4095,4096,4097,262149])
def test_large_hash_and_radix_boundaries(dtype,groups):
    rng=np.random.default_rng(871)
    a=(np.arange(262149)%groups).astype(dtype);rng.shuffle(a)
    b=(np.arange(262153)%groups).astype(dtype);rng.shuffle(b)
    compare(a,b)

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_continuous_large_radix(dtype):
    rng=np.random.default_rng(882)
    a=rng.normal(size=(513,517,3)).astype(dtype)
    b=rng.normal(size=(519,511,3)).astype(dtype)
    compare(a,b,channel_axis=-1)

@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('axis',[None,-1])
@pytest.mark.parametrize('policy',['warn','raise'])
@pytest.mark.parametrize('values',[[1e-50,2e-50,3e-50,4e-50],
                                  [1e100,2e100,3e100,4e100],
                                  [-np.inf,0,np.inf,np.nan]])
def test_floating_cast_policies(dtype,axis,policy,values):
    a=np.array([[0,1],[2,3]],dtype)
    b=np.array(values,np.float64).reshape(2,2)
    with np.errstate(over=policy,under=policy,invalid=policy):
        compare(a,b,channel_axis=axis)

@pytest.mark.parametrize('dtype',[np.int16,np.int64,np.float32,np.float64])
def test_tiny_sparse_ranges(dtype):
    a=np.array([-32768,0,32767],dtype)
    b=np.array([-30000,30000,0],dtype)
    compare(a,b)

@pytest.mark.parametrize('dtype',[np.uint8,np.int16,np.float32,np.float64])
@pytest.mark.parametrize('layout',['planar','reverse-pixels','reverse-channels','zero-pixels','zero-channels','nonlinear'])
def test_linear_view_input_copy_policy(dtype,layout):
    class Observed(np.ndarray):
        copies=0
        def copy(self,*args,**kwargs):
            type(self).copies+=1
            return np.asarray(self).copy(*args,**kwargs)
    a=(np.arange(17*19*3).reshape(17,19,3)%31).astype(dtype)
    b=(np.arange(11*13*3).reshape(11,13,3)%37).astype(dtype)
    if layout=='planar':a,b=np.moveaxis(np.ascontiguousarray(np.moveaxis(a,-1,0)),0,-1),np.moveaxis(np.ascontiguousarray(np.moveaxis(b,-1,0)),0,-1)
    elif layout=='reverse-pixels':a,b=a[::-1,::-1],b[::-1,::-1]
    elif layout=='reverse-channels':a,b=a[...,::-1],b[...,::-1]
    elif layout=='zero-pixels':a,b=np.broadcast_to(a[:1,:1],a.shape),np.broadcast_to(b[:1,:1],b.shape)
    elif layout=='zero-channels':a,b=np.broadcast_to(a[...,:1],a.shape),np.broadcast_to(b[...,:1],b.shape)
    else:a,b=a[:,::2],b[:,::2]
    expected=reference(a,b,channel_axis=-1)
    actual,flags=xx._core.match_histograms(a.view(Observed),b.view(Observed),expected.dtype,True,np.dtype(dtype).kind=='u',False)
    np.testing.assert_array_equal(actual,expected)
    assert actual.strides==expected.strides
    assert not any(flags)
    assert Observed.copies==(2 if layout=='nonlinear' else 0)

@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
def test_negative_and_zero_grayscale_views(dtype):
    a=(np.arange(357)%31).astype(dtype)[::-2]
    b=np.broadcast_to(np.array([3],dtype),(191,))
    compare(a,b)
