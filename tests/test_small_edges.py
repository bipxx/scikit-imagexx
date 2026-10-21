"""Reference contracts for reflected Laplace and Roberts filters."""
import warnings
import numpy as np
import pytest
from skimage import filters as original
import skimagexx as xx

NAMES=('laplace','roberts','roberts_pos_diag','roberts_neg_diag')
REAL=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64)


def sample(dtype,shape,seed=411):
    rng=np.random.default_rng(seed);kind=np.dtype(dtype).kind
    if kind=='b':return rng.random(shape)>.5
    if kind in 'iu':
        info=np.iinfo(dtype);return rng.integers(int(info.min),int(info.max)+1,shape,dtype=dtype)
    return rng.normal(size=shape).astype(dtype)


@pytest.fixture(autouse=True,params=[1,4])
def native(request):
    previous=xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous)
    xx.set_num_threads(1)


def check(name,image,**options):
    with warnings.catch_warnings(record=True) as expected_notices:
        warnings.simplefilter('always');expected=getattr(original,name)(image,**options)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always');actual=getattr(xx.filters,name)(image,**options)
    assert [(type(w.message),str(w.message)) for w in notices]==[(type(w.message),str(w.message)) for w in expected_notices]
    assert actual.dtype==expected.dtype and actual.shape==expected.shape
    tolerance=2e-6 if expected.dtype==np.float32 else 2e-13
    np.testing.assert_allclose(actual,expected,rtol=tolerance,atol=tolerance)
    assert not np.shares_memory(actual,image)
    return actual,expected


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('shape',[(1,1),(1,17),(17,1),(2,2),(3,7),(17,19),(129,131)])
def test_shapes_and_dtypes(name,dtype,shape):
    image=sample(dtype,shape)
    check(name,image)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('layout',['reversed','stepped','fortran','unaligned','readonly'])
def test_layouts(name,dtype,layout):
    image=sample(dtype,(33,35),412)
    if layout=='reversed':image=image[::-1,::-1]
    elif layout=='stepped':image=np.tile(image,(2,2))[::2,::2]
    elif layout=='fortran':image=np.asfortranarray(image)
    elif layout=='readonly':image.flags.writeable=False
    else:
        storage=np.empty(image.nbytes+1,dtype=np.uint8)
        view=np.ndarray(image.shape,dtype=dtype,buffer=storage,offset=1)
        view[:]=image;image=view
    check(name,image)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('scale',[.1,1.,10000.])
def test_floating_rounding(name,dtype,scale):
    image=(np.random.default_rng(413).normal(size=(257,263))*scale).astype(dtype)
    check(name,image)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_signed_zero(name,dtype):
    image=np.zeros((17,19),dtype=dtype)
    image[::2,::3]=-0.
    actual,expected=check(name,image)
    np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('case',['complex-mask','complex','object','nonfinite','subclass','empty','volume'])
def test_unsupported_options_remain_explicit(name,case):
    image=np.ones((5,7),dtype=np.float32);options={}
    if case=='complex-mask':options['mask']=np.ones_like(image,dtype=np.complex64)
    elif case=='complex':image=image.astype(np.complex64)
    elif case=='object':image=image.astype(object)
    elif case=='nonfinite':image[1,2]=np.nan
    elif case=='subclass':image=np.ma.array(image)
    elif case=='empty':image=image[:0]
    else:image=image[None,...]
    if name=='laplace' and case in ('empty','volume'):
        check(name,image);return
    with pytest.raises(NotImplementedError):getattr(xx.filters,name)(image,**options)
    xx.set_fallback_enabled(True)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:expected=getattr(original,name)(image,**options)
        except Exception as error:
            with pytest.warns(xx.FallbackWarning):
                with pytest.raises(type(error)):getattr(xx.filters,name)(image,**options)
            return
    with pytest.warns(xx.FallbackWarning):actual=getattr(xx.filters,name)(image,**options)
    assert actual.dtype==expected.dtype
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('ksize',[np.int64(3),5,7])
def test_laplace_kernel_metadata(ksize):
    image=np.arange(35,dtype=np.float64).reshape(5,7)
    check('laplace',image,ksize=ksize)


@pytest.mark.parametrize('name',NAMES[1:])
def test_underflow_warning_mode_uses_explicit_fallback(name):
    image=np.full((5,7),np.finfo(np.float32).tiny,dtype=np.float32)
    with np.errstate(under='warn'):
        with pytest.raises(NotImplementedError):getattr(xx.filters,name)(image)


@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('shape',[(7,),(1,),(2,5,7),(2,1,17,1),(1,1,17),(1,17,1),(3,3,5,7)])
@pytest.mark.parametrize('ksize',[3,5,8])
def test_laplace_dimensions_and_kernel_metadata(dtype,shape,ksize):
    check('laplace',sample(dtype,shape,419),ksize=ksize)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',REAL)
def test_large_parallel_arrays(name,dtype):
    check(name,sample(dtype,(513,517),420))


def test_all_finite_half_bit_patterns_decode_exactly():
    words=np.arange(65536,dtype=np.uint16)
    half=words[(words&0x7c00)!=0x7c00].view(np.float16)
    actual=xx._core.edge_half(half)
    expected=half.astype(np.float32)
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
    for name in NAMES:check(name,half.reshape(248,256))


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',REAL)
def test_numeric_masks(name,dtype):
    image=sample(np.float32,(19,23),421)
    mask=sample(dtype,image.shape,422)
    mask[3:8,7:12]=0
    before=mask.copy()
    if dtype==np.float16:
        with pytest.raises(RuntimeError,match='data type not supported'):getattr(original,name)(image,mask=mask)
        with pytest.raises(RuntimeError,match='data type not supported'):getattr(xx.filters,name)(image,mask=mask)
        return
    check(name,image,mask=mask)
    np.testing.assert_array_equal(mask,before)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('shape',[(),(23,),(1,23),(19,1),(19,23)])
@pytest.mark.parametrize('value',[False,True])
def test_masks_erode_before_broadcasting(name,shape,value):
    image=sample(np.float64,(19,23),423)
    check(name,image,mask=np.full(shape,value,dtype=bool))


@pytest.mark.parametrize('dtype',[np.uint8,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('shape',[(17,),(5,7,9),(5,5,7,9),(3,3,3,3,7),(0,7)])
def test_laplace_nd_masks(dtype,shape):
    image=sample(dtype,shape,424)
    mask=np.ones(shape,dtype=bool)
    if mask.size:mask.flat[mask.size//2]=False
    check('laplace',image,mask=mask)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('layout',['reverse','stepped','fortran','broadcast','alias'])
def test_strided_and_aliasing_masks(name,layout):
    image=sample(np.float32,(33,35),425)
    mask=np.ones(image.shape,dtype=np.float32)
    mask[7:13,11:23]=0
    if layout=='reverse':mask=mask[::-1,::-1]
    elif layout=='stepped':mask=np.tile(mask,(2,2))[::2,::2]
    elif layout=='fortran':mask=np.asfortranarray(mask)
    elif layout=='broadcast':mask=np.ones((35,),dtype=bool)
    else:mask=image
    before=image.copy()
    check(name,image,mask=mask)
    np.testing.assert_array_equal(image,before)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_nonfinite_masks_use_nonzero_truth(name,dtype):
    image=sample(np.float32,(17,19),426)
    mask=np.ones(image.shape,dtype=dtype)
    mask[5,6]=np.nan;mask[7,8]=np.inf;mask[9,10]=-np.inf
    check(name,image,mask=mask)


@pytest.mark.parametrize('name',NAMES)
def test_mask_multiplication_preserves_negative_zero(name):
    image=sample(np.float32,(17,19),427)
    actual,expected=check(name,image,mask=np.zeros(image.shape,dtype=bool))
    np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',['>i2','>u2','>i4','>u4','>i8','>u8','>f2','>f4','>f8'])
@pytest.mark.parametrize('layout',['contiguous','reverse','strided'])
def test_non_native_byte_order(name,dtype,layout):
    image=sample(np.dtype(dtype).newbyteorder('='),(17,19),428).astype(dtype)
    if layout=='reverse':image=image[::-1,::-1]
    elif layout=='strided':image=np.tile(image,(2,2))[::2,::2]
    check(name,image)


@pytest.mark.parametrize('dtype',['>i2','>u4','>i8','>f4','>f8'])
@pytest.mark.parametrize('name',NAMES)
def test_non_native_masks(name,dtype):
    image=sample(np.float64,(19,23),429)
    mask=np.ones(image.shape,dtype=dtype);mask[7:13,9:17]=0
    check(name,image,mask=mask[::-1,::-1])


@pytest.mark.parametrize('dtype',['>f2','>f4','>f8'])
def test_laplace_non_native_volume(dtype):
    image=sample(np.dtype(dtype).newbyteorder('='),(3,7,19),430).astype(dtype)
    check('laplace',image,ksize=7)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('mask_shape',[(0,7),(3,4),(2,5,7)])
def test_mask_shape_errors_follow_upstream(name,mask_shape):
    image=sample(np.float32,(5,7),431);mask=np.ones(mask_shape,dtype=bool)
    with pytest.raises(ValueError) as expected:getattr(original,name)(image,mask=mask)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning):
        with pytest.raises(ValueError) as actual:getattr(xx.filters,name)(image,mask=mask)
    assert str(actual.value)==str(expected.value)


@pytest.mark.parametrize('name',NAMES)
def test_scalar_half_mask_error(name):
    image=sample(np.float64,(17,19),432);mask=np.asarray(1,dtype=np.float16)
    with pytest.raises(RuntimeError,match='data type not supported'):getattr(original,name)(image,mask=mask)
    with pytest.raises(RuntimeError,match='data type not supported'):getattr(xx.filters,name)(image,mask=mask)


def test_empty_half_mask_is_valid_for_empty_laplace():
    check('laplace',np.empty((0,7),dtype=np.float16),mask=np.empty((0,7),dtype=np.float16))


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.uint8,np.float16,np.float32,np.float64])
def test_readonly_memory_mapped_images_and_masks(name,dtype,tmp_path):
    path=tmp_path/'image.raw'
    data=sample(dtype,(17,19),433)
    writable=np.memmap(path,dtype=dtype,mode='w+',shape=data.shape)
    writable[:]=data;writable.flush();del writable
    image=np.memmap(path,dtype=dtype,mode='r',shape=data.shape)
    actual,expected=check(name,image)
    assert type(actual) is type(expected) is np.ndarray
    mask_path=tmp_path/'mask.raw'
    mask_write=np.memmap(mask_path,dtype=bool,mode='w+',shape=data.shape)
    mask_write[:]=True;mask_write[7:11,8:15]=False;mask_write.flush();del mask_write
    mask=np.memmap(mask_path,dtype=bool,mode='r',shape=data.shape)
    check(name,image,mask=mask)
    np.testing.assert_array_equal(image,data)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_signaling_nan_masks_preserve_warning_behavior(name,dtype):
    image=sample(np.float32,(17,19),434)
    words=np.full(image.shape,0x3f800000 if dtype==np.float32 else 0x3ff0000000000000,
                  dtype=np.uint32 if dtype==np.float32 else np.uint64)
    words[7,9]=0x7f800001 if dtype==np.float32 else 0x7ff0000000000001
    check(name,image,mask=words.view(dtype))


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('shape',[(1,513),(513,1)])
def test_skinny_arrays_and_scalar_masks(name,dtype,shape):
    image=sample(dtype,shape,435)
    check(name,image)
    for flag in (False,True):
        actual,expected=check(name,image,mask=np.asarray(flag,dtype=bool))
        np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('shape',[(1,262145),(262145,1)])
def test_parallel_skinny_arrays(name,dtype,shape):
    check(name,sample(dtype,shape,436))


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',['f4','f8'])
def test_equivalent_native_dtype_descriptors(name,dtype):
    alias=np.dtype('>'+dtype).newbyteorder('=')
    image=sample(np.dtype(dtype),(17,19),437).view(alias)
    assert image.dtype==np.dtype(dtype)
    check(name,image)
    if name=='laplace':check(name,image[None,...])
