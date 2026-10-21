"""RGBA composition, raw grayscale expansion and upstream metadata behavior."""
import warnings
import numpy as np
import pytest
from skimage import color as original
import skimagexx as xx

REAL=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float32,np.float64)
COPY=REAL+(np.float16,np.complex64,np.complex128)

@pytest.fixture(autouse=True,params=[1,4])
def native(request):
    previous=xx.set_fallback_enabled(False);xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous);xx.set_num_threads(1)


def check(name,image,*args,**kwargs):
    def run(module):
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            value=getattr(module,name)(image,*args,**kwargs)
        return value,[(type(w.message),str(w.message)) for w in notices]
    expected,expected_notices=run(original);actual,notices=run(xx.color)
    assert actual.shape==expected.shape and actual.dtype==expected.dtype
    assert notices==expected_notices
    if name=='rgba2rgb':
        tol=2e-6 if actual.dtype==np.float32 else 2e-13
        np.testing.assert_allclose(actual,expected,rtol=tol,atol=tol,equal_nan=True)
    else:np.testing.assert_array_equal(actual,expected)
    return actual,expected

@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('shape',[(4,),(7,4),(5,7,4),(2,5,7,4),(0,4)])
@pytest.mark.parametrize('background',[(1,1,1),(0,.3,.9),np.array([.2,.4,.6],dtype=np.float32)])
def test_rgba_dtypes_dimensions(dtype,shape,background):
    raw=np.random.default_rng(95).integers(0,256,shape,dtype=np.uint8)
    a=raw>127 if dtype==np.bool_ else raw.astype(dtype)
    if dtype in (np.float32,np.float64):a=a/dtype(91)-dtype(.7)
    check('rgba2rgb',a,background)

@pytest.mark.parametrize('dtype',[np.uint8,np.uint16,np.float32,np.float64])
@pytest.mark.parametrize('axis',[-4,-3,-2,-1,0,1,2,3])
def test_rgba_channel_axes(dtype,axis):
    a=np.random.default_rng(96).integers(0,256,(2,5,7,4),dtype=np.uint8).astype(dtype)
    if dtype in (np.float32,np.float64):a/=dtype(255)
    a=np.moveaxis(a,-1,axis)
    check('rgba2rgb',a,channel_axis=axis)

@pytest.mark.parametrize('name',['gray2rgb','gray2rgba'])
@pytest.mark.parametrize('dtype',COPY)
@pytest.mark.parametrize('shape',[(),(17,),(5,7),(2,5,7),(0,7)])
def test_expansion_dtypes_dimensions(name,dtype,shape):
    a=np.random.default_rng(97).integers(0,256,shape,dtype=np.uint8).astype(dtype)
    if dtype in (np.complex64,np.complex128):a=a+dtype(1j)
    options={'alpha':dtype(.7)} if name=='gray2rgba' and np.issubdtype(dtype,np.complexfloating) else {}
    check(name,a,**options)

@pytest.mark.parametrize('name',['gray2rgb','gray2rgba'])
@pytest.mark.parametrize('axis',[-4,-3,-2,-1,0,1,2,3])
def test_expansion_channel_axes(name,axis):
    a=np.random.default_rng(98).normal(size=(3,5,7)).astype(np.float32)
    check(name,a,channel_axis=axis)

@pytest.mark.parametrize('dtype',[np.uint8,np.int16,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('alpha',[0.,.3,1.,-1.,256.,np.float32(.7),np.nan])
def test_scalar_alpha_casts(dtype,alpha):
    a=np.arange(35,dtype=dtype).reshape(5,7)
    check('gray2rgba',a,alpha)

@pytest.mark.parametrize('dtype',[np.uint8,np.int16,np.float32,np.float64])
@pytest.mark.parametrize('shape',[(1,),(7,),(1,7),(5,1),(5,7)])
def test_alpha_broadcast_arrays(dtype,shape):
    a=np.arange(35,dtype=dtype).reshape(5,7)
    alpha=np.full(shape,1,dtype=dtype)
    check('gray2rgba',a,alpha)

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_nan_alpha_array_warning(dtype):
    a=np.ones((5,7),dtype=dtype);alpha=np.ones_like(a);alpha[2,3]=np.nan
    check('gray2rgba',a,alpha)

@pytest.mark.parametrize('name',['gray2rgb','gray2rgba','rgba2rgb'])
@pytest.mark.parametrize('layout',['reversed','stepped','fortran','readonly','unaligned'])
def test_layouts(name,layout):
    shape=(17,19,4) if name=='rgba2rgb' else (17,19)
    a=np.random.default_rng(99).random(shape).astype(np.float32)
    if layout=='reversed':a=a[::-1,::-1]
    if layout=='stepped':a=np.tile(a,(2,2,1) if a.ndim==3 else (2,2))[::2,::2]
    if layout=='fortran':a=np.asfortranarray(a)
    if layout=='readonly':a.flags.writeable=False
    if layout=='unaligned':
        b=np.ndarray(a.shape,dtype=a.dtype,buffer=bytearray(a.nbytes+1),offset=1);b[:]=a;a=b
    options={'alpha':a[::-1,::-1]} if name=='gray2rgba' else {}
    check(name,a,**options)

@pytest.mark.parametrize('name',['gray2rgb','gray2rgba','rgba2rgb'])
@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
def test_large_parallel(name,dtype):
    shape=(513,517,4) if name=='rgba2rgb' else (513,517)
    a=np.random.default_rng(100).integers(0,256,shape,dtype=np.uint8)
    if dtype!=np.uint8:a=(a/255).astype(dtype)
    check(name,a)

@pytest.mark.parametrize('name',['gray2rgb','gray2rgba'])
@pytest.mark.parametrize('dtype',['>f2','>f4','>f8','>i2','>u4','>c8','>c16','>U5','S5','U5','V5'])
def test_fixed_non_native_storage(name,dtype):
    kind=np.dtype(dtype).kind
    if kind=='V':
        a=np.zeros((5,7),dtype=dtype);a.view(np.uint8).reshape(-1)[:]=np.arange(a.nbytes,dtype=np.uint8)
    else:a=np.arange(35).astype(dtype).reshape(5,7)
    if name=='gray2rgba' and kind in 'SUV':return
    options={'alpha':np.asarray(.7+.3j,dtype=dtype)[()]} if name=='gray2rgba' and kind=='c' else {}
    check(name,a,**options)

@pytest.mark.parametrize('background',[(0,1),(.1,.2,.3,.4),(-.1,0,0),(0,1.1,0)])
def test_invalid_background(background):
    a=np.ones((5,7,4),dtype=np.float32)
    with pytest.raises(ValueError):original.rgba2rgb(a,background)
    with pytest.raises(ValueError):xx.color.rgba2rgb(a,background)

@pytest.mark.parametrize('axis',[None,np.int64(-1),[-1]])
def test_rgba_axis_type_errors(axis):
    a=np.ones((5,7,4),dtype=np.float32)
    with pytest.raises(TypeError):original.rgba2rgb(a,channel_axis=axis)
    with pytest.raises(TypeError):xx.color.rgba2rgb(a,channel_axis=axis)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('background',[(np.nan,.3,.7),(-0.,-0.,-0.)])
def test_rgba_background_nan_and_signed_zero(dtype,background):
    a=np.zeros((3,7,4),dtype=dtype);a[...,:3]=-0.;a[...,3]=.5
    actual,expected=check('rgba2rgb',a,background)
    finite=np.isfinite(expected)
    np.testing.assert_array_equal(np.signbit(actual[finite]),np.signbit(expected[finite]))

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_gray_bit_patterns_are_preserved(dtype):
    bits=np.array([0x80000000,0x7fc12345,0x3f800001],dtype=np.uint32) if dtype==np.float32 else np.array([0x8000000000000000,0x7ff8123456789000,0x3ff0000000000001],dtype=np.uint64)
    a=np.tile(bits.view(dtype),13)
    for name in ('gray2rgb','gray2rgba'):
        actual,expected=check(name,a)
        word=np.uint32 if dtype==np.float32 else np.uint64
        np.testing.assert_array_equal(actual.view(word),expected.view(word))

@pytest.mark.parametrize('dtype',[np.complex64,np.complex128,'datetime64[ns]','timedelta64[ns]'])
def test_alpha_missing_values_warn(dtype):
    kind=np.dtype(dtype).kind
    if kind=='c':a=np.ones((5,7),dtype=dtype);alpha=a.copy();alpha[2,3]=np.nan+1j
    else:
        a=np.arange(35,dtype=np.int64).reshape(5,7).astype(dtype)
        alpha=a.copy();alpha[2,3]=np.datetime64('NaT') if kind=='M' else np.timedelta64('NaT')
    check('gray2rgba',a,alpha)

@pytest.mark.parametrize('name',['gray2rgb','gray2rgba','rgba2rgb'])
def test_native_outputs_do_not_alias_input(name):
    a=np.ones((7,11,4) if name=='rgba2rgb' else (7,11),dtype=np.float32)
    result,_=check(name,a)
    assert not np.shares_memory(result,a)
    result.flat[0]=.2
    assert a.flat[0]==1

@pytest.mark.parametrize('dtype',[np.uint8,np.uint16,np.float32,np.float64])
def test_rgba_genuine_planar_storage(dtype):
    a=np.random.default_rng(111).integers(0,256,(17,19,4),dtype=np.uint8).astype(dtype)
    if dtype in (np.float32,np.float64):a/=dtype(255)
    planar=np.ascontiguousarray(np.moveaxis(a,-1,0))
    check('rgba2rgb',planar,channel_axis=0)

@pytest.mark.parametrize('name',['gray2rgb','gray2rgba','rgba2rgb'])
def test_unsupported_classes_are_explicit(name):
    a=np.ma.array(np.ones((5,7,4) if name=='rgba2rgb' else (5,7),dtype=np.float32))
    with pytest.raises(NotImplementedError):getattr(xx.color,name)(a)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning):actual=getattr(xx.color,name)(a)
    expected=getattr(original,name)(a)
    assert type(actual) is type(expected)
    np.testing.assert_array_equal(actual,expected)

@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
def test_alpha_broadcast_shape_error_preserves_prior_warning(dtype):
    a=np.ones((5,7),dtype=dtype);alpha=np.full((3,4),.3)
    xx.set_fallback_enabled(True)
    # A differently typed non-scalar alpha is outside the native casting path.
    # Its upstream warning and shape error remain visible through fallback.
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        with pytest.raises(ValueError,match='alpha.shape must match image.shape'):xx.color.gray2rgba(a,alpha)
    with warnings.catch_warnings(record=True) as expected_notices:
        warnings.simplefilter('always')
        with pytest.raises(ValueError,match='alpha.shape must match image.shape'):original.gray2rgba(a,alpha)
    actual=[(type(w.message),str(w.message)) for w in notices if not isinstance(w.message,xx.FallbackWarning)]
    assert actual==[(type(w.message),str(w.message)) for w in expected_notices]
