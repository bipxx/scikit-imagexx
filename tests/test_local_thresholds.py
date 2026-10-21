"""Local threshold contracts: integral ordering, reflection and dtype stages."""
import numpy as np
import pytest
from skimage import filters as original
import skimagexx as xx

NAMES=('threshold_niblack','threshold_sauvola')
DTYPES=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float32,np.float64)

@pytest.fixture(autouse=True,params=[1,4])
def native(request):
    previous=xx.set_fallback_enabled(False);xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous);xx.set_num_threads(1)


def compare(name,a,**options):
    expected=getattr(original,name)(a,**options)
    actual=getattr(xx.filters,name)(a,**options)
    assert actual.shape==expected.shape
    assert actual.dtype==expected.dtype
    # Identical integral accumulation order is required even when final output
    # is float32. Float64 output retains the project's strict base tolerance.
    tol=2e-6 if expected.dtype==np.float32 else 2e-13
    np.testing.assert_allclose(actual,expected,rtol=tol,atol=tol,equal_nan=True)
    return actual,expected

@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('shape',[(19,),(7,11),(3,7,11),(1,1),(1,9,1)])
@pytest.mark.parametrize('window',[1,3,15])
def test_dtype_dimension_window(name,dtype,shape,window):
    raw=np.random.default_rng(75).integers(0,256,shape,dtype=np.uint8)
    a=(raw>127) if dtype==np.bool_ else raw.astype(dtype)
    if dtype in (np.float32,np.float64):a=a/dtype(73)-dtype(2)
    compare(name,a,window_size=window)

@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('window',[(1,5),(7,3),np.int64(15),np.array([3,5],dtype=np.int64)])
def test_window_metadata_precision(name,dtype,window):
    a=np.random.default_rng(75).random((13,17)).astype(dtype)
    if dtype==np.uint8:a=np.random.default_rng(75).integers(0,256,a.shape,dtype=dtype)
    compare(name,a,window_size=window)

@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('k',[0.,-.2,.2,1.,np.float32(.17),np.float64(.17)])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_weight_promotions(name,k,dtype):
    a=np.random.default_rng(76).normal(size=(17,21)).astype(dtype)
    compare(name,a,k=k)

@pytest.mark.parametrize('r',[None,.7,-2.,np.float32(.37),np.float64(.37)])
@pytest.mark.parametrize('k',[.2,np.float64(.17)])
def test_sauvola_range_promotion(r,k):
    a=np.random.default_rng(77).normal(size=(17,21)).astype(np.float32)
    compare('threshold_sauvola',a,k=k,r=r)

@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('layout',['reversed','fortran','stepped','readonly','unaligned'])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_layouts(name,layout,dtype):
    a=np.random.default_rng(78).random((19,23)).astype(dtype)
    if layout=='reversed':a=a[::-1,::-1]
    if layout=='fortran':a=np.asfortranarray(a)
    if layout=='stepped':a=np.tile(a,(2,2))[::2,::2]
    if layout=='readonly':a.flags.writeable=False
    if layout=='unaligned':
        b=np.ndarray(a.shape,dtype=dtype,buffer=bytearray(a.nbytes+1),offset=1);b[:]=a;a=b
    compare(name,a)

@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
def test_large_parallel(name,dtype):
    a=np.random.default_rng(79).integers(0,256,(513,517),dtype=np.uint8)
    if dtype!=np.uint8:a=(a/255).astype(dtype)
    compare(name,a,window_size=(31,15))

@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('value',[0.,-0.,1.,-3.])
def test_constant_and_zero_sign(name,dtype,value):
    a=np.full((11,13),value,dtype=dtype)
    actual,expected=compare(name,a,k=0.)
    np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))

@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('window',[0,2,-1,(3,4),(3,),(),3.5,np.uint8(15)])
def test_invalid_window_contract(name,window):
    a=np.arange(77,dtype=np.float64).reshape(7,11)
    try:getattr(original,name)(a,window_size=window)
    except (ValueError,TypeError,IndexError,RuntimeError) as error:
        with pytest.raises(type(error)):getattr(xx.filters,name)(a,window_size=window)
    else:pytest.fail('reference accepted an expected-invalid window')

@pytest.mark.parametrize('name',NAMES)
def test_pipeline_swaps_backend(name):
    import skimage
    a=np.random.default_rng(80).integers(0,256,(37,41),dtype=np.uint8)
    def pipeline(backend):return a>getattr(backend.filters,name)(a,window_size=7)
    np.testing.assert_array_equal(pipeline(xx),pipeline(skimage))


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.uint64,np.int64])
def test_wide_integer_extrema(name,dtype):
    limits=np.iinfo(dtype)
    a=np.array([limits.min,limits.max,0,1,limits.max-1,limits.min+1]*13,dtype=dtype).reshape(6,13)
    compare(name,a,window_size=(3,5))

@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('value',[1.,1000.,1e6])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_nearly_constant_variance(name,value,dtype):
    a=np.full((37,41),value,dtype=dtype)
    a[::3,::7]=np.nextafter(dtype(value),dtype(np.inf))
    compare(name,a,window_size=31)

@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.int8,np.int16])
def test_wrapped_window_area_fallback(name,dtype):
    import warnings
    shape=(3,7) if dtype==np.int8 else (3,5,7)
    a=np.random.default_rng(87).random(shape).astype(np.float32)
    window=dtype(15 if dtype==np.int8 else 33)
    with pytest.raises(NotImplementedError):getattr(xx.filters,name)(a,window_size=window)
    xx.set_fallback_enabled(True)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        actual=getattr(xx.filters,name)(a,window_size=window)
    assert any(isinstance(w.message,xx.FallbackWarning) for w in notices)
    with warnings.catch_warnings(record=True) as reference_notices:
        warnings.simplefilter('always')
        expected=getattr(original,name)(a,window_size=window)
    remaining=[(type(w.message),str(w.message)) for w in notices if not isinstance(w.message,xx.FallbackWarning)]
    assert remaining==[(type(w.message),str(w.message)) for w in reference_notices]
    np.testing.assert_array_equal(actual,expected)

@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('kind',['float16','endian','nan','huge','array-weight'])
def test_explicit_unsupported_fallback(name,kind):
    a=np.random.default_rng(88).random((7,11)).astype(np.float64)
    options={}
    if kind=='float16':a=a.astype(np.float16)
    if kind=='endian':a=a.astype('>f8')
    if kind=='nan':a[3,5]=np.nan
    if kind=='huge':a[:]=1e200
    if kind=='array-weight':options['k']=np.linspace(.1,.3,11)
    with pytest.raises(NotImplementedError):getattr(xx.filters,name)(a,**options)
    xx.set_fallback_enabled(True)
    with np.errstate(all='ignore'):
        with pytest.warns(xx.FallbackWarning):actual=getattr(xx.filters,name)(a,**options)
        expected=getattr(original,name)(a,**options)
    np.testing.assert_array_equal(actual,expected)
