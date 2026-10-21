"""HSV, CIE-Lab and polar conversion contracts against original scikit-image."""
import warnings
import numpy as np
import pytest
from skimage import color as reference
import skimagexx as xx
from skimagexx.coverage import parity_tolerance

FUNCTIONS = ['rgb2hsv','hsv2rgb','xyz2lab','lab2xyz','rgb2lab','lab2rgb','lab2lch','lch2lab',
             'xyz2luv','luv2xyz','rgb2luv','luv2rgb']
DTYPES = [np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float32,np.float64]
CIE = ['xyz2lab','lab2xyz','rgb2lab','lab2rgb','xyz2luv','luv2xyz']


@pytest.fixture(autouse=True,params=[1,4])
def configure(request):
    previous=xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(1)
    xx.set_fallback_enabled(previous)


def compare(name,a,**kwargs):
    with warnings.catch_warnings(record=True) as old:
        warnings.simplefilter('always',UserWarning)
        warnings.simplefilter('ignore',RuntimeWarning)
        try:
            expected=getattr(reference,name)(a,**kwargs)
        except (ValueError,TypeError,IndexError,AttributeError) as error:
            with pytest.raises(type(error)) as actual:
                getattr(xx.color,name)(a,**kwargs)
            assert str(actual.value)==str(error)
            return
    with warnings.catch_warnings(record=True) as new:
        warnings.simplefilter('always',UserWarning)
        warnings.simplefilter('ignore',RuntimeWarning)
        actual=getattr(xx.color,name)(a,**kwargs)
    assert [(w.category,str(w.message)) for w in new]==[(w.category,str(w.message)) for w in old]
    assert actual.dtype==expected.dtype
    assert actual.shape==expected.shape
    rtol,atol=parity_tolerance('color.'+name,actual.dtype)
    np.testing.assert_allclose(actual,expected,rtol=rtol,atol=atol)
    if isinstance(a,np.ndarray):
        assert not np.shares_memory(a,actual)


@pytest.mark.parametrize('name',FUNCTIONS)
@pytest.mark.parametrize('dtype',DTYPES)
def test_source_dtypes_and_integer_extrema(name,dtype):
    dtype=np.dtype(dtype)
    if dtype.kind=='b':
        a=np.tile(np.array([0,1,2,128,255,0],dtype=np.uint8),19).view(np.bool_).reshape(-1,3)
    elif dtype.kind in 'ui':
        info=np.iinfo(dtype)
        a=np.tile(np.array([info.min,info.max,0,1,info.max//2,info.max//2+1],dtype=dtype),19).reshape(-1,3)
    else:
        a=np.random.default_rng(20).random((19,23,3)).astype(dtype)
    before=a.copy()
    compare(name,a)
    np.testing.assert_array_equal(a,before)


@pytest.mark.parametrize('name',CIE)
@pytest.mark.parametrize('illuminant',['A','B','C','D50','D55','D65','D75','E'])
@pytest.mark.parametrize('observer',['2','10','R'])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_all_white_points(name,illuminant,observer,dtype):
    a=np.random.default_rng(61).random((7,9,3)).astype(dtype)
    if name.startswith(('lab','luv')):
        a[...,0]*=100
        a[...,1:]=a[...,1:]*255-128
    compare(name,a,illuminant=illuminant.lower(),observer=observer.lower())


@pytest.mark.parametrize('name',FUNCTIONS)
@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('axis',[-4,-3,-2,-1,0,1,2,3,None,(1,)])
def test_nd_and_channel_axes(name,dtype,axis):
    a=np.random.default_rng(41).integers(0,256,(2,7,9,3),dtype=np.uint8)
    if dtype!=np.uint8:
        a=(a/255).astype(dtype)
    if axis is not None:
        a=np.moveaxis(a,-1,axis[0] if isinstance(axis,tuple) else axis)
    compare(name,a,channel_axis=axis)


@pytest.mark.parametrize('name',FUNCTIONS)
@pytest.mark.parametrize('shape',[(3,),(0,3),(2,0,3),(2,1,3),(),(2,),(3,4)])
def test_empty_single_pixel_and_shape_errors(name,shape):
    compare(name,np.zeros(shape,dtype=np.float32))


@pytest.mark.parametrize('name',FUNCTIONS)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_large_parallel_planar_strided_and_unaligned(name,dtype):
    a=np.random.default_rng(32).random((513,515,3)).astype(dtype)
    compare(name,a)
    compare(name,np.ascontiguousarray(np.moveaxis(a,-1,0)),channel_axis=0)
    compare(name,a[::-1,::-1])
    compare(name,a[:,::2,::-1])
    storage=np.empty(a.nbytes+1,dtype=np.uint8)
    unaligned=np.ndarray(a.shape,dtype=dtype,buffer=storage,offset=1)
    unaligned[:]=a
    compare(name,unaligned)


@pytest.mark.parametrize('name',['lab2lch','lch2lab'])
@pytest.mark.parametrize('channels',[3,4,7])
@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
def test_polar_extra_channels(name,channels,dtype):
    a=np.random.default_rng(72).integers(0,256,(19,23,channels),dtype=np.uint8)
    if dtype!=np.uint8:
        a=a.astype(dtype)
        a[...,2]=a[...,2]/dtype(255)*dtype(40)-dtype(20)
    compare(name,a)
    compare(name,np.ascontiguousarray(np.moveaxis(a,-1,0)),channel_axis=0)


@pytest.mark.parametrize('name',['lab2xyz','lab2rgb'])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_negative_z_warnings_and_scalar_promotion(name,dtype):
    for shape in [(3,),(1,3),(2,3,3),(513,515,3)]:
        a=np.broadcast_to(np.array([0,0,100],dtype=dtype),shape)
        compare(name,a)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_hsv_ties_grays_and_wrapped_byte_sectors(dtype):
    rgb=np.tile(np.array([[0,0,0],[1,1,1],[1,1,0],[0,1,1],[1,0,1],[1,0,0],[0,1,0],[0,0,1],[-1,-1,-1]],dtype=dtype),(19,1))
    compare('rgb2hsv',rgb)
    h=np.arange(-512,513,dtype=dtype)/dtype(6)
    a=np.stack([h,np.full_like(h,0.7),np.full_like(h,0.9)],axis=-1)
    compare('hsv2rgb',a)


@pytest.mark.parametrize('name',['rgb2hsv','hsv2rgb'])
def test_hsv_unsupported_nonfinite_is_explicit(name):
    a=np.array([[np.nan,1,1],[0,np.inf,0]],dtype=np.float64)
    with pytest.raises(NotImplementedError):
        getattr(xx.color,name)(a)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning):
        with np.errstate(all='ignore'):
            getattr(xx.color,name)(a)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_hue_cast_domain_and_sector_boundaries(dtype):
    for h in [0,1/6,np.nextafter(dtype(1/6),dtype(0)),1,-1,
              -2147483648/6,2147483647/6,2147483648/6,1e20]:
        a=np.tile(np.array([h,0.6,0.8],dtype=dtype),(19,1))
        hi=float(np.floor(dtype(a[0,0]*dtype(6))))
        if -2147483648 <= hi < 2147483648:
            compare('hsv2rgb',a)
        else:
            with pytest.raises(NotImplementedError):
                xx.color.hsv2rgb(a)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_hsv_signed_zero_reductions_and_tails(dtype):
    rgb=np.array([[np.copysign(0.0,-1 if bits & (1<<j) else 1) for j in range(3)] for bits in range(8)],dtype=dtype)
    rgb=np.concatenate((np.tile(rgb,(19,1)),np.array([[-0.0,-0.0,0.0]],dtype=dtype)))
    actual,expected=xx.color.rgb2hsv(rgb),reference.rgb2hsv(rgb)
    np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('name',['lab2lch','lch2lab','xyz2lab','lab2xyz','rgb2lab','lab2rgb','xyz2luv','luv2xyz','rgb2luv','luv2rgb'])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_nonfinite_signed_zero_and_polar_boundaries(name,dtype):
    a=np.array([[0,-0.0,0.0],[-0.0,-0.0,-0.0],[1,np.inf,np.nan],
                [1,0,np.pi/2],[1,1,np.pi],[1,2,1e20],
                [1,1e-30,-1e-30],[1,1,1e-14]],dtype=dtype)
    compare(name,np.tile(a,(19,1)))


@pytest.mark.parametrize('dtype',DTYPES+[np.float16,object])
def test_white_metadata_dtype_and_copies(dtype):
    for ill in ['A','B','C','D50','D55','D65','D75','E']:
        for obs in ['2','10','R']:
            expected=reference.xyz_tristimulus_values(illuminant=ill.lower(),observer=obs.lower(),dtype=dtype)
            actual=xx.color.xyz_tristimulus_values(illuminant=ill.lower(),observer=obs.lower(),dtype=dtype)
            assert actual.dtype==expected.dtype
            np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('ill,obs',[('unknown','2'),('D65','4'),(None,'2'),('D65',2)])
def test_white_errors(ill,obs):
    try:
        reference.xyz_tristimulus_values(illuminant=ill,observer=obs)
    except (ValueError,AttributeError) as error:
        with pytest.raises(type(error)) as actual:
            xx.color.xyz_tristimulus_values(illuminant=ill,observer=obs)
        assert str(actual.value)==str(error)


@pytest.mark.parametrize('name',FUNCTIONS)
def test_fallback_half_and_root_aliases(name):
    a=np.zeros((7,3),dtype=np.float16)
    with pytest.raises(NotImplementedError):
        getattr(xx.color,name)(a)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning):
        actual=getattr(xx.color,name)(a)
    np.testing.assert_array_equal(actual,getattr(reference,name)(a))
    assert getattr(xx,name) is getattr(xx.color,name)


@pytest.mark.parametrize('name',['xyz2luv','luv2xyz'])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('illuminant',['A','B','C','D50','D55','D65','D75','E'])
@pytest.mark.parametrize('observer',['2','10','R'])
def test_luv_branch_and_singular_boundaries(name,dtype,illuminant,observer):
    w=reference.xyz_tristimulus_values(illuminant=illuminant,observer=observer,dtype=dtype)
    denominator=np.array([1,15,3],dtype=dtype)@w
    v0=9*w[1]/denominator
    if name=='luv2xyz':
        branch=dtype(7.999625)
        levels=np.array([0.,-0.,-1.,np.nextafter(branch,dtype(-np.inf)),branch,
                         np.nextafter(branch,dtype(np.inf)),16.,100.],dtype=dtype)
        a=np.zeros((len(levels),3),dtype=dtype); a[:,0]=levels
        a[:,1]=dtype(10)
        a[:,2]=-v0*(dtype(13)*levels+np.finfo(dtype).eps)
    else:
        branch=dtype(.008856)
        levels=np.array([0.,-0.,-1.,np.nextafter(branch,dtype(-np.inf)),branch,
                         np.nextafter(branch,dtype(np.inf)),1.,100.],dtype=dtype)
        a=np.zeros((len(levels),3),dtype=dtype); a[:,1]=levels
        a[:,0]=-dtype(15)*levels-np.finfo(dtype).eps
    compare(name,np.tile(a,(19,1)),illuminant=illuminant,observer=observer)


@pytest.mark.parametrize('name',['xyz2luv','luv2xyz','rgb2luv','luv2rgb'])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_luv_signed_zero(name,dtype):
    a=np.tile(np.array([[0.,-0.,0.],[-0.,0.,-0.]],dtype=dtype),(19,1))
    actual=getattr(xx.color,name)(a)
    expected=getattr(reference,name)(a)
    np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))
