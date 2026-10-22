"""Weighted Lab distances, dtype promotion and branch boundaries versus skimage."""
from pathlib import Path
import warnings
import numpy as np
import pytest
from skimage import color as reference
import skimagexx as xx
from skimagexx.coverage import parity_tolerance

NAMES=['deltaE_ciede94','deltaE_ciede2000','deltaE_cmc']
TYPES=[np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float32,np.float64]


@pytest.fixture(autouse=True,params=[1,4])
def configure(request):
    old=xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(1)
    xx.set_fallback_enabled(old)


def compare(name,a,b,**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        try:
            expected=getattr(reference,name)(a,b,**kwargs)
        except (ValueError,TypeError,IndexError,AttributeError,OverflowError,ZeroDivisionError) as error:
            with pytest.raises(type(error)) as actual:getattr(xx.color,name)(a,b,**kwargs)
            assert str(actual.value)==str(error)
            return
        actual=getattr(xx.color,name)(a,b,**kwargs)
    assert type(actual)==type(expected)
    assert actual.shape==expected.shape and actual.dtype==expected.dtype
    # CMC may return float64 for two float32 scalar colors, while its Lab and
    # chroma intermediates still have float32 precision.
    single=np.asarray(a).dtype==np.asarray(b).dtype==np.float32
    rtol,atol=parity_tolerance('color.'+name,np.dtype(np.float32 if single else np.float64))
    np.testing.assert_allclose(actual,expected,rtol=rtol,atol=atol)
    if isinstance(actual,np.ndarray):
        assert not np.shares_memory(actual,a) and not np.shares_memory(actual,b)


def sample(dtype,shape=(19,23,3)):
    if dtype==np.bool_:return np.random.default_rng(51).integers(0,256,shape,dtype=np.uint8).view(np.bool_)
    if np.dtype(dtype).kind in 'ui':
        i=np.iinfo(dtype)
        return np.resize(np.array([i.min,i.max,0,1,i.max//2,i.max//2+1],dtype=dtype),shape)
    return (np.random.default_rng(32).random(shape)*200-100).astype(dtype)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('first',TYPES)
@pytest.mark.parametrize('second',TYPES)
def test_dtype_pairs_raw_lab_and_no_mutation(name,first,second):
    a,b=sample(first),sample(second)[::-1,::-1,::-1]
    aa,bb=a.copy(),b.copy()
    compare(name,a,b)
    np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('axis',[-4,-3,-2,-1,0,1,2,3])
@pytest.mark.parametrize('channels',[3,4,7])
def test_extra_channels_nd_axes(name,dtype,axis,channels):
    a,b=sample(dtype,(2,7,9,channels)),sample(dtype,(2,7,9,channels+2))
    compare(name,np.moveaxis(a,-1,axis),np.moveaxis(b,-1,axis),channel_axis=axis)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.uint8])
@pytest.mark.parametrize('shape1,shape2',[((3,),(3,)),((7,3),(3,)),((3,),(7,3)),((2,1,9,3),(1,7,1,4)),
    ((0,3),(1,3)),((2,0,3),(3,)),((2,3),(3,3)),((2,),(3,)),((0,),(3,)),((),(3,))])
def test_broadcast_scalar_empty_shape_errors(name,dtype,shape1,shape2):
    compare(name,np.ones(shape1,dtype=dtype),np.zeros(shape2,dtype=dtype))


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('axis',[None,4,-5,(0,),1.5])
def test_axis_error_and_ciede2000_modulo_contract(name,axis):
    compare(name,np.zeros((7,9,3)),np.zeros((7,9,3)),channel_axis=axis)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.uint16])
def test_large_layouts_readonly_and_unaligned(name,dtype):
    a,b=sample(dtype,(513,515,3)),sample(dtype,(513,515,3))[::-1]
    a.flags.writeable=False
    compare(name,a,b)
    compare(name,np.ascontiguousarray(np.moveaxis(a,-1,0)),np.moveaxis(b,-1,0),channel_axis=0)
    compare(name,a[:,::2,::-1],b[:,::2])
    storage=np.empty(a.nbytes+1,dtype=np.uint8)
    unaligned=np.ndarray(a.shape,dtype=dtype,buffer=storage,offset=1);unaligned[:]=a
    compare(name,unaligned,b)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_nonfinite_signed_zero_overflow_subnormals(name,dtype):
    values=np.array([[0.,-0.,0.],[-0.,0.,-0.],[np.inf,1,2],[np.nan,-np.inf,0],
        [np.finfo(dtype).max,0,0],[0,np.finfo(dtype).max,np.finfo(dtype).max],
        [np.finfo(dtype).tiny,1e-20,-1e-20]],dtype=dtype)
    compare(name,np.tile(values,(19,1)),np.zeros((3,),dtype=dtype))


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_identical_and_nearly_identical_colors(name,dtype):
    a=sample(dtype)
    compare(name,a,a)
    compare(name,a,np.nextafter(a,dtype(np.inf)))
    compare(name,a,a+dtype(1e-4))


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_random_scalar_pairs_and_scalar_broadcast(name,dtype):
    a,b=sample(dtype,(67,3)),sample(dtype,(67,3))[::-1].copy()
    for first,second in zip(a,b):compare(name,first,second)
    compare(name,a[0],b,kL=2,kC=.5)
    compare(name,a,b[0],kL=2,kC=.5)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_float32_weight_scalars(name,dtype):
    compare(name,sample(dtype),sample(dtype)[:,::-1],kL=np.float32(2),kC=np.float32(.7))


def test_ciede2000_float32_accuracy_against_float64_reference():
    rng=np.random.default_rng(342)
    a=(rng.random((67,71,3))*200-100).astype(np.float32)
    b=(rng.random(a.shape)*200-100).astype(np.float32)
    for image in (a,b):
        image[...,0]=np.abs(image[...,0])
        # Positive a* keeps this accuracy check away from the formula's
        # discontinuity at a 180-degree hue separation. Other tests exercise it.
        image[...,1]=np.abs(image[...,1])+10
    expected=reference.deltaE_ciede2000(a.astype(np.float64),b.astype(np.float64))
    np.testing.assert_allclose(xx.color.deltaE_ciede2000(a,b),expected,rtol=2e-6,atol=2.5e-4)
    np.testing.assert_allclose(reference.deltaE_ciede2000(a,b),expected,rtol=2e-6,atol=2.5e-4)


def test_cmc_scalar_promotion_retains_double_final_precision():
    a,b=np.ones(3,dtype=np.float32),np.zeros(3,dtype=np.float32)
    actual=xx.color.deltaE_cmc(a,b)
    assert actual.dtype==np.float64
    assert actual!=np.float64(np.float32(actual))
    compare('deltaE_cmc',a,b)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_ciede2000_sharma_reference_pairs_and_reversed_pairs(dtype):
    # Author's supplementary numerical test data (34 pairs):
    # https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/dataNprograms/ciede2000testdata.txt
    data=np.loadtxt(Path(__file__).parent/'data/ciede2000-sharma.txt')
    a,b=data[:,:3].astype(dtype),data[:,3:6].astype(dtype)
    compare('deltaE_ciede2000',a,b)
    compare('deltaE_ciede2000',b,a)
    # Published values have four decimal places; float32 itself can select
    # another side of the formula's 180-degree discontinuity on these inputs.
    if dtype==np.float64:
        # Public dispatch above must match the current NumPy provider, including
        # SVML's different branch on published pair10. Check the standalone
        # C++ libm profile separately against every published value.
        from skimagexx import _core
        canonical=_core.distance_metric(a,b,1,(1.,1.,1.,.045,.015),True,False)
        np.testing.assert_allclose(canonical,data[:,6],rtol=0,atol=5e-5)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_branch_angles_gray_chroma_and_lightness(name,dtype):
    h=np.array([0,164,345,180,360],dtype=dtype)*dtype(np.pi/180)
    h=np.concatenate((h,np.nextafter(h,dtype(-np.inf)),np.nextafter(h,dtype(np.inf))))
    a=np.stack((np.resize(np.array([0,15.999,16,50,100],dtype=dtype),h.shape),20*np.cos(h),20*np.sin(h)),axis=-1)
    b=a.copy();b[:,1:]*=-1
    compare(name,np.tile(a,(19,1)),np.tile(b,(19,1)))
    compare(name,np.tile(a,(19,1)),np.zeros((3,),dtype=dtype))


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('options',[{'kL':2},{'kC':.5},{'kL':0},{'kC':0},{'kL':-1.5},{'kC':2.5}])
def test_common_weight_parameters(name,dtype,options):
    compare(name,sample(dtype),sample(dtype)[:,::-1],**options)


@pytest.mark.parametrize('name',['deltaE_ciede94','deltaE_ciede2000'])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('hue',[0.,.5,2.,-1.])
def test_hue_weights(name,dtype,hue):
    compare(name,sample(dtype),sample(dtype)[:,::-1],kH=hue)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('k1,k2',[(.048,.014),(0,0),(-.045,.015),(.1,0),(.01,-.02)])
def test_cie94_chroma_coefficients(dtype,k1,k2):
    compare('deltaE_ciede94',sample(dtype),sample(dtype)[:,::-1],k1=k1,k2=k2)


@pytest.mark.parametrize('name',NAMES)
def test_half_and_parameter_array_fallback_and_root_aliases(name):
    a=np.zeros((7,9,3),dtype=np.float16)
    with pytest.raises(NotImplementedError):getattr(xx.color,name)(a,a)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning):actual=getattr(xx.color,name)(a,a)
    np.testing.assert_array_equal(actual,getattr(reference,name)(a,a))
    a=a.astype(np.float32)
    with pytest.warns(xx.FallbackWarning):actual=getattr(xx.color,name)(a,a,kL=np.ones((7,1)))
    np.testing.assert_array_equal(actual,getattr(reference,name)(a,a,kL=np.ones((7,1))))
    assert getattr(xx,name) is getattr(xx.color,name)
