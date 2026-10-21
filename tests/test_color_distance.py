"""CIE76 raw Lab distance and the nine-space routing API versus scikit-image."""
import warnings
import numpy as np
import pytest
from skimage import color as reference
import skimagexx as xx
from benchmarks.color_parity import check_routed_hsv

TYPES=[np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float32,np.float64]
SPACES=['RGB','HSV','RGB CIE','XYZ','YUV','YIQ','YPbPr','YCbCr','YDbDr']


@pytest.fixture(autouse=True,params=[1,4])
def configure(request):
    old=xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(1)
    xx.set_fallback_enabled(old)


def compare(name,*args,**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        try:
            expected=getattr(reference,name)(*args,**kwargs)
        except (ValueError,TypeError,IndexError,AttributeError,OverflowError) as error:
            with pytest.raises(type(error)) as actual:
                getattr(xx.color,name)(*args,**kwargs)
            assert str(actual.value)==str(error)
            return
        actual=getattr(xx.color,name)(*args,**kwargs)
    assert type(actual)==type(expected)
    assert actual.shape==expected.shape
    assert actual.dtype==expected.dtype
    single=actual.dtype==np.float32
    if name=='convert_colorspace' and args[1].lower()!='rgb' and args[2].lower()=='hsv' and not kwargs:
        check_routed_hsv(args[0],args[1],actual,expected)
    else:
        np.testing.assert_allclose(actual,expected,rtol=2e-6 if single else 2e-13,atol=2e-6 if single else 2e-13)
    if isinstance(actual,np.ndarray):
        if name=='convert_colorspace' and args[1].lower()==args[2].lower()=='rgb':
            assert actual is args[0]
        else:
            for a in args[:2] if name=='deltaE_cie76' else args[:1]:
                if isinstance(a,np.ndarray): assert not np.shares_memory(actual,a)


def sample(dtype,shape=(19,23,3)):
    if dtype==np.bool_:
        return np.random.default_rng(51).integers(0,256,shape,dtype=np.uint8).view(np.bool_)
    if np.dtype(dtype).kind in 'ui':
        info=np.iinfo(dtype)
        return np.resize(np.array([info.min,info.max,0,1,info.max//2,info.max//2+1],dtype=dtype),shape)
    return (np.random.default_rng(32).random(shape)*200-100).astype(dtype)


@pytest.mark.parametrize('first',TYPES)
@pytest.mark.parametrize('second',TYPES)
def test_distance_dtype_pairs_raw_values_and_no_mutation(first,second):
    a,b=sample(first),sample(second)[::-1,::-1,::-1]
    before_a,before_b=a.copy(),b.copy()
    compare('deltaE_cie76',a,b)
    np.testing.assert_array_equal(a,before_a)
    np.testing.assert_array_equal(b,before_b)


@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.int16])
@pytest.mark.parametrize('axis',[-4,-3,-2,-1,0,1,2,3])
@pytest.mark.parametrize('channels',[3,4,7])
def test_distance_extra_channels_and_nd_axes(dtype,axis,channels):
    a=sample(dtype,(2,7,9,channels))
    b=sample(dtype,(2,7,9,channels+2))
    compare('deltaE_cie76',np.moveaxis(a,-1,axis),np.moveaxis(b,-1,axis),channel_axis=axis)


@pytest.mark.parametrize('shape1,shape2',[((3,),(3,)),((7,3),(3,)),((2,1,9,3),(1,7,1,4)),
    ((0,3),(1,3)),((2,0,3),(3,)),((2,3),(3,3)),((2,),(3,)),((0,),(3,)),((),(3,))])
@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.uint8])
def test_distance_broadcast_empty_scalar_and_shape_errors(shape1,shape2,dtype):
    compare('deltaE_cie76',np.ones(shape1,dtype=dtype),np.zeros(shape2,dtype=dtype))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_distance_nonfinite_overflow_and_signed_zero(dtype):
    values=np.array([[0.,-0.,0.],[-0.,0.,-0.],[np.inf,1,2],[np.nan,-np.inf,0],
                     [np.finfo(dtype).max,0,0],[np.finfo(dtype).tiny,1e-20,-1e-20]],dtype=dtype)
    compare('deltaE_cie76',np.tile(values,(19,1)),np.zeros((3,),dtype=dtype))


@pytest.mark.parametrize('axis',[None,4,-5,(0,),1.5])
def test_distance_invalid_axes(axis):
    compare('deltaE_cie76',np.zeros((7,9,3)),np.zeros((7,9,3)),channel_axis=axis)


def test_distance_arraylikes_and_array_subclasses():
    a=[[20,30,-40],[0,1,2]]
    compare('deltaE_cie76',a,(20.,30.,40.))
    class Subclass(np.ndarray): pass
    compare('deltaE_cie76',np.asarray(a,dtype=np.float32).view(Subclass),np.ma.array(a,mask=True))


@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.uint16])
def test_distance_large_layouts_and_readonly(dtype):
    a,b=sample(dtype,(513,515,3)),sample(dtype,(513,515,3))[::-1]
    a.flags.writeable=False
    compare('deltaE_cie76',a,b)
    compare('deltaE_cie76',np.ascontiguousarray(np.moveaxis(a,-1,0)),np.moveaxis(b,-1,0),channel_axis=0)
    compare('deltaE_cie76',a[:,::2,::-1],b[:,::2])
    storage=np.empty(a.nbytes+1,dtype=np.uint8)
    unaligned=np.ndarray(a.shape,dtype=dtype,buffer=storage,offset=1)
    unaligned[:]=a
    compare('deltaE_cie76',unaligned,b)
    compare('deltaE_cie76',a,np.array([20,-10,5],dtype=dtype if dtype!=np.uint16 else np.int16))


@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('layout',range(4))
def test_distance_nonuniform_nd_rows(dtype,layout):
    rng=np.random.default_rng(86)
    a=rng.integers(0,256,(3,5,7,9,4),dtype=np.uint8).astype(dtype)
    b=rng.integers(0,256,a.shape,dtype=np.uint8).astype(dtype)
    if layout==0:
        first,second=a[::-1,:,::2,::2,:3],b[:,:,::-2,::-2,:3]
    elif layout==1:
        first,second=np.asfortranarray(a[...,:3]),b[...,2::-1]
    elif layout==2:
        first,second=a[:,:,0:1,:1,:3],b[:,:,::-1,:,:3]
    else:
        first=a.transpose(1,0,2,3,4)[...,:3]
        second=b.transpose(1,0,2,3,4)[:,::-1,:,:,2::-1]
    first.flags.writeable=False;second.flags.writeable=False
    compare('deltaE_cie76',first,second)


def test_distance_parallel_nonuniform_batches():
    a=np.random.default_rng(87).random((2,513,515,3),dtype=np.float32)
    compare('deltaE_cie76',a,a[::-1,::-1,:,::-1])


@pytest.mark.parametrize('first_type,second_type',[(np.uint16,np.float32),(np.int64,np.float64),(np.bool_,np.float32)])
def test_distance_mixed_dtype_nonuniform_rows(first_type,second_type):
    rng=np.random.default_rng(88)
    raw=rng.integers(0,256,(3,17,19,3),dtype=np.uint8)
    a=raw.view(np.bool_) if first_type==np.bool_ else raw.astype(first_type)
    b=rng.integers(0,256,raw.shape,dtype=np.uint8).astype(second_type)
    compare('deltaE_cie76',a[:,::-1],b[:,:,::-1,::-1])


@pytest.mark.parametrize('fromspace',SPACES)
@pytest.mark.parametrize('tospace',SPACES)
@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
def test_convert_all_space_pairs(fromspace,tospace,dtype):
    rgb=np.random.default_rng(72).integers(0,256,(7,9,3),dtype=np.uint8)
    if dtype!=np.uint8: rgb=(rgb/255).astype(dtype)
    image=reference.convert_colorspace(rgb,'RGB',fromspace)
    compare('convert_colorspace',image,fromspace.lower(),tospace.lower())


@pytest.mark.parametrize('fromspace',SPACES)
@pytest.mark.parametrize('dtype',TYPES)
def test_convert_raw_source_dtypes(fromspace,dtype):
    compare('convert_colorspace',sample(dtype),fromspace,'HSV')


@pytest.mark.parametrize('fromspace,tospace',[('RGB','HSV'),('HSV','XYZ'),('XYZ','YUV'),('YCbCr','RGB')])
@pytest.mark.parametrize('axis',[-4,-3,-2,-1,0,1,2,3,None,(1,)])
def test_convert_nd_axes(fromspace,tospace,axis):
    a=np.random.default_rng(73).random((2,7,9,3)).astype(np.float32)
    if axis is not None: a=np.moveaxis(a,-1,axis[0] if isinstance(axis,tuple) else axis)
    compare('convert_colorspace',a,fromspace,tospace,channel_axis=axis)


@pytest.mark.parametrize('a',[np.zeros((),dtype=np.float32),[1,2,3],np.empty((2,0,7)),None])
def test_convert_rgb_identity(a):
    assert xx.color.convert_colorspace(a,'RGB','rgb',channel_axis=None) is a


@pytest.mark.parametrize('fromspace,tospace',[('unknown','RGB'),('RGB','Lab'),(None,'RGB'),('RGB',2)])
def test_convert_invalid_spaces(fromspace,tospace):
    compare('convert_colorspace',np.zeros((7,9,3)),fromspace,tospace)


@pytest.mark.parametrize('name',['deltaE_cie76','convert_colorspace'])
def test_unsupported_half_fallback_and_alias(name):
    a=np.zeros((7,9,3),dtype=np.float16)
    args=(a,a) if name=='deltaE_cie76' else (a,'RGB','HSV')
    with pytest.raises(NotImplementedError): getattr(xx.color,name)(*args)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning): actual=getattr(xx.color,name)(*args)
    np.testing.assert_array_equal(actual,getattr(reference,name)(*args))
    assert getattr(xx,name) is getattr(xx.color,name)
