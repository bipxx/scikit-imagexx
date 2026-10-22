"""Adaptive equalization versus the pinned original, with strict dispatch."""
import warnings
import numpy as np
import pytest
from skimage.exposure import equalize_adapthist as original
import skimagexx as xx


@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    previous=xx.set_fallback_enabled(False);xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous);xx.set_num_threads(1)


def compare(image,**options):
    def call(function):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:result=function(image,**options)
            except Exception as error:result=error
        return result,[(w.category,str(w.message)) for w in caught]
    expected,expected_warnings=call(original)
    actual,actual_warnings=call(xx.exposure.equalize_adapthist)
    assert actual_warnings==expected_warnings
    if isinstance(expected,Exception):
        assert type(actual) is type(expected),(actual,expected)
        assert str(actual)==str(expected)
    else:
        assert type(actual) is type(expected) and actual.shape==expected.shape
        assert actual.dtype==expected.dtype and actual.dtype.metadata==expected.dtype.metadata
        if image.ndim==3 and image.shape[-1] in (3,4):
            np.testing.assert_allclose(actual,expected,rtol=2e-6 if actual.dtype==np.float32 else 2e-13,atol=2e-6 if actual.dtype==np.float32 else 2e-13)
        else:np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('dtype',[np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('shape',[(57,),(17,23),(7,9,11),(3,5,7,9)])
@pytest.mark.parametrize('bins',[7,32,256])
def test_real_images(dtype,shape,bins):
    image=np.random.default_rng(815).integers(0,256,shape,dtype=np.uint8).astype(dtype)
    if dtype==np.bool_:image=image>0
    elif np.issubdtype(dtype,np.floating):image=image/dtype(255)
    compare(image,kernel_size=3,nbins=bins)


@pytest.mark.parametrize('shape',[(1,1),(3,5),(11,17),(5,7,9),(2,3,5,7)])
@pytest.mark.parametrize('kind',['random','zero','constant','ramp'])
@pytest.mark.parametrize('clip',[0.,.001,.01,.1,1.,2.])
def test_histogram_clipping_and_edges(shape,kind,clip):
    image=np.random.default_rng(996).integers(0,65536,shape,dtype=np.uint16)
    if kind=='zero':image[:]=0
    elif kind=='constant':image[:]=13579
    elif kind=='ramp':image=np.arange(image.size,dtype=np.uint16).reshape(shape)
    compare(image,kernel_size=tuple(2+d for d in range(image.ndim)),clip_limit=clip)


@pytest.mark.parametrize('kernel',[None,1,2,3,8,31,(3,7),(2,4)])
@pytest.mark.parametrize('layout',['contiguous','reversed','fortran','readonly','broadcast','unaligned'])
def test_layouts_and_large_kernels(kernel,layout):
    image=np.random.default_rng(76).integers(0,65536,(13,19),dtype=np.uint16)
    if layout=='reversed':image=image[::-1,::2]
    elif layout=='fortran':image=np.asfortranarray(image)
    elif layout=='readonly':image.flags.writeable=False
    elif layout=='broadcast':image=np.broadcast_to(image[:1,:1],image.shape)
    elif layout=='unaligned':
        memory=np.empty(image.nbytes+1,dtype=np.uint8)
        unaligned=np.ndarray(image.shape,dtype=image.dtype,buffer=memory,offset=1)
        unaligned[:]=image;image=unaligned
    compare(image,kernel_size=kernel)


@pytest.mark.parametrize('dtype',[np.uint8,np.uint16,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('channels',[3,4])
@pytest.mark.parametrize('kernel',[None,3,(5,7)])
def test_color_adapter(dtype,channels,kernel):
    image=np.random.default_rng(18).integers(0,256,(17,23,channels),dtype=np.uint8).astype(dtype)
    if np.issubdtype(dtype,np.floating):image/=dtype(255)
    compare(image,kernel_size=kernel)


@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('shape',[(513,517),(65,67,69)])
def test_parallel_and_simd_boundaries(dtype,shape):
    image=np.random.default_rng(52).integers(0,256,shape,dtype=np.uint8).astype(dtype)
    if dtype!=np.uint8:image/=dtype(255)
    compare(image,kernel_size=tuple(11+2*d for d in range(len(shape))))


@pytest.mark.parametrize('dtype',[np.uint8,np.uint16,np.float32,np.float64])
@pytest.mark.parametrize('endian',[False,True])
def test_dtype_metadata(dtype,endian):
    tagged=np.dtype(dtype,metadata={'units':'linear','source':'parity'})
    if endian:tagged=tagged.newbyteorder('>')
    image=np.random.default_rng(32).integers(0,256,(17,19),dtype=np.uint8).astype(tagged)
    if np.issubdtype(dtype,np.floating):
        image=(image/255).astype(tagged)
    compare(image,kernel_size=5)


@pytest.mark.parametrize('kernel',[0,-1,(),(2,),(2,3,4),(0,-1),(-1,0)])
def test_kernel_errors(kernel):
    compare(np.arange(35,dtype=np.uint8).reshape(5,7),kernel_size=kernel)


@pytest.mark.parametrize('clip',[np.float16(.01),np.float32(.01),np.float64(.01),-1.,float('nan'),float('inf')])
def test_scalar_metadata(clip):
    compare(np.arange(19*23,dtype=np.uint16).reshape(19,23),kernel_size=7,clip_limit=clip)


def test_backend_swap():
    import skimage
    image=np.random.default_rng(3).integers(0,256,(31,37),dtype=np.uint8)
    def pipeline(backend):return backend.exposure.equalize_adapthist(image)>.5
    np.testing.assert_array_equal(pipeline(xx),pipeline(skimage))


@pytest.mark.parametrize('values',[[0.],[.25],[-.25],[-.5,.5],[-.5,0.],[0.,.5],[-.5,-0.,.5]])
@pytest.mark.parametrize('mode',['warn','raise','ignore'])
def test_half_conversion_errstate(values,mode):
    image=np.resize(np.array(values,dtype=np.float16),(7,9))
    with np.errstate(over=mode,invalid=mode):compare(image,kernel_size=3)


@pytest.mark.parametrize('ndim',[5,6,7,8])
def test_higher_dimensions(ndim):
    image=np.random.default_rng(ndim).integers(0,256,(2,)*ndim,dtype=np.uint8)
    compare(image,kernel_size=3 if ndim<7 else 2,nbins=17)


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('channels',[3,4])
def test_color_endian_and_metadata(dtype,channels):
    tagged=np.dtype(dtype,metadata={'units':'linear'}).newbyteorder('>')
    image=(np.random.default_rng(51).integers(0,256,(11,17,channels),dtype=np.uint8)/255).astype(tagged)
    compare(image,kernel_size=5)


@pytest.mark.parametrize('bins',[1,2,3,257,16384,65536])
def test_bin_extremes(bins):
    compare(np.arange(35,dtype=np.uint16).reshape(5,7),kernel_size=3,nbins=bins)


@pytest.mark.parametrize('shape',[(1,),(1,7,1),(1,1,1,1)])
def test_singleton_axes(shape):
    compare(np.full(shape,4321,dtype=np.uint16),kernel_size=3)


def test_memmap(tmp_path):
    image=np.memmap(tmp_path/'clahe.bin',mode='w+',dtype=np.uint16,shape=(17,19))
    image[:]=np.arange(image.size,dtype=np.uint16).reshape(image.shape)
    compare(image,kernel_size=5)


@pytest.mark.parametrize('dtype',[np.uint8,np.float16])
def test_zero_dimensional_kernel_error_order(dtype):
    image=np.arange(35,dtype=np.uint8).reshape(5,7).astype(dtype)
    if dtype==np.float16:image/=np.float16(255)
    compare(image,kernel_size=np.array(3))


@pytest.mark.parametrize('kernel',[float('nan'),float('inf')])
@pytest.mark.parametrize('dtype',[np.uint8,np.float16])
def test_kernel_conversion_error_order(kernel,dtype):
    image=np.arange(35,dtype=np.uint8).reshape(5,7).astype(dtype)
    if dtype==np.float16:image/=np.float16(255)
    compare(image,kernel_size=kernel)


@pytest.mark.parametrize('color',[False,True])
def test_unsupported_kernel_delegates_before_conversion(color,monkeypatch):
    import skimage.exposure as upstream
    shape=(5,7,3) if color else (5,7)
    image=np.arange(np.prod(shape),dtype=np.uint32).reshape(shape)
    sentinel=np.ones((5,7))
    def delegated(actual,kernel_size=None,clip_limit=.01,nbins=256):
        assert actual is image and kernel_size==2**31+1
        return sentinel
    monkeypatch.setattr(upstream,'equalize_adapthist',delegated)
    xx.set_fallback_enabled(True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert xx.exposure.equalize_adapthist(image,kernel_size=2**31+1) is sentinel
    assert len(caught)==1 and caught[0].category is xx.FallbackWarning


@pytest.mark.parametrize('dtype',[int,bool,np.int8,np.uint8,np.int16,np.uint16,np.int32,np.uint32,np.int64,np.uint64])
def test_bin_scalar_promotion(dtype):
    compare(np.arange(35,dtype=np.uint16).reshape(5,7),kernel_size=3,nbins=dtype(32))


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('layout',['fortran','transpose','reversed','readonly'])
def test_float_preparation_layouts(dtype,layout):
    image=(np.random.default_rng(21).integers(0,256,(17,23),dtype=np.uint8)/255).astype(dtype)
    image[0,0]=0
    if layout=='fortran':image=np.asfortranarray(image)
    elif layout=='transpose':image=image.T
    elif layout=='reversed':image=image[::-1,::2]
    else:image.flags.writeable=False
    compare(image,kernel_size=5)
