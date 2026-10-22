"""Gabor image responses, borders, storage, types and native composition."""
import inspect
import warnings
import numpy as np
import pytest
from scipy import ndimage
from skimage.filters import gabor as reference, gabor_kernel as reference_kernel
import skimagexx as xx

REAL=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64)
MODES=('reflect','constant','nearest','mirror','wrap','grid-mirror','grid-constant','grid-wrap')


@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    previous=xx.set_fallback_enabled(False);xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(1);xx.set_fallback_enabled(previous)


def compare(image,**options):
    def captured(function):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:value=function(image,**options)
            except Exception as error:value=error
        return value,[(w.category,str(w.message)) for w in caught]
    expected,ew=captured(reference);actual,aw=captured(xx.filters.gabor)
    assert aw==ew
    if isinstance(expected,Exception):
        assert (type(actual),str(actual))==(type(expected),str(expected));return
    assert type(actual) is type(expected) is tuple and len(actual)==2
    for a,e in zip(actual,expected):
        assert type(a) is type(e) is np.ndarray
        assert a.shape==e.shape and a.dtype==e.dtype and a.dtype.metadata==e.dtype.metadata
        if e.dtype.kind in 'biu':
            np.testing.assert_array_equal(a.view(np.uint8),e.view(np.uint8))
        else:
            tolerance=2e-6 if e.dtype==np.float32 else 2e-13
            np.testing.assert_allclose(a,e,rtol=tolerance,atol=tolerance)
    return actual


@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('shape',[(1,1),(3,5),(17,23)])
@pytest.mark.parametrize('mode',MODES)
def test_dtypes_borders(dtype,shape,mode):
    image=np.random.default_rng(104).integers(0,256,shape,dtype=np.uint8).astype(dtype)
    if np.issubdtype(dtype,np.floating):image/=dtype(255)
    compare(image,frequency=.2,theta=.37,mode=mode,cval=-3.25)


@pytest.mark.parametrize('dtype',[np.uint8,np.uint16,np.float32,np.float64])
@pytest.mark.parametrize('frequency',[.1,.3,.75,1.2])
@pytest.mark.parametrize('theta',[0.,.31,np.pi/2])
@pytest.mark.parametrize('sigmas',[(None,None),(1.5,3.)])
def test_parameters(dtype,frequency,theta,sigmas):
    image=np.random.default_rng(304).integers(0,256,(19,21),dtype=np.uint8).astype(dtype)
    if np.issubdtype(dtype,np.floating):image/=dtype(255)
    compare(image,frequency=frequency,theta=theta,sigma_x=sigmas[0],sigma_y=sigmas[1],offset=.43)


@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('layout',['reversed','fortran','broadcast','readonly','unaligned','endian','tagged'])
def test_layout_metadata(dtype,layout):
    image=np.random.default_rng(771).integers(0,256,(13,17),dtype=np.uint8).astype(dtype)
    if np.issubdtype(dtype,np.floating):image/=dtype(255)
    if layout=='reversed':image=image[::-1,::2]
    elif layout=='fortran':image=np.asfortranarray(image)
    elif layout=='broadcast':image=np.broadcast_to(image[:1,:1],image.shape)
    elif layout=='readonly':image.flags.writeable=False
    elif layout=='unaligned':
        temporary=np.empty(image.nbytes+1,dtype=np.uint8)
        image=np.ndarray(image.shape,dtype=image.dtype,buffer=temporary,offset=1);image[:]=dtype(1)
    elif layout=='endian':image=image.astype(image.dtype.newbyteorder('>'))
    elif layout=='tagged':image=image.view(np.dtype(image.dtype,metadata={'tag':'gabor'}))
    compare(image,frequency=.3,sigma_x=1.5,sigma_y=2.5,mode='mirror')


@pytest.mark.parametrize('dtype',[np.int8,np.uint8,np.int16,np.uint16,np.int32,np.uint32,np.int64,np.uint64])
@pytest.mark.parametrize('mode',['reflect','constant'])
def test_integer_extremes(dtype,mode):
    limits=np.iinfo(dtype)
    image=np.resize(np.array([limits.min,limits.max,0,1,limits.max//3],dtype=dtype),(11,17))
    compare(image,frequency=.23,theta=.71,mode=mode,cval=-17.3)


@pytest.mark.parametrize('dtype',[np.int8,np.uint8,np.int16,np.uint16,np.int32,np.uint32,np.int64,np.uint64])
def test_integer_cast_overflow(dtype):
    limits=np.iinfo(dtype)
    image=np.resize(np.array([limits.min,limits.max,0,1,limits.max//3],dtype=dtype),(7,11))
    compare(image,frequency=.4,sigma_x=.1,sigma_y=.1,offset=.3)


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('value',[np.nan,np.inf,-np.inf])
def test_nonfinite_neighborhoods(dtype,value):
    image=np.ones((13,17),dtype=dtype);image[6,7]=value
    compare(image,frequency=.4,sigma_x=.1,sigma_y=.1,offset=0)
    compare(image,frequency=.4,sigma_x=1.,sigma_y=2.,offset=.3,mode='constant',cval=value)


@pytest.mark.parametrize('dtype,bits,integer',[(np.float16,0x7c01,np.uint16),
    (np.float32,0x7f800001,np.uint32),(np.float64,0x7ff0000000000001,np.uint64)])
def test_signaling_nan(dtype,bits,integer):
    image=np.ones((9,13),dtype=dtype);image.view(integer)[4,6]=bits
    compare(image,frequency=.3,sigma_x=1.,sigma_y=2.)


@pytest.mark.parametrize('options',[{'frequency':0.},
    {'frequency':.2,'sigma_x':np.float16(300),'sigma_y':1,'n_stds':1,'mode':'nearest'},
    {'frequency':.2,'mode':'invalid'}, {'frequency':.2,'cval':1j}])
def test_whole_call_fallback_order(options):
    image=np.ones((5,7),dtype=np.float16);image.view(np.uint16)[2,3]=0x7c01
    xx.set_fallback_enabled(True)
    def outcome(function):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:value=function(image,**options)
            except Exception as error:value=error
        return value,[(w.category,str(w.message)) for w in caught if w.category is not xx.FallbackWarning]
    expected,ew=outcome(reference);actual,aw=outcome(xx.filters.gabor)
    assert aw==ew
    if isinstance(expected,Exception):assert (type(actual),str(actual))==(type(expected),str(expected))
    else:
        for a,e in zip(actual,expected):np.testing.assert_array_equal(a,e)


def test_boolean_bytes():
    image=np.resize(np.array([0,1,2,255],dtype=np.uint8),(9,13)).view(np.bool_)
    compare(image,frequency=.2,sigma_x=.1,sigma_y=.1,offset=.3)


@pytest.mark.parametrize('dtype',[np.bool_,np.uint8,np.int16,np.float32,np.float64])
@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('storage',['native','endian','unaligned','row-only'])
def test_zero_stride_boundary_sentinel(dtype,mode,storage):
    image=np.full((1,1),7,dtype=dtype)
    if storage=='endian':image=image.astype(image.dtype.newbyteorder('>'))
    elif storage=='unaligned':
        raw=np.empty(image.nbytes+1,dtype=np.uint8)
        image=np.ndarray((1,1),dtype=dtype,buffer=raw,offset=1);image[:]=7
    elif storage=='row-only':image=np.arange(13,dtype=np.uint8).reshape(1,13).astype(dtype)
    image=np.broadcast_to(image,(9,13))
    compare(image,frequency=.4,sigma_x=1.,sigma_y=1.5,mode=mode,cval=-.3)


def test_unsafe_distant_reflect_routes_upstream(monkeypatch):
    image=np.ones((5,7),dtype=np.float32)
    options=dict(frequency=.2,sigma_x=30.,sigma_y=1.,n_stds=1)
    with pytest.raises(NotImplementedError):xx.filters.gabor(image,**options)
    xx.set_fallback_enabled(True);sentinel=object();calls=[]
    def delegated(*args,**kwargs):calls.append((args,kwargs));return sentinel
    monkeypatch.setattr('skimage.filters.gabor',delegated)
    with pytest.warns(xx.FallbackWarning):assert xx.filters.gabor(image,**options) is sentinel
    assert len(calls)==1 and calls[0][0][0] is image


@pytest.mark.parametrize('shape',[(0,2),(2,0),(),(3,),(2,3,4)])
def test_shape_errors(shape):
    compare(np.zeros(shape),frequency=.2)


def test_signature():
    assert inspect.signature(xx.filters.gabor)==inspect.signature(reference)


@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.uint8])
def test_parallel_and_no_original_algorithm(dtype,monkeypatch):
    image=np.random.default_rng(24).integers(0,256,(257,263),dtype=np.uint8).astype(dtype)
    if np.issubdtype(dtype,np.floating):image/=dtype(255)
    options=dict(frequency=.4,sigma_x=1.5,sigma_y=2.,theta=.3)
    expected=reference(image,**options)
    def forbidden(*args,**kwargs):raise AssertionError('upstream algorithm body executed')
    monkeypatch.setattr('skimage.filters.gabor',forbidden)
    monkeypatch.setattr('skimage.filters.gabor_kernel',forbidden)
    monkeypatch.setattr(ndimage,'convolve',forbidden)
    xx.set_num_threads(1);serial=xx.filters.gabor(image,**options)
    xx.set_num_threads(4);parallel=xx.filters.gabor(image,**options)
    for s,p,e in zip(serial,parallel,expected):
        np.testing.assert_array_equal(s,p)
        if dtype==np.uint8:np.testing.assert_array_equal(s,e)
        else:np.testing.assert_allclose(s,e,rtol=2e-6 if dtype==np.float32 else 2e-13,atol=2e-6 if dtype==np.float32 else 2e-13)


@pytest.mark.parametrize('single',[False,True])
def test_convolution_order_independent_of_kernel_generator(single):
    # Isolate convolution from transcendental provider rounding, including the
    # strict epsilon cutoff and independent real/imaginary footprints.
    image=np.random.default_rng(512).normal(size=(17,23)).astype(np.float32 if single else np.float64)
    dtype=np.complex64 if single else np.complex128
    kernel=np.array([[0,2e-16+1j,1e-15],[.3j,.25,-.7j],[.2,0,.15-.12j]],dtype=dtype)
    for name,code in [('reflect',0),('constant',1),('nearest',2),('mirror',3),('wrap',4)]:
        actual=xx._core.gabor_filter(image,kernel,code,-.3)
        expected=(ndimage.convolve(image,kernel.real,mode=name,cval=-.3),ndimage.convolve(image,kernel.imag,mode=name,cval=-.3))
        for a,e in zip(actual,expected):np.testing.assert_array_equal(a,e)


def test_integer_kernel_math_provider():
    import math
    rng=np.random.default_rng(919)
    for index in range(80):
        frequency=float(rng.uniform(.08,.8));theta=float(rng.uniform(-3,3))
        sx=float(rng.uniform(.7,7));sy=float(rng.uniform(.7,7));offset=float(rng.uniform(-3,3))
        expected=reference_kernel(frequency,theta=theta,sigma_x=sx,sigma_y=sy,offset=offset)
        ct=math.cos(theta);st=math.sin(theta)
        actual=xx._core.gabor_kernel_accurate(*expected.shape,ct,st,sx**2,sy**2,2*np.pi*frequency,
                                           offset,1/(2*np.pi*sx*sy),False)
        np.testing.assert_array_equal(actual,expected)
