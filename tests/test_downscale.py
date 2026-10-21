"""Local-mean port contract: raw values, padding, dtypes and block geometry.

Test-first work in progress: keep this worktree separate until the native port
implements the contract and has original-library performance measurements.
"""
import inspect
import math
import re
import warnings
import numpy as np
import pytest
import skimage
from skimage import transform as original
import skimagexx as xx
from test_integral import REAL,sample


@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    previous=xx.set_fallback_enabled(False);xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous);xx.set_num_threads(1)


def check(image,factors,**options):
    with warnings.catch_warnings(record=True) as expected_notices:
        warnings.simplefilter('always')
        try:expected=original.downscale_local_mean(image,factors,**options)
        except Exception as error:
            with pytest.raises(type(error),match=re.escape(str(error))):xx.transform.downscale_local_mean(image,factors,**options)
            return
    snapshot=np.array(image,copy=True)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always');actual=xx.transform.downscale_local_mean(image,factors,**options)
    assert [(type(w.message),str(w.message)) for w in notices]==[(type(w.message),str(w.message)) for w in expected_notices]
    assert type(actual) is type(expected)
    a,e=np.asarray(actual),np.asarray(expected)
    assert a.shape==e.shape and a.dtype==e.dtype and a.strides==e.strides
    assert a.dtype.metadata==e.dtype.metadata
    assert not np.shares_memory(actual,image)
    np.testing.assert_array_equal(image,snapshot)
    # Mean reductions can differ in summation order. Bound absolute error by
    # the input scale and logarithmic reduction depth, rather than dividing by
    # a nearly cancelled mean. Half outputs additionally round to binary16.
    volume=math.prod(int(n) for n in factors) if isinstance(factors,tuple) else int(factors)**image.ndim
    accumulator=np.float32 if image.dtype.kind=='f' and image.dtype.itemsize<=4 else np.float64
    scale=max(1.,float(np.max(np.abs(np.asarray(image,dtype=np.float64)),initial=0)))
    if 'cval' in options and np.isscalar(options['cval']):
        try:scale=max(scale,abs(float(options['cval'])))
        except (TypeError,ValueError):pass
    atol=8*np.finfo(accumulator).eps*scale*max(1.,math.log2(max(1,volume)))
    rtol=2*np.finfo(e.dtype).eps
    np.testing.assert_allclose(a,e,rtol=rtol,atol=atol)
    return actual,expected


def test_signature():
    assert inspect.signature(xx.transform.downscale_local_mean)==inspect.signature(original.downscale_local_mean)


@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('shape,factors',[((),2),((19,),3),((13,17),(2,3)),((2,5,7),(1,2,3)),((2,3,5,7),(1,2,2,3)),((0,7),(2,3)),((3,0,5),(2,3,2))])
def test_raw_dtypes_dimensions_and_padding(dtype,shape,factors):
    check(sample(dtype,shape),factors)


@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('layout',['fortran','reversed','stepped','readonly','unaligned','endian','permuted','zero-stride','tagged'])
def test_storage(dtype,layout):
    image=sample(dtype,(5,7,11))
    if layout=='fortran':image=np.asfortranarray(image)
    elif layout=='reversed':image=image[::-1,::-1,::-1]
    elif layout=='stepped':image=np.tile(image,(2,2,2))[::2,::2,::2]
    elif layout=='readonly':image.flags.writeable=False
    elif layout=='endian':image=image.astype(image.dtype.newbyteorder('>'))
    elif layout=='permuted':image=image.transpose(2,0,1)
    elif layout=='zero-stride':image=np.broadcast_to(image[:1],image.shape)
    elif layout=='tagged':image=image.astype(np.dtype(dtype,metadata={'units':'raw'}))
    else:
        raw=np.empty(image.nbytes+1,dtype=np.uint8)
        copied=np.ndarray(image.shape,dtype=image.dtype,buffer=raw,offset=1);copied[:]=image;image=copied
    check(image,(2,3,2));check(image,(1,1,1))


@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('constant',[0,1,-2,1.75,np.float32(.125)])
def test_padding_casts_and_unused_constant(dtype,constant):
    check(sample(dtype,(7,11)),(2,3),cval=constant)
    check(sample(dtype,(6,12)),(2,3),cval=constant)


@pytest.mark.parametrize('dtype',[np.uint8,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('factors',[1,2,3,np.int32(2),(1,1),(1,3),(3,1),(3,5),(16,16)])
def test_factors_and_ignored_clip(dtype,factors):
    image=sample(dtype,(17,19));check(image,factors,clip=False);check(image,factors,clip=object())


@pytest.mark.parametrize('factors',[0,-1,(0,2),(2,-1),(1,),(),[2,3],np.array([2,3]),np.array(2),True,(True,True),1.0,2.0,(2.0,3.0),None,'two'])
def test_factor_validation(factors):
    check(np.arange(35).reshape(5,7),factors)


@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('factors',[(2,2),(3,5),(4,8),(1,1)])
def test_simd_and_worker_boundaries(dtype,factors):
    check(sample(dtype,(513,517)),factors)


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('factors',[(128,128),(129,127),(257,257)])
@pytest.mark.parametrize('layout',['C','F'])
def test_large_reduction_blocks(dtype,factors,layout):
    image=np.full((257,263),1.1,dtype=dtype,order=layout)
    check(image,factors,cval=1.1)
    image=sample(dtype,(257,263))
    if layout=='F':image=np.asfortranarray(image)
    check(image,factors)


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('shape',[(),(17,),(7,11)])
def test_signed_zero(dtype,shape):
    image=np.full(shape,-0.,dtype=dtype)
    actual,expected=check(image,2)
    np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))


def test_memory_map(tmp_path):
    image=np.memmap(tmp_path/'image.dat',mode='w+',dtype=np.float32,shape=(7,11))
    image[:]=np.arange(77).reshape(7,11);check(image,(2,3))


def test_backend_swap_pipeline():
    image=np.random.default_rng(990).integers(0,256,(37,41,3),dtype=np.uint8)
    def pipeline(backend):
        gray=backend.color.rgb2gray(image)
        return backend.transform.downscale_local_mean(backend.filters.gaussian(gray,.75),(2,3))
    np.testing.assert_allclose(pipeline(xx),pipeline(skimage),rtol=2e-13,atol=2e-13)


def test_half_exact_storage_and_rounding_ties():
    words=np.arange(65536,dtype=np.uint16)
    image=words.view(np.float16)
    finite=image[np.isfinite(image)]
    actual,expected=check(finite,1)
    np.testing.assert_array_equal(actual.view(np.uint16),expected.view(np.uint16))
    # Adjacent finite half values have float32-exact midpoints. Their means
    # must round ties to even, including subnormal/normal exponent boundaries.
    words=np.arange(0x7bff,dtype=np.uint16)
    pairs=np.stack((words.view(np.float16),(words+1).view(np.float16)),axis=-1)
    for image in (pairs,-pairs):
        actual,expected=check(image,(1,2))
        np.testing.assert_array_equal(actual.view(np.uint16),expected.view(np.uint16))
        actual,expected=check(np.tile(image.reshape(1,-1),(1,5)),(1,2))
        np.testing.assert_array_equal(actual.view(np.uint16),expected.view(np.uint16))


def test_noncanonical_boolean_and_large_integer_casts():
    image=np.array([[0,2,128,255,1],[255,0,0,4,8]],dtype=np.uint8).view(np.bool_)
    check(image,(2,2));check(image,(1,1))
    words=np.array([0,2**53-1,2**53,2**53+1,2**63-1,2**63,2**63+1,2**64-2049,2**64-1],dtype=np.uint64)
    for image in (words,words[::-1]):check(image,2);check(image,1)


@pytest.mark.parametrize('image',[np.array([[np.inf,-np.inf]]),np.array([[np.nan,1.]]),np.full((3,5),np.finfo(np.float64).max)])
def test_exceptional_working_ranges_require_explicit_fallback(image):
    with pytest.raises(NotImplementedError,match='outside native coverage'):
        xx.transform.downscale_local_mean(image,(2,2))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');expected=original.downscale_local_mean(image,(2,2))
        xx.set_fallback_enabled(True)
        with pytest.warns(xx.FallbackWarning):actual=xx.transform.downscale_local_mean(image,(2,2))
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
def test_nondefault_underflow_policy_is_explicit(dtype):
    image=np.ones((3,5),dtype=dtype)
    with np.errstate(under='raise'):
        with pytest.raises(NotImplementedError,match='outside native coverage'):
            xx.transform.downscale_local_mean(image,(2,2))


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('factors',[(1,1),(2,2),(4,4)])
def test_vector_negative_zero_and_parallel_tails(dtype,factors):
    actual,expected=check(np.full((513,517),-0.,dtype=dtype),factors)
    np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))
