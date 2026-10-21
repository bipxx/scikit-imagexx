"""Ordered prefix sums, raw dtype conversion, modulo overflow and storage."""
import inspect
import re
import warnings
import numpy as np
import pytest
from skimage import transform as original
import skimagexx as xx

REAL=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64)


@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    previous=xx.set_fallback_enabled(False);xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous);xx.set_num_threads(1)


def sample(dtype,shape,seed=543):
    rng=np.random.default_rng(seed);dtype=np.dtype(dtype)
    if dtype.kind=='b':return rng.random(shape)>.5
    if dtype.kind in 'iu':
        limits=np.iinfo(dtype)
        return rng.integers(int(limits.min),int(limits.max)+1,shape,dtype=dtype)
    return rng.normal(size=shape).astype(dtype)


def check(image,**options):
    with warnings.catch_warnings(record=True) as expected_notices:
        warnings.simplefilter('always')
        try:expected=original.integral_image(image,**options)
        except Exception as error:
            with pytest.raises(type(error),match=re.escape(str(error))):xx.transform.integral_image(image,**options)
            return None,None
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always');actual=xx.transform.integral_image(image,**options)
    assert [(type(w.message),str(w.message)) for w in notices]==[(type(w.message),str(w.message)) for w in expected_notices]
    assert type(actual) is type(expected)
    assert actual.shape==expected.shape and actual.dtype==expected.dtype
    assert actual.dtype.metadata==expected.dtype.metadata
    assert actual.strides==expected.strides
    if expected is image:assert actual is image
    else:assert not np.shares_memory(actual,image)
    np.testing.assert_array_equal(actual,expected,strict=True)
    if expected.dtype.kind=='f':np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))
    return actual,expected


def test_signature():
    assert inspect.signature(xx.transform.integral_image)==inspect.signature(original.integral_image)


@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('shape',[(),(1,),(19,),(13,17),(2,5,7),(2,3,5,7),(0,7),(3,0,5)])
def test_default_dtypes_and_dimensions(dtype,shape):
    check(sample(dtype,shape))


@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('layout',['fortran','reversed','stepped','readonly','unaligned','endian'])
def test_storage(dtype,layout):
    image=sample(dtype,(17,19))
    if layout=='fortran':image=np.asfortranarray(image)
    elif layout=='reversed':image=image[::-1,::-1]
    elif layout=='stepped':image=np.tile(image,(2,2))[::2,::2]
    elif layout=='readonly':image.flags.writeable=False
    elif layout=='endian':image=image.astype(image.dtype.newbyteorder('>'))
    else:
        storage=np.empty(image.nbytes+1,dtype=np.uint8)
        copied=np.ndarray(image.shape,dtype=image.dtype,buffer=storage,offset=1);copied[...]=image;image=copied
    check(image)


@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('target',[np.float64,np.dtype('>f8'),np.dtype(np.float64,metadata={'sum':'raw'})])
def test_explicit_double(dtype,target):
    check(sample(dtype,(17,19)),dtype=target)


@pytest.mark.parametrize('dtype',REAL[:9])
@pytest.mark.parametrize('target',[np.int64,np.uint64,np.dtype('>i8'),np.dtype(np.uint64,metadata={'sum':'modulo'})])
def test_modular_integer_targets(dtype,target):
    check(sample(dtype,(17,19)),dtype=target)


@pytest.mark.parametrize('target',[np.float32,np.dtype('>f4'),np.dtype(np.float32,metadata={'sum':'single'})])
def test_single_precision_stages(target):
    check(sample(np.float32,(17,19)),dtype=target)


@pytest.mark.parametrize('dtype',REAL)
def test_dtype_tags(dtype):
    check(sample(dtype,(17,19)).astype(np.dtype(dtype,metadata={'units':'raw','source':'parity'})))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape',[(19,),(3,17),(5,7,3)])
def test_negative_zero_and_cancellation(dtype,shape):
    count=int(np.prod(shape));image=np.resize(np.array([-0.,0.,1e8,1.,-1e8,-0.],dtype=dtype),count).reshape(shape)
    check(image)
    check(image,dtype=dtype)


def test_noncanonical_bool():
    image=np.array([[0,2,128,255,1],[255,0,0,4,8]],dtype=np.uint8).view(np.bool_)
    check(image);check(image,dtype=np.uint64);check(image,dtype=np.float64)


def test_large_integer_cast_boundaries():
    unsigned=np.array([0,2**53-1,2**53,2**53+1,2**53+2,2**63-1025,2**63-1,2**63,2**63+1,2**63+1025,2**64-2049,2**64-1025,2**64-2,2**64-1,1],dtype=np.uint64).reshape(3,5)
    signed=np.array([-2**63,-2**63+1,-2**53-1,-1,0,1,2**53-1,2**53+1,2**63-1],dtype=np.int64).reshape(3,3)
    for image in (unsigned,unsigned[::-1,::-1],signed):
        check(image);check(image,dtype=np.float64);check(image,dtype=np.int64);check(image,dtype=np.uint64)


@pytest.mark.parametrize('image',[np.array('text'),np.array('2020-01-01',dtype='datetime64[D]'),np.array((1,2.),dtype=[('a','i4'),('b','f8')]),np.array(None,dtype=object)])
def test_non_numeric_scalar_identity(image):
    check(image);check(image,dtype='this-is-not-a-dtype')


@pytest.mark.parametrize('dtype',[np.uint8,np.int64,np.uint64,np.float32,np.float64])
@pytest.mark.parametrize('shape',[(262145,),(513,517),(1,262145),(262145,1),(67,131,33)])
def test_worker_partitions(dtype,shape):
    check(sample(dtype,shape))


@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.complex64])
@pytest.mark.parametrize('target',[np.int8,np.float16,'this-is-not-a-dtype'])
def test_scalar_retains_identity_before_dtype_validation(dtype,target):
    image=np.array(7,dtype=dtype)
    check(image,dtype=target)


@pytest.mark.parametrize('image',[np.array([[np.inf,-np.inf]]),np.array([[np.nan,1.]]),np.full((3,5),np.finfo(np.float64).max)])
def test_exceptional_arithmetic_is_explicit_fallback(image):
    with pytest.raises(NotImplementedError):xx.transform.integral_image(image)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');expected=original.integral_image(image)
        xx.set_fallback_enabled(True)
        with pytest.warns(xx.FallbackWarning):actual=xx.transform.integral_image(image)
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('dtype',[np.uint64,np.float32,np.float64])
@pytest.mark.parametrize('axes',[(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)])
@pytest.mark.parametrize('shape',[(3,5,7),(67,131,33)])
def test_physical_axis_permutations(dtype,axes,shape):
    image=sample(dtype,shape).transpose(axes)
    check(image);check(image[::-1,::-1,::-1])


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('policy',['warn','raise'])
def test_floating_error_policies_on_safe_working_ranges(dtype,policy):
    image=sample(dtype,(17,19))
    with np.errstate(all=policy):
        check(image)
        if dtype==np.float32:check(image,dtype=np.float32)


@pytest.mark.parametrize('dtype',REAL)
def test_numpy_scalar_identity(dtype):
    check(dtype(1))


@pytest.mark.parametrize('dtype',REAL[:9])
@pytest.mark.parametrize('length',[1,2,3,4,7,8,15,16,17,31,32,33,65])
def test_integer_simd_chunk_boundaries(dtype,length):
    image=sample(dtype,(length,))
    check(image);check(image,dtype=np.int64);check(image,dtype=np.uint64)
