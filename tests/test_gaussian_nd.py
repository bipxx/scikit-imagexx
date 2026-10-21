"""Gaussian axes, precision stages, channels, boundaries and output contracts."""
import re,warnings
import numpy as np
import pytest
from skimage import filters as original
import skimagexx as xx

REAL=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64)

@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    previous=xx.set_fallback_enabled(False);xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous);xx.set_num_threads(1)


def sample(dtype,shape,seed=541):
    rng=np.random.default_rng(seed);dtype=np.dtype(dtype)
    if dtype.kind=='b':return np.asarray(rng.random(shape)>.5,dtype=dtype)
    if dtype.kind in 'iu':
        limits=np.iinfo(dtype)
        return np.asarray(rng.integers(int(limits.min),int(limits.max)+1,shape,dtype=dtype))
    return np.asarray(rng.normal(size=shape),dtype=dtype)


def check(image,**options):
    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter('always')
        try:expected=original.gaussian(image,**options)
        except Exception as error:
            with pytest.raises(type(error),match=re.escape(str(error))):xx.filters.gaussian(image,**options)
            return None,None
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always');actual=xx.filters.gaussian(image,**options)
    assert [(type(w.message),str(w.message)) for w in notices]==[(type(w.message),str(w.message)) for w in expected_warnings]
    assert actual.shape==expected.shape and actual.dtype==expected.dtype
    assert actual.dtype.metadata==expected.dtype.metadata
    tolerance=2e-6 if expected.dtype==np.float32 else 2e-13
    np.testing.assert_allclose(actual,expected,rtol=tolerance,atol=tolerance,equal_nan=True)
    return actual,expected


@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('shape',[(),(17,),(17,19),(2,5,7),(1,5,1,7),(0,7)])
@pytest.mark.parametrize('preserve',[False,True])
def test_shapes_and_real_dtypes(dtype,shape,preserve):
    check(sample(dtype,shape),sigma=.75,preserve_range=preserve)


@pytest.mark.parametrize('sigma',[0,1e-16,.5,1.,2.5,np.float16(.75),np.float32(.75),np.float64(.75),np.int8(2),np.bool_(False),np.bool_(True),np.nan])
@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
def test_scalar_sigma_precision(sigma,dtype):
    check(sample(dtype,(13,17)),sigma=sigma)


@pytest.mark.parametrize('mode',['reflect','constant','nearest','mirror','wrap','grid-mirror','grid-constant','grid-wrap'])
@pytest.mark.parametrize('shape',[(1,),(1,17),(17,1),(13,17),(2,5,7)])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_boundary_modes(mode,shape,dtype):
    check(sample(dtype,shape),sigma=1.25,mode=mode,cval=.123456789)


@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('sigmas',[(0,.5,1),(1,0,2),(1e-16,.75,1.25)])
def test_per_axis_sigma_and_boundary(dtype,sigmas):
    check(sample(dtype,(3,7,19)),sigma=sigmas,mode=['wrap','mirror','constant'],cval=.25)


@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('axis',[-3,-1,0,1,2,5])
@pytest.mark.parametrize('sigma',[.75,(.5,1.25),(0,.5,1.25)])
def test_channel_axes(dtype,axis,sigma):
    check(sample(dtype,(3,5,7)),sigma=sigma,channel_axis=axis,mode='reflect')


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('layout',['fortran','reverse','stepped','unaligned','readonly','endian'])
def test_layouts(dtype,layout):
    image=sample(dtype,(17,19))
    if layout=='fortran':image=np.asfortranarray(image)
    elif layout=='reverse':image=image[::-1,::-1]
    elif layout=='stepped':image=np.tile(image,(2,2))[::2,::2]
    elif layout=='readonly':image.flags.writeable=False
    elif layout=='endian':image=image.astype(image.dtype.newbyteorder('>'))
    else:
        storage=np.empty(image.nbytes+1,dtype=np.uint8)
        view=np.ndarray(image.shape,dtype=dtype,buffer=storage,offset=1);view[:]=image;image=view
    check(image,sigma=(.5,1.25),mode='mirror')


@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('target',[np.float32,np.float64])
@pytest.mark.parametrize('layout',['contiguous','fortran','reversed','endian'])
def test_output_buffer_precision_and_identity(dtype,target,layout):
    image=sample(dtype,(17,19));output_dtype=np.dtype(target)
    if layout=='endian':output_dtype=output_dtype.newbyteorder('>')
    expected_out=np.empty(image.shape,dtype=output_dtype);actual_out=np.empty_like(expected_out)
    if layout=='fortran':expected_out=np.asfortranarray(expected_out);actual_out=np.asfortranarray(actual_out)
    if layout=='reversed':expected_out=expected_out[::-1,::-1];actual_out=actual_out[::-1,::-1]
    expected=original.gaussian(image,sigma=(.5,1.25),out=expected_out)
    actual=xx.filters.gaussian(image,sigma=(.5,1.25),out=actual_out)
    assert expected is expected_out and actual is actual_out
    tolerance=2e-6 if target==np.float32 else 2e-13
    np.testing.assert_allclose(actual,expected,rtol=tolerance,atol=tolerance)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_inplace_output(dtype):
    image=sample(dtype,(17,19));expected=image.copy();actual=image.copy()
    original.gaussian(expected,.75,out=expected)
    assert xx.filters.gaussian(actual,.75,out=actual) is actual
    tolerance=2e-6 if dtype==np.float32 else 2e-13
    np.testing.assert_allclose(actual,expected,rtol=tolerance,atol=tolerance)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape',[(1,262145),(262145,1),(513,517),(3,301,317)])
def test_parallel_images_and_skinny_shapes(dtype,shape):
    check(sample(dtype,shape),sigma=.75,mode='reflect')


@pytest.mark.parametrize('sigma',[0,.5,np.float32(.75),(0,1.25)])
def test_dtype_metadata(sigma):
    tagged=np.dtype('float32',metadata={'units':'linear'})
    check(sample(np.float32,(17,19)).astype(tagged),sigma=sigma)


@pytest.mark.parametrize('dtype,level',[(np.float32,1e30),(np.float64,1e150)])
@pytest.mark.parametrize('shape',[(13,17),(1,17),(3,5,7)])
def test_large_values_keep_symmetric_accumulation(dtype,level,shape):
    image=np.full(shape,level,dtype=dtype)
    image.flat[::3]*=-1
    check(image,sigma=.75,mode='reflect')


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('target',[np.float32,np.float64])
def test_no_axis_cast_warnings_and_signaling_nans(dtype,target):
    if dtype==np.float16:word=0x7c01;unsigned=np.uint16
    elif dtype==np.float32:word=0x7f800001;unsigned=np.uint32
    else:word=0x7ff0000000000001;unsigned=np.uint64
    image=np.full((3,5),word,dtype=unsigned).view(dtype)
    expected_out=np.zeros(image.shape,dtype=target);actual_out=np.zeros_like(expected_out)
    with warnings.catch_warnings(record=True) as expected_notices:
        warnings.simplefilter('always');original.gaussian(image,0,out=expected_out)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always');xx.filters.gaussian(image,0,out=actual_out)
    assert [(type(w.message),str(w.message)) for w in notices]==[(type(w.message),str(w.message)) for w in expected_notices]
    bits=np.uint32 if target==np.float32 else np.uint64
    np.testing.assert_array_equal(actual_out.view(bits),expected_out.view(bits))


@pytest.mark.parametrize('policy',['warn','raise'])
def test_no_axis_cast_side_effects(policy):
    image=np.array([[1e300,1e-300,-1e300],[1.,0.,-1.]])
    expected_out=np.zeros(image.shape,dtype=np.float32);actual_out=np.zeros_like(expected_out)
    def execute(fn,out):
        with np.errstate(over=policy,under=policy):
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always')
                try:fn(image,0,out=out)
                except FloatingPointError as error:return [(type(error),str(error))]
                return [(type(w.message),str(w.message)) for w in notices]
    assert execute(xx.filters.gaussian,actual_out)==execute(original.gaussian,expected_out)
    np.testing.assert_array_equal(actual_out,expected_out)


@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('shape',[(19,23,3),(67,131,33)])
@pytest.mark.parametrize('mode',['reflect','constant','nearest','mirror','wrap'])
def test_channel_interior_spans(dtype,shape,mode):
    check(sample(dtype,shape),sigma=.75,channel_axis=2,mode=mode,cval=.25)
