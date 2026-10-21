"""Difference of Gaussians and unsharp masking use the same native blur stages."""
import re,warnings
import numpy as np
import pytest
from skimage import filters as original
import skimagexx as xx

NAMES=('difference_of_gaussians','unsharp_mask')
REAL=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64)

@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    previous=xx.set_fallback_enabled(False);xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous);xx.set_num_threads(1)


def sample(dtype,shape):
    rng=np.random.default_rng(542);dtype=np.dtype(dtype)
    if dtype.kind=='b':return np.asarray(rng.random(shape)>.5,dtype=dtype)
    if dtype.kind in 'iu':
        limits=np.iinfo(dtype);return np.asarray(rng.integers(int(limits.min),int(limits.max)+1,shape,dtype=dtype))
    return np.asarray(rng.normal(size=shape),dtype=dtype)


def check(name,image,**options):
    arguments=(image,1.) if name=='difference_of_gaussians' else (image,)
    if name=='difference_of_gaussians' and 'low_sigma' in options:arguments=(image,options.pop('low_sigma'))
    with warnings.catch_warnings(record=True) as expected_notices:
        warnings.simplefilter('always')
        try:expected=getattr(original,name)(*arguments,**options)
        except Exception as error:
            with pytest.raises(type(error),match=re.escape(str(error))):getattr(xx.filters,name)(*arguments,**options)
            return
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always');actual=getattr(xx.filters,name)(*arguments,**options)
    assert [(type(w.message),str(w.message)) for w in notices]==[(type(w.message),str(w.message)) for w in expected_notices]
    assert np.shape(actual)==np.shape(expected) and actual.dtype==expected.dtype
    assert actual.dtype.metadata==expected.dtype.metadata
    tolerance=2e-6 if expected.dtype==np.float32 else 2e-13
    np.testing.assert_allclose(actual,expected,rtol=tolerance,atol=tolerance)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('shape',[(),(17,),(17,19),(3,5,7),(0,7)])
def test_default_real_inputs(name,dtype,shape):
    check(name,sample(dtype,shape))


@pytest.mark.parametrize('low,high',[(0,None),(.5,None),(.5,1.25),((0,.75),(1,1.25)),(np.float32(.75),np.float64(1.25))])
@pytest.mark.parametrize('mode',['reflect','constant','nearest','mirror','wrap'])
@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
def test_difference_sigmas_and_boundaries(low,high,mode,dtype):
    check('difference_of_gaussians',sample(dtype,(13,17)),low_sigma=low,high_sigma=high,mode=mode,cval=.25)


@pytest.mark.parametrize('radius',[0,.5,np.float32(.75),(.5,1.25)])
@pytest.mark.parametrize('amount',[0,-1,.5,2.,np.float32(.75),np.float64(.75)])
@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('preserve',[False,True])
def test_unsharp_formula_stages(radius,amount,dtype,preserve):
    check('unsharp_mask',sample(dtype,(13,17)),radius=radius,amount=amount,preserve_range=preserve)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('axis',[-3,-1,0,1,2])
@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
def test_channel_axes(name,axis,dtype):
    check(name,sample(dtype,(3,5,7)),channel_axis=axis)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.uint8,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('layout',['fortran','reverse','stepped','readonly','endian'])
def test_layouts(name,dtype,layout):
    image=sample(dtype,(17,19))
    if layout=='fortran':image=np.asfortranarray(image)
    elif layout=='reverse':image=image[::-1,::-1]
    elif layout=='stepped':image=np.tile(image,(2,2))[::2,::2]
    elif layout=='readonly':image.flags.writeable=False
    else:image=image.astype(image.dtype.newbyteorder('>'))
    check(name,image)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_dtype_metadata(name,dtype):
    check(name,sample(dtype,(17,19)).astype(np.dtype(dtype,metadata={'units':'linear'})))


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('shape',[(513,517),(1,262145),(262145,1)])
def test_parallel_calls(name,dtype,shape):
    check(name,sample(dtype,shape))


@pytest.mark.parametrize('value',[np.bool_(False),np.bool_(True)])
@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
def test_numpy_boolean_parameters(value,dtype):
    image=sample(dtype,(13,17))
    check('unsharp_mask',image,radius=value,amount=value)
    check('difference_of_gaussians',image,low_sigma=value)
