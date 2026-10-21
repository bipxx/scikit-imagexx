"""Reference contracts for Farid derivatives, axes, boundaries and masks."""
import warnings
import re
import numpy as np
import pytest
from skimage import filters as original
import skimagexx as xx

NAMES=('farid','farid_h','farid_v')
REAL=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64)


@pytest.fixture(autouse=True,params=[1,4])
def native(request):
    previous=xx.set_fallback_enabled(False);xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous);xx.set_num_threads(1)


def sample(dtype,shape,seed=511):
    rng=np.random.default_rng(seed);kind=np.dtype(dtype).kind
    if kind=='b':data=rng.random(shape)>.5
    elif kind in 'iu':
        info=np.iinfo(dtype);data=rng.integers(int(info.min),int(info.max)+1,shape,dtype=dtype)
    else:data=rng.normal(size=shape).astype(dtype)
    return np.asarray(data,dtype=dtype)


def check(name,image,**options):
    with warnings.catch_warnings(record=True) as expected_notices:
        warnings.simplefilter('always')
        try:expected=getattr(original,name)(image,**options)
        except Exception as expected_error:
            with pytest.raises(type(expected_error),match=re.escape(str(expected_error))):
                getattr(xx.filters,name)(image,**options)
            return None,None
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always');actual=getattr(xx.filters,name)(image,**options)
    assert [(type(w.message),str(w.message)) for w in notices]==[(type(w.message),str(w.message)) for w in expected_notices]
    assert actual.shape==expected.shape and actual.dtype==expected.dtype
    tolerance=2e-6 if expected.dtype==np.float32 else 2e-13
    np.testing.assert_allclose(actual,expected,rtol=tolerance,atol=tolerance)
    assert not np.shares_memory(actual,image)
    return actual,expected


@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('shape',[(),(17,),(17,19),(2,5,7),(2,3,5,7),(0,7)])
def test_generic_dimensions_and_dtypes(dtype,shape):
    check('farid',sample(dtype,shape))


@pytest.mark.parametrize('name',NAMES[1:])
@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('shape',[(1,1),(1,17),(17,1),(17,19)])
def test_directional_shapes_and_dtypes(name,dtype,shape):
    check(name,sample(dtype,shape,512))


@pytest.mark.parametrize('axis',[None,0,1,-1,-2,(0,1),(1,0),(0,0),(0,),()])
@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
def test_axes_and_repeated_axes(axis,dtype):
    check('farid',sample(dtype,(13,17),513),axis=axis)


@pytest.mark.parametrize('axis',[None,0,1,2,-1,-2,-3,(0,2),(2,0),(0,0),()])
@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
def test_volume_axes(axis,dtype):
    check('farid',sample(dtype,(3,5,7),514),axis=axis)


@pytest.mark.parametrize('mode',['reflect','mirror','nearest','wrap','constant','grid-mirror','grid-wrap','grid-constant'])
@pytest.mark.parametrize('axis',[None,0])
@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
def test_boundaries_and_constant_fill(mode,axis,dtype):
    check('farid',sample(dtype,(13,17),515),axis=axis,mode=mode,cval=.123456789)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.uint8,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('layout',['reverse','stepped','fortran','unaligned','readonly'])
def test_layouts(name,dtype,layout):
    image=sample(dtype,(17,19),516)
    if layout=='reverse':image=image[::-1,::-1]
    elif layout=='stepped':image=np.tile(image,(2,2))[::2,::2]
    elif layout=='fortran':image=np.asfortranarray(image)
    elif layout=='readonly':image.flags.writeable=False
    else:
        storage=np.empty(image.nbytes+1,dtype=np.uint8)
        view=np.ndarray(image.shape,dtype=dtype,buffer=storage,offset=1);view[:]=image;image=view
    check(name,image)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',['>i2','>u2','>i4','>u4','>i8','>u8','>f2','>f4','>f8'])
def test_nonnative_storage(name,dtype):
    check(name,sample(np.dtype(dtype).newbyteorder('='),(17,19),517).astype(dtype)[::-1])


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
def test_large_parallel_arrays(name,dtype):
    check(name,sample(dtype,(513,517),518))


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('shape',[(1,513),(513,1)])
def test_skinny_arrays(name,dtype,shape):
    check(name,sample(dtype,shape,519))


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',REAL)
def test_numeric_masks(name,dtype):
    image=sample(np.float32,(19,23),520)
    mask=np.ones(image.shape,dtype=dtype);mask[5:11,7:13]=0
    if dtype==np.float16:
        with pytest.raises(RuntimeError,match='data type not supported'):getattr(original,name)(image,mask=mask)
        with pytest.raises(RuntimeError,match='data type not supported'):getattr(xx.filters,name)(image,mask=mask)
    else:check(name,image,mask=mask[::-1,::-1])


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('shape',[(),(23,),(1,23),(19,1),(19,23)])
@pytest.mark.parametrize('flag',[False,True])
def test_eroded_masks_broadcast_after_erosion(name,shape,flag):
    actual,expected=check(name,sample(np.float64,(19,23),521),mask=np.full(shape,flag,dtype=bool))
    if not flag:np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))


@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('shape',[(17,),(5,7,9),(5,5,7,9),(0,7)])
def test_volume_masks(dtype,shape):
    image=sample(dtype,shape,522);mask=np.ones(shape,dtype=bool)
    if mask.size:mask.flat[mask.size//2]=False
    check('farid',image,mask=mask)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype,level',[(np.float32,1e15),(np.float64,1e150)])
def test_constant_cancellation_keeps_convolution_order(name,dtype,level):
    # Reordering the weighted sum can erase the reference's small residual.
    check(name,np.full((13,17),level,dtype=dtype))


def test_empty_axis_sequence_ignores_unconsumed_boundary_options():
    check('farid',sample(np.float32,(5,7),523),axis=[],mode='unused-invalid-mode',cval=None)


@pytest.mark.parametrize('dtype',['f4','f8'])
@pytest.mark.parametrize('name',NAMES)
def test_equivalent_native_dtype_descriptors(dtype,name):
    alias=np.dtype('>'+dtype).newbyteorder('=')
    check(name,sample(np.dtype(dtype),(17,19),524).view(alias))


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.uint8,np.float16,np.float32,np.float64])
def test_readonly_memory_mapped_inputs_and_masks(name,dtype,tmp_path):
    image=sample(dtype,(17,19),525)
    path=tmp_path/'image.raw';stored=np.memmap(path,dtype=dtype,mode='w+',shape=image.shape)
    stored[:]=image;stored.flush();del stored
    mapped=np.memmap(path,dtype=dtype,mode='r',shape=image.shape)
    check(name,mapped,mask=np.ones((19,),dtype=np.int16))
    np.testing.assert_array_equal(mapped,image)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',['>i2','>f2','>f4','>f8'])
def test_nonnative_masks(name,dtype):
    mask=np.ones((17,19),dtype=dtype);mask[5:11,7:13]=0
    if mask.dtype.type==np.float16:
        with pytest.raises(RuntimeError):getattr(original,name)(sample(np.float32,mask.shape),mask=mask)
        with pytest.raises(RuntimeError):getattr(xx.filters,name)(sample(np.float32,mask.shape),mask=mask)
    else:check(name,sample(np.float32,mask.shape),mask=mask[::-1,::-1])


@pytest.mark.parametrize('name',NAMES[1:])
@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('policy',['warn','raise','call','print','log'])
def test_directional_subnormal_policy(name,dtype,policy):
    minimum=np.nextafter(dtype(0),dtype(1))
    image=np.zeros((17,19),dtype=dtype);image[1::2,1::2]=minimum
    with np.errstate(under=policy):
        actual,expected=check(name,image)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
def test_magnitude_nondefault_underflow_policy_is_explicit(dtype):
    image=np.full((17,19),np.finfo(dtype).tiny,dtype=dtype)
    with np.errstate(under='warn'):
        with pytest.raises(NotImplementedError):xx.filters.farid(image)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('case',['nan','inf','complex','subclass','arraylike'])
def test_unsupported_contracts_use_explicit_reference_fallback(name,case):
    image=sample(np.float32,(17,19),526)
    if case=='nan':image[7,9]=np.nan
    elif case=='inf':image[7,9]=np.inf
    elif case=='complex':image=image.astype(np.complex64)
    elif case=='subclass':image=np.ma.array(image)
    else:image=image.tolist()
    with pytest.raises(NotImplementedError):getattr(xx.filters,name)(image)
    xx.set_fallback_enabled(True)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('ignore')
        try:expected=getattr(original,name)(image)
        except Exception as error:
            with pytest.warns(xx.FallbackWarning):
                with pytest.raises(type(error)):getattr(xx.filters,name)(image)
            return
    with pytest.warns(xx.FallbackWarning):actual=getattr(xx.filters,name)(image)
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('options',[{'axis':3},{'axis':[0,3]},{'mode':'bad'},
                                   {'mode':['reflect','nearest']},{'cval':None},
                                   {'mask':np.ones((3,4),dtype=bool)}])
def test_invalid_metadata_matches_reference_through_fallback(options):
    image=sample(np.float32,(5,7),527)
    with pytest.raises((ValueError,TypeError,RuntimeError,IndexError)) as expected:original.farid(image,**options)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning):
        with pytest.raises(type(expected.value)) as actual:xx.filters.farid(image,**options)
    assert str(actual.value)==str(expected.value)


@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('shape',[(1,262145),(262145,1)])
@pytest.mark.parametrize('name',NAMES)
def test_parallel_skinny_arrays(name,dtype,shape):
    check(name,sample(dtype,shape,528))


@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
def test_scalar_mask_on_rank_zero(dtype):
    check('farid',np.asarray(1,dtype=dtype),mask=np.asarray(False))


@pytest.mark.parametrize('shape',[(2,3,2,3,5),(1,2,1,2,1,7),(1,)*7,(1,)*8])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('axis',[0,-1])
def test_high_dimensional_kernel_plans(shape,dtype,axis):
    check('farid',sample(dtype,shape,529),axis=axis,mode='wrap')


@pytest.mark.parametrize('dtype,level',[(np.float32,1e15),(np.float64,1e150)])
@pytest.mark.parametrize('axis',[None,0,-1,(2,0)])
def test_volume_constant_cancellation(dtype,level,axis):
    check('farid',np.full((3,5,7),level,dtype=dtype),axis=axis)
