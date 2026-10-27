"""Logical dtype equality must not depend on NumPy descriptor object identity."""
import importlib
import warnings
import numpy as np
import pytest
import skimagexx as xx
from skimagexx.coverage import native_functions,native_dtypes,needs_color_input,default_call,assert_default_parity,DERIVED_MOMENTS_FUNCTIONS,COORDS_MOMENTS_FUNCTIONS

CASES=[(api,dtype) for api in native_functions() for dtype in native_dtypes(api)]


@pytest.fixture(autouse=True,params=[1,4])
def native(request):
    previous=xx.set_fallback_enabled(False);xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous);xx.set_num_threads(1)


def alias(dtype):
    return np.dtype({'uint8':'>u1','float32':'>f4','float64':'>f8'}[dtype]).newbyteorder('=')


@pytest.mark.parametrize('api,dtype',CASES,ids=[api+'-'+dtype for api,dtype in CASES])
def test_every_native_default_accepts_equivalent_descriptor(api,dtype):
    descriptor=alias(dtype)
    assert descriptor==np.dtype(dtype) and descriptor.isnative
    rng=np.random.default_rng(439)
    gray=rng.integers(0,256,(9,11),dtype=np.uint8)
    rgb=rng.integers(0,256,(9,11,3),dtype=np.uint8)
    if dtype!='uint8':gray=(gray/255).astype(dtype);rgb=(rgb/255).astype(dtype)
    image=(rgb if needs_color_input(api) else gray).view(descriptor)
    arguments,options=default_call(api,image)
    module,name=api.rsplit('.',1)
    original=getattr(importlib.import_module('skimage.'+module),name)
    native=getattr(importlib.import_module('skimagexx.'+module),name)
    with warnings.catch_warnings(record=True) as expected_notices:
        warnings.simplefilter('always');expected=original(*arguments,**options)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always');actual=native(*arguments,**options)
    assert [(type(w.message),str(w.message)) for w in notices]==[(type(w.message),str(w.message)) for w in expected_notices]
    if expected is image:assert actual is image
    if isinstance(expected,tuple):
        assert type(actual) is type(expected) and len(actual)==len(expected)
    pairs=zip(actual,expected) if isinstance(expected,tuple) else [(actual,expected)]
    for a,b in pairs:
        if api=='segmentation.relabel_sequential' and type(b).__name__=='ArrayMap':
            assert type(a).__name__==type(b).__name__
            for field in ('in_values','out_values'):
                value,reference=getattr(a,field),getattr(b,field)
                assert value.shape==reference.shape and value.dtype==reference.dtype
                np.testing.assert_array_equal(value,reference)
                assert value.dtype.metadata==reference.dtype.metadata
            continue
        assert np.asarray(a).dtype==np.asarray(b).dtype and np.asarray(a).shape==np.asarray(b).shape
        assert_default_parity(api,np.dtype(dtype),a,b,
                              image=arguments if api == 'measure.mesh_surface_area' else arguments[0] if api in DERIVED_MOMENTS_FUNCTIONS or api in COORDS_MOMENTS_FUNCTIONS or api == 'measure.perimeter' or api == 'measure.euler_number' else image)


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('name',['img_as_float','img_as_float32','img_as_float64'])
@pytest.mark.parametrize('force_copy',[False,True])
def test_conversion_identity_and_copy_with_equivalent_descriptor(dtype,name,force_copy):
    from skimage import util
    image=np.linspace(0,1,99,dtype=dtype).reshape(9,11).view(alias(dtype))
    expected=getattr(util,name)(image,force_copy=force_copy)
    actual=getattr(xx.util,name)(image,force_copy=force_copy)
    assert (actual is image)==(expected is image)
    assert np.shares_memory(actual,image)==np.shares_memory(expected,image)
    assert actual.dtype==expected.dtype
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('operation',['bounds','edge'])
def test_logical_dtype_checks_still_reject_opposite_byte_order(dtype,operation):
    opposite=np.dtype(dtype).newbyteorder('S')
    image=np.arange(35,dtype=dtype).reshape(5,7).astype(opposite)
    assert not image.dtype.isnative
    with pytest.raises(TypeError):
        if operation=='bounds':xx._core.intensity_bounds(image)
        else:xx._core.small_edge(image,0)
