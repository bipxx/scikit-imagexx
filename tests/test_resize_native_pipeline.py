"""Direct candidate tests before public resize/rescale registration."""
import importlib.util
from pathlib import Path
import sys
import warnings
import numpy as np
import pytest
from skimage import transform as reference
import skimagexx as xx


def candidate_module():
    name='skimagexx._resize'
    path=Path(__file__).parents[1]/'python/skimagexx/_resize.py'
    loaded=sys.modules.get(name)
    if loaded is not None and Path(loaded.__file__).resolve()==path.resolve():return loaded
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


candidate=candidate_module()


@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    previous=xx.set_fallback_enabled(False);xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous);xx.set_num_threads(1)


def observe(function,*args,**kwargs):
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:result,error=function(*args,**kwargs),None
        except Exception as caught:result,error=None,caught
    return result,error,[(type(w.message),str(w.message)) for w in notices]


def check(name,image,target,**options):
    snapshot=image.tobytes()
    expected,error,notices=observe(getattr(reference,name),image,target,**options)
    actual,actual_error,actual_notices=observe(getattr(candidate,name),image,target,**options)
    assert actual_notices==notices
    assert image.tobytes()==snapshot
    if error is not None:
        assert type(actual_error) is type(error)
        assert str(actual_error)==str(error)
        return
    if actual_error is not None:raise actual_error
    assert type(actual) is type(expected)
    assert actual.shape==expected.shape and actual.strides==expected.strides
    assert actual.dtype==expected.dtype and actual.dtype.metadata==expected.dtype.metadata
    assert not np.shares_memory(image,actual)
    if expected.dtype.kind in 'biu' or options.get('order')==0:np.testing.assert_array_equal(actual,expected)
    else:
        tol=3e-6 if expected.dtype==np.float32 else 3e-13
        np.testing.assert_allclose(actual,expected,rtol=tol,atol=tol,equal_nan=True)
        if np.all(expected==0):assert actual.tobytes()==expected.tobytes()


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('pattern',[[0.],[-0.],[0.,-0.],[-0.,0.],[np.nan,0.,-0.],[np.nan]])
@pytest.mark.parametrize('order',[0,1,3])
def test_zero_signs_and_nan_bounds(dtype,pattern,order):
    check('resize',np.resize(np.array(pattern,dtype),(5,7)),(3,11),order=order,anti_aliasing=False)


@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.uint64,np.int64])
@pytest.mark.parametrize('order',[0,1,3])
@pytest.mark.parametrize('target',[(),(1,),(3,)])
def test_scalar_geometry(dtype,order,target):
    check('resize',np.asarray(1,dtype=dtype),target,order=order,anti_aliasing=False)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('cval',[np.nan,np.inf,-np.inf,3.,-.3])
@pytest.mark.parametrize('order',[0,1,3])
def test_constant_padding_extremes(dtype,cval,order):
    check('resize',np.arange(35,dtype=dtype).reshape(5,7),(9,11),order=order,
          mode='constant',cval=cval,anti_aliasing=False)


@pytest.mark.parametrize('order',[0,None])
def test_noncanonical_bool(order):
    image=np.resize(np.array([0,1,2,128,255],np.uint8),35).view(np.bool_).reshape(5,7)
    check('resize',image,(9,11),order=order,anti_aliasing=False)


@pytest.mark.parametrize('dtype',[np.longlong,np.ulonglong,np.intp,np.uintp])
@pytest.mark.parametrize('order',[0,1,3])
def test_integer_aliases(dtype,order):
    check('resize',np.arange(35,dtype=dtype).reshape(5,7),(3,11),order=order)


@pytest.mark.parametrize('order',[0.,1.,1.5,2.5,3.])
@pytest.mark.parametrize('target',[(5,7),(3,11)])
def test_fractional_order_identity(order,target):
    check('resize',np.arange(35,dtype=np.float64).reshape(5,7),target,order=order,anti_aliasing=False)


@pytest.mark.parametrize('cval',['1',[],{},None,complex(1,2)])
@pytest.mark.parametrize('target',[(5,7),(3,11)])
def test_constant_scalar_validation(cval,target):
    check('resize',np.arange(35,dtype=np.float64).reshape(5,7),target,cval=cval,
          clip=False,anti_aliasing=False)


@pytest.mark.parametrize('order',[0,1,1.5,3])
@pytest.mark.parametrize('anti_aliasing',[False,True])
@pytest.mark.parametrize('clip',[False,True])
@pytest.mark.parametrize('preserve_range',[False,True])
@pytest.mark.parametrize('pattern',[[0.,-0.,1.,np.nan,-1.],[np.nan],[0.,-0.]])
@pytest.mark.parametrize('mode',['reflect','constant'])
def test_opposite_endian_half_identity(order,anti_aliasing,clip,preserve_range,pattern,mode):
    if (candidate._scipy_profile()[1] and (order==0 or not preserve_range)
            and not hasattr(xx._core,'resize_half_identity')):
        pytest.skip('native Half identity helper pending; public registration remains gated')
    image=np.resize(np.asarray(pattern,np.float16),(3,5)).astype('>f2')
    options=dict(order=order,anti_aliasing=anti_aliasing,clip=clip,preserve_range=preserve_range,mode=mode)
    if anti_aliasing:options['anti_aliasing_sigma']=0
    check('resize',image,image.shape,**options)


def test_native_pipeline_does_not_call_python_image_algorithms(monkeypatch):
    from scipy import ndimage
    image=np.arange(35,dtype=np.uint64).reshape(5,7)+2**53
    expected=reference.resize(image,(3,11),order=0,mode='constant',cval=.3,anti_aliasing=True)
    def forbidden(*args,**kwargs):raise AssertionError('Python image algorithm called')
    for module,name in [(reference,'resize'),(ndimage,'zoom'),(ndimage,'gaussian_filter'),
                        (np,'clip'),(np,'min'),(np,'max'),(np,'nanmin'),(np,'nanmax')]:
        monkeypatch.setattr(module,name,forbidden)
    actual=candidate.resize(image,(3,11),order=0,mode='constant',cval=.3,anti_aliasing=True)
    np.testing.assert_array_equal(actual,expected)
