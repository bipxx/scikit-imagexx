"""Complex Gabor kernel parity, planning, and strict native dispatch."""
import inspect
import warnings
import numpy as np
import pytest
from skimage.filters import gabor_kernel as reference
import skimagexx as xx


@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    previous=xx.set_fallback_enabled(False);xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(1);xx.set_fallback_enabled(previous)


def compare(frequency,**options):
    expected=reference(frequency,**options)
    actual=xx.filters.gabor_kernel(frequency,**options)
    assert type(actual) is type(expected) is np.ndarray
    assert actual.shape==expected.shape and actual.dtype==expected.dtype
    assert actual.dtype.metadata==expected.dtype.metadata
    tolerance=3e-7 if actual.dtype.type==np.complex64 else 3e-14
    # Relative error alone is ill-conditioned at the real/imaginary zeroes;
    # bound absolute error by the maximum envelope, never the image scale.
    scale=float(np.max(np.abs(expected)))
    np.testing.assert_allclose(actual.real,expected.real,rtol=tolerance,atol=tolerance*scale)
    np.testing.assert_allclose(actual.imag,expected.imag,rtol=tolerance,atol=tolerance*scale)
    return actual


@pytest.mark.parametrize('dtype',[np.complex64,np.complex128])
@pytest.mark.parametrize('frequency',[.05,.13,.2,.7,1.,-.3])
@pytest.mark.parametrize('theta',[0.,.23,np.pi/4,np.pi/2,2.7])
def test_frequency_rotation(dtype,frequency,theta):
    compare(frequency,theta=theta,dtype=dtype)


@pytest.mark.parametrize('dtype',[np.complex64,np.complex128])
@pytest.mark.parametrize('bandwidth',[.2,.5,1.,2.,4.])
@pytest.mark.parametrize('sigmas',[(None,None),(1.3,None),(None,2.7),(2.,5.),(-2.,3.),(-3.,-2.)])
def test_envelope(dtype,bandwidth,sigmas):
    compare(.3,theta=.71,bandwidth=bandwidth,sigma_x=sigmas[0],sigma_y=sigmas[1],dtype=dtype)


@pytest.mark.parametrize('dtype',[np.complex64,np.complex128])
@pytest.mark.parametrize('offset',[0.,.2,np.pi/2,np.pi,-2.9,99999.,100001.])
@pytest.mark.parametrize('n_stds',[0.,-2.,1.,3.,7.])
def test_phase_and_extent(dtype,offset,n_stds):
    compare(.2,sigma_x=2.,sigma_y=3.,theta=.53,offset=offset,n_stds=n_stds,dtype=dtype)


@pytest.mark.parametrize('scalar',[float,np.float16,np.float32,np.float64,np.int16,np.int64])
@pytest.mark.parametrize('dtype',[np.complex64,np.complex128])
def test_scalar_promotion(scalar,dtype):
    compare(scalar(1),theta=scalar(1),bandwidth=scalar(2),sigma_x=scalar(2),
            sigma_y=scalar(3),n_stds=scalar(2),offset=scalar(1),dtype=dtype)


@pytest.mark.parametrize('dtype',[np.dtype(np.complex64,metadata={'tag':'gabor'}),
                                np.dtype(np.complex128,metadata={'tag':'gabor'})])
def test_dtype_metadata(dtype):
    compare(.2,dtype=dtype)


def test_signature():
    assert inspect.signature(xx.filters.gabor_kernel)==inspect.signature(reference)


@pytest.mark.parametrize('options',[{'dtype':np.float32},{'dtype':'not-a-dtype'}])
def test_dtype_errors(options):
    def outcome(function):
        try:function(.2,**options)
        except Exception as error:return type(error),str(error)
    assert outcome(xx.filters.gabor_kernel)==outcome(reference)


@pytest.mark.parametrize('options',[{'frequency':0.},{'frequency':float('nan')},
    {'frequency':.2,'sigma_x':0.},{'frequency':.2,'dtype':np.clongdouble},
    {'frequency':.2,'theta':[1.]},{'frequency':.2,'dtype':'>c16'},
    {'frequency':.2,'sigma_x':np.float16(300),'sigma_y':1,'n_stds':1},
    {'frequency':.2,'sigma_x':np.float16(200),'sigma_y':np.float16(200),'n_stds':1}])
def test_explicit_unsupported_calls(options):
    if (isinstance(options.get('sigma_x'),np.float16)
            and type(np.float16(1)**2) is np.float64):
        # NumPy 1.24 computes these scalar powers/products in float64. The
        # same narrow values therefore stay inside the finite native core.
        compare(options['frequency'],**{k:v for k,v in options.items() if k!='frequency'});return
    # Extended complex types are a separate API contract even on platforms
    # where their storage width happens to equal complex128.
    with pytest.raises(NotImplementedError):xx.filters.gabor_kernel(**options)
    xx.set_fallback_enabled(True)
    def outcome(function):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:value=function(**options)
            except Exception as error:value=error
        return value,[(w.category,str(w.message)) for w in caught if w.category is not xx.FallbackWarning]
    actual,aw=outcome(xx.filters.gabor_kernel);expected,ew=outcome(reference)
    assert aw==ew
    if isinstance(expected,Exception):assert (type(actual),str(actual))==(type(expected),str(expected))
    else:np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('scalar',[np.int8(12),np.int16(200)])
def test_wrapped_scalar_squares(scalar):
    options=dict(sigma_x=scalar,sigma_y=1,n_stds=1)
    if (scalar**2)>0:
        # NumPy 1.24 promotes these powers; NumPy 2 preserves narrow width.
        compare(.2,**options)
    else:
        with pytest.raises(NotImplementedError):xx.filters.gabor_kernel(.2,**options)
        xx.set_fallback_enabled(True)
        with pytest.warns(xx.FallbackWarning):actual=xx.filters.gabor_kernel(.2,**options)
        np.testing.assert_array_equal(actual,reference(.2,**options))


def test_scalar_overflow_raise_order():
    options=dict(sigma_x=np.float16(300),sigma_y=1,n_stds=1)
    # Older NumPy promotes this scalar power to float64 instead.
    with np.errstate(over='ignore'):
        if np.isfinite(options['sigma_x']**2):return
    xx.set_fallback_enabled(True)
    def outcome(function):
        with warnings.catch_warnings(record=True) as caught,np.errstate(over='raise'):
            warnings.simplefilter('always')
            try:function(.2,**options)
            except Exception as error:result=(type(error),str(error))
        return result,[(w.category,str(w.message)) for w in caught if w.category is not xx.FallbackWarning]
    assert outcome(xx.filters.gabor_kernel)==outcome(reference)


@pytest.mark.parametrize('dtype',[np.complex64,np.complex128])
def test_large_parallel_and_no_original_body(dtype,monkeypatch):
    options=dict(sigma_x=90.,sigma_y=100.,theta=.3,dtype=dtype)
    expected=reference(.2,**options)
    def forbidden(*args,**kwargs):raise AssertionError('original Gabor body executed')
    monkeypatch.setattr('skimage.filters.gabor_kernel',forbidden)
    xx.set_num_threads(1);serial=xx.filters.gabor_kernel(.2,**options)
    xx.set_num_threads(4);parallel=xx.filters.gabor_kernel(.2,**options)
    np.testing.assert_array_equal(serial,parallel)
    tolerance=3e-7 if dtype==np.complex64 else 3e-14
    np.testing.assert_allclose(serial,expected,rtol=tolerance,atol=tolerance*np.max(np.abs(expected)))
