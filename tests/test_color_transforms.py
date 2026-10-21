"""Each native color transform versus the pinned upstream implementation."""
import warnings
import numpy as np
import pytest
from skimage import color as original
import skimagexx as xx

FUNCTIONS = ['rgb2yuv','yuv2rgb','rgb2yiq','yiq2rgb','rgb2ypbpr','ypbpr2rgb',
             'rgb2ycbcr','ycbcr2rgb','rgb2ydbdr','ydbdr2rgb','rgb2rgbcie',
             'rgbcie2rgb','rgb2xyz','xyz2rgb']
DTYPES = [np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,
          np.uint64,np.int64,np.float32,np.float64]


@pytest.fixture(autouse=True, params=[1,4])
def configuration(request):
    previous = xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(1)
    xx.set_fallback_enabled(previous)


def compare(name, a, **options):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        try:
            expected = getattr(original,name)(a, **options)
        except (ValueError, TypeError, IndexError, OverflowError, AttributeError) as error:
            with pytest.raises(type(error)) as actual:
                getattr(xx.color,name)(a, **options)
            assert str(actual.value) == str(error)
            return
        actual = getattr(xx.color,name)(a, **options)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    tol = 2e-6 if actual.dtype == np.float32 else 2e-13
    absolute = tol
    if actual.dtype == np.float32:
        # BLAS may fuse products; cancellation amplifies relative error near
        # zero. Bound absolute dot-product roundoff using the reference matrix.
        from skimage.color import colorconv
        left,right = name.split('2')
        matrix = getattr(colorconv,right+'_from_'+left)
        magnitude = np.max(np.abs(a)[np.isfinite(a)],initial=0)
        absolute += 4*np.finfo(np.float32).eps*np.max(np.sum(np.abs(matrix),axis=1))*magnitude
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=absolute)
    if isinstance(a,np.ndarray):
        assert not np.shares_memory(actual,a)


@pytest.mark.parametrize('name',FUNCTIONS)
@pytest.mark.parametrize('dtype',DTYPES)
def test_integer_widths_and_floating_extrema(name,dtype):
    dtype = np.dtype(dtype)
    if dtype.kind == 'b':
        a = np.tile(np.eye(3,dtype=dtype),(19,1))
    elif dtype.kind in 'ui':
        info = np.iinfo(dtype)
        a = np.tile(np.array([info.min,info.max,0,1,info.max//2,info.max//2+1],dtype=dtype),(19,1)).reshape(-1,3)
    else:
        a = np.random.default_rng(5).uniform(-2,2,(19,23,3)).astype(dtype)
        a[0,:3] = [[0,0,0],[1,1,1],[-0.0,0.04045,0.0031308]]
    before = a.copy()
    compare(name,a)
    np.testing.assert_array_equal(a,before)
    if dtype.kind in 'ui':
        bits = np.random.default_rng(92).integers(0,256,67*3*dtype.itemsize,dtype=np.uint8)
        compare(name,bits.view(dtype).reshape(67,3))


@pytest.mark.parametrize('name',FUNCTIONS)
@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('axis',[-4,-3,-2,-1,0,1,2,3,None,(1,)])
def test_arbitrary_dimensions_and_channel_axes(name,dtype,axis):
    a = np.random.default_rng(81).integers(0,256,(2,7,9,3),dtype=np.uint8)
    if dtype != np.uint8:
        a = (a/255).astype(dtype)
    if axis is not None:
        a = np.moveaxis(a,-1,axis[0] if isinstance(axis,tuple) else axis)
    compare(name,a,channel_axis=axis)


@pytest.mark.parametrize('name',FUNCTIONS)
@pytest.mark.parametrize('shape',[(3,),(7,3),(0,3),(2,0,3),(2,1,5,3)])
def test_one_pixel_empty_and_nd(name,shape):
    compare(name,np.zeros(shape,dtype=np.float64))


@pytest.mark.parametrize('name',FUNCTIONS)
@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
def test_parallel_strides_tails_and_alignment(name,dtype):
    a = np.random.default_rng(33).integers(0,256,(513,515,3),dtype=np.uint8)
    if dtype != np.uint8:
        a = (a/255).astype(dtype)
    compare(name,a)
    compare(name,a[::-1,::3,::-1])
    compare(name,a[::-1,::-1])
    compare(name,np.ascontiguousarray(np.moveaxis(a,-1,0)),channel_axis=0)
    storage = np.empty(a.nbytes+1,dtype=np.uint8)
    unaligned = np.ndarray(a.shape,dtype=dtype,buffer=storage,offset=1)
    unaligned[:] = a
    compare(name,unaligned)


@pytest.mark.parametrize('name',FUNCTIONS)
@pytest.mark.parametrize('shape',[(),(1,),(2,),(2,4),(2,3,4)])
def test_invalid_shape(name,shape):
    compare(name,np.zeros(shape))


@pytest.mark.parametrize('name',FUNCTIONS)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_nonfinite_and_srgb_boundary_pixels(name,dtype):
    a = np.tile(np.array([[np.nan,0,1],[0,np.inf,1],[-np.inf,1,0],
                         [0.04045,np.nextafter(0.04045,0),np.nextafter(0.04045,1)],
                         [0.0031308,np.nextafter(0.0031308,0),np.nextafter(0.0031308,1)],
                         [1e-300,-1e-300,-0.0]],dtype=dtype),(17,1))
    compare(name,a)


@pytest.mark.parametrize('name',FUNCTIONS)
@pytest.mark.parametrize('axis',[3,-3,(),(0,1),1.5])
def test_invalid_channel_axes(name,axis):
    compare(name,np.zeros((7,3)),channel_axis=axis)


@pytest.mark.parametrize('name',FUNCTIONS)
def test_noncanonical_boolean_storage(name):
    a = np.tile(np.array([0,1,2,128,255,0],dtype=np.uint8),19).view(np.bool_).reshape(-1,3)
    compare(name,a)


@pytest.mark.parametrize('name',FUNCTIONS)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_broadcast_and_uniform_strided_views(name,dtype):
    pixel = np.array([0.2,0.6,0.9],dtype=dtype)
    compare(name,np.broadcast_to(pixel,(19,3)))
    compare(name,np.broadcast_to(np.arange(19,dtype=dtype)[:,None]/dtype(19),(19,3)))
    a = np.random.default_rng(7).random((3,14,26)).astype(dtype)
    compare(name,a[:,:,::2],channel_axis=0)
    compare(name,a[:,::2,::2],channel_axis=0)
    compare(name,a[:,::-1,::-1],channel_axis=0)


@pytest.mark.parametrize('name',FUNCTIONS)
def test_array_like_and_identity(name):
    compare(name,[[0,1,0],[1,0,1]])
    assert getattr(xx,name) is getattr(xx.color,name)


@pytest.mark.parametrize('name',FUNCTIONS)
def test_unsupported_dtype_uses_explicit_fallback(name):
    a = np.zeros((7,3),dtype=np.float16)
    with pytest.raises(NotImplementedError):
        getattr(xx.color,name)(a)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning):
        actual = getattr(xx.color,name)(a)
    np.testing.assert_array_equal(actual,getattr(original,name)(a))


def test_color_pipeline_can_swap_backend():
    def pipeline(backend,a):
        return backend.color.xyz2rgb(backend.color.rgb2xyz(a))
    import skimage
    a = np.random.default_rng(9).random((17,21,3))
    np.testing.assert_allclose(pipeline(xx,a),pipeline(skimage,a),rtol=2e-13,atol=2e-13)
