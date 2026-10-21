"""Stain image kernels and metadata versus original scikit-image."""
import warnings
import numpy as np
import pytest
from skimage import color as original
import skimagexx as xx
from skimagexx.coverage import parity_tolerance

NAMES=('separate_stains','combine_stains','rgb2hed','hed2rgb')
DTYPES=('bool','uint8','int8','uint16','int16','uint32','int32','uint64','int64','float32','float64')
FAMILIES=('hed','hdx','fgx','bex','rbd','gdx','hax','bro','bpx','ahx','hpx')


@pytest.fixture(autouse=True,params=[1,4])
def configure(request):
    previous=xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(1); xx.set_fallback_enabled(previous)


def compare(name,image,matrix=None,**kwargs):
    args=(image,matrix) if name in NAMES[:2] else (image,)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        expected=getattr(original,name)(*args,**kwargs)
        actual=getattr(xx.color,name)(*args,**kwargs)
    assert actual.dtype==expected.dtype
    assert actual.shape==expected.shape
    rtol,atol=parity_tolerance('color.'+name,np.asarray(image).dtype)
    scale=1 if matrix is None or not np.size(matrix) else max(1,float(np.max(np.sum(np.abs(matrix),axis=0))))
    np.testing.assert_allclose(actual,expected,rtol=rtol,atol=atol*scale)
    assert not np.shares_memory(image,actual)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',DTYPES)
def test_dtypes_and_extrema(name,dtype):
    dt=np.dtype(dtype)
    if dt.kind=='b': a=np.tile(np.array([0,1,2,128,255,0],dtype='u1'),19).view(bool).reshape(-1,3)
    elif dt.kind in 'ui':
        limits=np.iinfo(dt)
        a=np.tile(np.array([limits.min,limits.max,0,1,limits.max//2,limits.max//2+1],dtype=dt),19).reshape(-1,3)
    else: a=np.random.default_rng(13).random((19,23,3)).astype(dtype)
    before=a.copy()
    matrix=original.hed_from_rgb if name=='separate_stains' else original.rgb_from_hed
    compare(name,a,matrix)
    np.testing.assert_array_equal(a,before)


@pytest.mark.parametrize('name',NAMES[:2])
@pytest.mark.parametrize('family',FAMILIES)
@pytest.mark.parametrize('dtype',['float32','float64'])
def test_stain_families(name,family,dtype):
    a=np.random.default_rng(21).random((13,17,3)).astype(dtype)
    matrix=getattr(original,family+'_from_rgb' if name=='separate_stains' else 'rgb_from_'+family)
    compare(name,a,matrix)


@pytest.mark.parametrize('name',NAMES[:2])
@pytest.mark.parametrize('columns',[0,1,2,3,7,17])
@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('matrix_dtype',['float16','float32','float64','int16','bool'])
def test_custom_matrices_and_channels(name,columns,dtype,matrix_dtype):
    a=np.random.default_rng(9).random((7,19,3)).astype(dtype)
    matrix=np.random.default_rng(5).uniform(-2,2,(3,columns)).astype(matrix_dtype)
    compare(name,a,matrix)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('axis',[-4,-3,-2,-1,0,1,2,3,None,(1,)])
def test_nd_channel_axes(name,axis):
    a=np.random.default_rng(6).random((2,7,19,3)).astype('float32')
    if axis is not None: a=np.moveaxis(a,-1,axis[0] if isinstance(axis,tuple) else axis)
    matrix=original.hed_from_rgb if name=='separate_stains' else original.rgb_from_hed
    compare(name,a,matrix,channel_axis=axis)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('shape',[(3,),(0,3),(2,0,3),(2,1,3)])
def test_empty_and_single_pixels(name,shape):
    matrix=original.hed_from_rgb if name=='separate_stains' else original.rgb_from_hed
    compare(name,np.zeros(shape,dtype='float32'),matrix)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',['float32','float64'])
def test_parallel_layouts_and_readonly_inputs(name,dtype):
    a=np.random.default_rng(3).random((519,521,3)).astype(dtype)
    matrix=original.hed_from_rgb if name=='separate_stains' else original.rgb_from_hed
    for view in (a,a[::-1,::-1],a[:,::2,::-1],np.broadcast_to(a[0,0],a.shape)):
        compare(name,view,matrix)
    planar=np.ascontiguousarray(np.moveaxis(a,-1,0))
    compare(name,planar,matrix,channel_axis=0)
    unaligned=np.ndarray(a.shape,dtype=dtype,buffer=bytearray(a.nbytes+1),offset=1); unaligned[:]=a
    unaligned.flags.writeable=False
    compare(name,unaligned,matrix)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',['float32','float64'])
def test_floors_nonfinite_and_signed_zero(name,dtype):
    a=np.array([[0.,-0.,1.],[-1,np.inf,np.nan],[1e-8,1e-6,1e-4],
                [1e30,-1e30,-np.inf]],dtype=dtype)
    matrix=original.hed_from_rgb if name=='separate_stains' else original.rgb_from_hed
    compare(name,np.tile(a,(19,1)),matrix)


def test_matrix_metadata_and_mutable_aliases():
    for family in FAMILIES:
        for name in (family+'_from_rgb','rgb_from_'+family):
            actual=getattr(xx.color,name); expected=getattr(original,name)
            assert actual.dtype==expected.dtype
            np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-13)
    matrix=xx.color.hed_from_rgb
    saved=matrix.copy()
    try:
        matrix[0,0]+=1
        a=np.array([[.2,.3,.8]])
        np.testing.assert_allclose(xx.color.rgb2hed(a),original.separate_stains(a,matrix),rtol=2e-13,atol=2e-13)
    finally: matrix[:]=saved


@pytest.mark.parametrize('name',NAMES)
def test_half_fallback_and_root_aliases(name):
    a=np.zeros((7,3),dtype='float16')
    matrix=original.hed_from_rgb if name=='separate_stains' else original.rgb_from_hed
    args=(a,matrix) if name in NAMES[:2] else (a,)
    with pytest.raises(NotImplementedError): getattr(xx.color,name)(*args)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning): actual=getattr(xx.color,name)(*args)
    np.testing.assert_array_equal(actual,getattr(original,name)(*args))
    assert getattr(xx,name) is getattr(xx.color,name)
