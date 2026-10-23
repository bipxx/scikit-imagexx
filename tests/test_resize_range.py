"""Typed clipping bounds: precision, NaNs, native scans and runtime zero ties."""
import warnings
import numpy as np
import pytest
import skimagexx as xx

TYPES=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,
       np.uint64,np.int64,np.float32,np.float64)


@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    old=xx.set_fallback_enabled(False);xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(old);xx.set_num_threads(1)


def sample(dtype,shape):
    rng=np.random.default_rng(4852)
    n=int(np.prod(shape))
    if dtype is np.bool_:
        return np.resize(np.array([0,1,2,128,255],np.uint8),n).view(np.bool_).reshape(shape)
    if np.dtype(dtype).kind in 'iu':return rng.bit_generator.random_raw(n).astype(dtype).reshape(shape)
    a=rng.normal(size=n).astype(dtype)
    if n:a[::7]=np.nan
    return a.reshape(shape)


def check(image):
    snapshot=image.tobytes()
    nan=bool(np.isnan(image).any()) if image.dtype.kind=='f' else False
    value=bool((~np.isnan(image)).any()) if image.dtype.kind=='f' else True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        expected=(np.nanmin(image),np.nanmax(image)) if nan else (np.min(image),np.max(image))
    low,high,has_nan,has_value,runtime_order=xx._core.resize_range(image)
    assert (has_nan,has_value)==(nan,value)
    mixed=image.dtype.kind=='f' and bool(((image==0)&np.signbit(image)).any()) and bool(((image==0)&~np.signbit(image)).any())
    assert runtime_order==bool(not value or (mixed and (expected[0]==0 or expected[1]==0)))
    for actual,wanted in zip((low,high),expected):
        wanted=np.asarray(wanted)
        assert actual.shape==() and actual.dtype==wanted.dtype
        assert actual.dtype.metadata==wanted.dtype.metadata
        assert actual.tobytes()==wanted.tobytes()
    assert image.tobytes()==snapshot


@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('shape',[(),(1,),(3,),(4,),(7,),(8,),(15,),(16,),(31,),(32,),(33,),(1031,),(257,1025)])
def test_types_vector_tails_and_workers(dtype,shape):
    check(sample(dtype,shape))


@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('layout',['fortran','reverse','step','broadcast','readonly','unaligned','endian','tagged'])
def test_layouts(dtype,layout):
    a=sample(dtype,(17,19))
    if layout=='fortran':a=np.asfortranarray(a)
    elif layout=='reverse':a=a[::-1,::-1]
    elif layout=='step':a=a[::2,::2]
    elif layout=='broadcast':a=np.broadcast_to(a[:1],a.shape)
    elif layout=='readonly':a.flags.writeable=False
    elif layout=='endian':a=a.astype(a.dtype.newbyteorder('>'))
    elif layout=='tagged':a=a.view(np.dtype(a.dtype,metadata={'unit':'raw'}))
    else:
        original=a;a=np.ndarray(a.shape,a.dtype,buffer=bytearray(a.nbytes+1),offset=1);a[...]=original
    check(a)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('size',[1,3,4,7,8,15,16,17,31,32,33,37,1031,257*1025])
@pytest.mark.parametrize('pattern',[[0.],[-0.],[0.,-0.],[-0.,0.],[np.nan,-0.,0.],
    [np.nan],[np.inf],[-np.inf],[-np.inf,np.nan,np.inf],[np.nan,0.,-0.,1.],[-1.,0.,-0.]])
def test_special_values_and_zero_signs(dtype,size,pattern):
    check(np.resize(np.array(pattern,dtype),size))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('layout',['fortran','reverse','step','broadcast','endian'])
def test_zero_reduction_layout_order(dtype,layout):
    a=np.resize(np.array([np.nan,0.,-0.,0.,-0.],dtype),(17,19))
    if layout=='fortran':a=np.asfortranarray(a)
    elif layout=='reverse':a=a[::-1,::-1]
    elif layout=='step':a=a[::2,::2]
    elif layout=='endian':a=a.astype(a.dtype.newbyteorder('>'))
    else:a=np.broadcast_to(a[:1],a.shape)
    check(a)


@pytest.mark.parametrize('dtype',[np.uint64,np.int64])
def test_full_width_precision(dtype):
    if dtype is np.uint64:a=np.array([2**53+1,2**63+4095,2**64-1],dtype)
    else:a=np.array([-2**63,-2**53-1,2**53+1,2**63-1],dtype)
    check(a)


def test_no_python_image_algorithms(monkeypatch):
    images=[sample(np.uint64,(1031,)),np.resize(np.array([np.nan,0.,-0.]),1031),np.full(1031,np.nan)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        expected=[(np.nanmin(a),np.nanmax(a)) for a in images]
    def forbidden(*args,**kwargs):raise AssertionError('Python image reduction called')
    for name in ['min','max','nanmin','nanmax','minimum','maximum','fmin','fmax']:
        monkeypatch.setattr(np,name,forbidden)
    for a,wanted in zip(images,expected):
        low,high,*_=xx._core.resize_range(a)
        for actual,value in zip((low,high),wanted):assert actual.tobytes()==np.asarray(value).tobytes()


@pytest.mark.parametrize('dtype',TYPES)
def test_empty(dtype):
    with pytest.raises(ValueError,match='zero-size array to reduction operation minimum which has no identity'):
        xx._core.resize_range(np.empty((0,7),dtype))
