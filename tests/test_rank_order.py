"""Exact ranks, reconstruction, label widths and storage against upstream."""
import inspect,re,warnings
import numpy as np
import pytest
from skimage.filters import rank_order as reference
import skimagexx as xx

REAL=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64)

@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    previous=xx.set_fallback_enabled(False);workers=xx.get_num_threads();xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(workers);xx.set_fallback_enabled(previous)

def compare(image):
    before=image.copy()
    try:expected=reference(image)
    except Exception as error:
        with pytest.raises(type(error),match=re.escape(str(error))):xx.filters.rank_order(image)
        return
    actual=xx.filters.rank_order(image)
    assert type(actual) is type(expected) is tuple and len(actual)==2
    for a,e in zip(actual,expected):
        assert type(a) is type(e) and a.shape==e.shape and a.dtype==e.dtype
        assert a.dtype.metadata==e.dtype.metadata
        np.testing.assert_array_equal(a,e)
        assert a.tobytes()==e.tobytes()
    np.testing.assert_array_equal(actual[1][actual[0]],image)
    np.testing.assert_array_equal(image,before)
    assert not np.shares_memory(image,actual[0]) and not np.shares_memory(image,actual[1])

def sample(dtype,shape):
    rng=np.random.default_rng(301)
    a=rng.integers(0,127,shape)
    if dtype==np.bool_:return a.astype(bool)
    if np.dtype(dtype).kind=='f':return (a/32-1).astype(dtype)
    return a.astype(dtype)

def test_signature():assert inspect.signature(xx.filters.rank_order)==inspect.signature(reference)

@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('shape',[(),(0,),(0,7),(1,),(7,),(8,),(9,),(17,19),(2,3,5),(2,1,3,7)])
def test_shapes_types(dtype,shape):compare(sample(dtype,shape))

@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('layout',['f','reverse','strided','broadcast','readonly','endian','unaligned','metadata','scalar'])
def test_storage(dtype,layout):
    a=sample(dtype,(17,19))
    if layout=='f':a=np.asfortranarray(a)
    if layout=='reverse':a=a[::-1,::-1]
    if layout=='strided':a=a[:,::2]
    if layout=='broadcast':a=np.broadcast_to(a[0,0],a.shape)
    if layout=='readonly':a.flags.writeable=False
    if layout=='endian':a=a.astype(a.dtype.newbyteorder('S'))
    if layout=='unaligned':
        buffer=np.empty(a.nbytes+1,np.uint8);b=np.ndarray(a.shape,dtype=dtype,buffer=buffer,offset=1);b[:]=a;a=b
    if layout=='metadata':a=a.view(np.dtype(dtype,metadata={'rank':'original'}))
    if layout=='scalar':a=a[0,0]
    compare(a)

@pytest.mark.parametrize('dtype',[np.uint8,np.uint64,np.int64,np.float32,np.float64])
@pytest.mark.parametrize('size',[254,255,256,257,2047,2048,2049,65535,65536,262145])
def test_label_widths_and_radix_boundaries(dtype,size):
    compare(sample(dtype,(size,)))

@pytest.mark.parametrize('dtype',[np.int8,np.uint8,np.int16,np.uint16,np.int32,np.uint32,np.int64,np.uint64])
def test_full_integer_values(dtype):
    info=np.iinfo(dtype)
    a=np.array([info.min,info.max,0,1,info.max-1,info.min+1,info.max//2],dtype=dtype)
    compare(np.resize(a,(519,521)))

@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('kind',['continuous','infinite','negative-zero','positive-zero','subnormal'])
def test_float_values(dtype,kind):
    a=np.random.default_rng(123).normal(-2,7,(517,519)).astype(dtype)
    if kind=='infinite':a.flat[::19]=np.inf;a.flat[1::31]=-np.inf
    if kind=='negative-zero':a[:]=dtype(-0.)
    if kind=='positive-zero':a[:]=dtype(0.)
    if kind=='subnormal':a=np.resize(np.array([0,np.nextafter(dtype(0),dtype(1)),np.nextafter(dtype(0),dtype(-1))],dtype=dtype),a.shape)
    compare(a)

@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('kind',['nan','mixed-zero'])
def test_sort_provider_cases_explicit_fallback(dtype,kind):
    a=np.array([1,np.nan,np.nan,3] if kind=='nan' else [0.,-0.,0.,-0.],dtype=dtype)
    with pytest.raises(NotImplementedError,match='outside native coverage'):xx.filters.rank_order(a)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning):actual=xx.filters.rank_order(a)
    expected=reference(a)
    for value,original in zip(actual,expected):assert value.tobytes()==original.tobytes()

def test_memmap(tmp_path):
    a=np.memmap(tmp_path/'image.bin',dtype=np.int64,mode='w+',shape=(17,19));a[:]=sample(np.int64,a.shape)
    compare(a)
    labels,_=xx.filters.rank_order(a)
    assert labels.filename is None and labels._mmap is None and labels.mode is None and labels.offset is None

@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('size',[7,8,9,31,32,33,257,262145])
@pytest.mark.parametrize('mismatch',[False,True])
def test_uniform_and_late_mismatch(dtype,size,mismatch):
    a=np.full(size,True if dtype==np.bool_ else 7,dtype=dtype)
    if mismatch:a[-1]=0
    compare(a)

@pytest.mark.parametrize('values',[[2,2,2],[0,2,2],[1,255,0],[128,255,128],[0,0,128]])
def test_noncanonical_boolean_storage_explicit_fallback(values):
    a=np.array(values,np.uint8).view(bool)
    with pytest.raises(NotImplementedError,match='outside native coverage'):xx.filters.rank_order(a)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning):actual=xx.filters.rank_order(a)
    expected=reference(a)
    for value,original in zip(actual,expected):assert value.tobytes()==original.tobytes()

@pytest.mark.parametrize('dtype',[np.uint16,np.uint64,np.int64,np.float32,np.float64])
@pytest.mark.parametrize('size',[4095,4096,4097,65535,65536,262145])
def test_dense_table_size_and_label_boundaries(dtype,size):
    values=np.arange(257,dtype=dtype)
    if np.dtype(dtype).kind=='f':values=values*np.nextafter(dtype(0),dtype(1))
    compare(np.resize(values[::-1],size))

@pytest.mark.parametrize('dtype',[np.uint64,np.int64,np.float32,np.float64])
@pytest.mark.parametrize('unique',[4095,4096,4097])
@pytest.mark.parametrize('size',[65535,65536,262145])
def test_sparse_table_capacity_and_radix_fallback(dtype,unique,size):
    if dtype==np.uint64:
        values=np.arange(unique,dtype=dtype)*np.uint64(257)
        values+=np.uint64(np.iinfo(dtype).max-int(values[-1]))
    elif dtype==np.int64:values=np.arange(unique,dtype=dtype)*257+np.iinfo(dtype).min
    else:values=np.arange(unique,dtype=dtype)*dtype(.125)
    compare(np.resize(values[np.random.default_rng(407).permutation(unique)],size))

@pytest.mark.parametrize('size',[65535,65536,262145])
def test_dense_full_uint16_rank_range(size):
    compare(np.resize(np.arange(min(size,65536),dtype=np.uint16)[::-1],size))

@pytest.mark.parametrize('span',[65534,65535,65536])
@pytest.mark.parametrize('size',[4096,65535,65536])
def test_dense_span_limit_near_uint64_max(span,size):
    values=np.array([np.iinfo(np.uint64).max-span,np.iinfo(np.uint64).max],dtype=np.uint64)
    compare(np.resize(values,size))
