"""Sparse correlation order, padding, storage, promotion and floating policies."""
import inspect,re,warnings
import numpy as np
import pytest
from skimage.filters import correlate_sparse as reference
import skimagexx as xx

REAL=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64)
MODES=('reflect','mirror','nearest','edge','symmetric','constant','wrap','valid')

@pytest.fixture(autouse=True,params=[1,4])
def settings(request):
    fallback=xx.set_fallback_enabled(False);workers=xx.get_num_threads();xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(workers);xx.set_fallback_enabled(fallback)

def compare(image,kernel,mode='reflect'):
    before=image.copy();kb=np.array(kernel,copy=True)
    def captured(fn):
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter('always')
            try:result=fn();error=None
            except Exception as e:result=None;error=e
        return result,error,[(type(m.message),str(m.message)) for m in messages]
    expected,error,messages=captured(lambda:reference(image,kernel,mode))
    actual,failure,reported=captured(lambda:xx.filters.correlate_sparse(image,kernel,mode))
    assert reported==messages
    if error:
        assert type(failure) is type(error),(failure,error)
        assert str(failure)==str(error)
    else:
        assert failure is None,repr(failure)
        assert type(actual) is type(expected)
        assert actual.shape==expected.shape and actual.dtype==expected.dtype
        assert actual.dtype.metadata==expected.dtype.metadata
        np.testing.assert_array_equal(actual,expected)
        if actual.dtype.kind=='f':np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))
        assert not np.shares_memory(actual,image)
    np.testing.assert_array_equal(image,before);np.testing.assert_array_equal(kernel,kb)

def sample(dtype,shape):
    a=np.random.default_rng(231).integers(0,19,shape).astype(dtype)
    return a/np.dtype(dtype).type(8) if np.dtype(dtype).kind=='f' else a

@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('shape',[(13,),(7,9),(3,4,5)])
@pytest.mark.parametrize('kind',['sparse','dense','zero','corner'])
def test_real_types_modes_and_sparse_order(dtype,mode,shape,kind):
    a=sample(dtype,shape);k=np.zeros((3,)*len(shape))
    if kind=='dense':k=np.random.default_rng(831).uniform(-1,1,k.shape)
    elif kind=='sparse':k.flat[1]=.25;k.flat[-1]=-1.125
    elif kind=='corner':k.flat[0]=1;k.flat[-1]=-.25
    compare(a,k,mode)

@pytest.mark.parametrize('dtype',[np.uint8,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('layout',['F','reverse','stride','endian','unaligned','metadata','readonly'])
@pytest.mark.parametrize('mode',MODES)
def test_storage(dtype,layout,mode):
    a=sample(dtype,(9,13));k=np.array([[1.,0.,.25],[0.,-1.,0.],[0.,.5,0.]])
    if layout=='F':a=np.asfortranarray(a)
    elif layout=='reverse':a=a[::-1,::-1]
    elif layout=='stride':a=a[:,::2]
    elif layout=='endian':a=a.astype(a.dtype.newbyteorder('>'))
    elif layout=='unaligned':a=np.ndarray(a.shape,dtype=a.dtype,buffer=bytearray(a.nbytes+1),offset=1);a[:]=sample(dtype,a.shape)
    elif layout=='metadata':a=a.view(np.dtype(a.dtype,metadata={'unit':'raw'}))
    else:a.flags.writeable=False
    compare(a,k,mode)

@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('mode',MODES)
def test_kernel_storage(dtype,mode):
    k=sample(dtype,(3,7))[:,::-1].astype(np.dtype(dtype).newbyteorder('>'))
    compare(sample(np.float32,(8,11)),k,mode)

@pytest.mark.parametrize('shape',[(1,),(1,1),(1,2),(2,1),(1,1,2),(0,3),(2,0),(0,)])
@pytest.mark.parametrize('mode',MODES[:-1])
def test_singletons_empty_and_large_padding(shape,mode):
    compare(sample(np.float32,shape),np.ones((7,)*len(shape)),mode)

@pytest.mark.parametrize('shape',[(3,),(5,7),(3,4,5)])
def test_even_valid_kernel(shape):
    compare(sample(np.float32,shape),np.ones((2,)*len(shape)),'valid')

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('mode',['reflect','constant','valid'])
@pytest.mark.parametrize('policy',['warn','raise','ignore'])
@pytest.mark.parametrize('case',['nan','infinity','overflow-product','overflow-add','underflow','signaling-image','signaling-kernel'])
def test_floating_status(dtype,mode,policy,case):
    a=np.arange(1,66,dtype=dtype).reshape(5,13);k=np.array([[0,.25,0],[1,-1,0],[0,0,2]],dtype=dtype)
    if case=='nan':a.flat[3]=np.nan;a.flat[9]=-np.nan
    elif case=='infinity':a.flat[3]=np.inf;a.flat[9]=-np.inf
    elif case=='overflow-product':a[:]=np.finfo(dtype).max
    elif case=='overflow-add':a[:]=np.finfo(dtype).max;k[:]=1
    elif case=='underflow':a[:]=np.finfo(dtype).tiny;k[:]=np.finfo(dtype).tiny
    else:
        target=a if case=='signaling-image' else k
        target.view(np.uint32 if dtype==np.float32 else np.uint64).flat[1]=0x7f800001 if dtype==np.float32 else 0x7ff0000000000001
    with np.errstate(all=policy):compare(a,k,mode)

@pytest.mark.parametrize('mode',['reflect','valid'])
@pytest.mark.parametrize('corner',[0,1,2])
def test_memmap(tmp_path,mode,corner):
    a=np.memmap(tmp_path/'sparse.bin',mode='w+',shape=(9,13),dtype=np.float32);a[:]=sample(np.float32,a.shape)
    k=np.zeros((3,3));k[0,0]=corner;k[-1,-1]=.25
    compare(a,k,mode)

@pytest.mark.parametrize('shape',[(513,515),(65,65,65)])
def test_large_worker_boundaries(shape):
    k=np.zeros((3,)*len(shape));k.flat[0]=.25;k.flat[-1]=-.125;k.flat[len(k.flat)//2]=1
    compare(sample(np.float32,shape),k)

def test_signature():
    assert inspect.signature(xx.filters.correlate_sparse)==inspect.signature(reference)

def test_explicit_fallback():
    a=np.ones((3,3),dtype=complex);k=np.ones((3,3))
    with pytest.raises(NotImplementedError,match='outside native coverage'):xx.filters.correlate_sparse(a,k)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('policy',['warn','raise','ignore'])
@pytest.mark.parametrize('mode',['reflect','valid'])
def test_combined_status_in_last_worker(dtype,policy,mode):
    a=np.ones((515,517),dtype=dtype);k=np.array([[0,.5,0],[0,1,0],[0,0,2]],dtype=dtype)
    a[-1,-5:]=[np.finfo(dtype).max,np.finfo(dtype).tiny,np.inf,-np.inf,np.nan]
    with np.errstate(all=policy):compare(a,k,mode)

@pytest.mark.parametrize('dtype',[np.int64,np.uint64,np.longlong,np.ulonglong])
@pytest.mark.parametrize('mode',MODES)
def test_full_width_integer_conversion(dtype,mode):
    limits=np.iinfo(dtype)
    a=np.resize(np.array([limits.min,limits.max,limits.max-1,0,1],dtype=dtype),(9,13))
    compare(a,np.array([[0,1,0],[.25,-.125,0],[0,0,1]]),mode)

@pytest.mark.parametrize('mode',MODES[:-1])
def test_even_kernel_error(mode):
    compare(np.ones((3,4)),np.ones((2,3)),mode)

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('corner',[0,1,.25])
@pytest.mark.parametrize('mode',MODES)
def test_metadata_depends_on_first_product(dtype,corner,mode):
    a=sample(dtype,(9,13)).view(np.dtype(dtype,metadata={'unit':'raw'}))
    k=np.zeros((3,3));k[0,0]=corner;k[-1,-1]=.125
    compare(a,k,mode)
