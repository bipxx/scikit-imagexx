"""Median contract cases; native coverage is registered only after these pass."""
import inspect,warnings
import numpy as np
import pytest
from skimage.filters import median as reference
import skimagexx as xx

REAL=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64)
MODES=('nearest','reflect','mirror','wrap','constant','grid-mirror','grid-wrap','grid-constant')

@pytest.fixture(autouse=True,params=[1,4])
def settings(request):
    previous=xx.set_fallback_enabled(False);workers=xx.get_num_threads();xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(workers);xx.set_fallback_enabled(previous)

def compare(image,footprint=None,**options):
    before=np.array(image,copy=True)
    def captured(fn):
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter('always')
            try:result=fn();error=None
            except Exception as e:result=None;error=e
        return result,error,[(type(m.message),str(m.message)) for m in messages]
    expected,error,messages=captured(lambda:reference(image,footprint,**options))
    actual,failure,reported=captured(lambda:xx.filters.median(image,footprint,**options))
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
        assert not np.shares_memory(actual,image)
    np.testing.assert_array_equal(image,before)

def sample(dtype,shape):
    a=np.random.default_rng(3241).integers(0,31,shape).astype(dtype)
    return a/np.dtype(dtype).type(8) if np.dtype(dtype).kind=='f' else a

@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('shape',[(),(33,),(9,11),(3,5,7),(2,3,4,5)])
@pytest.mark.parametrize('mode',MODES)
def test_default_real_types_and_dimensions(dtype,shape,mode):
    compare(sample(dtype,shape),mode=mode)

@pytest.mark.parametrize('dtype',[np.uint8,np.int16,np.float32,np.float64])
@pytest.mark.parametrize('shape',[(33,),(9,11),(3,5,7)])
@pytest.mark.parametrize('kind',['dense','sparse','even','one','two','false'])
@pytest.mark.parametrize('mode',MODES)
def test_footprints(dtype,shape,kind,mode):
    size=2 if kind=='even' else 3
    footprint=np.ones((size,)*len(shape),dtype=bool)
    if kind=='sparse':footprint.flat[1::2]=False
    elif kind in ('one','two','false'):
        footprint[:]=False
        if kind!='false':footprint.flat[0]=True
        if kind=='two':footprint.flat[-1]=True
    compare(sample(dtype,shape),footprint,mode=mode)

@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('mode',['nearest','constant'])
@pytest.mark.parametrize('value',[-300.,300.,1e12+1,np.inf,np.nan])
def test_constant_storage_casts(dtype,mode,value):
    compare(sample(dtype,(3,5)),mode=mode,cval=value)

@pytest.mark.parametrize('dtype',[np.int64,np.uint64,np.longlong,np.ulonglong])
@pytest.mark.parametrize('shape',[(33,),(7,9)])
@pytest.mark.parametrize('mode',['nearest','constant','reflect'])
def test_full_width_integer_values(dtype,shape,mode):
    limits=np.iinfo(dtype)
    values=np.array([limits.min,limits.min+1,0,1,limits.max-1,limits.max],dtype=dtype)
    compare(np.resize(values,shape),mode=mode)

@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('layout',['F','reverse','strided','endian','unaligned','metadata','readonly','broadcast','zero-strides'])
@pytest.mark.parametrize('mode',MODES)
def test_storage(dtype,layout,mode):
    a=sample(dtype,(9,13))
    if layout=='F':a=np.asfortranarray(a)
    elif layout=='reverse':a=a[::-1,::-1]
    elif layout=='strided':a=a[:,::2]
    elif layout=='endian':a=a.astype(a.dtype.newbyteorder('>'))
    elif layout=='unaligned':a=np.ndarray(a.shape,dtype=a.dtype,buffer=bytearray(a.nbytes+1),offset=1);a[:]=sample(dtype,a.shape)
    elif layout=='metadata':a=a.view(np.dtype(a.dtype,metadata={'unit':'raw'}))
    elif layout=='readonly':a.flags.writeable=False
    elif layout=='broadcast':a=np.broadcast_to(a[:1],a.shape)
    else:a=np.broadcast_to(np.array(2,dtype=dtype),a.shape)
    compare(a,mode=mode,cval=5)

@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.uint8])
@pytest.mark.parametrize('shape',[(0,),(0,3),(2,0),(1,1),(1,2),(2,1)])
@pytest.mark.parametrize('mode',MODES)
def test_empty_and_singletons(dtype,shape,mode):
    compare(sample(dtype,shape),mode=mode)

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('value',[np.nan,np.inf,-np.inf])
def test_nd_nonfinite_values(dtype,mode,value):
    a=sample(dtype,(9,13));a[2,4]=value;a[7,9]=value
    compare(a,mode=mode)

@pytest.mark.parametrize('dtype',[np.uint8,np.int16,np.float32,np.float64])
@pytest.mark.parametrize('layout',['C','strided','alias'])
def test_output_identity_and_aliasing(dtype,layout):
    image=sample(dtype,(9,13));a=image.copy();b=image.copy()
    if layout=='alias':out_a,out_b=a,b
    elif layout=='strided':out_a=np.empty((9,26),dtype=dtype)[:,::2];out_b=np.empty((9,26),dtype=dtype)[:,::2]
    else:out_a=np.empty_like(a);out_b=np.empty_like(b)
    expected=reference(a,out=out_a);actual=xx.filters.median(b,out=out_b)
    assert expected is out_a and actual is out_b
    np.testing.assert_array_equal(out_b,out_a)

@pytest.mark.parametrize('shape',[(513,515),(65,65,65)])
def test_large_worker_boundaries(shape):
    compare(sample(np.float32,shape))

def test_signature():
    assert inspect.signature(xx.filters.median)==inspect.signature(reference)

def test_rank_behavior_explicit_fallback():
    a=sample(np.uint8,(9,13));footprint=np.ones((3,3),dtype=bool)
    with pytest.raises(NotImplementedError,match='outside native coverage'):xx.filters.median(a,footprint,behavior='rank')
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning):actual=xx.filters.median(a,footprint,behavior='rank')
    np.testing.assert_array_equal(actual,reference(a,footprint,behavior='rank'))

@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('target',REAL)
@pytest.mark.parametrize('kind',['dense','one','two'])
def test_nd_output_dtypes(dtype,target,kind):
    footprint=np.ones((3,3),dtype=bool)
    if kind!='dense':
        footprint[:]=False;footprint[0,0]=True
        if kind=='two':footprint[-1,-1]=True
    compare(sample(dtype,(7,9)),footprint,out=target,mode='constant',cval=300)

@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('mode',['nearest','constant'])
@pytest.mark.parametrize('value',[-300.,300.,1e12+1,np.inf])
def test_one_dimensional_constant_casts(dtype,mode,value):
    compare(sample(dtype,(33,)),mode=mode,cval=value)

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('kind',['sparse','one','two'])
def test_nonfinite_sparse_footprints(dtype,mode,kind):
    a=sample(dtype,(9,13));a[0,0]=np.nan;a[2,4]=-np.inf;a[7,9]=np.inf
    footprint=np.zeros((3,3),dtype=bool);footprint[0,0]=True
    if kind!='one':footprint[-1,-1]=True
    if kind=='sparse':footprint[1,0]=True;footprint[0,1]=True
    compare(a,footprint,mode=mode)

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('mode',MODES)
def test_one_dimensional_infinities(dtype,mode):
    a=sample(dtype,(33,));a[0]=np.inf;a[9]=-np.inf;a[-1]=np.inf
    compare(a,mode=mode)

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('mode',['nearest','reflect','mirror','wrap','grid-mirror','grid-wrap'])
def test_unused_nan_constant_is_supported(dtype,mode):
    compare(sample(dtype,(33,)),mode=mode,cval=np.nan)

@pytest.mark.parametrize('footprint',[np.array([[1,0,0],[0,0,0],[0,0,0]],dtype=bool),np.ones((1,2),dtype=bool)])
def test_fractional_constant_and_different_output(footprint):
    compare(np.ones((7,9),dtype=np.float32),footprint,out=np.float64,mode='constant',cval=300.3)

@pytest.mark.parametrize('dtype',[np.uint8,np.int64,np.uint64,np.float32,np.float64])
@pytest.mark.parametrize('window',[11,31,63])
@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('reverse',[False,True])
def test_sliding_rank_windows(dtype,window,mode,reverse):
    a=sample(dtype,(257,))
    if reverse:a=a[::-1]
    compare(a,np.ones(window,dtype=bool),mode=mode,cval=3.25)

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('window',[11,31,63])
@pytest.mark.parametrize('mode',MODES)
def test_sliding_rank_infinities(dtype,window,mode):
    a=sample(dtype,(257,));a[0]=np.inf;a[129]=-np.inf;a[-1]=np.inf
    compare(a,np.ones(window,dtype=bool),mode=mode)

@pytest.mark.parametrize('dtype',[np.int64,np.uint64])
@pytest.mark.parametrize('window',[11,31,63])
def test_sliding_rank_full_width(dtype,window):
    limits=np.iinfo(dtype)
    a=np.resize(np.array([limits.min,limits.min+1,0,1,2**53+1,limits.max-1,limits.max],dtype=dtype),257)
    compare(a,np.ones(window,dtype=bool))

@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('taps',[10,12,16,17,25,27,31,32])
def test_medium_simd_order_statistics(dtype,taps):
    mask=np.zeros((6,6),dtype=bool);mask.flat[:taps]=True
    compare(sample(dtype,(17,21)),mask)

@pytest.mark.parametrize('shape',[(3,3),(5,5)])
def test_mutated_footprint_does_not_use_stale_plan(shape):
    a=sample(np.float32,(17,21));mask=np.ones(shape,dtype=bool)
    compare(a,mask);mask.flat[::2]=False;compare(a,mask)
