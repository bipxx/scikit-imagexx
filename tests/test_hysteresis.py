"""Hysteresis thresholds, promotion, connectivity and label-zero behavior."""
import inspect,re,warnings
import numpy as np
import pytest
from skimage.filters import apply_hysteresis_threshold as reference
import skimagexx as xx

REAL=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64)

@pytest.fixture(autouse=True,params=[1,4])
def settings(request):
    previous=xx.set_fallback_enabled(False);workers=xx.get_num_threads();xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(workers);xx.set_fallback_enabled(previous)

def compare(image,low,high):
    before=np.array(image,copy=True);bounds=[np.array(v,copy=True) for v in (low,high)]
    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter('always')
        try:expected=reference(image,low,high)
        except Exception as error:
            with warnings.catch_warnings(record=True) as actual_warnings:
                warnings.simplefilter('always')
                with pytest.raises(type(error),match=re.escape(str(error))):xx.filters.apply_hysteresis_threshold(image,low,high)
            assert [(type(w.message),str(w.message)) for w in actual_warnings]==[(type(w.message),str(w.message)) for w in expected_warnings]
            return
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter('always');actual=xx.filters.apply_hysteresis_threshold(image,low,high)
    assert [(type(w.message),str(w.message)) for w in actual_warnings]==[(type(w.message),str(w.message)) for w in expected_warnings]
    assert type(actual) is type(expected) and actual.shape==expected.shape and actual.dtype==expected.dtype==np.dtype(bool)
    assert actual.dtype.metadata==expected.dtype.metadata
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(image,before)
    for value,copy in zip((low,high),bounds):np.testing.assert_array_equal(value,copy)

def sample(dtype,shape):
    a=np.random.default_rng(982).integers(0,127,shape)
    if dtype==np.bool_:return a>63
    if np.dtype(dtype).kind=='f':return (a/32-1).astype(dtype)
    return a.astype(dtype)

def test_signature():assert inspect.signature(xx.filters.apply_hysteresis_threshold)==inspect.signature(reference)

@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('shape',[(),(0,),(0,7),(1,),(7,),(8,),(9,),(17,19),(3,5,7),(1,3,1,5)])
@pytest.mark.parametrize('bounds',[(0,1),(1,0),(.25,.75),(-1,2)])
def test_shapes_types(dtype,shape,bounds):compare(sample(dtype,shape),*bounds)

@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('layout',['f','reverse','strided','broadcast','readonly','endian','unaligned','metadata','scalar'])
def test_storage(dtype,layout):
    a=sample(dtype,(17,19))
    if layout=='f':a=np.asfortranarray(a)
    if layout=='reverse':a=a[::-1,::-1]
    if layout=='strided':a=a[:,::2]
    if layout=='broadcast':a=np.broadcast_to(a[:1,:1],a.shape)
    if layout=='readonly':a.flags.writeable=False
    if layout=='endian':a=a.astype(a.dtype.newbyteorder('S'))
    if layout=='unaligned':
        buffer=np.empty(a.nbytes+1,np.uint8);b=np.ndarray(a.shape,dtype=dtype,buffer=buffer,offset=1);b[:]=a;a=b
    if layout=='metadata':a=a.view(np.dtype(dtype,metadata={'hysteresis':'input'}))
    if layout=='scalar':a=a[0,0]
    compare(a,np.float32(.25),np.float64(.75))

@pytest.mark.parametrize('dtype',[np.uint8,np.int64,np.uint64,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('bound_type',[int,float,np.int64,np.uint64,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('kind',['scalars','zero-d','arrays','rows','expand'])
def test_threshold_promotion_and_broadcast(dtype,bound_type,kind):
    a=sample(dtype,(7,9));low=bound_type(0);high=bound_type(1)
    if kind=='zero-d':low=np.array(low);high=np.array(high)
    if kind=='arrays':low=np.full(a.shape,low);high=np.full(a.shape,high)
    if kind=='rows':low=np.full((7,1),low);high=np.full((1,9),high)
    if kind=='expand':a=a[:1];low=np.full((7,1),low);high=np.full((1,9),high)
    compare(a,low,high)

@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('low',[np.nan,np.inf,-np.inf,0.])
@pytest.mark.parametrize('high',[np.nan,np.inf,-np.inf,1.])
def test_nonfinite_and_background(dtype,low,high):
    a=np.array([[0,1,2,np.nan],[np.inf,-np.inf,-0.,3]],dtype=dtype)
    compare(a,low,high)

@pytest.mark.parametrize('dtype',[np.int64,np.uint64,np.float32,np.float64])
def test_array_nan_background_and_threshold_layout(dtype):
    a=sample(dtype,(17,19));low=np.full(a.shape,.25);high=np.full(a.shape,.75)
    low[::3,::2]=np.nan;high[1::3,1::2]=np.nan
    compare(a[::-1],low[::-1],np.asfortranarray(high))

@pytest.mark.parametrize('dtype',[np.int8,np.uint8,np.int16,np.uint16,np.int32,np.uint32,np.int64,np.uint64])
@pytest.mark.parametrize('kind',['native','mixed-signed','float'])
def test_full_integer_values(dtype,kind):
    info=np.iinfo(dtype);a=np.resize(np.array([info.min,info.max,0,1,info.max-1],dtype=dtype),(17,19))
    low=dtype(0);high=dtype(info.max-1)
    if kind=='mixed-signed':low=np.int64(-1);high=np.uint64(2**63-2)
    if kind=='float':low=np.float32(.5);high=np.float64(2**63)
    compare(a,low,high)

@pytest.mark.parametrize('dtype',[np.int8,np.uint8,np.int64,np.uint64,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('high',[-129,-1,128,256,2**63,2**64-1])
def test_python_integer_bounds(dtype,high):compare(sample(dtype,(7,9)),0,high)

@pytest.mark.parametrize('pattern',['random','diagonal','snake','no-seeds','all-weak','equal','background'])
def test_parallel_components_and_boundaries(pattern):
    a=np.random.default_rng(193).integers(0,3,(513,515),dtype=np.uint8)
    low=0;high=1
    if pattern=='diagonal':a[:]=0;np.fill_diagonal(a,1);a[0,0]=2
    if pattern=='snake':a[:]=0;a[::2]=1;a[1::4,-1]=1;a[3::4,0]=1;a[0,0]=2
    if pattern=='no-seeds':high=3
    if pattern=='all-weak':low=-1
    if pattern=='equal':low=high
    if pattern=='background':low=np.full(a.shape,0.);low[0,0]=np.nan;a[0,0]=2
    compare(a,low,high)

def test_connectivity_is_face_only():
    a=np.array([[2,0,0],[0,1,0],[0,0,1]],np.uint8)
    actual=xx.filters.apply_hysteresis_threshold(a,0,1)
    np.testing.assert_array_equal(actual,[[True,False,False],[False,False,False],[False,False,False]])

def test_memmap(tmp_path):
    a=np.memmap(tmp_path/'input.bin',dtype=np.float32,mode='w+',shape=(17,19));a[:]=sample(np.float32,a.shape)
    compare(a,.25,.75)

@pytest.mark.parametrize('scalar',[False,True])
@pytest.mark.parametrize('array_bounds',[False,True])
def test_exact_mixed_integer_background(scalar,array_bounds):
    a=np.array([2**63-1,0,-1],np.int64);low=np.int64(2**63-1);high=np.uint64(2**63-2)
    if scalar:a=a[0]
    if array_bounds:low=np.full(np.shape(a),low);high=np.full(np.shape(a),high)
    compare(a,low,high)

@pytest.mark.parametrize('dtype,integer,bits',[(np.float16,np.uint16,0x7c01),(np.float32,np.uint32,0x7f800001),(np.float64,np.uint64,0x7ff0000000000001)])
@pytest.mark.parametrize('role',[0,1,2])
@pytest.mark.parametrize('policy',['warn','raise','ignore'])
def test_signaling_nan_cast_policies(dtype,integer,bits,role,policy):
    nan=np.array([bits],integer).view(dtype)[0]
    args=[np.array([0,1,2],dtype),0,1]
    if role==0:args[0][1]=nan
    else:
        args[role]=np.array([0,0,0] if role==1 else [1,1,1],dtype);args[role][1]=nan
    with np.errstate(invalid=policy):compare(*args)

@pytest.mark.parametrize('shape',[(65,65,65),(3,45,45,45)])
def test_parallel_higher_dimensions(shape):
    a=np.random.default_rng(196).integers(0,3,shape,dtype=np.uint8)
    compare(a,0,1)

@pytest.mark.parametrize('bounds',[(False,True),(0,1),(.25,.75)])
def test_noncanonical_boolean_storage(bounds):
    a=np.resize(np.array([0,1,2,128,255],np.uint8).view(bool),(17,19))
    compare(a,*bounds)

def test_explicit_fallback_for_complex():
    a=np.array([1+2j,0,3+1j])
    with pytest.raises(NotImplementedError,match='outside native coverage'):xx.filters.apply_hysteresis_threshold(a,0,1)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning):actual=xx.filters.apply_hysteresis_threshold(a,0,1)
    np.testing.assert_array_equal(actual,reference(a,0,1))

@pytest.mark.parametrize('dtype,integer,bits',[(np.float16,np.uint16,0x7c01),(np.float32,np.uint32,0x7f800001),(np.float64,np.uint64,0x7ff0000000000001)])
@pytest.mark.parametrize('image_dtype',[np.uint64,np.float32,np.float64])
@pytest.mark.parametrize('role',[1,2])
@pytest.mark.parametrize('policy',['warn','raise','ignore'])
@pytest.mark.parametrize('size',[3,33])
def test_signaling_threshold_fields_with_mixed_images(dtype,integer,bits,image_dtype,role,policy,size):
    field=np.zeros(size,dtype=dtype) if role==1 else np.ones(size,dtype=dtype)
    field.view(integer)[1]=bits
    args=[np.resize(np.array([0,1,2],dtype=image_dtype),size),0,1];args[role]=field
    with np.errstate(invalid=policy):compare(*args)

@pytest.mark.parametrize('dtype',[np.longlong,np.ulonglong])
@pytest.mark.parametrize('shape',[(),(17,19)])
def test_platform_integer_aliases(dtype,shape):
    compare(sample(dtype,shape),0,2**63)

def test_python_scalar_image_explicit_fallback():
    a=2**63+1
    with pytest.raises(NotImplementedError,match='outside native coverage'):xx.filters.apply_hysteresis_threshold(a,0,float(2**63))
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning):actual=xx.filters.apply_hysteresis_threshold(a,0,float(2**63))
    np.testing.assert_array_equal(actual,reference(a,0,float(2**63)))

@pytest.mark.parametrize('shape',[(513,515),(65,65,65)])
def test_large_broadcast_fields_across_worker_boundaries(shape):
    a=np.random.default_rng(211).random(shape,dtype=np.float32)
    low=np.linspace(.1,.4,shape[0]*shape[-1],dtype=np.float32).reshape((shape[0],)+(1,)*(len(shape)-2)+(shape[-1],)) if len(shape)>2 else np.linspace(.1,.4,shape[0],dtype=np.float32)[:,None]
    high=np.linspace(.6,.9,shape[1],dtype=np.float32).reshape((1,shape[1])+(1,)*(len(shape)-2))
    compare(a,low,high)

@pytest.mark.parametrize('shape',[(2,3),(3,2),(1,6),(6,1)])
def test_exhaustive_connectivity_masks(shape):
    # Independent masks include strong pixels outside weak: upstream selects
    # the entire label-zero background when any such pixel is present.
    from scipy.ndimage import label
    positions=np.arange(6)
    for weak_bits in range(64):
        weak=((weak_bits>>positions)&1).astype(np.uint8).reshape(shape)
        labels,count=label(weak)
        for strong_bits in range(64):
            strong=((strong_bits>>positions)&1).astype(np.uint8).reshape(shape)
            selected=np.zeros(count+1,dtype=bool)
            selected[labels[strong!=0]]=True
            actual=xx._core.hysteresis_connect(weak,strong)
            np.testing.assert_array_equal(actual,selected[labels])
