"""Exact corner order, raw scalar conversion, coordinate semantics and storage."""
import inspect
import re
import warnings
import numpy as np
import pytest
from skimage import transform as original
import skimagexx as xx
from test_integral import REAL, sample


@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    previous=xx.set_fallback_enabled(False);xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous);xx.set_num_threads(1)


def check(table,start,end):
    with warnings.catch_warnings(record=True) as expected_notices:
        warnings.simplefilter('always')
        try:expected=original.integrate(table,start,end)
        except Exception as error:
            with pytest.raises(type(error),match=re.escape(str(error))):xx.transform.integrate(table,start,end)
            return
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always');actual=xx.transform.integrate(table,start,end)
    assert [(type(w.message),str(w.message)) for w in notices]==[(type(w.message),str(w.message)) for w in expected_notices]
    assert type(actual) is type(expected)
    assert actual.dtype==expected.dtype and actual.dtype.metadata==expected.dtype.metadata
    assert actual.shape==expected.shape and actual.strides==expected.strides
    assert not np.shares_memory(actual,table)
    np.testing.assert_array_equal(actual,expected,strict=True)
    np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))


def windows(shape,count,seed=881):
    rng=np.random.default_rng(seed)
    a=np.column_stack([rng.integers(0,n,count) for n in shape])
    b=np.column_stack([rng.integers(0,n,count) for n in shape])
    return np.minimum(a,b),np.maximum(a,b)


def test_signature():
    assert inspect.signature(xx.transform.integrate)==inspect.signature(original.integrate)


@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('shape',[(19,),(11,13),(3,5,7),(2,3,2,5),(2,)*8,(2,)*9])
def test_raw_tables_and_dimensions(dtype,shape):
    table=sample(dtype,shape);first,last=windows(shape,17)
    check(table,first,last)
    check(table,tuple(0 for _ in shape),tuple(n-1 for n in shape))
    check(table,first-np.array(shape),last-np.array(shape))


@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('layout',['fortran','reversed','stepped','readonly','unaligned','endian','zero-stride','permuted','tagged'])
def test_storage(dtype,layout):
    table=sample(dtype,(5,7,11))
    if layout=='fortran':table=np.asfortranarray(table)
    elif layout=='reversed':table=table[::-1,::-1,::-1]
    elif layout=='stepped':table=np.tile(table,(2,2,2))[::2,::2,::2]
    elif layout=='readonly':table.flags.writeable=False
    elif layout=='endian':table=table.astype(table.dtype.newbyteorder('>'))
    elif layout=='zero-stride':table=np.broadcast_to(table[:1],table.shape)
    elif layout=='permuted':table=table.transpose(2,0,1)
    elif layout=='tagged':table=table.astype(np.dtype(dtype,metadata={'units':'raw'}))
    else:
        storage=np.empty(table.nbytes+1,dtype=np.uint8)
        copied=np.ndarray(table.shape,dtype=table.dtype,buffer=storage,offset=1);copied[...]=table;table=copied
    first,last=windows(table.shape,33);check(table,first,last)


@pytest.mark.parametrize('dtype',REAL)
def test_scalar_and_empty_windows(dtype):
    check(np.array(1,dtype=dtype),(),())
    check(dtype(1),(),())
    check(sample(dtype,(7,11)),np.empty((0,2),dtype=int),np.empty((0,2),dtype=int))


@pytest.mark.parametrize('length',[1,2,3,4,7,8,15,16,17,31,32,33,255,256,257,511,512,513,1025])
@pytest.mark.parametrize('dtype',[np.uint64,np.float32,np.float64])
def test_simd_and_block_tails(dtype,length):
    first,last=windows((17,19),length);check(sample(dtype,(17,19)),first,last)


@pytest.mark.parametrize('dtype',[np.uint64,np.float32,np.float64])
def test_parallel_window_partitions(dtype):
    first,last=windows((17,19),65539);check(sample(dtype,(17,19)),first,last)


@pytest.mark.parametrize('start,end',[
    ((0,0),(4,6)),((1,2),(-1,-1)),((-5,-7),(-1,-1)),
    ((-6,-8),(-1,-1)),((-10,-14),(-6,-8)),
    (0,2),([1],[3]),([[1],[2]],[[3],[4]]),
    ([[0,1],[1,2]],(3,5)),((0,1),[[2,3],[3,5]]),
    ((0,0),(5,6)),((0,0),(4,7)),((3,4),(1,2)),
    ((-20,-20),(-20,-20)),
])
def test_coordinate_normalization_broadcasting_and_errors(start,end):
    check(np.arange(35,dtype=float).reshape(5,7),start,end)


@pytest.mark.parametrize('dtype',[np.int8,np.int16,np.int32,np.int64,np.uint8,np.uint16,np.uint32])
def test_integer_coordinate_widths(dtype):
    first,last=windows((7,11),17)
    check(np.arange(77).reshape(7,11),first.astype(dtype),last.astype(dtype))


@pytest.mark.parametrize('layout',['fortran','reversed','stepped','readonly','unaligned','endian','zero-stride','tagged'])
def test_integer_coordinate_storage(layout):
    first,last=windows((7,11),33)
    def storage(array):
        if layout=='fortran':return np.asfortranarray(array)
        if layout=='reversed':return array[::-1]
        if layout=='stepped':return np.repeat(array,2,axis=0)[::2]
        if layout=='readonly':array.flags.writeable=False;return array
        if layout=='endian':return array.astype('>i8')
        if layout=='zero-stride':return np.broadcast_to(array[:1],array.shape)
        if layout=='tagged':return array.view(np.dtype(array.dtype,metadata={'coordinates':'pixels'}))
        raw=np.empty(array.nbytes+1,dtype=np.uint8)
        result=np.ndarray(array.shape,dtype=array.dtype,buffer=raw,offset=1);result[:]=array;return result
    check(sample(np.float64,(7,11)),storage(first),storage(last))


def test_integer_coordinate_broadcasting_ignores_unused_end_rows():
    table=sample(np.float64,(7,11))
    check(table,np.array([[1,2]]),np.array([[3,5],[4,6]]))
    check(table,np.array([[1,2],[2,3]]),np.array([[4,6]]))


def test_raw_integer_conversion_boundaries():
    unsigned=np.array([0,2**53-1,2**53,2**53+1,2**53+2,2**63-1025,2**63-1,2**63,2**63+1,2**63+1025,2**64-2049,2**64-1025,2**64-2,2**64-1],dtype=np.uint64)
    signed=np.array([-2**63,-2**63+1,-2**53-1,-1,0,1,2**53-1,2**53+1,2**63-1],dtype=np.int64)
    for table in (unsigned,signed):
        check(table,np.zeros((table.size,1),dtype=int),np.arange(table.size)[:,None])
        first,last=windows(table.shape,257);check(table,first,last)


def test_exhaustive_finite_binary16_and_noncanonical_bool():
    bits=np.arange(65536,dtype=np.uint16);bits=bits[(bits&0x7c00)!=0x7c00]
    table=bits.view(np.float16)
    check(table,np.zeros((len(table),1),dtype=int),np.arange(len(table))[:,None])
    table=np.arange(256,dtype=np.uint8).view(np.bool_)
    check(table,np.zeros((256,1),dtype=int),np.arange(256)[:,None])


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
def test_ordered_cancellation_and_signed_zero(dtype):
    table=np.resize(np.array([-0.,0.,1000.,1.,-1000.,-0.],dtype=dtype),35).reshape(5,7)
    first,last=windows(table.shape,257);check(table,first,last)
    with np.errstate(all='raise'):check(table,first,last)


def test_only_sampled_pixels_affect_fallback():
    table=np.arange(35,dtype=float).reshape(5,7);table[2,3]=np.nan
    check(table,(0,0),(4,6))
    with pytest.raises(NotImplementedError):xx.transform.integrate(table,(0,0),(2,3))


@pytest.mark.parametrize('table',[
    np.array([np.nan]),np.array([np.inf]),np.array([-np.inf]),
    np.array([[np.finfo(float).max,-np.finfo(float).max],[-np.finfo(float).max,np.finfo(float).max]]),
])
def test_exceptional_arithmetic_explicit_fallback(table):
    first=tuple(1 if n>1 else 0 for n in table.shape);last=tuple(n-1 for n in table.shape)
    with pytest.raises(NotImplementedError):xx.transform.integrate(table,first,last)
    xx.set_fallback_enabled(True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');expected=original.integrate(table,first,last)
    with pytest.warns(xx.FallbackWarning):actual=xx.transform.integrate(table,first,last)
    np.testing.assert_array_equal(actual,expected)


def test_memory_mapped_table(tmp_path):
    table=np.memmap(tmp_path/'table.dat',dtype=np.float32,mode='w+',shape=(7,11))
    table[:]=np.arange(77).reshape(7,11)
    check(table,(1,2),(5,8));check(table[::-1,::-1],(1,2),(5,8))


@pytest.mark.parametrize('shape',[(0,),(0,3),(3,0)])
def test_empty_table_bounds(shape):
    check(np.empty(shape),tuple(0 for _ in shape),tuple(0 for _ in shape))


@pytest.mark.parametrize('kind',['complex','object','subclass','float-coordinates'])
def test_unsupported_calls_use_explicit_fallback(kind):
    table=np.arange(35,dtype=float).reshape(5,7);first=(1,2);last=(3,5)
    if kind=='complex':table=table.astype(np.complex128)
    elif kind=='object':table=table.astype(object)
    elif kind=='subclass':
        class Table(np.ndarray):pass
        table=table.view(Table)
    else:first=(1.,2.)
    with pytest.raises(NotImplementedError):xx.transform.integrate(table,first,last)
    xx.set_fallback_enabled(True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:expected=original.integrate(table,first,last)
        except Exception as error:
            with pytest.warns(xx.FallbackWarning),pytest.raises(type(error),match=re.escape(str(error))):xx.transform.integrate(table,first,last)
            return
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.warns(xx.FallbackWarning):actual=xx.transform.integrate(table,first,last)
    np.testing.assert_array_equal(actual,expected)


def test_pipeline_can_swap_backend():
    image=np.random.default_rng(4).integers(0,256,(17,19),dtype=np.uint8)
    first,last=windows(image.shape,257)
    expected=original.integrate(original.integral_image(image),first,last)
    actual=xx.transform.integrate(xx.transform.integral_image(image),first,last)
    np.testing.assert_array_equal(actual,expected)
