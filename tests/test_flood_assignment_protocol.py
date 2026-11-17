"""Prepared changed-binder regressions; NOT run by source-only preparation.

Requires an explicitly admitted changed private core or genuine installed core.
The inputs are bounded descriptors/words; no original algorithm is requested.
"""
import ctypes
import importlib
import pytest
from test_flood_fill_public import target, genuine_global_worker_and_caller


@pytest.fixture
def assign(target):
    np,module=target
    core=module._core if hasattr(module,'_core')else importlib.import_module('skimagexx._core')
    from util_views_fenv import FEnv,backend
    libc,helper=backend()
    def state():
        v=FEnv();assert libc.fegetenv(ctypes.byref(v))==0;return v
    def call(output,mask,values):
        outer=state();before=state()
        try:
            try:return core.flood_assign(output,mask,values)
            finally:
                after=state()
                assert (before.control,before.status,before.mxcsr)==(after.control,after.status,after.mxcsr)
        finally:
            assert helper.reference_fenv_restore(ctypes.byref(outer),32)==0
            assert bytes(state())==bytes(outer)
    return call


def guarded(np,dtype,layout,shape=(3,4)):
    physical=tuple(2*s for s in shape)if layout=='gapped'else shape
    count=1
    for n in physical:count*=n
    store=np.full(64+count*dtype.itemsize,0xa5,dtype=np.uint8)
    a=np.ndarray(physical,dtype=dtype,buffer=store,offset=32,order='F'if layout=='F'else'C')
    if layout=='reverse':a=a[tuple(slice(None,None,-1)for _ in shape)]
    elif layout=='gapped':a=a[tuple(slice(None,None,2)for _ in shape)]
    a[...] = np.arange(a.size).reshape(a.shape)%7
    return store,a


def clone_view(np,store,a):
    raw=store.copy()
    offset=a.__array_interface__['data'][0]-store.__array_interface__['data'][0]
    return raw,np.ndarray(a.shape,dtype=a.dtype,buffer=raw,offset=offset,strides=a.strides)


DTYPES=('bool','i1','u2','i8','f2','f4','f8','c16','>u4','metadata-f4')
LAYOUTS=('C','F','reverse','gapped')


@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('layout',LAYOUTS)
def test_assignment_guarded_POD_layout_and_C_selected_order(target,assign,dtype,layout):
    np,_=target
    dtype=np.dtype('f4',metadata={'units':'assignment'})if dtype=='metadata-f4'else np.dtype(dtype)
    store,a=guarded(np,dtype,layout);expected_store,expected=clone_view(np,store,a)
    pattern=(np.arange(12).reshape(3,4)%3)!=1
    mask=np.array(pattern,order='F')if layout=='C'else pattern[::-1,::-1]
    n=int(np.count_nonzero(mask));values=np.empty(n,dtype=a.dtype)
    values[...] = np.arange(n)%5+1
    before_mask=mask.tobytes();before_values=values.tobytes();metadata=a.dtype.metadata
    selected=0
    for index in np.ndindex(a.shape):
        if mask[index]:expected[index]=values[selected];selected+=int(bool(mask[index]))
    assert assign(a,mask,values)is None
    assert store.tobytes()==expected_store.tobytes()
    assert mask.tobytes()==before_mask and values.tobytes()==before_values
    assert a.dtype.metadata==metadata


@pytest.mark.parametrize('shape,selected',(((),True),((),False),((0,),False),((2,0,3),False),((1,4,1),True),((2,2,3),True)))
def test_assignment_scalar_empty_singleton_and_multiaxis(target,assign,shape,selected):
    np,_=target
    store,a=guarded(np,np.dtype('i4'),'C',shape);expected_store,expected=clone_view(np,store,a)
    mask=np.full(shape,selected,dtype=bool);values=np.full(a.size if selected else 0,23,dtype=a.dtype)
    if selected:expected[...]=23
    assert assign(a,mask,values)is None
    assert store.tobytes()==expected_store.tobytes()


def test_assignment_zero_output_and_mask_strides_keep_last_selected_value(target,assign):
    np,_=target
    raw=np.array([91,0xa5],dtype='u1');a=np.ndarray((2,3),dtype='u1',buffer=raw,strides=(0,0))
    flag=np.array([True],dtype=bool);mask=np.ndarray((2,3),dtype=bool,buffer=flag,strides=(0,0))
    values=np.array([1,2,3,4,5,6],dtype='u1')
    assert assign(a,mask,values)is None and raw.tobytes()==bytes([6,0xa5])


def test_assignment_live_mask_output_alias_not_snapshotted(target,assign):
    np,_=target
    raw=np.array([True,True,True,True],dtype=bool);mask=raw[::-1]
    values=np.array([False,True,False],dtype=bool)
    assert assign(raw,mask,values)is None
    assert raw.tobytes()==bytes([0,1,0,1])


def test_assignment_values_alias_partial_count_failure_keeps_read_order(target,assign):
    np,_=target
    # Both memcpy ranges are disjoint; no undefined overlapping-byte copy.
    a=np.array([1,2,3],dtype='i8');mask=np.array([False,True,True])
    with pytest.raises(ValueError,match='value count mismatch'):assign(a,mask,a)
    assert a.tolist()==[1,1,1]


@pytest.mark.parametrize('rhs_shape',((),(2,2)))
def test_assignment_C_contiguous_owned_RHS_uses_flat_order(target,assign,rhs_shape):
    np,_=target
    values=np.empty(rhs_shape,dtype='i4');values[...] = np.arange(values.size).reshape(rhs_shape)+19
    a=np.zeros(values.size,dtype='i4');mask=np.ones(a.size,dtype=bool)
    assert assign(a,mask,values)is None and a.tobytes()==values.tobytes()


@pytest.mark.parametrize('values_size,expected',((1,[11,2,3]),(4,[11,12,13])))
def test_assignment_count_errors_keep_exact_partial_writes(target,assign,values_size,expected):
    np,_=target
    a=np.array([1,2,3],dtype='i4');mask=np.ones(3,dtype=bool);values=np.arange(11,11+values_size,dtype=a.dtype)
    with pytest.raises(ValueError,match='value count mismatch'):assign(a,mask,values)
    assert a.tolist()==expected


@pytest.mark.parametrize('fault,message,kind',(
    ('readonly','read-only',ValueError),('shape','mask shape',ValueError),
    ('maskdtype','matching numeric dtype',TypeError),('rhsdtype','matching numeric dtype',TypeError),
    ('rhsview','descriptor mismatch',ValueError),('object','numeric POD dtype',TypeError)))
def test_assignment_validation_priority_no_write(target,assign,fault,message,kind):
    np,_=target
    a=np.array([1,2],dtype='i4');mask=np.array([True,True]);values=np.array([7,8],dtype='i4')
    if fault=='readonly':a.flags.writeable=False;mask=np.zeros((2,1),dtype=bool)
    elif fault=='shape':mask=np.zeros(3,dtype=bool)
    elif fault=='maskdtype':mask=np.ones(2,dtype='u1')
    elif fault=='rhsdtype':values=np.array([7,8],dtype='u4')
    elif fault=='rhsview':values=values[:]
    elif fault=='object':a=np.array([1,2],dtype=object);values=np.array([7,8],dtype=object)
    before=a.tobytes()
    with pytest.raises(kind,match=message):assign(a,mask,values)
    assert a.tobytes()==before
