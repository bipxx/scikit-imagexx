"""Exact footprint utility values, views, records and sequence contracts."""
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import warnings
import numpy as np
import pytest
from skimage import morphology as reference
import skimagexx as xx

_path=Path(__file__).with_name('test_footprint_contracts.py')
_spec=importlib.util.spec_from_file_location('footprint_contract_provider',_path)
contracts=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(contracts)
if os.environ.get('SKIMAGEXX_FOOTPRINT_UTIL_CANDIDATE'):
    _spec=importlib.util.spec_from_file_location('skimagexx._footprint_util_candidate',os.environ['SKIMAGEXX_FOOTPRINT_UTIL_CANDIDATE'])
    native=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(native)
    native._core=importlib.import_module(os.environ['SKIMAGEXX_FOOTPRINT_UTIL_BACKEND'])
else:native=importlib.import_module('skimagexx._footprint_util')

CASES=[case for case in contracts.CASES if case[1] in ('mirror_footprint','pad_footprint')]


@pytest.fixture(autouse=True,params=[1,4])
def workers(request):
    previous=xx.set_fallback_enabled(False)
    threads=xx.get_num_threads();other=native._core.get_num_threads()
    xx.set_num_threads(request.param);native._core.set_num_threads(request.param)
    yield
    native._core.set_num_threads(other);xx.set_num_threads(threads);xx.set_fallback_enabled(previous)


def check(name,*args,**kwargs):
    expected=contracts.observe(getattr(reference,name),args,kwargs)
    actual=contracts.observe(getattr(native,name),args,kwargs)
    assert actual==expected


@pytest.mark.parametrize('case',CASES,ids=[case[0] for case in CASES])
def test_frozen_utility_contract(case):
    _,name,args,kwargs=case
    check(name,*args,**kwargs)


DTYPES=['bool','u1','i1','u2','i2','u4','i4','u8','i8','f2','f4','f8',
        'longdouble','c8','c16','clongdouble','S0','S5','U0','U5','V0','V7',
        'datetime64[ns]','timedelta64[D]','>i2','>f8',
        np.dtype([('a','i2'),('b','S3'),('c',('f4',(2,)))]),
        np.dtype('u1',metadata={'test':'footprint'}),
        np.dtype([('a','i2'),('b','O')]),'object']


@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('layout',['C','F','reverse','step','broadcast','readonly','unaligned'])
@pytest.mark.parametrize('name',['mirror_footprint','pad_footprint'])
def test_arbitrary_records(dtype,layout,name):
    a=np.empty((4,6,8),dtype)
    a[...] = b"abcdefg" if a.dtype.kind=="V" and a.dtype.fields is None else 3
    if layout=='F':a=np.asfortranarray(a)
    elif layout=='reverse':a=a[::-1,::-1,::-1]
    elif layout=='step':a=a[::2,::2,::2]
    elif layout=='broadcast':a=np.broadcast_to(a[:1],a.shape)
    elif layout=='readonly':a.flags.writeable=False
    elif layout=='unaligned':
        # Reference-bearing dtypes must be initialized by NumPy, not exposed
        # as arbitrary pointers in a bytearray-backed fake object array.
        if not a.dtype.hasobject:
            raw=np.ndarray(a.shape,a.dtype,buffer=bytearray(a.nbytes+1),offset=1)
            raw[...] = a;a=raw
    check(name,a)


@pytest.mark.parametrize('shape',[(0,2),(1,1),(2,3),(3,5),(2,4,6),(1,2,3,4),(513,1024)])
@pytest.mark.parametrize('front',[False,True])
def test_raw_bytes_and_ownership(shape,front):
    rng=np.random.default_rng(445)
    a=rng.integers(0,256,size=int(np.prod(shape))*8,dtype=np.uint8).view(np.float64).reshape(shape)
    with np.errstate(all='ignore'):
        expected=reference.pad_footprint(a,pad_end=not front)
        actual=native.pad_footprint(a,pad_end=not front)
    assert actual.dtype==expected.dtype and actual.strides==expected.strides
    assert actual.tobytes()==expected.tobytes()
    assert actual.flags.owndata and actual.flags.writeable
    assert not np.shares_memory(a,actual)


@pytest.mark.parametrize('name',['mirror_footprint','pad_footprint'])
def test_object_identity_and_lifetime(name):
    first,second=object(),object()
    a=np.asarray([[first,second],[second,first]],dtype=object)
    before=(sys.getrefcount(first),sys.getrefcount(second))
    out=getattr(native,name)(a)
    expected=getattr(reference,name)(a)
    assert out.tolist()==expected.tolist()
    assert all(value is first or value is second or value==0 for value in out.flat)
    if name=='mirror_footprint':assert np.shares_memory(a,out)
    else:assert not np.shares_memory(a,out)
    del out,expected
    assert (sys.getrefcount(first),sys.getrefcount(second))==before


@pytest.mark.parametrize('shape',[(2,4,6),(3,5),(0,2),(1,2,3,4)])
@pytest.mark.parametrize('fail_after',[None,1,2])
def test_stateful_padding_truthiness(shape,fail_after):
    def exercise(module):
        a=np.ones(shape,dtype='i2');calls=[]
        class Direction:
            def __bool__(self):
                calls.append(len(calls))
                if fail_after is not None and len(calls)>fail_after:raise RuntimeError('direction exhausted')
                a.fill(len(calls))
                return len(calls)%2==0
        with warnings.catch_warnings(record=True) as notices:
            try:result,error=module.pad_footprint(a,pad_end=Direction()),None
            except Exception as caught:result,error=None,(type(caught),str(caught))
        return contracts.describe(result,[a]) if result is not None else None,error,calls,a.tobytes(),notices
    assert exercise(native)==exercise(reference)


@pytest.mark.parametrize('name',['mirror_footprint','pad_footprint'])
def test_array_interface_protocol(name):
    def exercise(module):
        a=np.arange(24).reshape(4,6);calls=[]
        class Array:
            @property
            def __array_interface__(self):calls.append('interface');return a.__array_interface__
        out=getattr(module,name)(Array())
        return contracts.describe(out,[a]),calls
    assert exercise(native)==exercise(reference)


@pytest.mark.parametrize('name',['mirror_footprint','pad_footprint'])
def test_no_upstream_forwarding(name,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('upstream forwarding')
    monkeypatch.setattr(reference,name,forbidden)
    actual=getattr(native,name)(np.ones((2,4),dtype='u1'))
    assert isinstance(actual,np.ndarray)


@pytest.mark.parametrize('name',['mirror_footprint','pad_footprint'])
def test_variable_width_string_provider(name):
    if not hasattr(np,'dtypes') or not hasattr(np.dtypes,'StringDType'):
        pytest.skip('variable-width StringDType requires NumPy2')
    a=np.array([['alpha','beta'],['gamma','delta']],dtype=np.dtypes.StringDType())
    check(name,a)


@pytest.mark.parametrize('name',['mirror_footprint','pad_footprint'])
@pytest.mark.parametrize('shape',[(2,),(3,)])
@pytest.mark.parametrize('dtype',['datetime64','timedelta64','V20'])
def test_unspecified_time_unit_and_void(name,shape,dtype):
    a=np.zeros(shape,dtype)
    def observe(module):
        try:
            out=getattr(module,name)(a)
            return out.dtype,out.shape,out.strides,out.tobytes(),None
        except Exception as error:return None,None,None,None,(type(error),str(error))
    assert observe(native)==observe(reference)
