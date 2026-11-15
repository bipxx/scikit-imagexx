"""Descriptor/one-fixture setup only: no ufunc application/error constructor."""
import builtins,json
import numpy as np
import pytest
import draw_multiblock_lbp_observer_v2 as observer
from mark_boundaries_exception_codec import encode_exception_args


@pytest.fixture(autouse=True)
def no_targets(monkeypatch):
    old=builtins.__import__
    def refuse(name,*args,**options):
        if name.split('.')[0]in('skimage','scipy','skimagexx'):raise AssertionError('ORIGINAL_OR_PROVIDER_REQUEST_FORBIDDEN')
        return old(name,*args,**options)
    monkeypatch.setattr(builtins,'__import__',refuse)


def test_casting_shape_descriptor_values_and_alias_tokens():
    src=np.dtype('float64');dst=np.dtype('int64')
    row=encode_exception_args((np.multiply,'same_kind',src,dst,0,[src,np.multiply]),np)
    values=row['items'];u=values[0]
    assert(u['name'],u['nin'],u['nout'],u['numpy_namespace_identity'])==('multiply',2,1,True)
    assert values[2]['str']==src.str and values[3]['str']==dst.str
    assert values[2]['type']['qualname']==type(src).__qualname__
    assert values[-1]['items'][0]['object_index']==values[2]['object_index']
    assert values[-1]['items'][1]['object_index']==u['object_index'];json.dumps(row,allow_nan=False)


def test_distinct_equal_dtype_and_nested_no_loop_shape():
    a=np.dtype('int64',copy=True);b=np.dtype('int64',copy=True);assert a==b and a is not b
    row=encode_exception_args((np.multiply,(a,b),[a,b],None),np)
    assert row['items'][1]['items'][0]['object_index']!=row['items'][1]['items'][1]['object_index']
    assert row['items'][2]['items'][0]['object_index']==row['items'][1]['items'][0]['object_index']
    json.dumps(row,allow_nan=False)


@pytest.mark.parametrize('kind',('structured','subarray','metadata'))
def test_unassessed_dtype_descriptors_refused(kind):
    dtype=np.dtype([('field','i4')])if kind=='structured'else np.dtype(('f8',(2,)))if kind=='subarray'else np.dtype('f8',metadata={'units':'m'})
    with pytest.raises(AssertionError,match='UNASSESSED_EXCEPTION_DTYPE'):encode_exception_args((np.multiply,dtype),np)


def test_arbitrary_descriptor_refused():
    with pytest.raises(TypeError,match='Unsupported'):encode_exception_args((object(),),np)


@pytest.mark.parametrize('form',('casting','no-loop'))
def test_actual_observer_inert_typed_error_branch_preserves_state(form):
    class InertDescriptorError(TypeError):pass
    # Mimic display-as-base name mutation without constructing a NumPy error.
    InertDescriptorError.__name__='TypeError'
    calls=[];policy=np.geterr(),np.geterrcall(),np.getbufsize();printing=np.get_printoptions()
    src=np.dtype('float64');dst=np.dtype('int64')
    arguments=(np.multiply,'same_kind',src,dst,0)if form=='casting'else(np.multiply,(src,dst))
    def inert(*args,**options):calls.append(True);raise InertDescriptorError(*arguments)
    spec=observer.cases()[-1];row=observer.observe(inert,spec,np)
    assert calls==[True]and row['error']['type']=='TypeError'
    assert row['error']['qualname']==InertDescriptorError.__qualname__
    assert row['error']['module']==InertDescriptorError.__module__
    assert row['error']['args']==encode_exception_args(arguments,np)
    assert row['before']==row['after']and all(row['checks'].values())
    assert row['supplied_call_attempts']==1 and row['result']is None
    assert (np.geterr(),np.geterrcall(),np.getbufsize())==policy and np.get_printoptions()==printing
    json.dumps(row,allow_nan=False)
