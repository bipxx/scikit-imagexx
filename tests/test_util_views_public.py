"""All admitted observations, with process-local diagnostics kept out of parity."""
import copy
import ctypes
import inspect
import json
from pathlib import Path
import numpy as np
import pytest
import skimagexx._array_views as arrays
import skimagexx._slice_along_axes as slicing
from array_views_contract import cases as array_cases
from array_views_observer import observe as array_observe
from slice_along_axes_contract import cases as slice_cases
from slice_along_axes_observer import observe as slice_observe

DATA=json.loads((Path(__file__).parent/'fixtures/util_views_originals_v1.json').read_bytes())
PROFILE='minimum' if np.__version__.startswith('1.24.') else 'modern'
GOLD={family:{r['id']:r for r in DATA['families'][family][PROFILE]['rows']} for family in ('slice','array_views')}


def comparable(row):
    row=copy.deepcopy(row)
    # Raw fields remain intact in the saved fixture. Instruction addresses and
    # nonsharing allocation distances are explicitly diagnostic in raw admission.
    for key in ('target_immediate_fenv','controller_after_observer_fenv','caller_restore_fenv'):
        row.pop(key,None)
    for alias in row.get('aliases',{}).values():
        if alias.get('pointer_delta_role')=='nonsharing-diagnostic-only':
            alias['pointer_delta']=None
    return row


def captured_FP_call(call, expected):
    # Identical reference_fenv_restore envelope, with explicit genuine/private binding.
    from util_views_fenv import captured_call
    return captured_call(call, expected)


@pytest.mark.parametrize('workers',(1,4))
@pytest.mark.parametrize('spec',slice_cases(),ids=lambda s:s['id'])
def test_slice_saved_public(spec,workers):
    previous=slicing._core.get_num_threads()
    slicing._core.set_num_threads(workers)
    try:
        expected=GOLD['slice'][spec['id']]
        actual=slice_observe(captured_FP_call(slicing.slice_along_axes,expected),spec,np)
        # Observe's outer pre/post flags are its inherited caller context. Exact
        # captured incoming/outgoing target flags were checked in the envelope.
        assert actual['fenv_before']==actual['fenv_after']
        actual=comparable(actual);expected=comparable(expected)
        for row in (actual,expected):
            row.pop('fenv_before');row.pop('fenv_after')
        assert actual==expected
        assert slicing._core.get_num_threads()==workers
    finally:slicing._core.set_num_threads(previous)


@pytest.mark.parametrize('workers',(1,4))
@pytest.mark.parametrize('spec',array_cases(),ids=lambda s:s['id'])
def test_array_saved_public(spec,workers):
    previous=arrays._core.get_num_threads();arrays._core.set_num_threads(workers)
    try:
        call=arrays.view_as_blocks if spec['api'].endswith('blocks') else arrays.view_as_windows
        actual=array_observe(call,spec,np)
        assert comparable(actual)==comparable(GOLD['array_views'][spec['id']])
        assert arrays._core.get_num_threads()==workers
    finally:arrays._core.set_num_threads(previous)


def test_single_component_native_entry_ownership():
    assert arrays._core is slicing._core
    for name in ('util_view_as_blocks','util_view_as_windows','util_slice_view','util_slice_copy_into'):
        assert inspect.isbuiltin(getattr(arrays._core,name))
    assert inspect.signature(slicing.slice_along_axes).__str__()=='(image, slices, axes=None, copy=False)'
    assert inspect.signature(arrays.view_as_blocks).__str__()=='(arr_in, block_shape)'
    assert inspect.signature(arrays.view_as_windows).__str__()=='(arr_in, window_shape, step=1)'


def test_native_entries_are_required(monkeypatch):
    def refused(*a,**k):raise RuntimeError('native entry refused')
    a=np.arange(24,dtype=np.int16).reshape(4,6)
    for name,call in (
        ('util_view_as_blocks',lambda:arrays.view_as_blocks(a,(2,2))),
        ('util_view_as_windows',lambda:arrays.view_as_windows(a,(2,2))),
        ('util_slice_view',lambda:slicing.slice_along_axes(a,[(1,3)])),
        ('util_slice_copy_into',lambda:slicing.slice_along_axes(a,[(1,3)],copy=True))):
        with monkeypatch.context() as patch:
            patch.setattr(arrays._core,name,refused)
            with pytest.raises(RuntimeError,match='native entry refused'):call()


def test_slice_copy_boundary_rejects_alias_object_readonly_nonowning():
    a=np.arange(6,dtype=np.int16)
    with pytest.raises(ValueError,match='overlaps'):slicing._core.util_slice_copy_into(a,a)
    b=np.empty_like(a);b.flags.writeable=False
    with pytest.raises(ValueError,match='owning writable C'):slicing._core.util_slice_copy_into(a,b)
    with pytest.raises(ValueError,match='owning writable C'):slicing._core.util_slice_copy_into(a,b.view())
    a=np.array([object()],dtype=object);b=np.empty_like(a)
    with pytest.raises(TypeError,match='POD'):slicing._core.util_slice_copy_into(a,b)


def test_supplementary_zero_item_and_native_shape_stride():
    for dtype in ('V0','S3','>i2'):
        a=np.empty((4,6),dtype=dtype)
        b=arrays.view_as_blocks(a,(2,2));w=arrays.view_as_windows(a,(2,3),step=2)
        assert b.shape==(2,3,2,2) and b.strides==(a.strides[0]*2,a.strides[1]*2,*a.strides)
        assert w.shape==(2,2,2,3) and w.strides==(a.strides[0]*2,a.strides[1]*2,*a.strides)
        result=slicing.slice_along_axes(a,[(0,2)],copy=True)
        assert result.shape==(2,6) and result.dtype==a.dtype and result.flags.owndata


def test_subclass_and_safe_overflow_metadata_remain_named_providers(monkeypatch):
    def refused(*a,**k):raise AssertionError('provider domain entered native metadata')
    monkeypatch.setattr(arrays._core,'util_view_as_blocks',refused)
    monkeypatch.setattr(arrays._core,'util_view_as_windows',refused)
    class Subclass(np.ndarray):pass
    a=np.arange(24,dtype=np.int16).reshape(4,6).view(Subclass)
    assert arrays.view_as_blocks(a,(2,2)).shape==(2,3,2,2)
    assert arrays.view_as_windows(a,(2,2)).shape==(3,5,2,2)
    # Only zero-length or singleton high-stride axes: every reachable logical
    # address is an allocated slot. No arbitrary unsafe as_strided input.
    maximum=np.iinfo(np.intp).max
    empty=np.ndarray((0,2),dtype='u1',buffer=bytearray(2),strides=(maximum,1))
    assert empty.strides==(maximum,1)
    block=arrays.view_as_blocks(empty,(2,1))
    assert block.shape==(0,2,2,1) and block.strides==(-2,1,maximum,1)
    singleton=np.ndarray((1,2),dtype='u1',buffer=bytearray(2),strides=(maximum,1));singleton[:]=[[5,6]]
    assert singleton.strides==(maximum,1)
    window=arrays.view_as_windows(singleton,(1,1),step=(2,1))
    assert window.shape==(1,2,1,1) and window.strides==(*singleton[::2,:].strides,*singleton.strides)
    assert window.ravel().tolist()==[5,6]
