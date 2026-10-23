"""Provider-stage parity. Private bootstrap/bridge is not whole-package proof.

The frozen598 observer and observations are reused without changing its hooks
or assertions. Only its factory is injected when checking native execution.
"""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import warnings

import numpy as np
import pytest
from skimage import segmentation as reference_namespace
from skimage.util._map_array import ArrayMap as ReferenceMap
import skimagexx as xx

ROOT=Path(__file__).parents[1]
if os.environ.get('SKIMAGEXX_COMPONENT_CONSUMERS_MODULE_DIR'):
    import sys
    sys.path.insert(0,os.environ['SKIMAGEXX_COMPONENT_CONSUMERS_MODULE_DIR'])
    core=importlib.import_module('_component_consumers_core')
else:core=xx._core

candidate=importlib.import_module('skimagexx._component_consumers')
assert candidate._core is core
spec=importlib.util.spec_from_file_location('arraymap_frozen_provider_observer',ROOT/'tests/test_arraymap_provider_contracts.py')
contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
matrix=json.loads(contract.MATRIX.read_text())


@pytest.fixture(autouse=True,params=[1,4])
def workers(request):
    before=core.get_num_threads();policy=xx.set_fallback_enabled(False)
    core.set_num_threads(request.param)
    yield
    core.set_num_threads(before);xx.set_fallback_enabled(policy)


@pytest.mark.parametrize('case',contract.CASES)
def test_all_frozen_provider_observations(case,monkeypatch):
    calls=[]
    def factory(labels):
        result=xx.segmentation.relabel_sequential(labels)
        assert type(result[1]) is candidate.ArrayMap
        calls.append(core.get_num_threads())
        return result
    # Explicit test-only factory injection; no upstream algorithm executes.
    monkeypatch.setattr(reference_namespace,'relabel_sequential',factory)
    actual=contract.probe('reference',case)['observation']
    assert calls==[core.get_num_threads()]
    assert actual==matrix['reference'][contract.profile()][case]


def test_private_or_installed_binding_identity_and_gil_assertions():
    assert core.consumer_dense_assignment_version==1
    assert core.component_consumers_interface_version==1
    assert core.consumer_dense_assign.__module__==core.__name__
    if os.environ.get('SKIMAGEXX_REQUIRE_GIL_ASSERTIONS')=='1':
        assert core._consumer_gil_assertions is True
    if os.environ.get('SKIMAGEXX_DISABLE_SIMD'):assert not core.has_avx2()


def output_observation(cls,kind,action):
    mapping=cls(np.array([0,2,4],np.uint16),np.array([2,3,4],np.int16))
    events=[];retained=[]
    original_zeros=np.zeros
    class Destination(np.ndarray):
        def __setitem__(self,key,value):
            events.append(dict(stage='subclass-setitem',same_keys=key is mapping.in_values,same_values=value is mapping.out_values))
            super().__setitem__(key,value)
    class Box:
        def __setitem__(self,key,value):
            events.append(dict(stage='object-setitem',same_keys=key is mapping.in_values,same_values=value is mapping.out_values,
                               keys=contract.array(np.asarray(key)),values=contract.array(np.asarray(value))))
            if kind=='object-error':
                warnings.warn('custom assignment warning',UserWarning)
                raise ValueError('custom assignment failed')
    if kind in ('alias-values','alias-keys'):
        mapping.in_values=np.array([1,2,0],np.uint16)
        mapping.out_values=np.array([10,20,30],np.uint16)
    if kind=='float-payload':
        mapping.out_values=np.array([0x7f800001,0x3f800000,0x80000000],np.uint32).view(np.float32)
    def allocator(length,*,dtype):
        events.append(dict(stage='allocate',length=contract.scalar(length),dtype=np.dtype(dtype).str))
        if kind=='step':
            owner=np.full(int(length)*2+1,77,dtype);out=owner[1::2];retained.append(owner)
        elif kind=='reverse':
            owner=np.full(int(length),77,dtype);out=owner[::-1];retained.append(owner)
        elif kind=='readonly':
            out=original_zeros(length,dtype=dtype);out.flags.writeable=False
        elif kind=='subclass':out=original_zeros(length,dtype=dtype).view(Destination)
        elif kind in ('object','object-error'):out=Box()
        elif kind=='none':out=None
        elif kind=='alias-values':out=mapping.out_values
        elif kind=='alias-keys':out=mapping.in_values
        else:out=original_zeros(length,dtype=dtype)
        retained.append(out)
        return out
    np.zeros=allocator
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            result=mapping.__array__() if action=='direct' else np.asarray(mapping)
            error=None
        except Exception as caught:result=None;error=dict(type=type(caught).__name__,message=str(caught))
        finally:np.zeros=original_zeros
    return dict(result=contract.array(result) if isinstance(result,np.ndarray) else dict(type=type(result).__name__),
        error=error,events=events,warnings=[dict(type=type(item.message).__name__,message=str(item.message)) for item in notices],
        retained=[contract.array(value) if isinstance(value,np.ndarray) else dict(type=type(value).__name__) for value in retained],
        returned_allocator_result=bool(error is None and retained and result is retained[-1]),keys=contract.array(mapping.in_values),values=contract.array(mapping.out_values),cache_is_none=mapping._array is None)


@pytest.mark.parametrize('kind',['plain','step','reverse','readonly','subclass','object','object-error','none','alias-values','alias-keys','float-payload'])
@pytest.mark.parametrize('action',['direct','asarray'])
def test_safe_custom_allocator_outputs_and_ownership(kind,action):
    assert output_observation(candidate.ArrayMap,kind,action)==output_observation(ReferenceMap,kind,action)


def attribute_observation(base):
    events=[];armed=False
    class Tracking(base):
        def __getattribute__(self,name):
            if armed and name in ('in_values','out_values'):events.append(name)
            return super().__getattribute__(name)
    mapping=Tracking(np.array([0,2,4],np.uint16),np.array([2,3,4],np.int16))
    original=np.zeros
    def allocator(length,*,dtype):
        nonlocal armed
        output=original(length,dtype=dtype);armed=True
        return output
    np.zeros=allocator
    try:result=mapping.__array__()
    finally:armed=False;np.zeros=original
    return events,contract.array(result)


def test_fresh_rhs_values_precede_index_keys():
    expected=attribute_observation(ReferenceMap)
    assert expected[0]==['out_values','in_values']
    assert attribute_observation(candidate.ArrayMap)==expected


def test_max_callable_is_bound_before_key_attribute_hook():
    def run(base):
        events=[];armed=False;original=np.max
        def replacement(keys):events.append('replacement-max');return np.int64(6)
        class Tracking(base):
            def __getattribute__(self,name):
                if armed and name=='in_values':np.max=replacement;events.append('keys')
                return super().__getattribute__(name)
        mapping=Tracking(np.array([0,2,4],np.uint16),np.array([2,3,4],np.int16));armed=True
        try:value=len(mapping)
        finally:np.max=original
        return value,events
    assert run(candidate.ArrayMap)==run(ReferenceMap)==(5,['keys'])


def test_ordinary_max_scan_and_assignment_remain_native(monkeypatch):
    seen=[];old_max=core.consumer_max;old_assign=core.consumer_dense_assign
    def maximum(keys):seen.append('native-max');return old_max(keys)
    def assign(*args):seen.append('native-assignment');return old_assign(*args)
    def reject(*args,**kwargs):raise AssertionError('upstream ArrayMap forwarded')
    monkeypatch.setattr(core,'consumer_max',maximum);monkeypatch.setattr(core,'consumer_dense_assign',assign)
    monkeypatch.setattr(ReferenceMap,'__init__',reject)
    mapping=candidate.ArrayMap(np.array([0,2,4],np.uint16),np.array([2,3,4],np.int16))
    assert len(mapping)==5
    np.testing.assert_array_equal(mapping.__array__(),[2,0,3,0,4])
    assert seen==['native-max','native-max','native-assignment']
