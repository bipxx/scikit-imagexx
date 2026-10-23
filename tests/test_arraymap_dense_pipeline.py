"""Private dense protocol proof; actual whole-package acceptance is separate."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import warnings
import numpy as np
import pytest
import skimagexx as xx
from skimage.util._map_array import ArrayMap as OriginalArrayMap

ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


if os.environ.get('SKIMAGEXX_COMPONENT_CONSUMERS_MODULE_DIR'):
    sys.path.insert(0,os.environ['SKIMAGEXX_COMPONENT_CONSUMERS_MODULE_DIR'])
    import _component_consumers_core as core
else:core=xx._core
assert core.component_consumers_interface_version==1
components=load('skimagexx._dense_label_adapter',ROOT/'python/skimagexx/_components.py');components._core=core
candidate=load('skimagexx._dense_consumer_adapter',ROOT/'python/skimagexx/_component_consumers.py');candidate._core=core;candidate._label=components.label
contract=load('_dense_reference_contract',ROOT/'tests/test_arraymap_dense_contracts.py')
original_construct=contract.construct
matrix=json.loads(contract.MATRIX.read_text())


def construct(spec,native=False):
    if not native:return original_construct(spec,False)
    # Fixture construction mirrors the sealed descriptors. Every operation
    # under test and every returned-map raster runs the private native module.
    keys=contract.layout(np.array(spec['keys'],dtype=spec['key_dtype']),spec.get('key_layout','C'))
    values=np.arange(keys.size,dtype=np.int16).reshape(keys.shape)+2
    values=contract.layout(values,spec.get('value_layout','C'))
    if spec.get('origin')=='returned':
        _,forward,inverse=candidate.relabel_sequential(keys,offset=spec['offset'])
        mapping=forward if spec['direction']=='forward' else inverse
        keys,values=mapping.in_values,mapping.out_values
    else:mapping=candidate.ArrayMap(keys,values)
    cache=spec.get('cache','none')
    if cache=='alias-values':mapping._array=values
    elif cache!='none':
        mapping._array=np.arange(5,dtype=np.int16)
        if cache=='readonly':mapping._array.flags.writeable=False
    return mapping,keys,values,mapping._array


contract.construct=construct


@pytest.fixture(autouse=True,params=[1,4])
def workers(request):
    before=core.get_num_threads();policy=xx.set_fallback_enabled(False)
    core.set_num_threads(request.param);assert core.get_num_threads()==request.param
    yield
    core.set_num_threads(before);xx.set_fallback_enabled(policy)


@pytest.mark.parametrize('spec',contract.CASES,ids=[s['id'] for s in contract.CASES])
def test_frozen_dense_protocol(spec):
    profile=matrix['profiles'][contract.profile()]
    expected=matrix['observations'][profile['observations'][spec['id']]]
    assert contract.observe(spec)==expected
    assert contract.observe(spec,True)==expected


def test_private_native_provenance_and_debug_assertions():
    assert candidate.ArrayMap is not OriginalArrayMap and candidate._core is core
    assert core.consumer_dense.__module__==core.__name__ and core.consumer_scatter.__module__==core.__name__
    if os.environ.get('SKIMAGEXX_REQUIRE_GIL_ASSERTIONS')=='1':assert core._consumer_gil_assertions is True


@pytest.mark.parametrize('index',[0,-1,np.int64(0),np.uint64(999),np.uint64(2**63),np.uint64(2**64-1),
    2**63,2**100,-2**100,True,np.bool_(True),
    np.array(0),np.array(True),np.array([0,1]),np.array([0,999]),np.array([0,0]),slice(None)])
@pytest.mark.parametrize('readonly',[False,True])
def test_checked_scalar_array_bool_index_and_readonly_order(index,readonly):
    def run(cls):
        keys=np.array([0,3,4],np.int16);values=np.array([2,3,4],np.int16)
        mapping=cls(keys,values);cache=np.arange(5,dtype=np.int16)
        if readonly:cache.flags.writeable=False
        mapping._array=cache
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            try:mapping[index]=7;error=None
            except Exception as caught:error=(type(caught).__name__,str(caught))
        return error,[(type(n.message).__name__,str(n.message)) for n in notices],contract.state(mapping,keys,values,cache)
    assert run(candidate.ArrayMap)==run(OriginalArrayMap)


@pytest.mark.parametrize('nonzero',[False,True])
def test_compacted_key_base_and_ownership(nonzero):
    expected=np.flatnonzero(np.arange(5) if nonzero else np.zeros(5))
    keys,_=core.consumer_compact(np.arange(5) if nonzero else np.zeros(5))
    assert contract.array(keys)==contract.array(expected)
    assert keys.base.shape==expected.base.shape and keys.base.strides==expected.base.strides
    assert keys.base.flags.owndata==expected.base.flags.owndata


@pytest.mark.parametrize('keys,values',[
    ([0,2],np.array([3,4],np.int16)),([0,2],[3,4]),
    ([True,False,True],[3,4]),([[0,2],[2,0]],[[3,4],[5,6]]),
    ([0.5,1.0],[3,4]),
])
@pytest.mark.parametrize('dtype',[None,np.float32,'not-a-dtype'])
def test_checked_sequence_keys_and_allocation_precedence(keys,values,dtype):
    def run(cls):
        mapping=cls(keys,values)
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            try:output=contract.array(mapping.__array__(dtype=dtype));error=None
            except Exception as caught:output=None;error=(type(caught).__name__,str(caught))
        return output,error,[(type(n.message).__name__,str(n.message)) for n in notices],mapping._array is None,mapping.in_values is keys,mapping.out_values is values
    assert run(candidate.ArrayMap)==run(OriginalArrayMap)
