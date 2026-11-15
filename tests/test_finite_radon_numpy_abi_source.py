"""Pure NumPy ABI/policy-source guards; no NumPy import or compilation."""
import ast
import hashlib
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/finite-radon-numpy-abi-v1.json'


def test_lowest_feature_api_runtime_dispatch_and_frozen_inverse():
    d=json.loads(CONTRACT.read_bytes());raw=(ROOT/d['binder']).read_bytes();text=raw.decode()
    assert hashlib.sha256(raw).hexdigest()==d['binder_sha256']
    assert '#define NPY_TARGET_VERSION NPY_1_23_API_VERSION'in text
    assert 'PyArray_GetNDArrayCFeatureVersion()>=0x00000012u'in text
    assert '#if NPY_ABI_VERSION'not in text
    for slot in (25,29,46):assert 'PyUFunc_API['+str(slot)+']'in text
    assert 'invalid_cast_notice();'in text and 'Py_XDECREF(object)'in text
    row=d['inverse'];assert raw.decode()==row['current']and hashlib.sha256(row['predecessor'].encode()).hexdigest()==row['predecessor_raw_sha256']
    assert 'NPY_TARGET_VERSION 0x00000012'in row['predecessor']


def test_policy_abi_signatures_and_unchanged_math_public():
    d=json.loads(CONTRACT.read_bytes())
    assert d['official_API_slots']=={'25':'int(char*,int*,int*,PyObject**)','29':'int(int,PyObject*,int,int*)','46':'int(const char*,int)'}
    assert d['NumPy254_requires_python']=='>=3.12'
    for name,pin in d['unchanged_source_raw_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==pin
    assert d['new_original_calls']==d['new_native_leaf_compiles']==0
    assert d['same_binary_scope']=='cp311:NumPy1.24.4+2.4.0;cp313separate:exact2.5.4'


def test_current_benchmark_source_context_is_strict_successor():
    d=json.loads((ROOT/'coverage/contracts/finite-radon-benchmark-execution-v3.json').read_bytes());old=json.loads((ROOT/d['predecessor_context']['path']).read_bytes())
    assert d['cases']==old['cases']and d['routes']==old['routes']and d['method']==old['method']
    assert all(x is None for x in d['unbound'].values())
    binder='src/bind_finite_radon.cpp';raw=(ROOT/binder).read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(raw).hexdigest()==d['production_source_canonical_sha256'][binder]
    assert hashlib.sha256(raw).hexdigest()!=old['production_source_canonical_sha256'][binder]
    row=d['ABI_production_predecessors'][binder]
    for f in row['inverse_fragments']:
        current=f['current'].encode();assert raw.count(current)==1;raw=raw.replace(current,f['predecessor'].encode(),1)
    assert hashlib.sha256(raw).hexdigest()==row['predecessor_canonical_sha256']


@pytest.mark.parametrize('method',('frt2','ifrt2'))
def test_original_algorithms_and_policyproviders_not_replaced(method):
    text=(ROOT/'python/skimagexx/_finite_radon.py').read_text();tree=ast.parse(text)
    assert any(isinstance(n,ast.FunctionDef)and n.name==method for n in tree.body)
    binder=(ROOT/'src/bind_finite_radon.cpp').read_text()
    assert 'PyArray_Cast'not in binder and 'PyArray_CopyInto'not in binder
    assert 'feclearexcept'not in binder and 'setcsr'not in binder
