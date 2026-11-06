"""Prescott saved-gold/routing tests on real C++; no new original/vendor calls.

A private component may be selected explicitly. Corename/version substitution is
metadata-only routing coverage, never a natural Prescott-host/public-core claim.
"""
import importlib.util
import ast
import inspect
import json
import os
from pathlib import Path
import sys
import struct
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

import skimagexx
native=load('skimagexx._matrix_prescott_candidate',ROOT/'python/skimagexx/_matrix_transform.py')
path=os.environ.get('MATRIX_TRANSFORM_PRIVATE_BACKEND')
backend=load('_matrix_math_core',path)if path and '_matrix_math_core'not in sys.modules else sys.modules.get('_matrix_math_core',skimagexx._core)
native._core=backend
contract=load('_prescott_saved_contract',ROOT/'tests/test_matrix_transform_contracts.py')
SAVED=json.loads((ROOT/'coverage/contracts/matrix-transform.json').read_text())
SPECS=[s for s in contract.cases()if s['id']in{f'geometry/D{d}/{model}/N{n}'for d in(2,3)for model in('identity','translation','zero')for n in(1,7)}]

@pytest.fixture(autouse=True)
def default_workers():
    prior=backend.get_num_threads();backend.set_num_threads(1)
    yield
    backend.set_num_threads(prior)

@pytest.mark.parametrize('workers',(1,4))
@pytest.mark.parametrize('spec',SPECS,ids=lambda s:s['id'])
def test_saved_dyadic_whole_observations_use_real_prescott(monkeypatch,spec,workers):
    backend.set_num_threads(workers)
    monkeypatch.setattr(native,'_runtime_profile',lambda:'1.24.4')
    monkeypatch.setattr(native,'_THREAD_QUERY',lambda:1)
    monkeypatch.setattr(native,'_RUNTIME_CORENAME','Prescott')
    def forbidden(*args,**kwargs):raise AssertionError('no materialized/image/provider algorithm allowed')
    monkeypatch.setattr(native,'_materialized',forbidden)
    before=backend.matrix_transform_v3_completed()
    actual=contract.observe(native.matrix_transform,spec)
    assert actual==SAVED['profiles'][contract.profile()]['observations'][spec['id']]
    assert backend.matrix_transform_v3_completed()==before+1

@pytest.mark.parametrize('k',(3,4,5))
@pytest.mark.parametrize('order',('C','F'))
@pytest.mark.parametrize('residue',(0,8))
def test_original_layout_selects_exact_v3_before_copies(k,order,residue):
    owner=np.zeros(k*k+1,np.float64)
    matrix=np.ndarray((k,k),np.float64,buffer=owner,offset=residue,order=order)
    matrix[:]=np.eye(k)
    homogeneous=np.ones((1,k),np.float64)
    assert matrix.ctypes.data%16==residue
    expected=(9 if residue==0 else 10)if order=='C'else 8
    assert native._prepared_profile(homogeneous,matrix,'1.24.4','Prescott')==expected
    assert native._prepared_profile(np.ones((2,k)),matrix,'1.24.4','Prescott')==7
    stepped=matrix[:,::2]
    if stepped.shape!=(k,k): # Actual square non-BLASable carrier, no original algorithm.
        wide=np.zeros((k,2*k));wide[:,::2]=matrix;stepped=wide[:,::2]
    assert native._prepared_profile(homogeneous,stepped,'1.24.4','Prescott')==2

@pytest.mark.parametrize('profile',(7,8,9,10))
def test_v3_builtin_owned_outputs_and_old_entries_refuse(profile):
    assert inspect.isbuiltin(backend.matrix_transform_run_v3)
    assert backend.matrix_transform_run_v3.__module__==backend.__name__
    assert backend.matrix_transform_v3_interface=='matrix-transform-math-v3'and backend.matrix_transform_v3_capabilities==1023
    p=np.array([[1.,2.],[3.,4.]])if profile==7 else np.array([[1.,2.]])
    m=np.eye(3);p.flags.writeable=False
    before=backend.matrix_transform_v3_completed();out,status=backend.matrix_transform_run_v3(p,m,profile)
    assert status==0 and out.tobytes()==p.tobytes()and backend.matrix_transform_v3_completed()==before+1
    assert out.flags.owndata and out.flags.c_contiguous and out.flags.writeable
    assert not np.shares_memory(out,p)and not np.shares_memory(out,m)
    assert backend.matrix_transform_run(p,m,profile)==(None,1)
    assert backend.matrix_transform_run_v2(p,m,profile)==(None,1)

@pytest.mark.parametrize('attribute,value',(('matrix_transform_v3_capabilities',1022),('matrix_transform_v3_interface','wrong'),('matrix_transform_run_v3',lambda *args:None)))
def test_v3_impostors_rejected_without_call(monkeypatch,attribute,value):
    monkeypatch.setattr(backend,attribute,value)
    assert native._checked_entry(7)is None and native._checked_entry(10)is None
    assert native._checked_entry(1)is not None and native._checked_entry(6)is not None


def test_actual_cpp_descriptor_refusals_no_protocol_replay():
    seen=[]
    class Provider:
        def __array__(self,*args,**kwargs):seen.append(1);raise AssertionError('conversion')
    with pytest.raises(TypeError):backend.matrix_transform_run_v3(Provider(),np.eye(3),7)
    assert not seen
    for dtype in('>f8','f4'):
        with pytest.raises(TypeError):backend.matrix_transform_run_v3(np.ones((2,2),dtype=dtype),np.eye(3),7)

RAW=json.loads((ROOT/'coverage/contracts/matrix-prescott-reference-v1.json').read_text())
RAW5=[s for s in RAW['cases']if s['spec']['kind']=='gemv'and s['spec']['k']==5]
@pytest.mark.parametrize('row',RAW5,ids=lambda r:r['spec']['id'])
@pytest.mark.parametrize('workers',(1,4))
def test_raw_captured_k5_mapping_uses_exact_words(row,workers):
    backend.set_num_threads(workers);spec=row['spec'];is_n=spec['calls'][0][0]=='dgemv_n_PRESCOTT';lda=spec['lda']
    def value(w):return struct.unpack('<d',bytes.fromhex(w)[::-1])[0]
    a=[value(w)for w in row['inputs']['A']];x=[value(w)for w in row['inputs']['x']]
    matrix=np.array([[a[i+j*lda]if is_n else a[j+i*lda]for j in range(5)]for i in range(5)],np.float64)
    profile=8 if is_n else(10 if spec['a_mod16']else 9)
    actual,status=backend.matrix_transform_run_v3(np.array([x[:4]],np.float64),matrix,profile)
    raw=[value(w)for w in row['expected']];den=raw[-1]if raw[-1]!=0. else float.fromhex('0x1p-52')
    expected=np.array([[v/den for v in raw[:-1]]],np.float64)
    assert status==0 and actual.tobytes()==expected.tobytes()


def test_existing_pipeline_witness_counts_actual_v3_return():
    source=ROOT/'tests/test_matrix_transform_native_pipeline.py'
    tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='guarded')
    env=dict(BACKEND=backend,ENTRY=backend.matrix_transform_run,ENTRY_V2=backend.matrix_transform_run_v2,ENTRY_V3=backend.matrix_transform_run_v3,sys=sys)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),env)
    (value,status),calls=env['guarded'](lambda:backend.matrix_transform_run_v3(np.array([[1.,2.],[3.,4.]]),np.eye(3),7))
    assert status==0 and calls==['real-checked-C++']and env['LAST_ATTEMPTS']==1
    assert value.tobytes()==np.array([[1.,2.],[3.,4.]]).tobytes()
