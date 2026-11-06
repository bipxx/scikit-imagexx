"""Saved contingency metadata/words and analytic prefix/suffix phase controls."""
import ast
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import warnings
import pytest
from label_pair_metrics_contract import fixture
from label_pair_metrics_observer import observation

ROOT=Path(__file__).parents[1]
DATA=json.loads((ROOT/'coverage/contracts/label-pair-metrics-v1.json').read_bytes())
CASES={s['id']:s for s in DATA['cases']}
IDS=[s['id']for s in DATA['cases']if s['api']=='metrics.contingency_table']


@pytest.fixture
def refuse_originals(monkeypatch):
    import skimage.metrics as original
    import skimagexx._compat as compat
    calls=[]
    def deny(*args,**kwargs):calls.append(True);raise AssertionError('Whole original metric/fallback forbidden')
    for name in ('contingency_table','adapted_rand_error','variation_of_information'):
        monkeypatch.setattr(original,name,deny)
    monkeypatch.setattr(importlib.import_module('skimage.metrics._contingency_table'),'contingency_table',deny)
    monkeypatch.setattr(compat,'fallback',deny)
    yield
    assert calls==[]


@pytest.mark.parametrize('case',IDS)
@pytest.mark.parametrize('workers',(1,4))
def test_saved_contingency_public(case,workers,refuse_originals):
    import numpy as np,skimagexx as xx
    from skimagexx._contingency_table import contingency_table
    spec=CASES[case];row=next(r for r in DATA['original_observations'][np.__version__]if r['id']==case)
    args,options,buffers=fixture(spec,np);before=observation((args,options,buffers),np)
    assert before==row['before']
    entry=xx._core.label_pair_counts_preweighted;getter=xx._core.get_num_threads
    assert inspect.isbuiltin(entry)
    calls=[];accepted=[];events=[];previous=sys.getprofile()
    def trace(frame,event,value):
        if event=='c_return'and(value is entry or value is getter):calls.append(value)
        if event=='return'and frame.f_code is contingency_table.__code__:accepted.append(frame.f_locals.get('native_result')is not None)
    old_workers=xx.get_num_threads();old_handler=np.geterrcall();xx.set_num_threads(workers)
    try:
        np.seterrcall(lambda message,flags:events.append({'event':'FP-handler','message':message,'flags':flags}))
        with np.errstate(**spec.get('policy',{})),warnings.catch_warnings(record=True)as caught:
            warnings.simplefilter('always');sys.setprofile(trace)
            try:
                if row['error'] is None:
                    result=contingency_table(*args,**options)
                    assert observation(result,np)==row['result']
                else:
                    with pytest.raises(Exception)as error:contingency_table(*args,**options)
                    assert dict(type=type(error.value).__name__,module=type(error.value).__module__,message=str(error.value))==row['error']
            finally:sys.setprofile(previous)
        if row['error'] is None:
            assert calls==[getter,entry]and accepted==[True]
        assert observation((args,options,buffers),np)==before==row['after']
        assert events==row['events']
        assert [{'category':type(x.message).__name__,'message':str(x.message)}for x in caught]==row['warnings']
        assert xx.get_num_threads()==workers
    finally:sys.setprofile(previous);np.seterrcall(old_handler);xx.set_num_threads(old_workers)


def test_normalization_callback_uses_current_labels_once(monkeypatch,refuse_originals):
    import numpy as np,skimagexx as xx
    import skimagexx._contingency_table as owned
    truth=np.ones(6,np.int16);test=np.array([0,2,2,1,1,2],np.int16)
    events=[];prefix=[];real_isin=np.isin;old=xx.get_num_threads();old_handler=np.geterrcall()
    def isin(*args,**kwargs):prefix.append(True);return real_isin(*args,**kwargs)
    def handler(message,flags):
        events.append((message,flags));truth[:]=[2,2,2,3,3,3];xx.set_num_threads(4)
    monkeypatch.setattr(np,'isin',isin);np.seterrcall(handler)
    try:
        with np.errstate(all='ignore',invalid='call'):
            result=owned.contingency_table(truth,test,ignore_labels=(1,),normalize=True,sparse_type='array')
        assert events==[('invalid value',8)]and prefix==[True]
        assert result.shape==(4,3)and result.indptr.tolist()==[0,0,0,2,4]
        assert result.indices.tolist()==[0,2,1,2]
        assert all(x==bytes.fromhex('000000000000f8ff')for x in [result.data[i:i+1].tobytes()for i in range(4)])
        assert truth.tolist()==[2,2,2,3,3,3]and xx.get_num_threads()==4
    finally:np.seterrcall(old_handler);xx.set_num_threads(old)


def test_callback_raise_stops_before_suffix(refuse_originals):
    import numpy as np,skimagexx as xx
    from skimagexx._contingency_table import contingency_table
    labels=np.ones(3,np.int16);calls=[];entry=xx._core.label_pair_counts_preweighted;getter=xx._core.get_num_threads;previous=sys.getprofile();old=np.geterrcall()
    def trace(frame,event,value):
        if event=='c_return'and(value is entry or value is getter):calls.append(value)
    def handler(message,flags):raise RuntimeError('stop normalization handler')
    np.seterrcall(handler)
    try:
        with np.errstate(all='ignore',invalid='call'):
            sys.setprofile(trace)
            try:
                with pytest.raises(RuntimeError,match='stop normalization handler'):
                    contingency_table(labels,labels,ignore_labels=(1,),normalize=True)
            finally:sys.setprofile(previous)
        assert calls==[]and labels.tolist()==[1,1,1]
    finally:np.seterrcall(old)


def test_prepared_weights_not_reignored_or_renormalized(monkeypatch,refuse_originals):
    """Analytic source-provider phase witness, not a new original gold."""
    import numpy as np
    from skimagexx._contingency_table import contingency_table
    truth=np.array([0,1,1,0],np.int16);test=np.zeros(4,np.int16);calls=[];real=np.count_nonzero
    def denominator(value):
        calls.append(value.copy());result=real(value);truth[:]=[1,1,1,2];return result
    monkeypatch.setattr(np,'count_nonzero',denominator)
    result=contingency_table(truth,test,ignore_labels=(0,),normalize=True,sparse_type='array')
    assert len(calls)==1 and calls[0].tolist()==[0.,1.,1.,0.]
    assert result.indptr.tolist()==[0,0,1,2]and result.indices.tolist()==[0,0]
    assert result.data.tobytes()==np.array([1.,0.],np.float64).tobytes()


@pytest.mark.parametrize('dtype',('int16','int64'))
@pytest.mark.parametrize('sparse_type',('array','matrix'))
def test_structural_factory_never_coalesces_again(monkeypatch,dtype,sparse_type,refuse_originals):
    import numpy as np,scipy.sparse as sparse
    import skimagexx._contingency_table as owned
    original=sparse.csr_array;calls=[]
    def factory(value,*args,**kwargs):
        assert isinstance(value,tuple)and len(value)==3,'Numerical coordinate coalescing provider forbidden'
        calls.append(True);return original(value,*args,**kwargs)
    monkeypatch.setattr(owned.sparse,'csr_array',factory)
    truth=np.array([0,1,1,2,2,2],dtype=dtype);test=np.array([0,2,2,1,1,2],dtype=dtype)
    result=owned.contingency_table(truth,test,sparse_type=sparse_type)
    assert calls==[True]and result.shape==(3,3)
    assert result.data.tolist()==[1.,2.,2.,1.]
    assert result.indices.dtype==np.dtype(np.int64 if dtype=='int64'else np.int32)
    assert result.data.base.shape==(6,)and result.indices.base.shape==(6,)
    assert result.__dict__.get('_has_canonical_format')==(True if sparse_type=='array'else None)


def test_unaligned_integer_storage_is_normalized(refuse_originals):
    import numpy as np
    from skimagexx._contingency_table import contingency_table
    owner=np.zeros(6*8+1,np.uint8);truth=np.ndarray((6,),np.int64,buffer=owner,offset=1);truth[:]=[0,1,1,2,2,2]
    before=truth.tobytes();result=contingency_table(truth,truth,sparse_type='array')
    assert result.data.tolist()==[1.,2.,3.]and result.indices.tolist()==[0,1,2]and truth.tobytes()==before


@pytest.mark.parametrize('kind',('dtype','swapped','rank','length','strided','unaligned','mixed','snan'))
def test_preweighted_private_refusal(kind):
    import numpy as np
    from skimagexx import _core
    truth=np.array([0,1],np.int64);test=truth.copy();weights=np.array([1.,1.],np.float64)
    if kind=='dtype':weights=weights.astype(np.float32)
    elif kind=='swapped':weights=weights.astype('>f8')
    elif kind=='rank':truth=truth.reshape(1,2)
    elif kind=='length':test=test[:1]
    elif kind=='strided':weights=np.ones(4)[::2]
    elif kind=='unaligned':weights=np.ndarray((2,),np.float64,buffer=np.zeros(17,np.uint8),offset=1)
    elif kind=='mixed':weights[:]=[.25,.5]
    else:weights.view(np.uint64)[0]=0x7ff0000000000001
    before=[x.tobytes()for x in (truth,test,weights)]
    assert _core.label_pair_counts_preweighted(truth,test,weights,1)is None
    assert before==[x.tobytes()for x in (truth,test,weights)]


def test_contingency_prospective_scope_without_science():
    contract=json.loads((ROOT/'coverage/contracts/contingency-table-frontend-v1.json').read_bytes())
    assert contract['original_call_budget']==0 and contract['policy']=='exact0ULP logicalCSR plus capturedmetadata'
    assert len(IDS)==8 and contract['saved_reference_ids']==IDS
    tree=ast.parse((ROOT/'python/skimagexx/_contingency_table.py').read_bytes())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='contingency_table')
    assert [x.arg for x in function.args.args]==['im_true','im_test']
    assert [x.arg for x in function.args.kwonlyargs]==['ignore_labels','normalize','sparse_type']
    assert 'data /= np.count_nonzero(data)'in ast.unparse(function)


def test_contingency_sdk_default_and_focused_source():
    text=(ROOT/'tests/cmake_consumer/label_pair_metrics_consumer.cpp').read_text()
    assert '--preweighted-only'in text and 'contingency_preweighted_controls::run()'in text
    assert 'if(!rand_only)std::cout<<"installed preweighted controls="'in text
    assert '--rand-only'in text and 'label_pair_controls::run()'in text
