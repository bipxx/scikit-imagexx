"""Unregistered public-shaped finite ARE partition, using saved originals only."""
import inspect
import json
from pathlib import Path
import sys
import warnings
import pytest
from label_pair_metrics_contract import fixture
from label_pair_metrics_observer import observation

DATA=json.loads((Path(__file__).parents[1]/'coverage/contracts/label-pair-metrics-v1.json').read_bytes())
CASES={s['id']:s for s in DATA['cases']}
IDS=[r['id']for r in DATA['original_observations']['1.24.4']if r['id'].startswith('adapted_rand_error/')and r['error']is None and '/table-'not in r['id']]


@pytest.fixture(autouse=True)
def refuse_original_metrics(monkeypatch):
    import skimage.metrics as original
    import skimagexx._compat as compat
    calls=[]
    def deny(*args,**kwargs):calls.append(True);raise AssertionError('Original metric/fallback forbidden')
    for name in ('contingency_table','adapted_rand_error','variation_of_information'):monkeypatch.setattr(original,name,deny)
    monkeypatch.setattr(compat,'fallback',deny)
    yield
    assert calls==[]


@pytest.mark.parametrize('case',IDS)
@pytest.mark.parametrize('workers',(1,4))
def test_saved_public_rand(case,workers):
    import numpy as np,skimagexx as xx
    from skimagexx._label_pair_metrics import adapted_rand_error,_are_from_i64
    spec=CASES[case];row=next(r for r in DATA['original_observations'][np.__version__]if r['id']==case)
    args,options,buffers=fixture(spec,np);before=observation((args,options,buffers),np);assert before==row['before']
    calls=[];accepted=[];events=[];names=['label_pair_check_storage','get_num_threads','label_pair_rand_terms','label_pair_rand_divide','label_pair_rand_fscore','label_pair_rand_error'];entries={getattr(xx._core,n):n for n in names}
    def trace(frame,event,value):
        if event=='c_return'and value in entries:calls.append(entries[value])
        if event=='return'and frame.f_code is _are_from_i64.__code__:accepted.append(frame.f_locals.get('native_result')is not None)
    old_workers=xx.get_num_threads();old_handler=np.geterrcall();previous=sys.getprofile();xx.set_num_threads(workers);np.seterrcall(lambda message,flags:events.append({'event':'FP-handler','message':message,'flags':flags}))
    try:
        with np.errstate(**spec.get('policy',{})),warnings.catch_warnings(record=True)as caught:
            warnings.simplefilter('always');sys.setprofile(trace)
            try:result=adapted_rand_error(*args,**options)
            finally:sys.setprofile(previous)
        assert observation(result,np)==row['result']
        assert observation((args,options,buffers),np)==before==row['after']
        assert events==row['events']and [{'category':type(v.message).__name__,'message':str(v.message)}for v in caught]==row['warnings']
        assert calls==['label_pair_check_storage']*2+['get_num_threads','label_pair_rand_terms','label_pair_rand_divide','label_pair_rand_divide','label_pair_rand_fscore','label_pair_rand_error']
        assert accepted==[True]and xx.get_num_threads()==workers
    finally:sys.setprofile(previous);np.seterrcall(old_handler);xx.set_num_threads(old_workers)


def test_public_bad_alpha_saved_error():
    import numpy as np
    from skimagexx._label_pair_metrics import adapted_rand_error
    spec=CASES['adapted_rand_error/bad-alpha'];args,options,_=fixture(spec,np)
    with pytest.raises(ValueError)as caught:adapted_rand_error(*args,**options)
    row=next(r for r in DATA['original_observations'][np.__version__]if r['id']==spec['id'])
    assert dict(type=type(caught.value).__name__,module=type(caught.value).__module__,message=str(caught.value))==row['error']


def test_public_shape_error_precedes_table_and_alpha():
    import numpy as np
    from skimagexx._label_pair_metrics import adapted_rand_error
    with pytest.raises(ValueError,match='^Input images must have the same dimensions.$'):
        adapted_rand_error(np.ones((2,3),np.int16),np.ones((3,2),np.int16),table=object(),alpha=-1)


@pytest.mark.parametrize('layout',('F-readonly','reverse'))
def test_public_layout_and_lifetime(layout):
    import numpy as np
    from skimagexx._label_pair_metrics import adapted_rand_error
    first=np.array([[0,1,1],[2,2,2]],dtype=np.int16,order='F');second=first.copy(order='K')
    if layout=='reverse':first=first[:,::-1];second=second[:,::-1]
    first.flags.writeable=False;before=[observation(v,np)for v in (first,second)];result=adapted_rand_error(first,second)
    assert [v.tobytes().hex()for v in result]==['0000000000000000','000000000000f03f','000000000000f03f']
    assert [observation(v,np)for v in (first,second)]==before
    del first,second
    assert all(type(v)is np.float64 and v.dtype.metadata is None for v in result)


@pytest.mark.parametrize('dtype',('int8','uint8','int16','uint16','int32','uint32','int64','uint64'))
def test_public_integer_conversion(dtype):
    import numpy as np
    from skimagexx._label_pair_metrics import adapted_rand_error
    labels=np.array([[0,1,1],[2,2,2]],dtype=dtype);before=labels.tobytes();values=adapted_rand_error(labels,labels)
    assert [v.tobytes().hex()for v in values]==['0000000000000000','000000000000f03f','000000000000f03f']
    assert labels.tobytes()==before


@pytest.mark.parametrize('frontier',('table','none','float','negative','large-label','empty','ignore-protocol'))
def test_public_unassessed_frontier(frontier):
    import numpy as np
    from skimagexx._label_pair_metrics import adapted_rand_error
    first=np.array([1],np.int64);second=first.copy();options={}
    if frontier=='table':options['table']=object()
    elif frontier=='none':first=None
    elif frontier=='float':first=first.astype(float)
    elif frontier=='negative':first[0]=-1
    elif frontier=='large-label':first[0]=2**20
    elif frontier=='empty':first=first[:0];second=second[:0]
    else:options['ignore_labels']=object()
    with pytest.raises(NotImplementedError):adapted_rand_error(first,second,**options)


def test_public_untrusted_span_refused_before_copy():
    import numpy as np
    from skimagexx._label_pair_metrics import adapted_rand_error
    first=np.lib.stride_tricks.as_strided(np.ones(1,np.int64),shape=(6,),strides=(8,))
    with pytest.raises(ValueError):adapted_rand_error(first,np.ones(6,np.int64))


def test_public_signature_defaults_and_no_registration():
    import skimagexx._label_pair_metrics as owned
    signature=inspect.signature(owned.adapted_rand_error)
    assert str(signature)=='(image_true=None, image_test=None, *, table=None, ignore_labels=(0,), alpha=0.5)'
    assert owned.adapted_rand_error.__module__=='skimagexx._label_pair_metrics'


@pytest.mark.parametrize('unaligned',('truth','test','both'))
def test_public_unaligned_ordinary_storage(unaligned):
    import numpy as np,skimagexx as xx
    from skimagexx._label_pair_metrics import adapted_rand_error,_are_from_i64
    def carrier(offset):
        owner=np.zeros(6*8+offset,np.uint8)
        image=np.ndarray((2,3),dtype=np.int64,buffer=owner,offset=offset)
        image[:]=[[0,1,1],[2,2,2]]
        return image
    truth=carrier(int(unaligned in ('truth','both')));test=carrier(int(unaligned in ('test','both')))
    assert truth.flags.aligned==(unaligned=='test')and test.flags.aligned==(unaligned=='truth')
    before=[observation(v,np)for v in (truth,test)];entry=xx._core.label_pair_rand_terms;getter=xx._core.get_num_threads;calls=[];accepted=[];intake=[];previous=sys.getprofile()
    def trace(frame,event,value):
        if event=='c_return'and(value is entry or value is getter):calls.append(value)
        if event=='call'and frame.f_code is _are_from_i64.__code__:
            a,b=frame.f_locals['truth'],frame.f_locals['test']
            intake.append(a.flags.aligned and b.flags.aligned and a.flags.c_contiguous and b.flags.c_contiguous)
        if event=='return'and frame.f_code is _are_from_i64.__code__:accepted.append(frame.f_locals.get('native_result')is not None)
    try:
        sys.setprofile(trace)
        result=adapted_rand_error(truth,test)
    finally:sys.setprofile(previous)
    assert calls==[getter,entry]and accepted==intake==[True]
    assert [v.tobytes().hex()for v in result]==['0000000000000000','000000000000f03f','000000000000f03f']
    assert [observation(v,np)for v in (truth,test)]==before


@pytest.mark.parametrize('dtype',('int8','uint8','int16','uint16','int32','uint32','int64','uint64'))
@pytest.mark.parametrize('unaligned',(False,True))
def test_public_integer_metadata_value_words_and_native_intake(dtype,unaligned):
    import numpy as np,skimagexx as xx
    from skimagexx._label_pair_metrics import adapted_rand_error,_are_from_i64
    tagged=np.dtype(dtype,metadata={'tag':7})
    owner=np.zeros(6*tagged.itemsize+int(unaligned),np.uint8)
    truth=np.ndarray((2,3),dtype=tagged,buffer=owner,offset=int(unaligned))
    truth[:]=[[0,1,1],[2,2,2]];test=truth.copy()
    before=[observation(a,np)for a in(truth,test)];entry=xx._core.label_pair_rand_terms;getter=xx._core.get_num_threads
    calls=[];accepted=[];intake=[];previous=sys.getprofile()
    def trace(frame,event,value):
        if event=='c_return'and(value is getter or value is entry):calls.append(value)
        if event=='call'and frame.f_code is _are_from_i64.__code__:
            arrays=[frame.f_locals[n]for n in('truth','test')]
            intake.append(all(a.dtype==np.dtype('int64')and a.dtype.metadata is None and a.flags.aligned and a.flags.c_contiguous for a in arrays))
        if event=='return'and frame.f_code is _are_from_i64.__code__:accepted.append(frame.f_locals.get('native_result')is not None)
    try:
        sys.setprofile(trace);result=adapted_rand_error(truth,test)
    finally:sys.setprofile(previous)
    # Same admitted perfect-label values and already captured exact words.
    assert [a.tobytes().hex()for a in result]==['0000000000000000','000000000000f03f','000000000000f03f']
    assert calls==[getter,entry]and accepted==intake==[True]
    assert [observation(a,np)for a in(truth,test)]==before
