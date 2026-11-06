"""Saved exact finite ARE stages; no original algorithm or public registration."""
import inspect
import json
from pathlib import Path
import sys
import warnings
import pytest
from test_label_pair_metrics_counts_binding import decode

DATA=json.loads((Path(__file__).parents[1]/'coverage/contracts/label-pair-metrics-v1.json').read_bytes())
CASES={s['id']:s for s in DATA['cases']}
IDS=[r['id']for r in DATA['original_observations']['1.24.4']if r['id'].startswith('adapted_rand_error/')and r['error']is None and '/table-'not in r['id']]


@pytest.mark.parametrize('case',IDS)
@pytest.mark.parametrize('workers',(1,4))
def test_saved_rand_stages(case,workers):
    import numpy as np,skimagexx as xx
    from skimagexx._label_pair_metrics import _are_from_i64
    row=next(r for r in DATA['original_observations'][np.__version__]if r['id']==case);spec=CASES[case]
    truth,test=[decode(v,np).reshape(-1).astype(np.int64)for v in row['before']['items'][0]['items']]
    ignored=np.array(spec.get('ignore',[0]),np.int64);before=[x.tobytes()for x in (truth,test,ignored)]
    names=['get_num_threads','label_pair_rand_terms','label_pair_rand_divide','label_pair_rand_fscore','label_pair_rand_error'];entries={getattr(xx._core,n):n for n in names};assert all(inspect.isbuiltin(x)for x in entries)
    calls=[];accepted=[];events=[]
    def trace(frame,event,value):
        if event=='c_return'and value in entries:calls.append(entries[value])
        if event=='return'and frame.f_code is _are_from_i64.__code__:accepted.append(frame.f_locals.get('native_result')is not None)
    def handler(message,flags):events.append({'event':'FP-handler','message':message,'flags':flags})
    old_workers=xx.get_num_threads();old_handler=np.geterrcall();previous=sys.getprofile();xx.set_num_threads(workers);np.seterrcall(handler)
    try:
        with np.errstate(**spec.get('policy',{})),warnings.catch_warnings(record=True)as caught:
            warnings.simplefilter('always');sys.setprofile(trace)
            try:actual=_are_from_i64(truth,test,ignored,float(spec.get('alpha',.5)))
            finally:sys.setprofile(previous)
        assert calls==['get_num_threads','label_pair_rand_terms','label_pair_rand_divide','label_pair_rand_divide','label_pair_rand_fscore','label_pair_rand_error']and accepted==[True]
        assert type(actual)is tuple and len(actual)==3
        assert [{'type':type(v).__name__,'dtype':v.dtype.str,'raw':v.tobytes().hex()}for v in actual]==row['result']['items']
        assert events==row['events']
        assert [{'category':type(v.message).__name__,'message':str(v.message)}for v in caught]==row['warnings']
        assert before==[x.tobytes()for x in (truth,test,ignored)]and xx.get_num_threads()==workers
    finally:sys.setprofile(previous);np.seterrcall(old_handler);xx.set_num_threads(old_workers)


@pytest.mark.parametrize('alpha',(-1.,2.,pytest.param(1<<4096,id='huge-int')))
def test_bad_alpha_after_terms(alpha):
    import numpy as np,skimagexx as xx
    from skimagexx._label_pair_metrics import _are_from_i64
    calls=[];entry=xx._core.label_pair_rand_terms;divide=xx._core.label_pair_rand_divide;previous=sys.getprofile()
    def trace(frame,event,value):
        if event=='c_return'and(value is entry or value is divide):calls.append(value)
    try:
        sys.setprofile(trace)
        with pytest.raises(ValueError,match='^alpha must be between 0 and 1$'):_are_from_i64(np.array([1],np.int64),np.array([1],np.int64),np.array([],np.int64),alpha)
    finally:sys.setprofile(previous)
    assert calls==[entry]


@pytest.mark.parametrize('policy',('ignore','raise','change'))
def test_scalar_delivery_order_and_policy_change(policy):
    import numpy as np,skimagexx as xx
    from skimagexx._label_pair_metrics import _are_from_i64
    labels=np.array([1],np.int64);ignored=np.array([],np.int64);events=[];old=np.geterrcall();previous=sys.getprofile();calls=[]
    divide=xx._core.label_pair_rand_divide;score=xx._core.label_pair_rand_fscore;error=xx._core.label_pair_rand_error
    def trace(frame,event,value):
        if event=='c_return'and(value is divide or value is score or value is error):calls.append(value)
    def handler(message,flags):events.append((message,flags));np.seterr(invalid='ignore')
    np.seterrcall(handler)
    try:
        with np.errstate(all='ignore',invalid='call'if policy=='change'else policy):
            sys.setprofile(trace)
            try:
                if policy=='raise':
                    with pytest.raises(FloatingPointError,match='invalid value encountered in scalar divide'):_are_from_i64(labels,labels,ignored)
                else:
                    values=_are_from_i64(labels,labels,ignored)
                    assert all(v.tobytes().hex()=='000000000000f8ff'for v in values)
            finally:sys.setprofile(previous)
        assert events==([('invalid value',8)]if policy=='change'else[])
        assert calls==([divide]if policy=='raise'else[divide,divide,score,error])
    finally:sys.setprofile(previous);np.seterrcall(old)


@pytest.mark.parametrize('kind',('negative','fractional','joint-excess','alpha','snan-divide','snan-error','inf-error','huge'))
def test_rand_typed_refusal(kind):
    import numpy as np
    from skimagexx import _core
    if kind=='negative':result=_core.label_pair_rand_divide(-1.,2.)
    elif kind=='fractional':result=_core.label_pair_rand_divide(.5,2.)
    elif kind=='joint-excess':result=_core.label_pair_rand_fscore(3.,2.,2.,.5)
    elif kind=='alpha':result=_core.label_pair_rand_fscore(0.,0.,0.,2.)
    elif kind=='snan-divide':result=_core.label_pair_rand_divide(np.array([0x7ff0000000000001],np.uint64).view(np.float64)[0],2.)
    elif kind=='snan-error':result=_core.label_pair_rand_error(0x7ff0000000000001)
    elif kind=='inf-error':result=_core.label_pair_rand_error(0x7ff0000000000000)
    else:result=_core.label_pair_rand_divide(float(2**41),float(2**41))
    assert result is None


@pytest.mark.parametrize('alpha',(float('nan'),object()))
def test_unassessed_alpha_explicit(alpha):
    import numpy as np
    from skimagexx._label_pair_metrics import _are_from_i64
    labels=np.array([1],np.int64)
    with pytest.raises(NotImplementedError,match='alpha protocol/nonfinite'):_are_from_i64(labels,labels,np.array([],np.int64),alpha)


def test_distinct_scalar_warnings_survive_default_filter():
    import numpy as np
    from skimagexx._label_pair_metrics import _are_from_i64
    with np.errstate(all='warn'),warnings.catch_warnings(record=True)as caught:
        warnings.simplefilter('default')
        _are_from_i64(np.ones(6,np.int64),np.arange(1,7,dtype=np.int64),np.array([],np.int64),0.)
    assert [str(w.message)for w in caught]==['invalid value encountered in scalar divide']*2
    assert len({w.lineno for w in caught})==2
