"""Saved logical-CSR private component checks, not public API registration."""
import base64
import gc
import inspect
import json
from pathlib import Path
import sys
import zlib
import pytest
ROOT=Path(__file__).parents[1]
DATA=json.loads((ROOT/'coverage/contracts/label-pair-metrics-v1.json').read_bytes())
CASES={s['id']:s for s in DATA['cases']}
IDS=[r['id']for r in DATA['original_observations']['1.24.4']if r['id'].startswith('contingency_table/')and r['error']is None]


def decode(value,np):
    result=np.empty(value['shape'],dtype=value['dtype']);result.flat[:]=np.frombuffer(zlib.decompress(base64.b64decode(value['raw_zlib_base64'])),dtype=value['dtype']);return result


@pytest.mark.parametrize('case',IDS)
@pytest.mark.parametrize('workers',(1,4))
def test_saved_logical_counts(case,workers):
    import numpy as np,skimagexx as xx
    from skimagexx._label_pair_metrics import _counts_from_i64
    row=next(r for r in DATA['original_observations'][np.__version__]if r['id']==case);spec=CASES[case]
    truth,test=[decode(v,np).reshape(-1).astype(np.int64)for v in row['before']['items'][0]['items']]
    truth.flags.writeable=False;ignored=np.array(spec.get('ignore')or[],dtype=np.int64)
    before=[v.tobytes()for v in (truth,test,ignored)];entry=xx._core.label_pair_counts;getter=xx._core.get_num_threads;calls=[];gets=[];accepted=[]
    assert inspect.isbuiltin(entry)
    def trace(frame,event,value):
        if event=='c_return'and value is entry:calls.append(True)
        if event=='c_return'and value is getter:gets.append(True)
        if event=='return'and frame.f_code is _counts_from_i64.__code__:accepted.append(frame.f_locals.get('native_result')is not None)
    old=xx.get_num_threads();previous=sys.getprofile();xx.set_num_threads(workers)
    try:
        sys.setprofile(trace)
        try:actual=_counts_from_i64(truth,test,ignored,spec.get('normalize',False))
        finally:sys.setprofile(previous)
        assert calls==gets==accepted==[True]and xx.get_num_threads()==workers
    finally:xx.set_num_threads(old)
    shape,indptr,indices,values,kept,status=actual;expected=row['result']
    assert shape==tuple(expected['shape']) and kept==sum(int(x)not in(spec.get('ignore')or[])for x in truth)
    assert status==(1 if case.endswith('allignored-normalize')else 0)
    assert np.array_equal(indptr,decode(expected['indptr'],np).astype(np.int64))
    assert np.array_equal(indices,decode(expected['indices'],np).astype(np.int64))
    assert values.tobytes()==decode(expected['data'],np).tobytes()
    for value,dtype in [(indptr,np.int64),(indices,np.int64),(values,np.float64)]:
        assert type(value)is np.ndarray and value.dtype==np.dtype(dtype)and value.dtype.metadata is None
        assert value.ndim==1 and value.strides==(8,)and value.flags.owndata and value.base is None and value.flags.writeable and value.flags.aligned
        assert all(not np.shares_memory(value,source)for source in (truth,test,ignored))
    assert all(not np.shares_memory(a,b)for i,a in enumerate(actual[1:4])for b in actual[1:4][i+1:])
    assert before==[v.tobytes()for v in (truth,test,ignored)]
    payload=values.tobytes();del truth,test,ignored,actual;gc.collect();assert values.tobytes()==payload


@pytest.mark.parametrize('kind',('dtype','swapped','rank','length','strided','unaligned','negative','empty','ignored-size','workers'))
def test_private_count_refusal(kind):
    import numpy as np
    from skimagexx import _core
    truth=np.array([0,1],np.int64);test=truth.copy();ignored=np.array([],np.int64);workers=1
    if kind=='dtype':truth=truth.astype(np.float64)
    elif kind=='swapped':truth=truth.astype('>i8')
    elif kind=='rank':truth=truth.reshape(1,2)
    elif kind=='length':test=test[:1]
    elif kind=='strided':truth=np.arange(4,dtype=np.int64)[::2]
    elif kind=='unaligned':truth=np.ndarray((2,),dtype=np.int64,buffer=np.zeros(17,np.uint8),offset=1)
    elif kind=='negative':truth[0]=-1
    elif kind=='empty':truth=truth[:0];test=test[:0]
    elif kind=='ignored-size':ignored=np.zeros(65,np.int64)
    elif kind=='workers':workers=1025
    before=[v.tobytes()for v in (truth,test,ignored)]
    assert _core.label_pair_counts(truth,test,ignored,False,workers)is None
    assert before==[v.tobytes()for v in (truth,test,ignored)]
