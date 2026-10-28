"""Actual map binding/native acceptance against the immutable50-case records."""
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from skimagexx import _map_array as implementation
if __package__:
    from .map_array_contract import cases
    from .test_map_array_public import observe
else:
    from map_array_contract import cases
    from test_map_array_public import observe

CASES=cases();MATRIX=Path(__file__).parents[1]/'coverage/contracts/map-array.json'
STAGE='map_array_apply'


def test_map_array_binding_signature_and_frozen_return_partition():
    matrix=json.loads(MATRIX.read_bytes())
    assert str(inspect.signature(implementation.map_array))==matrix['signature']
    partitions=[{row['id'] for row in rows if row['error'] is None} for rows in matrix['original_observations'].values()]
    assert len(partitions)==2 and len(partitions[0])==37 and partitions[0]==partitions[1]
    assert callable(getattr(implementation._core,STAGE,None)) and implementation._core.get_num_threads()==1


@pytest.mark.parametrize('spec',CASES,ids=[row['id'] for row in CASES])
@pytest.mark.parametrize('workers',(1,4))
def test_map_array_pipeline(spec,workers,monkeypatch,tmp_path):
    import skimage.util as public
    import skimage.util._map_array as original
    import skimage.util._remap as primitive
    matrix=json.loads(MATRIX.read_bytes());expected=next(row for row in matrix['original_observations'][np.__version__] if row['id']==spec['id'])
    core=implementation._core;get_workers=core.get_num_threads
    previous=get_workers();prior_profile=sys.getprofile();calls=[];returns=[];exceptions=[];accepted=[];snapshots=[];forbidden=[]
    def reject(*args,**kwargs):
        forbidden.append(True)
        raise AssertionError('original map_array algorithm or Cython mapping loop forwarding')
    def snapshot():
        value=get_workers();snapshots.append(value);return value
    def profile(frame,event,arg):
        if getattr(arg,'__module__',None)==core.__name__ and getattr(arg,'__name__',None)==STAGE:
            if event=='c_call':calls.append(STAGE)
            elif event=='c_return':returns.append(STAGE)
            elif event=='c_exception':exceptions.append(STAGE)
        if event=='return' and frame.f_code is implementation.map_array.__code__:
            accepted.append(frame.f_locals.get('native_result') is not None)
    try:
        core.set_num_threads(workers)
        with monkeypatch.context() as patch:
            patch.setattr(public,'map_array',reject);patch.setattr(original,'map_array',reject)
            patch.setattr(primitive,'_map_array',reject);patch.setattr(core,'get_num_threads',snapshot)
            sys.setprofile(profile)
            try:actual=observe(implementation.map_array,spec,np)
            finally:sys.setprofile(prior_profile)
        assert not forbidden and actual==expected
        if expected['error'] is None:
            assert calls==returns==[STAGE] and not exceptions and accepted==[True] and snapshots==[workers]
        else:
            assert not returns and accepted==[False]
            assert calls==exceptions and len(calls)<=1 and snapshots==([workers] if calls else [])
        assert get_workers()==workers
        (tmp_path/'candidate.json').write_text(json.dumps(dict(id=spec['id'],workers=workers,observation=actual,
            actual_C_calls=calls,actual_C_returns=returns,actual_C_exceptions=exceptions,native_accepted=accepted,
            worker_snapshots=snapshots,forbidden=forbidden))+'\n')
    finally:sys.setprofile(prior_profile);core.set_num_threads(previous)


@pytest.mark.parametrize('workers',(0,1025))
def test_map_array_binding_worker_boundary(workers):
    index=np.array([0,1],np.uint8);output=np.full(2,99,np.uint16)
    with pytest.raises(ValueError,match='worker snapshot'):
        implementation._core.map_array_apply(index,output,index,output,workers)
    np.testing.assert_array_equal(output,[99,99])
