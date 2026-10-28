"""Actual owned C entries plus immutable79-row original-first observations.

Private component execution is not whole-product/public registration proof.
Native admission is an additional assertion; original snapshots never change.
"""
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from skimagexx import _level_set as implementation
if __package__:
    from .level_set_contract import cases
    from .test_level_set_public import observe
else:
    from level_set_contract import cases
    from test_level_set_public import observe

CASES=cases()
MATRIX=Path(__file__).parents[1]/'coverage/contracts/level-set.json'
COMMON=('one-axis','default','volume','four-axis','singleton','empty-first',
        'empty-last','shape-list','bench-256','bench-1024','bench-volume','bench-thin')
NATIVE_IDS={name+'/'+case for name in ('checkerboard_level_set','disk_level_set') for case in COMMON}
NATIVE_IDS.update('checkerboard_level_set/'+case for case in
                  ('empty-shape','square-unit','square-two','square-oversized','square-negative','square-bool'))
NATIVE_IDS.update('disk_level_set/'+case for case in
                  ('radius-zero','radius-negative','radius-fraction','radius-exact-boundary',
                   'center-integer','center-outside','center-broadcast',
                   'strict-boundary-below','strict-boundary-equal','strict-boundary-above'))
STAGES=('level_set_checkerboard','level_set_disk')


def test_level_set_binding_and_signatures():
    record=json.loads(MATRIX.read_bytes())
    assert len(NATIVE_IDS)==40
    assert all(callable(getattr(implementation._core,name,None)) for name in STAGES)
    for api,signature in record['signatures'].items():
        assert str(inspect.signature(getattr(implementation,api.split('.')[-1])))==signature
    assert implementation._core.get_num_threads()==1


@pytest.mark.parametrize('workers',(1,4))
@pytest.mark.parametrize('spec',CASES,ids=[row['id'] for row in CASES])
def test_level_set_pipeline(spec,workers,monkeypatch,tmp_path):
    import skimage.segmentation as public
    import skimage.segmentation.morphsnakes as original
    record=json.loads(MATRIX.read_bytes())
    expected=next(row for row in record['original_observations'][np.__version__] if row['id']==spec['id'])
    name=spec['api'].split('.')[-1];candidate=getattr(implementation,name)
    calls=[];accepted=[];forbidden=[];snapshots=[]
    core=implementation._core;get_workers=core.get_num_threads
    previous_workers=get_workers();previous_profile=sys.getprofile()
    def reject(*args,**kwargs):
        forbidden.append(True)
        raise AssertionError('whole original level-set algorithm forwarding')
    def snapshot():
        value=get_workers();snapshots.append(value);return value
    def profile(frame,event,arg):
        if event=='c_return' and getattr(arg,'__module__',None)==core.__name__ and getattr(arg,'__name__',None) in STAGES:
            calls.append(arg.__name__)
        if event=='return' and frame.f_code is candidate.__code__:
            accepted.append(frame.f_locals.get('native_result') is not None)
    try:
        core.set_num_threads(workers)
        with monkeypatch.context() as patch:
            for module in (public,original):
                for function in ('checkerboard_level_set','disk_level_set'):patch.setattr(module,function,reject)
            patch.setattr(core,'get_num_threads',snapshot)
            sys.setprofile(profile)
            try:actual=observe(candidate,spec,np)
            finally:sys.setprofile(previous_profile)
        if spec['benchmark'] and actual['result'] is not None:actual['result'].pop('raw_zlib_base64')
        assert not forbidden and actual==expected
        if spec['id'] in NATIVE_IDS:
            assert calls==[STAGES[0 if name=='checkerboard_level_set' else 1]]
            assert accepted==[True] and snapshots==[workers]
        else:
            assert accepted==[False]
            assert len(calls)<=1 and snapshots==([workers] if calls else [])
        assert get_workers()==workers
        (tmp_path/'candidate.json').write_text(json.dumps(dict(id=spec['id'],workers=workers,
            observation=actual,actual_C_returns=calls,native_accepted=accepted,
            worker_snapshots=snapshots,forbidden=forbidden))+'\n')
    finally:
        sys.setprofile(previous_profile);core.set_num_threads(previous_workers)


@pytest.mark.parametrize('workers',(0,1025))
def test_level_set_binding_rejects_invalid_worker_snapshot(workers):
    core=implementation._core
    with pytest.raises(ValueError,match='worker snapshot'):
        core.level_set_checkerboard((2,3),2,workers)
    with pytest.raises(ValueError,match='worker snapshot'):
        core.level_set_disk((2,3),(1,1),2.,workers)


@pytest.mark.parametrize('name',('checkerboard_level_set','disk_level_set'))
def test_level_set_maximum_rank_preserves_mgrid_error(name,monkeypatch):
    maximum=(np._core if hasattr(np,'_core') else np.core).multiarray.MAXDIMS
    shape=(1,)*maximum
    with pytest.raises(ValueError) as original:
        np.mgrid[[slice(size) for size in shape]]
    def reject(*args,**kwargs):raise AssertionError('native admission or whole algorithm forwarding at mgrid rank limit')
    import skimage.segmentation as public
    import skimage.segmentation.morphsnakes as source
    monkeypatch.setattr(implementation._core,'level_set_checkerboard',reject)
    monkeypatch.setattr(implementation._core,'level_set_disk',reject)
    monkeypatch.setattr(public,name,reject);monkeypatch.setattr(source,name,reject)
    with pytest.raises(ValueError) as actual:getattr(implementation,name)(shape)
    assert str(actual.value)==str(original.value)
