"""Saved-only public checks; reference capture is never performed by tests."""
import gc
import importlib
import inspect
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from draw_bezier_contract import cases
from draw_bezier_observer import observe

ROOT=Path(__file__).parents[1]
MATRIX=json.loads((ROOT/'coverage/contracts/draw-bezier-v1.json').read_bytes())
CASES=cases()


@pytest.mark.skipif(not MATRIX['original_observations'],reason='Once-only original capture is pending; no candidate parity yet')
@pytest.mark.parametrize('spec',CASES,ids=lambda s:s['id'])
@pytest.mark.parametrize('workers',(1,4))
def test_bezier_saved_public(spec,workers,monkeypatch):
    import skimagexx as xx
    from skimagexx._bezier_curve import bezier_curve
    original=importlib.import_module('skimage.draw.draw')
    compiled=importlib.import_module('skimage.draw._draw')
    old_public,old_compiled=original.bezier_curve,compiled._bezier_curve
    def denied(*a,**k):raise AssertionError('Original algorithm forbidden in saved owned partition')
    monkeypatch.setattr(original,'_bezier_curve',denied)
    monkeypatch.setattr(compiled,'_bezier_curve',denied)
    assert str(inspect.signature(bezier_curve))==MATRIX['signature']
    entry,getter=xx._core.draw_bezier_curve,xx._core.get_num_threads
    assert inspect.isbuiltin(entry)
    calls=[];snapshots=[];accepted=[]
    def profile(frame,event,value):
        assert not(event=='call' and frame.f_code is old_public.__code__)
        assert not(event=='c_call' and value is old_compiled)
        if event=='c_return' and value is entry:calls.append(True)
        if event=='c_return' and value is getter:snapshots.append(True)
        if event=='return' and frame.f_code is bezier_curve.__code__:accepted.append(frame.f_locals.get('native_result') is not None)
    old=xx.get_num_threads();fallback=xx.set_fallback_enabled(False);prior=sys.getprofile();xx.set_num_threads(workers)
    try:
        sys.setprofile(profile)
        try:actual=observe(bezier_curve,spec,np)
        finally:sys.setprofile(prior)
        assert xx.get_num_threads()==workers
    finally:xx.set_num_threads(old);xx.set_fallback_enabled(fallback)
    expected=next(r for r in MATRIX['original_observations'][np.__version__] if r['id']==spec['id'])
    assert actual==expected
    success=expected['error'] is None
    assert calls==snapshots==([True] if success else [])
    assert accepted==[success]


def test_bezier_signature_and_owner_lifetime():
    from skimagexx._bezier_curve import bezier_curve
    assert str(inspect.signature(bezier_curve))==MATRIX['signature']
    # Source-analytic point case, not a fresh original/reference capture.
    rows,columns=bezier_curve(2,3,2,3,2,3,1.)
    for value in (rows,columns):
        assert value.dtype==np.dtype(np.intp) and value.shape==(1,)
        assert value.flags.owndata and value.base is None and value.strides==(np.dtype(np.intp).itemsize,)
    assert rows is not columns and not np.shares_memory(rows,columns)
    saved=columns.tobytes();rows[0]=99;del rows;gc.collect()
    assert columns.tobytes()==saved and columns.tolist()==[3]


@pytest.mark.parametrize('shape',((),(5,),[],[5]))
def test_bezier_original_unsafe_shortshape_refused(shape,monkeypatch):
    import skimagexx as xx
    from skimagexx._bezier_curve import bezier_curve
    monkeypatch.setattr(xx._core,'draw_bezier_curve',lambda *a:pytest.fail('quarantined shape entered kernel'))
    monkeypatch.setattr(xx._core,'get_num_threads',lambda:pytest.fail('quarantined shape budget lookup'))
    with pytest.raises(NotImplementedError,match='original-safe domain'):
        bezier_curve(0,0,1,4,6,6,1.,shape=shape)


def test_bezier_current_global_snapshot_and_native_decline(monkeypatch):
    import skimagexx as xx
    from skimagexx._bezier_curve import bezier_curve
    calls=[]
    monkeypatch.setattr(xx._core,'get_num_threads',lambda:(calls.append('worker') or 4))
    def decline(*args):
        calls.append(('native',args[-1]));return None
    monkeypatch.setattr(xx._core,'draw_bezier_curve',decline)
    with pytest.raises(NotImplementedError,match='outside assessed native coverage'):
        bezier_curve(0,0,1,4,6,6,1.)
    assert calls==['worker',('native',4)]
