"""Saved-original ellipsoid parity through the true native component."""
import gc
import inspect
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from ellipsoid_contract import cases
from ellipsoid_observer import observe

ROOT=Path(__file__).parents[1]
CASES=cases()
MATRIX=json.loads((ROOT/'coverage/contracts/ellipsoid.json').read_bytes())


def _public():
    import skimagexx
    from skimagexx._draw_ellipsoid import ellipsoid
    assert skimagexx.draw.ellipsoid is ellipsoid
    assert str(inspect.signature(ellipsoid))=='(a, b, c, spacing=(1.0, 1.0, 1.0), levelset=False)'
    return skimagexx,ellipsoid


@pytest.mark.parametrize('spec',CASES,ids=[s['id'] for s in CASES])
@pytest.mark.parametrize('workers',(1,4))
def test_ellipsoid_saved_public_pipeline(spec,workers,monkeypatch):
    xx,call=_public()
    import skimage.draw.draw3d as original
    def denied(*args,**kwargs):
        raise AssertionError('original ellipsoid forbidden on advertised native/boundary partition')
    monkeypatch.setattr(original,'ellipsoid',denied)
    builtin=xx._core.draw_ellipsoid;getter=xx._core.get_num_threads
    assert inspect.isbuiltin(builtin)
    witnessed=[];snapshots=[];accepted=[]
    def profile(frame,event,arg):
        if event=='c_return' and arg is builtin:witnessed.append(True)
        if event=='c_return' and arg is getter:snapshots.append(True)
    def trace(frame,event,arg):
        if frame.f_code is call.__code__ and event=='return':
            accepted.append(frame.f_locals.get('native_result') is not None)
        return trace
    old=xx.get_num_threads();xx.set_num_threads(workers)
    old_profile,old_trace=sys.getprofile(),sys.gettrace()
    try:
        sys.setprofile(profile);sys.settrace(trace)
        row=observe(call,spec,np)
    finally:
        sys.setprofile(old_profile);sys.settrace(old_trace);xx.set_num_threads(old)
    expected=next(r for r in MATRIX['original_observations'][np.__version__] if r['id']==spec['id'])
    assert row==expected
    success=expected['error'] is None
    assert len(witnessed)==len(snapshots)==int(success)
    assert accepted==[success]


@pytest.mark.parametrize('levelset',(False,True))
def test_ellipsoid_owning_lifetime_and_repeat(levelset,monkeypatch):
    xx,call=_public()
    import skimage.draw.draw3d as original
    monkeypatch.setattr(original,'ellipsoid',lambda *a,**k:pytest.fail('provider'))
    first=call(2,3,4,levelset=levelset);saved=first.copy();gc.collect()
    other=call(2,3,4,levelset=levelset)
    assert first.flags.owndata and first.flags.writeable and first.flags.c_contiguous
    assert first.base is None and other.base is None and not np.shares_memory(first,other)
    assert first.dtype==np.dtype('float64' if levelset else 'bool')
    assert first.dtype.metadata is None
    other[:]=0
    np.testing.assert_array_equal(first,saved)


@pytest.mark.parametrize('kind',('axis','spacing','truth','numpy','nonfinite','bad-spacing'))
def test_ellipsoid_protocol_provider_without_replay(kind,monkeypatch):
    xx,call=_public()
    import skimage.draw.draw3d as original
    events=[]
    class Protocol:
        def __le__(self,other):events.append('compare');raise AssertionError('early protocol')
        def __float__(self):events.append('float');raise AssertionError('early protocol')
        def __bool__(self):events.append('truth');raise AssertionError('early protocol')
    obj=Protocol();a,b,c=2,3,4;spacing=(1.,1.,1.);levelset=False
    if kind=='axis':a=obj
    if kind=='spacing':spacing=obj
    if kind=='truth':levelset=obj
    if kind=='numpy':a=np.float64(2)
    if kind=='nonfinite':a=float('inf')
    if kind=='bad-spacing':spacing=(0.,1.,1.)
    seen=[];token=object()
    def provider(*args,**kwargs):seen.append((args,kwargs));return token
    monkeypatch.setattr(original,'ellipsoid',provider)
    monkeypatch.setattr(xx._core,'draw_ellipsoid',lambda *a:pytest.fail('unexpected native admission'))
    assert call(a,b,c,spacing=spacing,levelset=levelset) is token
    assert not events and len(seen)==1
    assert seen[0][0][0] is a and seen[0][1]['spacing'] is spacing and seen[0][1]['levelset'] is levelset


def test_ellipsoid_native_refusal_worker_once_and_argument_preservation(monkeypatch):
    xx,call=_public()
    import skimage.draw.draw3d as original
    seen=[];budgets=[];token=object();spacing=[1.,1.,1.]
    monkeypatch.setattr(xx._core,'get_num_threads',lambda:budgets.append(1) or 1)
    monkeypatch.setattr(xx._core,'draw_ellipsoid',lambda *args:seen.append(args))
    original_seen=[]
    monkeypatch.setattr(original,'ellipsoid',lambda *a,**k:original_seen.append((a,k)) or token)
    assert call(2,3,4,spacing=spacing) is token
    assert budgets==[1] and seen==[((2.,3.,4.),(1.,1.,1.),False,1)]
    assert original_seen[0][1]['spacing'] is spacing


@pytest.mark.parametrize('value',(2**50+1,2**62,2**64,2**1024))
@pytest.mark.parametrize('position',('axis','spacing'))
def test_ellipsoid_large_builtin_int_preserves_source_provider(value,position,monkeypatch):
    xx,call=_public()
    import skimage.draw.draw3d as original
    axes=(value,3,4) if position=='axis' else (2,3,4)
    spacing=(value,1.,1.) if position=='spacing' else (1.,1.,1.)
    seen=[];token=object()
    def provider(*args,**kwargs):seen.append((args,kwargs));return token
    monkeypatch.setattr(original,'ellipsoid',provider)
    monkeypatch.setattr(xx._core,'draw_ellipsoid',lambda *a:pytest.fail('largeinteger native admission'))
    monkeypatch.setattr(xx._core,'get_num_threads',lambda:pytest.fail('largeinteger worker lookup'))
    assert call(*axes,spacing=spacing) is token
    assert len(seen)==1 and seen[0][0]==axes and seen[0][1]['spacing'] is spacing
    assert seen[0][0][0] is axes[0]


@pytest.mark.parametrize('position',('axis','spacing'))
def test_ellipsoid_integer_cap_inside_reaches_typed_prepare(position,monkeypatch):
    xx,call=_public()
    import skimage.draw.draw3d as original
    axes=(2**50,3,4) if position=='axis' else (2,3,4)
    spacing=(2**50,1.,1.) if position=='spacing' else (1.,1.,1.)
    token=object();seen=[];budgets=[]
    monkeypatch.setattr(original,'ellipsoid',lambda *a,**k:pytest.fail('inside cap refused before typedprepare'))
    monkeypatch.setattr(xx._core,'get_num_threads',lambda:budgets.append(1) or 1)
    monkeypatch.setattr(xx._core,'draw_ellipsoid',lambda *a:seen.append(a) or token)
    assert call(*axes,spacing=spacing) is token
    assert budgets==[1] and seen==[(tuple(map(float,axes)),tuple(map(float,spacing)),False,1)]
