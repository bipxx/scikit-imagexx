"""Frozen ellipse perimeter originals through true owned builtin component."""
import base64
import gc
import importlib
import inspect
import json
from pathlib import Path
import sys
import zlib
import numpy as np
import pytest
from ellipse_perimeter_contract import cases,fixture
from ellipse_perimeter_observer import observe
ROOT=Path(__file__).parents[1]
MATRIX=json.loads((ROOT/'coverage/contracts/ellipse-perimeter.json').read_bytes())
CASES=cases()


def public():
    import skimagexx as xx
    from skimagexx._draw_ellipse_perimeter import ellipse_perimeter
    assert xx.draw.ellipse_perimeter is ellipse_perimeter
    assert str(inspect.signature(ellipse_perimeter))==MATRIX['signature']
    return xx,ellipse_perimeter


@pytest.mark.parametrize('spec',CASES,ids=[s['id'] for s in CASES])
@pytest.mark.parametrize('workers',(1,4))
def test_ellipse_perimeter_saved_public_pipeline(spec,workers,monkeypatch):
    xx,call=public();original=importlib.import_module('skimage.draw.draw');compiled=importlib.import_module('skimage.draw._draw')
    old_public,old_compiled=original.ellipse_perimeter,compiled._ellipse_perimeter
    def denied(*a,**k):raise AssertionError('original provider forbidden in owned captured partition')
    monkeypatch.setattr(original,'_ellipse_perimeter',denied);monkeypatch.setattr(compiled,'_ellipse_perimeter',denied)
    entry,getter=xx._core.draw_ellipse_perimeter,xx._core.get_num_threads
    assert inspect.isbuiltin(entry)
    returns=[];snapshots=[];accepted=[]
    def profile(frame,event,value):
        assert not(event=='call' and frame.f_code is old_public.__code__)
        assert not(event=='c_call' and value is old_compiled)
        if event=='c_return' and value is entry:returns.append(True)
        if event=='c_return' and value is getter:snapshots.append(True)
        if event=='return' and frame.f_code is call.__code__:accepted.append(frame.f_locals.get('native_result') is not None)
    old=xx.get_num_threads();xx.set_num_threads(workers);prior=sys.getprofile()
    try:
        sys.setprofile(profile)
        try:row=observe(call,spec,np)
        finally:sys.setprofile(prior)
        assert xx.get_num_threads()==workers
    finally:xx.set_num_threads(old)
    expected=next(r for r in MATRIX['original_observations'][np.__version__] if r['id']==spec['id'])
    assert row==expected
    success=expected['error'] is None
    assert returns==snapshots==([True] if success else [])
    assert accepted==([] if spec['kind'] in ('error-missing','error-unexpected-keyword') else [success])


@pytest.mark.parametrize('angle',(float('nan'),float('inf'),-float('inf')))
def test_ellipse_point_earlyreturn_ignores_shape_and_nonfinite_angle(angle,monkeypatch):
    xx,call=public();compiled=importlib.import_module('skimage.draw._draw')
    monkeypatch.setattr(compiled,'_ellipse_perimeter',lambda *a:pytest.fail('provider'))
    class IgnoredShape:
        def __len__(self):raise AssertionError('point inspectedshape')
        def __getitem__(self,key):raise AssertionError('point indexedshape')
    result=call(3,-4,0,0,orientation=angle,shape=IgnoredShape())
    saved=next(r for r in MATRIX['original_observations'][np.__version__] if r['id']=='ellipse-perimeter/point')['result']['items']
    for a,e in zip(result,saved):
        assert a.ndim==0 and a.flags.owndata and a.base is None
        assert a.tobytes()==zlib.decompress(base64.b64decode(e['raw_zlib_base64']))


@pytest.mark.parametrize('shape',((),(8,),[],[8]))
def test_ellipse_short_builtin_shapes_refuse_without_provider(shape,monkeypatch):
    xx,call=public();compiled=importlib.import_module('skimage.draw._draw')
    monkeypatch.setattr(compiled,'_ellipse_perimeter',lambda *a:pytest.fail('unsafe provider'))
    monkeypatch.setattr(xx._core,'get_num_threads',lambda:pytest.fail('unsafe budgetlookup'))
    with pytest.raises(NotImplementedError,match='outside the assessed safe boundary'):call(0,0,2,3,shape=shape)


@pytest.mark.parametrize('angle',(None,'0.3',complex(1,0)))
def test_ellipse_orientation_exact_CAPI_error_no_worker_or_native(angle,monkeypatch):
    xx,call=public()
    monkeypatch.setattr(xx._core,'get_num_threads',lambda:pytest.fail('orientationerrorbudgetlookup'))
    monkeypatch.setattr(xx._core,'draw_ellipse_perimeter',lambda *a:pytest.fail('orientationerrornative'))
    with pytest.raises(TypeError,match='must be real number, not '+type(angle).__name__):call(0,0,2,3,orientation=angle)


def test_ellipse_protocol_conversion_order_and_provider_no_replay(monkeypatch):
    xx,call=public();compiled=importlib.import_module('skimage.draw._draw');events=[];token=object();shape=object();seen=[]
    class Index:
        def __init__(self,name,value):self.name,self.value=name,value
        def __index__(self):events.append(self.name);return self.value
    class Angle:
        def __float__(self):events.append('angle');return 0.
    monkeypatch.setattr(compiled,'_ellipse_perimeter',lambda *a:seen.append(a) or token)
    monkeypatch.setattr(xx._core,'get_num_threads',lambda:pytest.fail('provider-onlybudgetlookup'))
    assert call(Index('r',0),Index('c',0),Index('rr',2),Index('cr',3),Angle(),shape) is token
    assert events==['r','c','rr','cr','angle'] and seen==[(0,0,2,3,0.,shape)]


def test_ellipse_worker_snapshot_after_callback_and_owning_lifetime(monkeypatch):
    xx,call=public();old=xx.get_num_threads();getter=xx._core.get_num_threads;snapshots=[]
    class Angle:
        def __float__(self):xx.set_num_threads(1);return 0.
    def snapshot():value=getter();snapshots.append(value);return value
    monkeypatch.setattr(xx._core,'get_num_threads',snapshot)
    try:
        xx.set_num_threads(4);first=call(0,0,2,3,Angle());assert snapshots==[1]
        copies=tuple(a.copy() for a in first);gc.collect();other=call(0,0,2,3)
        assert first[0] is not first[1] and not np.shares_memory(*first)
        assert all(a.flags.owndata and a.base is None and a.flags.writeable for a in first)
        assert all(a is not b and not np.shares_memory(a,b) for a in first for b in other)
        for a,e in zip(first,copies):np.testing.assert_array_equal(a,e)
    finally:xx.set_num_threads(old)


@pytest.mark.parametrize('protocol_shape',(False,True))
@pytest.mark.parametrize('enabled',(False,True))
def test_ellipse_provider_paths_follow_global_fallback_policy(protocol_shape,enabled,monkeypatch):
    xx,call=public();compiled=importlib.import_module('skimage.draw._draw');seen=[];token=object()
    entry=xx._core.draw_ellipse_perimeter
    assert inspect.isbuiltin(entry)
    monkeypatch.setattr(compiled,'_ellipse_perimeter',lambda *args:seen.append(args) or token)
    shape=object() if protocol_shape else (16,20)
    # The typed radius exceeds the native contract and declines before geometry.
    radius=2 if protocol_shape else 32769
    previous=xx.set_fallback_enabled(enabled)
    try:
        if enabled:
            with pytest.warns(xx.FallbackWarning,match='skimage.draw._draw._ellipse_perimeter'):
                assert call(0,0,radius,1,shape=shape) is token
            assert seen==[(0,0,radius,1,0.,shape)]
        else:
            with pytest.raises(NotImplementedError,match='fallback is disabled'):
                call(0,0,radius,1,shape=shape)
            assert not seen
    finally:xx.set_fallback_enabled(previous)
