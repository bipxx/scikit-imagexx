"""Frozen-row public integration and actual native acceptance, no recapture."""
import importlib
import inspect
import json
from pathlib import Path
import sys
import pytest
if __package__:
    from .draw_coordinates_contract import APIS,cases
    from .draw_coordinates_observer import SIGNATURES,observe
else:
    from draw_coordinates_contract import APIS,cases
    from draw_coordinates_observer import SIGNATURES,observe

CASES=cases();MATRIX=Path(__file__).parents[1]/'coverage/contracts/draw-coordinates.json'
PROVIDER_KINDS={
    'draw.line':{'float-integral','float-fractional','index-provider','overflow-start'},
    'draw.rectangle':{'missing','both','mismatch'},
    'draw.disk':{'radius-zero','negative-radius','center-length1','center-length3','radius-provider','fp-raise-zero'},
    'draw.ellipse':{'zero-row-radius','zero-both','negative-radius','center-provider','rotation-provider','fp-call-zero','shape-empty'}}
NATIVE_IDS=tuple(spec['id']for spec in CASES if spec['kind']not in PROVIDER_KINDS[spec['api']])


def test_draw_eligible_partition_is_frozen_separately_from_observations():
    assert len(NATIVE_IDS)==68
    assert {api:sum(spec['api']==api and spec['id']in NATIVE_IDS for spec in CASES)for api in APIS}=={
        'draw.line':18,'draw.rectangle':21,'draw.disk':13,'draw.ellipse':16}


@pytest.mark.parametrize('api',APIS)
def test_draw_public_export_signature(api):
    import skimagexx as xx
    from skimagexx import _draw_coordinates as owned
    name=api.split('.')[1]
    assert getattr(xx.draw,name)is getattr(owned,name)
    assert str(inspect.signature(getattr(xx.draw,name)))==SIGNATURES[api]


@pytest.mark.parametrize('spec',CASES,ids=[spec['id']for spec in CASES])
@pytest.mark.parametrize('workers',(1,4))
def test_draw_captured_public_pipeline(spec,workers,monkeypatch):
    import numpy as np
    import skimagexx as xx
    from skimagexx import _draw_coordinates as owned
    expected=next(row for row in json.loads(MATRIX.read_bytes())['original_observations'][np.__version__]if row['id']==spec['id'])
    call=getattr(xx.draw,spec['api'].split('.')[1]);accepted=spec['id']in NATIVE_IDS
    entry_name={'draw.line':'draw_line','draw.rectangle':'draw_rectangle','draw.disk':'draw_ellipse','draw.ellipse':'draw_ellipse'}[spec['api']]
    entry=getattr(owned._core,entry_name);assert inspect.isbuiltin(entry)
    assert entry.__module__  # Retain its true component/public owner; never spoof it.
    implementation=owned.ellipse if spec['api']=='draw.disk'else call
    returns=[];admission=[];snapshots=[]
    previous=xx.get_num_threads();fallback=xx.set_fallback_enabled(False);xx.set_num_threads(workers)
    previous_profile=sys.getprofile();get_workers=owned._core.get_num_threads
    def worker_snapshot():
        value=get_workers();snapshots.append(value);return value
    def denied(*args,**kwargs):raise AssertionError('owned drawing forwarded to an original whole API')
    def profile(frame,event,value):
        if event=='c_return'and value is entry:returns.append(entry_name)
        if event=='return'and frame.f_code is implementation.__code__:
            admission.append(frame.f_locals.get('native_result')is not None)
    public=importlib.import_module('skimage.draw');original=importlib.import_module('skimage.draw.draw')
    compiled=importlib.import_module('skimage.draw._draw')
    try:
        with monkeypatch.context()as patch:
            for name in ('line','rectangle','disk','ellipse'):
                patch.setattr(public,name,denied);patch.setattr(original,name,denied)
            if accepted and spec['api']=='draw.line':patch.setattr(compiled,'_line',denied)
            patch.setattr(owned._core,'get_num_threads',worker_snapshot)
            sys.setprofile(profile)
            try:actual=observe(call,spec,np)
            finally:sys.setprofile(previous_profile)
        assert actual==expected
        if accepted:
            assert returns==[entry_name]and admission==[True]
            assert snapshots==[workers]
        else:
            assert not any(admission)
        assert xx.get_num_threads()==workers
    finally:
        sys.setprofile(previous_profile);xx.set_num_threads(previous);xx.set_fallback_enabled(fallback)
