"""Actual circle component/public wrapper against immutable original rows."""
import importlib
import inspect
import json
from pathlib import Path
import sys
import pytest
if __package__:
    from .circle_perimeter_contract import cases
    from .circle_perimeter_observer import SIGNATURE,observe
else:
    from circle_perimeter_contract import cases
    from circle_perimeter_observer import SIGNATURE,observe

MATRIX=Path(__file__).parents[1]/'coverage/contracts/circle-perimeter.json'
UNSAFE_KINDS={'shape-short','shape-empty'}
PROVIDER_OR_ERROR_KINDS={'float-integral','float-fractional','overflow-origin','shape-float','invalid-method','bytes-method','none-method','list-method'}
CASES=tuple(spec for spec in cases()if spec['kind']not in UNSAFE_KINDS)
NATIVE_IDS=tuple(spec['id']for spec in CASES if spec['kind']not in PROVIDER_OR_ERROR_KINDS)


@pytest.fixture(scope='module')
def expected_rows():
    matrix=json.loads(MATRIX.read_bytes())
    return {profile:{row['id']:row for row in rows}for profile,rows in matrix['original_observations'].items()}


def test_circle_perimeter_frozen_partition_and_signature():
    import skimagexx as xx
    from skimagexx import _circle_perimeter as owned
    assert len(CASES)==69 and len(NATIVE_IDS)==57
    assert xx.draw.circle_perimeter is owned.circle_perimeter
    assert str(inspect.signature(owned.circle_perimeter))==SIGNATURE
    entry=owned._core.draw_circle_perimeter
    assert inspect.isbuiltin(entry)and entry.__module__=='skimagexx._core'


@pytest.mark.parametrize('spec',CASES,ids=[spec['id']for spec in CASES])
@pytest.mark.parametrize('workers',(1,4))
def test_circle_perimeter_saved_public_contract(spec,workers,monkeypatch,expected_rows):
    import numpy as np
    import skimagexx as xx
    from skimagexx import _circle_perimeter as owned
    expected=expected_rows[np.__version__][spec['id']]
    accepted=spec['id']in NATIVE_IDS
    entry=owned._core.draw_circle_perimeter
    native_returns=[];admissions=[];snapshots=[]
    previous=xx.get_num_threads();fallback=xx.set_fallback_enabled(False);xx.set_num_threads(workers)
    old_profile=sys.getprofile();get_workers=owned._core.get_num_threads
    def snapshot():
        value=get_workers();snapshots.append(value);return value
    def denied(*args,**kwargs):raise AssertionError('native circle forwarded to original whole algorithm')
    def profile(frame,event,value):
        if event=='c_return'and value is entry:native_returns.append('circle')
        if event=='return'and frame.f_code is owned.circle_perimeter.__code__:
            admissions.append(frame.f_locals.get('native_result')is not None)
    public=importlib.import_module('skimage.draw');module=importlib.import_module('skimage.draw.draw')
    compiled=importlib.import_module('skimage.draw._draw')
    try:
        with monkeypatch.context()as patch:
            patch.setattr(public,'circle_perimeter',denied);patch.setattr(module,'circle_perimeter',denied)
            if accepted:patch.setattr(compiled,'_circle_perimeter',denied)
            patch.setattr(owned._core,'get_num_threads',snapshot)
            sys.setprofile(profile)
            try:actual=observe(xx.draw.circle_perimeter,spec,np)
            finally:sys.setprofile(old_profile)
        assert actual==expected
        if accepted:
            assert native_returns==['circle']and admissions==[True]and snapshots==[workers]
        else:assert not any(admissions)
        assert xx.get_num_threads()==workers
    finally:
        sys.setprofile(old_profile);xx.set_num_threads(previous);xx.set_fallback_enabled(fallback)


@pytest.mark.parametrize('method',('bresenham','andres'))
@pytest.mark.parametrize('workers',(1,4))
def test_circle_perimeter_long_native_admission(method,workers,monkeypatch):
    import numpy as np
    import skimagexx as xx
    from skimagexx import _circle_perimeter as owned
    compiled=importlib.import_module('skimage.draw._draw')
    def denied(*args,**kwargs):raise AssertionError('long circle left owned native partition')
    old=xx.get_num_threads();fallback=xx.set_fallback_enabled(False);xx.set_num_threads(workers)
    try:
        with monkeypatch.context()as patch:
            patch.setattr(compiled,'_circle_perimeter',denied)
            result=xx.draw.circle_perimeter(0,0,32768,method)
        assert isinstance(result,tuple)and len(result)==2 and len(result[0])>50000
        assert all(value.dtype==np.dtype(np.intp)and value.flags.owndata and value.base is None for value in result)
        assert not np.shares_memory(*result)and xx.get_num_threads()==workers
    finally:xx.set_num_threads(old);xx.set_fallback_enabled(fallback)
