"""Draw contracts and fixture safety, valid with clean or inherited imports."""
import ast
import base64
import builtins
from contextlib import contextmanager
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import zlib
import pytest
if __package__:
    from .draw_coordinates_contract import APIS,MAX_OUTPUT_ELEMENTS,UNASSESSED,cases,fixture,benchmark_cases
    from .draw_coordinates_observer import SOURCE_SHA,SIGNATURES,BINARY_SHA,FROZEN_FUNCTION_BODY_SHA256,scientific_profile,reference_runtime,encode,observe
else:
    from draw_coordinates_contract import APIS,MAX_OUTPUT_ELEMENTS,UNASSESSED,cases,fixture,benchmark_cases
    from draw_coordinates_observer import SOURCE_SHA,SIGNATURES,BINARY_SHA,FROZEN_FUNCTION_BODY_SHA256,scientific_profile,reference_runtime,encode,observe

CASES=cases();MATRIX=Path(__file__).parents[1]/'coverage/contracts/draw-coordinates.json'


@contextmanager
def without_draw_activity():
    """Scope the safety assertion to fixture work, preserving inherited modules."""
    modules={n:m for n,m in sys.modules.items()if n=='skimage.draw'or n.startswith('skimage.draw.')}
    previous=sys.getprofile();original_import=builtins.__import__;original_dynamic=importlib.import_module
    def guarded_import(name,globals=None,locals=None,fromlist=(),level=0):
        assert not(name=='skimage.draw'or name.startswith('skimage.draw.')or name=='skimage'and 'draw'in(fromlist or())),'fixture attempted draw import'
        return original_import(name,globals,locals,fromlist,level)
    def guarded_dynamic(name,package=None):
        assert not(name=='skimage.draw'or name.startswith('skimage.draw.')),'fixture attempted draw import'
        return original_dynamic(name,package)
    def profile(frame,event,value):
        module=frame.f_globals.get('__name__','')if event=='call'else (getattr(value,'__module__','')or '')if event=='c_call'else ''
        assert not(module=='skimage.draw'or module.startswith('skimage.draw.')),'fixture attempted draw call'
    builtins.__import__=guarded_import;importlib.import_module=guarded_dynamic;sys.setprofile(profile)
    try:yield
    finally:
        sys.setprofile(previous);builtins.__import__=original_import;importlib.import_module=original_dynamic
    assert {n:m for n,m in sys.modules.items()if n=='skimage.draw'or n.startswith('skimage.draw.')}==modules


def test_draw_coordinates_prospective_manifest():
    matrix=json.loads(MATRIX.read_bytes());assert matrix['apis']==list(APIS)and matrix['cases']==CASES
    assert len(CASES)==88 and len({s['id']for s in CASES})==88
    assert matrix['signatures']==SIGNATURES and matrix['source_canonical_lf_sha256']==SOURCE_SHA and matrix['binary_sha256']==BINARY_SHA
    assert matrix['native_implementation']=='pending'and matrix['unassessed']==list(UNASSESSED)and matrix['benchmark_cases']==benchmark_cases()
    for name,pin in matrix['source_freeze'].items():assert hashlib.sha256((MATRIX.parents[2]/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin


@pytest.mark.parametrize('spec',CASES,ids=[s['id']for s in CASES])
def test_draw_coordinates_fixture_without_original(spec):
    np=scientific_profile()
    with without_draw_activity():
        args,options,events,buffers=fixture(spec,np)
        assert not events and all(value.size<=32 for value in buffers.values());json.dumps(encode((args,options,buffers),np),allow_nan=False)


@pytest.mark.parametrize('profile',('1.24.4','2.5.4'))
def test_draw_coordinates_saved_original_rows(profile):
    rows=json.loads(MATRIX.read_bytes())['original_observations']
    if not rows:pytest.skip('Prospective only; original capture awaits review')
    observed=rows[profile];assert len(observed)==len(CASES)and[r['id']for r in observed]==[s['id']for s in CASES]
    for row in observed:
        assert all(row['checks'].values());pending=[row]
        while pending:
            value=pending.pop()
            if isinstance(value,dict):
                if 'raw_zlib_base64'in value:assert hashlib.sha256(zlib.decompress(base64.b64decode(value['raw_zlib_base64']))).hexdigest()==value['raw_sha256']
                pending.extend(value.values())
            elif isinstance(value,list):pending.extend(value)


@pytest.mark.parametrize('spec',CASES,ids=[s['id']for s in CASES])
@pytest.mark.parametrize('workers',(1,4))
def test_draw_coordinates_native_public_contract(spec,workers):
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    if spec['api']not in native_functions():pytest.skip('Native coordinate generation pending')
    import numpy as np
    expected=next(r for r in json.loads(MATRIX.read_bytes())['original_observations'][np.__version__]if r['id']==spec['id'])
    old=xx.get_num_threads();fallback=xx.set_fallback_enabled(False);xx.set_num_threads(workers)
    try:
        call=getattr(xx.draw,spec['api'].split('.')[1]);assert str(inspect.signature(call))==SIGNATURES[spec['api']]
        assert observe(call,spec,np)==expected
    finally:xx.set_num_threads(old);xx.set_fallback_enabled(fallback)


def test_draw_coordinate_fixture_preserves_inherited_original_imports():
    np=scientific_profile();module=importlib.import_module('skimage.draw.draw')
    with without_draw_activity():
        spec=next(s for s in CASES if s['id']=='line/horizontal')
        args,options,events,buffers=fixture(spec,np)
        assert args==(2,1,2,8)and not options and not events and not buffers
    assert sys.modules['skimage.draw.draw']is module


def test_fixture_guard_rejects_draw_import_before_importing():
    with pytest.raises(AssertionError,match='fixture attempted draw import'):
        with without_draw_activity():importlib.import_module('skimage.draw.draw')


def test_fixture_guard_rejects_synthetic_draw_call_before_body():
    namespace={'__name__':'skimage.draw.guard_standin'}
    exec("def standin(): raise RuntimeError('standin body must not execute')",namespace)
    with pytest.raises(AssertionError,match='fixture attempted draw call'):
        with without_draw_activity():namespace['standin']()


def test_draw_observer_function_bodies_preserve_original_capture():
    matrix=json.loads(MATRIX.read_bytes());migration=matrix['observer_migration']
    assert migration['original_test_file_sha256']=='bca3c48e1b619e7bdfba40e38cf85e5873189ddcedf49c532bfe48d3df3398de'
    assert migration['original_capture_commit']=='66ba107'and migration['function_body_sha256']==FROZEN_FUNCTION_BODY_SHA256
    for name,pin in FROZEN_FUNCTION_BODY_SHA256.items():
        assert hashlib.sha256(inspect.getsource(globals()[name]).rstrip('\n').encode()).hexdigest()==pin
