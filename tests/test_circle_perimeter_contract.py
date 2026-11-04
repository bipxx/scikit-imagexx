"""Inert contract and frozen-original checks; no draw algorithm replay."""
import base64
import builtins
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import zlib
import pytest
from circle_perimeter_contract import API,MAX_OUTPUT_ELEMENTS,UNASSESSED,cases,fixture,benchmark_cases
from circle_perimeter_observer import SIGNATURE,SOURCE_SHA,BINARY_SHA,CYTHON_SOURCE_SHA,scientific_profile,encode

MATRIX=Path(__file__).parents[1]/'coverage/contracts/circle-perimeter.json'
CASES=cases()


@contextmanager
def no_original_imports():
    old=builtins.__import__;dynamic=importlib.import_module
    def checked(name,*args,**kwargs):
        assert not(name=='skimage'or name.startswith('skimage.')),'fixture attempted original import'
        return old(name,*args,**kwargs)
    def checked_dynamic(name,*args,**kwargs):
        assert not(name=='skimage'or name.startswith('skimage.')),'fixture attempted original import'
        return dynamic(name,*args,**kwargs)
    builtins.__import__=checked;importlib.import_module=checked_dynamic
    try:yield
    finally:builtins.__import__=old;importlib.import_module=dynamic


def test_circle_perimeter_prospective_manifest():
    matrix=json.loads(MATRIX.read_bytes())
    assert matrix['apis']==[API]and matrix['cases']==CASES
    assert len(CASES)==73 and len({case['id']for case in CASES})==73
    assert matrix['signature']==SIGNATURE and matrix['source_canonical_lf_sha256']==SOURCE_SHA
    assert matrix['cython_source_canonical_lf_sha256']==CYTHON_SOURCE_SHA and matrix['binary_sha256']==BINARY_SHA
    assert matrix['benchmark_cases']==benchmark_cases()and matrix['unassessed'][:len(UNASSESSED)]==list(UNASSESSED)
    assert matrix['native_implementation']=='pending'
    for name,pin in matrix['source_freeze'].items():
        assert hashlib.sha256((MATRIX.parents[2]/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin


@pytest.mark.parametrize('spec',CASES,ids=[case['id']for case in CASES])
def test_circle_perimeter_fixture_is_inert(spec):
    np=scientific_profile()
    with no_original_imports():
        args,options,events,buffers=fixture(spec,np)
        assert len(args)==3 and not events
        assert all(value.size<=2 for value in buffers.values())
        json.dumps(encode((args,options,buffers),np),allow_nan=False)


@pytest.mark.parametrize('profile',('1.24.4','2.5.4'))
def test_circle_perimeter_saved_rows(profile):
    matrix=json.loads(MATRIX.read_bytes())
    if not matrix['original_observations']:pytest.skip('Prospective original-only capture not run yet')
    rows=matrix['original_observations'][profile]
    excluded=set(matrix['capture_safety']['excluded_ids'])
    assert len(rows)==69 and [row['id']for row in rows]==[case['id']for case in CASES if case['id']not in excluded]
    assert matrix['capture']['original_calls_started_total']==140 and matrix['capture']['completed_observations_total']==138
    assert matrix['capture_safety']['original_calls_replayed']==0
    assert sum(row['error']is None for row in rows)==59 and sum(row['error']is not None for row in rows)==10
    for row in rows:
        assert row['api']==API and all(row['checks'].values())
        assert row['input_before']==row['input_after']
        pending=[row]
        while pending:
            value=pending.pop()
            if isinstance(value,dict):
                if 'raw_zlib_base64'in value:
                    raw=zlib.decompress(base64.b64decode(value['raw_zlib_base64']))
                    assert hashlib.sha256(raw).hexdigest()==value['raw_sha256']
                pending.extend(value.values())
            elif isinstance(value,list):pending.extend(value)


def test_circle_perimeter_fixture_import_guard():
    with no_original_imports():
        with pytest.raises(AssertionError,match='fixture attempted original import'):
            importlib.import_module('skimage.draw')
