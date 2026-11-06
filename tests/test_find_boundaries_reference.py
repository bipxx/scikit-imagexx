"""Pure source/safety/saved-data guards; installed intake has its own module."""
import ast
import base64
import hashlib
import json
import math
from pathlib import Path
import zlib
import pytest
from find_boundaries_contract import cases,logical_shape,output_bound,API,SIGNATURE,MAX_INPUT_ELEMENTS,MAX_OUTPUT_ELEMENTS
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/find-boundaries-v1.json'


def test_find_boundaries_prospective_scope_and_bounds():
    matrix=json.loads(CONTRACT.read_bytes());specs=cases()
    assert matrix['api']==API and matrix['public_signature']==SIGNATURE
    assert matrix['cases']==specs and len(specs)==len({s['id']for s in specs})==32
    assert {'thick','inner','outer','subpixel'}<={s['mode']for s in specs}
    assert {s['dtype']for s in specs}=={'bool','uint8','int8','uint16','int16','uint32','int32','uint64','int64'}
    for spec in specs:
        assert len(spec['values'])==math.prod(spec['shape']) and math.prod(spec['shape'])<=64
        assert 1<=len(spec['shape'])<=4
        assert max(spec['shape'])<=4
        shape=output_bound(spec)
        if any(n<0 for n in shape):assert spec['id'].endswith('empty-subpixel-error')
        else:assert math.prod(shape)<=MAX_OUTPUT_ELEMENTS
        assert math.prod(logical_shape(spec))<=MAX_INPUT_ELEMENTS


def test_find_boundaries_source_freeze_and_pure_boundary():
    matrix=json.loads(CONTRACT.read_bytes())
    for path,pin in matrix['source_freeze'].items():
        assert hashlib.sha256((ROOT/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin
    for path in ('tests/find_boundaries_contract.py','tests/find_boundaries_observer.py'):
        for node in ast.parse((ROOT/path).read_bytes()).body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[a.name for a in node.names]if isinstance(node,ast.Import)else[node.module or'']
                assert not any(n.split('.')[0]in('numpy','scipy','skimage','skimagexx')for n in names)
    assert matrix['original_observations']=={}or set(matrix['original_observations'])=={'1.24.4','2.5.4'}


def test_find_boundaries_saved_original_rows_without_replay():
    matrix=json.loads(CONTRACT.read_bytes())
    if not matrix['original_observations']:pytest.skip('Prospective; original capture not authorized by this test')
    def check(value):
        if isinstance(value,dict):
            if 'raw_zlib_base64'in value:
                raw=zlib.decompress(base64.b64decode(value['raw_zlib_base64']));assert hashlib.sha256(raw).hexdigest()==value['raw_sha256']
            for sub in value.values():check(sub)
        elif isinstance(value,list):
            for sub in value:check(sub)
    for rows in matrix['original_observations'].values():
        assert [r['id']for r in rows]==[s['id']for s in cases()]
        assert all(all(r['checks'].values())for r in rows)
        check(rows)
