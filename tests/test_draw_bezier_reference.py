"""Bezier prospective safety and saved observations; no implicit capture."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib
import pytest
if __package__:
    from .draw_bezier_contract import API, MAX_OUTPUT_ELEMENTS, cases, fixture
    from .draw_bezier_observer import encode, scientific_profile
    from .test_draw_coordinates_public import without_draw_activity
else:
    from draw_bezier_contract import API, MAX_OUTPUT_ELEMENTS, cases, fixture
    from draw_bezier_observer import encode, scientific_profile
    from test_draw_coordinates_public import without_draw_activity
ROOT = Path(__file__).parents[1]
MATRIX = ROOT/'coverage/contracts/draw-bezier-v1.json'
CASES = cases()


def test_bezier_prospective_cases_and_source_safety():
    matrix=json.loads(MATRIX.read_bytes())
    assert matrix['cases']==CASES and len(CASES)==len({s['id'] for s in CASES})==31
    assert not any(s['kind']=='shape-short' for s in CASES)
    assert matrix['max_output_elements']==MAX_OUTPUT_ELEMENTS==4096
    assert {'column-extremum-weighted','row-extremum-weighted','both-extrema-weighted','flat-quarter-weight'} <= {s['kind'] for s in CASES}
    for spec in CASES:
        for value in spec['args'][:6]:
            assert isinstance(value,str) or -3<=value<=21 or spec['kind']=='endpoint-overflow'
        assert isinstance(spec['args'][6],str) or spec['args'][6] in (0.,.25,.5,1.,2.)
        assert spec['shape'] is None or all(0<=n<=99 for n in spec['shape'])
    for path,pin in matrix['source_freeze'].items():
        assert hashlib.sha256((ROOT/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin
    sources={}
    for entry in matrix['reused_source_methods']:
        path=entry['path']
        if path not in sources:
            text=(ROOT/path).read_text(); sources[path]=(text,ast.parse(text))
        text,tree=sources[path]
        node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==entry['name'])
        start=min([node.lineno]+[n.lineno for n in node.decorator_list])
        body=''.join(text.splitlines(keepends=True)[start-1:node.end_lineno])
        assert hashlib.sha256(body.encode()).hexdigest()==entry['canonical_LF_sha256']
    source=(ROOT/'tests/draw_bezier_contract.py').read_text()
    assert not any(isinstance(n,ast.Import) and any(a.name.startswith(('skimage','scipy')) for a in n.names) for n in ast.parse(source).body)


@pytest.mark.parametrize('spec',CASES,ids=lambda s:s['id'])
def test_bezier_fixture_without_original_activity(spec):
    np=scientific_profile()
    with without_draw_activity():
        args,options,events,buffers=fixture(spec,np)
        assert not events and sum(a.size for a in buffers.values())<=4
        json.dumps(encode((args,options,buffers),np),allow_nan=False)


@pytest.mark.parametrize('profile',('1.24.4','2.5.4'))
def test_bezier_saved_original_rows_without_replay(profile):
    matrix=json.loads(MATRIX.read_bytes()); rows=matrix['original_observations']
    if not rows:
        pytest.skip('Prospective; no original capture authorized by this test')
    rows=rows[profile]
    assert [row['id'] for row in rows]==[s['id'] for s in CASES]
    for row in rows:
        assert all(row['checks'].values())
        pending=[row]
        while pending:
            value=pending.pop()
            if isinstance(value,dict):
                if 'raw_zlib_base64' in value:
                    assert hashlib.sha256(zlib.decompress(base64.b64decode(value['raw_zlib_base64']))).hexdigest()==value['raw_sha256']
                pending.extend(value.values())
            elif isinstance(value,list):
                pending.extend(value)
