"""Prospective-only controller intake; saved validation never recaptures originals."""
import base64
import hashlib
import json
import math
from pathlib import Path
import zlib
import pytest
if __package__:
    from .line_nd_contract import API,MAX_DIMENSIONS,MAX_POINTS,MAX_OUTPUT_ELEMENTS,UNASSESSED,cases,fixture,source_safe
    from .line_nd_observer import observe,encode,scientific_profile,SOURCE_SHA,LINSPACE_SOURCE_SHA,SIGNATURE
    from .test_draw_coordinates_public import without_draw_activity
else:
    from line_nd_contract import API,MAX_DIMENSIONS,MAX_POINTS,MAX_OUTPUT_ELEMENTS,UNASSESSED,cases,fixture,source_safe
    from line_nd_observer import observe,encode,scientific_profile,SOURCE_SHA,LINSPACE_SOURCE_SHA,SIGNATURE
    from test_draw_coordinates_public import without_draw_activity
ROOT = Path(__file__).parents[1]
MATRIX = ROOT/'coverage/contracts/line-nd.json'
CASES = cases()


def test_line_nd_prospective_matrix_and_source_safety():
    matrix = json.loads(MATRIX.read_bytes())
    assert matrix['api'] == API and matrix['signature'] == SIGNATURE
    assert matrix['cases'] == CASES and len(CASES) == len({s['id'] for s in CASES}) == 54
    assert matrix['original_source_sha256'] == SOURCE_SHA and matrix['numpy_linspace_source_sha256'] == LINSPACE_SOURCE_SHA
    assert matrix['limits'] == dict(max_dimensions=MAX_DIMENSIONS,max_points=MAX_POINTS,max_output_elements=MAX_OUTPUT_ELEMENTS)
    assert matrix['unassessed'] == list(UNASSESSED)
    assert matrix['numerical_policy']['max_ulp'] == 0
    for spec in CASES:
        source_safe(spec)
    for name,pin in matrix['source_freeze'].items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() == pin


@pytest.mark.parametrize('spec',CASES,ids=[s['id'] for s in CASES])
def test_line_nd_actual_controller_original_refused_intake(spec):
    np = scientific_profile()
    calls = []
    def refused(*args,**options):
        calls.append((args,options))
        raise RuntimeError('line-nd-original-refused-before-capture')
    with without_draw_activity():
        args,options,events,buffers = fixture(spec,np)
        assert not events and all(array.size <= 8 for array in buffers.values())
        json.dumps(encode((args,options,buffers),np),allow_nan=False)
        row = observe(refused,spec,np)
    assert len(calls) == 1 and row['result'] is None
    assert row['error'] == ['builtins','RuntimeError','line-nd-original-refused-before-capture']
    assert row['events'] == dict(container='list',items=[])
    assert not row['warnings'] and all(row['checks'].values())


@pytest.mark.parametrize('profile',('1.24.4','2.5.4'))
def test_line_nd_saved_original_rows_without_replay(profile):
    matrix = json.loads(MATRIX.read_bytes())
    if not matrix['original_observations']:
        pytest.skip('STOP: prospective intake only; separate root capture authorization required')
    rows = matrix['original_observations'][profile]
    assert len(rows) == 54 and [r['id'] for r in rows] == [s['id'] for s in CASES]
    for row in rows:
        assert row['api'] == API and all(row['checks'].values())
        pending = [row]
        while pending:
            value = pending.pop()
            if isinstance(value,dict):
                if 'raw_zlib_base64' in value:
                    raw = zlib.decompress(base64.b64decode(value['raw_zlib_base64']))
                    assert hashlib.sha256(raw).hexdigest() == value['raw_sha256']
                    assert len(raw) == math.prod(value['shape'])*int(value['dtype'][2:])
                pending.extend(value.values())
            elif isinstance(value,list):
                pending.extend(value)
