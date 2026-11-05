"""Prospective intake and saved parity; this suite never invokes originals."""
import base64
import hashlib
import json
import math
from pathlib import Path
import zlib
import pytest

if __package__:
    from .ellipsoid_contract import API, MAX_AXIS, MAX_GRID_AXIS, MAX_OUTPUT_ELEMENTS, UNASSESSED, cases, source_safe
    from .ellipsoid_observer import SIGNATURE, SOURCE_SHA, observe, scientific_profile
    from .test_draw_coordinates_public import without_draw_activity
else:
    from ellipsoid_contract import API, MAX_AXIS, MAX_GRID_AXIS, MAX_OUTPUT_ELEMENTS, UNASSESSED, cases, source_safe
    from ellipsoid_observer import SIGNATURE, SOURCE_SHA, observe, scientific_profile
    from test_draw_coordinates_public import without_draw_activity

ROOT = Path(__file__).parents[1]
MATRIX = ROOT / 'coverage/contracts/ellipsoid.json'
CASES = cases()


def test_ellipsoid_prospective_matrix_and_source_safety():
    matrix = json.loads(MATRIX.read_bytes())
    assert matrix['api'] == API and matrix['signature'] == SIGNATURE
    assert matrix['cases'] == CASES and len(CASES) == len({s['id'] for s in CASES}) == 28
    assert matrix['original_source_sha256'] == SOURCE_SHA
    assert matrix['limits'] == dict(max_axis=MAX_AXIS, max_grid_axis=MAX_GRID_AXIS, max_output_elements=MAX_OUTPUT_ELEMENTS)
    assert matrix['unassessed'] == list(UNASSESSED)
    for spec in CASES:
        assert source_safe(spec)['max_output_elements'] <= MAX_OUTPUT_ELEMENTS
    for name, pin in matrix['source_freeze'].items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == pin


@pytest.mark.parametrize('spec', CASES, ids=[s['id'] for s in CASES])
def test_ellipsoid_controller_original_refused_intake(spec):
    np = scientific_profile()
    calls = []
    def refused(*args, **options):
        calls.append((args, options))
        raise RuntimeError('ellipsoid-original-refused-before-capture')
    with without_draw_activity():
        row = observe(refused, spec, np)
    assert len(calls) == 1 and row['result'] is None
    assert row['error'] == ['builtins', 'RuntimeError', 'ellipsoid-original-refused-before-capture']
    assert not row['events'] and not row['warnings'] and all(row['checks'].values())


@pytest.mark.parametrize('profile', ('1.24.4', '2.5.4'))
def test_ellipsoid_saved_observations_without_replay(profile):
    matrix = json.loads(MATRIX.read_bytes())
    if not matrix['original_observations']:
        pytest.skip('Prospective only; capture requires reviewed controller and bounded source intake')
    rows = matrix['original_observations'][profile]
    assert [r['id'] for r in rows] == [s['id'] for s in CASES]
    assert sum(r['error'] is not None for r in rows) == 9
    for spec, row in zip(CASES, rows):
        assert row['api'] == API and all(row['checks'].values())
        assert not row['events'] and not row['warnings']
        if spec['kind'] == 'nonpositive-axis':
            assert row['error'] == ['builtins', 'ValueError', 'Parameters a, b, and c must all be > 0']
        if row['error'] is not None:
            assert row['result'] is None
            continue
        result = row['result']
        assert result['type'] == 'ndarray' and len(result['shape']) == 3
        assert result['dtype'] == ('<f8' if spec['levelset'] else '|b1')
        raw = zlib.decompress(base64.b64decode(result['raw_zlib_base64']))
        assert hashlib.sha256(raw).hexdigest() == result['raw_sha256']
        assert len(raw) == math.prod(result['shape']) * (8 if spec['levelset'] else 1)
        assert math.prod(result['shape']) <= source_safe(spec)['max_output_elements']
