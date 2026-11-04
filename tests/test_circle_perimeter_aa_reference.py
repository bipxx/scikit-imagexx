"""Source-safe intake and saved reference validation, without original replay."""
import base64
import hashlib
import json
import struct
import zlib
from pathlib import Path
import pytest
if __package__:
    from .circle_perimeter_aa_contract import API, MAX_OUTPUT_ELEMENTS, UNASSESSED, cases, fixture, source_safe, benchmark_cases
    from .circle_perimeter_aa_observer import observe, encode, scientific_profile, SOURCE_SHA, SOURCE_PYX_SHA, BINARY_SHA, SIGNATURE
    from .test_draw_coordinates_public import without_draw_activity
else:
    from circle_perimeter_aa_contract import API, MAX_OUTPUT_ELEMENTS, UNASSESSED, cases, fixture, source_safe, benchmark_cases
    from circle_perimeter_aa_observer import observe, encode, scientific_profile, SOURCE_SHA, SOURCE_PYX_SHA, BINARY_SHA, SIGNATURE
    from test_draw_coordinates_public import without_draw_activity
ROOT = Path(__file__).parents[1]
MATRIX = ROOT / 'coverage/contracts/circle-perimeter-aa.json'
CASES = cases()


def test_circle_perimeter_aa_prospective_source_safety():
    matrix = json.loads(MATRIX.read_bytes())
    assert matrix['api'] == API and matrix['cases'] == CASES
    assert len(CASES) == len({s['id'] for s in CASES}) == 48
    assert matrix['signature'] == SIGNATURE
    assert matrix['original_python_sha256'] == SOURCE_SHA
    assert matrix['original_pyx_sha256'] == SOURCE_PYX_SHA
    assert matrix['original_binary_sha256'] == BINARY_SHA
    assert matrix['unassessed'] == list(UNASSESSED)
    assert matrix['benchmark_cases'] == benchmark_cases()
    assert matrix['weight_policy']['max_ulp'] == 0
    assert matrix['native_implementation'] == 'pending'
    for spec in CASES:
        source_safe(spec)
    for spec in benchmark_cases():
        source_safe(spec, matrix['benchmark_max_output_elements'])
    for name, pin in matrix['source_freeze'].items():
        assert hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == pin


@pytest.mark.parametrize('spec', CASES, ids=[s['id'] for s in CASES])
def test_circle_perimeter_aa_controller_intake_originals_refused(spec):
    np = scientific_profile()
    calls = []
    def refused(*args, **options):
        calls.append((args, options))
        raise RuntimeError('circle-aa-original-refused-before-capture')
    with without_draw_activity():
        args, options, events, buffers = fixture(spec, np)
        assert not events and all(a.size <= 3 for a in buffers.values())
        json.dumps(encode((args, options, buffers), np), allow_nan=False)
        row = observe(refused, spec, np)
    assert len(calls) == 1
    assert row['error'] == ['builtins', 'RuntimeError', 'circle-aa-original-refused-before-capture']
    assert row['result'] is None and all(row['checks'].values())
    assert row['input_before'] == row['input_after']
    assert row['events']['items'] == []


@pytest.mark.parametrize('profile', ('1.24.4', '2.5.4'))
def test_circle_perimeter_aa_saved_original_rows(profile):
    matrix = json.loads(MATRIX.read_bytes())
    if not matrix['original_observations']:
        pytest.skip('Prospective only; safe once-only originals pending')
    rows = matrix['original_observations'][profile]
    assert len(rows) == 48 and [r['id'] for r in rows] == [s['id'] for s in CASES]
    for spec, row in zip(CASES, rows):
        assert row['api'] == API and all(row['checks'].values())
        assert row['input_before'] == row['input_after']
        assert not row['warnings']
        expected_events = ['r.__index__'] if spec['kind'] == 'provider-first' else ['r.__index__', 'c.__index__', 'radius.__index__'] if spec['kind'] == 'providers-all' else []
        assert row['events'] == dict(container='list', items=expected_events)
        if row['error'] is None:
            result = row['result']
            assert result['container'] == 'tuple' and len(result['items']) == 3
            arrays = result['items']
            assert [a['dtype'] for a in arrays] == ['<i8', '<i8', '<f8']
            assert len({tuple(a['shape']) for a in arrays}) == 1
            assert sum(a['shape'][0] for a in arrays) <= MAX_OUTPUT_ELEMENTS
            assert all(a['own'] and a['writeable'] and a['base_type'] == 'NoneType' for a in arrays)
            assert all(a['strides'] == ([8] if a['shape'][0] else [0]) for a in arrays)
            if spec['id'] in ('circle-aa/radius-0', 'circle-aa/radius--3', 'circle-aa/radius--1'):
                radius = spec['args'][2]
                expected = ([radius, 0, radius, 0, -radius, 0, -radius, 0], [0, radius, 0, -radius, 0, radius, 0, -radius], [1.] * 8)
                for array, values, format_code in zip(arrays, expected, ('q', 'q', 'd')):
                    assert zlib.decompress(base64.b64decode(array['raw_zlib_base64'])) == struct.pack('<8' + format_code, *values)
            assert all(not pair[1] for pair in row['output_input_aliases'])
            for i, pair in enumerate(row['output_pair_aliases']):
                assert pair[0] == (i // 3 == i % 3)
                if i // 3 != i % 3:
                    assert not pair[1]
        pending = [row]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                if 'raw_zlib_base64' in value:
                    raw = zlib.decompress(base64.b64decode(value['raw_zlib_base64']))
                    assert hashlib.sha256(raw).hexdigest() == value['raw_sha256']
                    assert len(raw) == value['shape'][0] * 8 if len(value['shape']) == 1 else True
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
