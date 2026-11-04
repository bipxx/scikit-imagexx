"""Source/reference metadata only: this file never calls original regionprops."""
import base64
import hashlib
import json
from pathlib import Path
import zlib
import pytest
if __package__:
    from .regionprops_state_contract import API, ALIASES, PROPERTIES, UNASSESSED, cases, source_safe
    from .regionprops_state_observer import SIGNATURE, SOURCE_SHA, UTILS_SHA
else:
    from regionprops_state_contract import API, ALIASES, PROPERTIES, UNASSESSED, cases, source_safe
    from regionprops_state_observer import SIGNATURE, SOURCE_SHA, UTILS_SHA
ROOT = Path(__file__).parents[1]
MATRIX = ROOT/'coverage/contracts/regionprops-state-v1.json'
CASES = cases()


def test_regionprops_state_frozen_partition():
    matrix = json.loads(MATRIX.read_bytes())
    discovery = json.loads((ROOT/'coverage/contracts/regionprops-runtime-surface.json').read_bytes())
    assert matrix['api'] == API and matrix['cases'] == CASES
    assert matrix['signature'] == SIGNATURE and matrix['properties'] == list(PROPERTIES)
    assert matrix['aliases'] == ALIASES and matrix['unassessed'] == list(UNASSESSED)
    assert matrix['original_sources'] == {'_regionprops.py': SOURCE_SHA, '_regionprops_utils.py': UTILS_SHA}
    assert all(discovery['aliases'][a] == target for a, target in ALIASES.items())
    assert len(CASES) == len({s['id'] for s in CASES}) == 29
    assert matrix['native_credit'] == 0
    for name, pin in matrix['source_freeze'].items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == pin


@pytest.mark.parametrize('spec', CASES, ids=[s['id'] for s in CASES])
def test_regionprops_state_prospective_source_safety(spec):
    bounds = source_safe(spec)
    assert bounds['input_elements'] <= 64 and bounds['maximum_label'] <= 3
    assert set(s[0] for s in spec['access_sequence']) <= {'attr', 'item', 'iter', 'mutate_input'}
    assert all(name in PROPERTIES or name in ALIASES or name == 'not_a_property'
               for operation, name in spec['access_sequence'] if operation in ('attr', 'item'))


@pytest.mark.parametrize('profile', ('1.24.4', '2.5.4'))
def test_regionprops_state_saved_rows_without_replay(profile):
    matrix = json.loads(MATRIX.read_bytes())
    if not matrix['original_observations']:
        pytest.skip('Prospective only; original capture pending')
    rows = matrix['original_observations'][profile]
    assert [r['id'] for r in rows] == [s['id'] for s in CASES]
    for spec, row in zip(CASES, rows):
        assert all(row['checks'].values())
        assert (row['error'] is not None) == (spec['rejection'] is not None)
        if row['error'] is None:
            assert row['result_container'] == 'list'
            assert all(len(r['steps']) == len(spec['access_sequence']) for r in row['regions'])
            for region in row['regions']:
                for step in region['steps']:
                    expected_error = step['name'] == 'not_a_property' or (step['operation'] == 'attr' and step['name'] == 'Area')
                    assert (step['error'] is not None) == expected_error
        pending = [row]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                if 'raw_zlib_base64' in value:
                    assert hashlib.sha256(zlib.decompress(base64.b64decode(value['raw_zlib_base64']))).hexdigest() == value['raw_sha256']
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
