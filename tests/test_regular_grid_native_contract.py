"""Source/dependency closure and literal SDK extraction; no scientific calls."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def test_regular_grid_complete_dependency_closure():
    data=json.loads((ROOT/'coverage/contracts/regular-grid-native-v1.json').read_bytes())
    assert data['API_completion'] is False and data['shared_consumer_unchanged'] is True
    for group in ('source_raw_sha256','reused_raw_sha256'):
        for name,pin in data[group].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==pin,name
    assert 'python/skimagexx/_regular_seeds.py' in data['reused_raw_sha256']
    assert 'tests/regular_grid_sdk_fixture.py' in data['source_raw_sha256']
    assert 'tests/fixtures/regular_grid_originals_v1.hpp' in data['source_raw_sha256']
    assert 'tests/cmake_consumer/regular_grid_consumer.hpp' in data['source_raw_sha256']


def test_regular_grid_sdk_literal_fixture():
    from regular_grid_sdk_fixture import render
    assert (ROOT/'tests/fixtures/regular_grid_originals_v1.hpp').read_text()==render()
