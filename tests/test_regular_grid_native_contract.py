"""Source/dependency closure and literal SDK extraction; no scientific calls."""
import hashlib
import json
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]


def test_regular_grid_complete_dependency_closure():
    data=json.loads((ROOT/'coverage/contracts/regular-grid-native-v2.json').read_bytes())
    assert data['API_completion'] is False and data['shared_consumer_unchanged'] is True
    predecessor=data['public_adapter_predecessor']
    assert predecessor['contract']=='coverage/contracts/regular-grid-native-v1.json'
    raw=(ROOT/predecessor['contract']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==predecessor['raw_sha256']=='58082bd2e38dc186a7dd4e7517d16004a434ae6d0d04ba1cdadc8e9523b61e9a'
    previous=json.loads(raw)
    raw=(ROOT/predecessor['context']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==predecessor['context_raw_sha256']
    context=json.loads(raw);assert context['path']=='tests/test_regular_grid_public.py'
    assert context['predecessor_source']=='47df67d4ae916701fdc15f5f9f536a4df9a58170'
    raw=(ROOT/context['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==context['current_raw_sha256']==data['source_raw_sha256'][context['path']]
    for fragment in context['inverse_fragments']:
        current,old=fragment['current'].encode(),fragment['predecessor'].encode()
        assert current and raw.count(current)==1
        raw=raw.replace(current,old,1)
    assert hashlib.sha256(raw).hexdigest()==context['predecessor_raw_sha256']==previous['source_raw_sha256'][context['path']]

    context=json.loads((ROOT/'coverage/contracts/next-util-benchmark-execution-v2.json').read_bytes())
    projection=runpy.run_path(str(ROOT/'benchmarks/next_util_execution.py'))['historical_bytes']
    for group in ('source_raw_sha256','reused_raw_sha256'):
        for name,pin in data[group].items():
            # Exact admitted inverse authenticates current bytes before restoring historical benchmark bytes.
            raw=projection(ROOT,name)if name in context['sources']else(ROOT/name).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==pin,name
    assert 'python/skimagexx/_regular_seeds.py' in data['reused_raw_sha256']
    assert 'tests/regular_grid_sdk_fixture.py' in data['source_raw_sha256']
    assert 'tests/fixtures/regular_grid_originals_v1.hpp' in data['source_raw_sha256']
    assert 'tests/cmake_consumer/regular_grid_consumer.hpp' in data['source_raw_sha256']


def test_regular_grid_sdk_literal_fixture():
    from regular_grid_sdk_fixture import render
    assert (ROOT/'tests/fixtures/regular_grid_originals_v1.hpp').read_text()==render()
