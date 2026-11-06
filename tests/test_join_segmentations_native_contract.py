"""Small source/dependency checks only; no scientific imports or target calls."""
import hashlib
import json
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]


def test_exact_source_dependency_closure():
    contract=json.loads((ROOT/'coverage/contracts/join-segmentations-native-v2.json').read_bytes())
    assert contract['api']=='segmentation.join_segmentations' and contract['API_completion'] is False
    context=json.loads((ROOT/'coverage/contracts/next-util-benchmark-execution-v2.json').read_bytes())
    projection=runpy.run_path(str(ROOT/'benchmarks/next_util_execution.py'))['historical_bytes']
    for group in ('source_raw_sha256','reused_raw_sha256'):
        for name,pin in contract[group].items():
            # Exact admitted inverse authenticates current bytes before restoring historical benchmark bytes.
            raw=projection(ROOT,name)if name in context['sources']else(ROOT/name).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==pin,name
    assert 'tests/fixtures/join_segmentations_originals_v1.hpp' in contract['source_raw_sha256']
    assert 'tests/fixtures/join_segmentations_originals_v1.json' in contract['reused_raw_sha256']


def test_original_packet_preserved():
    fixture=json.loads((ROOT/'tests/fixtures/join_segmentations_originals_v1.json').read_bytes())
    for name,pin in fixture['frozen_source_raw_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==pin,name
    assert fixture['actual_public_topcalls']==100 and fixture['native_calls']==fixture['timed_batches']==0
