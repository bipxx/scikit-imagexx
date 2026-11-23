"""Pure current integration gates; no scientific imports or algorithm bodies."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
from variation_of_information_source_context import provider_predecessor_source
ROOT=Path(__file__).resolve().parents[1]
CONTEXT=ROOT/'coverage/contracts/vi-reduction-integration-v2.json'
EXPECTED_CONTEXT_SHA256="4554220569f6be2f49e0b2e169f168d6ab9ed1d5c899e3d3dee5f0cd979eec6c"


def test_exact_correction_and_historical_source_projection():
    assert hashlib.sha256(CONTEXT.read_bytes()).hexdigest()==EXPECTED_CONTEXT_SHA256
    data=json.loads(CONTEXT.read_bytes())
    from tools.join_source_context import correction_predecessor_bytes
    for name,row in data['sources'].items():
        raw=correction_predecessor_bytes(ROOT,name,(ROOT/name).read_bytes())
        assert hashlib.sha256(raw).hexdigest()==row['predecessor_canonical_lf_sha256']
    raw=provider_predecessor_source(ROOT,data['native_source']['path'])
    assert hashlib.sha256(raw).hexdigest()==data['native_source']['raw_sha256']
    old=correction_predecessor_bytes(ROOT,data['native_source']['path'],(ROOT/data['native_source']['path']).read_bytes())
    assert hashlib.sha256(old).hexdigest()=='966f48ec3ce563548aaa23d4f3ede7ac69354d91bab3b2fd4ffef7d2c2e35955'


@pytest.mark.parametrize('profile',('minimum','modern','debug','windows'))
def test_actual_qualification_source_census_and_large_group(profile):
    spec=importlib.util.spec_from_file_location('_corrected_qualifier',ROOT/'tools/public_qualification.py')
    q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
    value=q.plan(ROOT,profile,308)
    assert(value['registry']['image_count'],value['registry']['registered_count'],value['product_members'],value['SDK_members'],len(value['SDK_families']),value['SDK_checks'])==(308,311,286,127,57,114)
    group='public-VI-large-saved-linux'
    if profile=='windows':
        assert group not in value['public_groups']
        assert value['public_groups']['public-VI-current-windows']==('test_vi_public_matmul_windows_public.py',)
        assert 'public-VI-current-public-linux' not in value['public_groups']
    else:
        assert value['public_groups'][group]==('test_variation_of_information_large_saved.py','test_vi_public_log2_provider_large.py')
        assert value['public_groups']['public-VI-current-public-linux']==('test_vi_public_matmul_provider_public.py',)
        assert 'public-VI-current-windows' not in value['public_groups']
    assert q.CURRENT_SCOPE_CONTRACT=='coverage/contracts/current-public-qualification-v19.json'
    large=json.loads((ROOT/q.CURRENT_SCOPE_CONTRACT).read_bytes())['VI_current_runtime_large_reference']
    assert(large['legacy_nodes_per_mode'],large['current_nodes_per_mode'])==(8,8)
    consumer=(ROOT/'tests/cmake_consumer/variation_of_information_enabled_consumer.cpp').read_text()
    assert 'vi_current_counts_controls()'in consumer and 'vi_large_saved_controls()'in consumer
    assert 'if(!accepted||result.captured_workers!=budget)'in consumer


def test_saved_historical_contexts_remain_immutable():
    # Earlier reviews stay history; no current word parity is inferred from them.
    legacy=(ROOT/'coverage/contracts/current-public-qualification-v6.json').read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(legacy).hexdigest()=='95f9dc56c2ad38e3c4dc3cafa10c14e23a4e3b38b370e2c4071f9c3342113cef'
    old=json.loads((ROOT/'coverage/contracts/abi-vi-cumulative-integration-v1.json').read_bytes())
    assert old['predecessor_source']=='8c4fb4307746a0eb5fac5d125b6384fa80e88621'
    current=json.loads((ROOT/'coverage/contracts/variation-of-information-benchmark-execution-v4.json').read_bytes())
    assert current['cases']==json.loads((ROOT/'coverage/contracts/variation-of-information-benchmark-execution-v3.json').read_bytes())['cases']
    assert all(v is None for v in current['unbound'].values())
