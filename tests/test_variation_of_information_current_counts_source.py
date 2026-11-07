"""Current dependency-overlay source controls only; no scientific imports."""
import hashlib
import json
from pathlib import Path
import pytest
from variation_of_information_source_context import historical_source
ROOT=Path(__file__).resolve().parents[1]
DATA=json.loads((ROOT/'coverage/contracts/variation-of-information-current-counts-v2.json').read_bytes())


@pytest.mark.parametrize('group',('owned_source_raw_sha256','unchanged_VI_raw_sha256'))
def test_exact_family_and_historical_VI_bytes(group):
    for path,digest in DATA[group].items():
        assert hashlib.sha256(historical_source(ROOT,path)).hexdigest()==digest,path


def test_current_dependency_scope_is_not_shared_or_whole105_claim():
    assert DATA['current_dependency_source']=='b87a24e4b39578b26cf8a2f9d3324ca59b8be6b3'
    assert DATA['dependency_domains']['src/label_pair_metrics.cpp']['current_raw']=='e8bc05ba11b1892d9cc8885c9e8d1373f3cfa715b57fd560221565d5889cd079'
    assert DATA['authority']==dict(original=0,capture=0,timing=0,hosted=0,genuine_binding=None)
    assert 'does NOT modify/import shared files'in DATA['overlay']
    assert 'Whole current105 SDK has not been built/qualified'in DATA['SDK']


def test_larger_controls_cover_logical_physical_and_worker_capture():
    source=(ROOT/'tests/variation_of_information_current_counts_controls.hpp').read_text()
    assert '68*72>n/4'in source and '32<=n/4'in source
    assert 'storage.data.size()!=n'in source and 'logical!=32'in source
    assert 'captured_workers!=captured'in source and 'explicit_one.captured_workers!=1'in source
    assert 'untouched.values[0]!=123.'in source and 'workers:{1,4,0}'in source
    assert 'std::memcmp(&a,&b,8)==0'in source
    public=(ROOT/'tests/test_variation_of_information_current_counts_public.py').read_text()
    assert 'data[:logical]'in public and 'data.size==indices.size==n'in public
    assert 'dense.tobytes()==fallback.tobytes()'in public
    assert 'Candidate'in DATA['candidate_equivalence']['accuracy_claim']


def test_original_fixtures_and_frozen_benchmark_scope_remain_exact():
    assert DATA['saved']['recipes_per_profile']==8 and DATA['saved']['numeric_words_cases']==7
    assert 'No new original gold'in DATA['candidate_equivalence']['accuracy_claim']
    assert DATA['predecessor_handoff_sha256']=='6a95087ca81d955aade678b4f6c303c7a3fc2aa36ac9375588d188406b67aaa2'
    assert DATA['independent_review_sha256']=='8186a89dbfbe3293a71067387a76e30d7d1aaec38525d547c2ac6b24688041de'
