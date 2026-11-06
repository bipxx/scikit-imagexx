"""Source-only admission of the frozen layout component into shared integration."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import runpy
import pytest
ROOT=Path(__file__).resolve().parents[1]


def test_layout_integrated_source_and_historical_driver_are_exact():
    driver=runpy.run_path(str(ROOT/'benchmarks/find_boundaries.py'))
    current=json.loads(driver['CONTRACT'].read_bytes());driver['validate_source_closure'](current)
    data=json.loads((ROOT/'coverage/contracts/find-boundaries-layout-optimization-v1.json').read_bytes())
    for name,pin in data['source_files_raw_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==pin,name
    raw=(ROOT/'src/find_boundaries.cpp').read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(driver['layout_predecessor']('src/find_boundaries.cpp',raw)).hexdigest()==data['shipping_leaf_canonical_sha256']
    previous=json.loads((ROOT/current['layout_successor']['contract']).read_bytes())
    assert current['cases']==data['cases']==previous['cases'] and len(current['cases'])==8
    assert current['sampler']==previous['sampler'] and current['method_sha256']==data['method_sha256']
    for name,pin in data['unchanged_benchmark_files_raw_sha256'].items():
        raw=(ROOT/name).read_bytes()
        if name=='benchmarks/find_boundaries.py':
            for f in current['layout_successor']['driver_inverse']:
                assert raw.count(f['current'].encode())==1
                raw=raw.replace(f['current'].encode(),f['predecessor'].encode(),1)
        assert hashlib.sha256(raw).hexdigest()==pin,name
    # Historical v3 refuses current leaf/helper; no overwritten contract or raw.
    for name in ('src/find_boundaries.cpp','tests/cmake_consumer/find_boundaries_consumer.hpp'):
        assert current['source_closure'][name]!=previous['source_closure'][name]
    assert all(current[k]is None for k in ('shared_CMake_sha256','genuine_core_sha256','prepared_run','prepared_artifact'))


@pytest.mark.parametrize('name',('src/find_boundaries.cpp','tests/cmake_consumer/find_boundaries_consumer.hpp','coverage/contracts/find-boundaries-layout-source-context-v1.json'))
def test_layout_source_tamper_refused(name):
    driver=runpy.run_path(str(ROOT/'benchmarks/find_boundaries.py'));c=json.loads(driver['CONTRACT'].read_bytes());c['source_closure'][name]='0'*64
    with pytest.raises(AssertionError):driver['validate_source_closure'](c)


def test_layout_future_selectors_and_full_grid_domain_are_retained():
    q=runpy.run_path(str(ROOT/'tools/public_qualification.py'))
    assert 'test_find_boundaries_layout_public.py'in q['GROUPS']['public-consumers']
    assert 'test_find_boundaries_layout_integration.py'in q['POST203_TESTS']
    for name in ('tests/test_find_boundaries_layout.cpp','tests/test_find_boundaries_layout_public.py','tests/test_find_boundaries_layout_contract.py','tests/test_find_boundaries_layout_integration.py','coverage/contracts/find-boundaries-layout-optimization-v1.json'):
        assert q['DRAW_SOURCE_PATHS'].count(name)==1
    assert q['applicable_public_groups'](q['GROUPS'],'minimum')['public-regular-grid-saved-linux']==('test_regular_grid_public.py',)
    assert 'public-regular-grid-saved-linux'not in q['applicable_public_groups'](q['GROUPS'],'windows')
    workflow=(ROOT/'.github/workflows/public-qualification.yml').read_text()
    assert 'tests/test_find_boundaries_layout_integration.py' in workflow
