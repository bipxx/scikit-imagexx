"""Source-only contract for the boundary layout component; no target imports."""
import ast
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=json.loads((ROOT/'coverage/contracts/find-boundaries-f-row-optimization-v1.json').read_bytes())


def test_layout_source_closure_and_exact_shipping_inverse():
    for name,pin in DATA['source_files_raw_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==pin,name
    raw=(ROOT/'src/find_boundaries.cpp').read_bytes().replace(b'\r\n',b'\n')
    for fragment in DATA['shipping_inverse_fragments']:
        current=fragment['current'].encode();assert raw.count(current)==1
        raw=raw.replace(current,fragment['shipping'].encode(),1)
    assert hashlib.sha256(raw).hexdigest()==DATA['shipping_leaf_canonical_sha256']=='9f910ba308ab90189c6a8c6386a1fbf40ffab9093437ad1f0dfad392e2c53885'


def test_frozen_public_cases_method_and_losses_are_preserved():
    old=ROOT/DATA['benchmark_predecessor']['contract'];raw=old.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==DATA['benchmark_predecessor']['raw_sha256']
    previous=json.loads(raw);assert DATA['cases']==previous['cases']and len(DATA['cases'])==8
    assert DATA['method_sha256']==previous['method_sha256']
    for name,pin in DATA['unchanged_benchmark_files_raw_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==pin
    text=(ROOT/'tools/montage_sustained.py').read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='measure')
    assert hashlib.sha256((ast.get_source_segment(text,node)+'\n').encode()).hexdigest()==DATA['method_sha256']
    result=ROOT/DATA['historical_public_result']['path'];assert hashlib.sha256(result.read_bytes()).hexdigest()==DATA['historical_public_result']['raw_sha256']
    assert [r['case_id']for r in json.loads(result.read_bytes())['results']if r['speedup']<1]==DATA['historical_public_result']['losses']


def test_component_gate_and_unchanged_API_frontier():
    assert DATA['status']=='COMPONENT_ONLY_PENDING_INDEPENDENT_REVIEW_GENUINE_SOURCE_BINDING_AND_PERFORMANCE_GO'
    assert DATA['original_calls_authorized']==DATA['public_original_comparison_runs']==0
    header=(ROOT/'include/skimagexx/find_boundaries.hpp').read_bytes();assert hashlib.sha256(header).hexdigest()==DATA['unchanged_header_raw_sha256']
    old=json.loads((ROOT/DATA['benchmark_predecessor']['contract']).read_bytes())
    assert old['source_closure']['src/find_boundaries.cpp']!=DATA['source_files_raw_sha256']['src/find_boundaries.cpp']
    assert DATA['active_benchmark_still_frozen_and_will_refuse_changed_source']is True
    assert DATA['assessed_domain_expansion']is False


def test_F_route_only_changes_metadata_and_frozen_driver_stays_closed():
    import runpy
    import pytest
    raw=(ROOT/'src/find_boundaries.cpp').read_text()
    fragment=DATA['shipping_inverse_fragments'][0]
    predecessor=raw.replace(fragment['current'],fragment['shipping'],1)
    assert hashlib.sha256(predecessor.encode()).hexdigest()==DATA['predecessor_leaf_canonical_sha256']
    route=fragment['current']
    assert 'return run<T,Boolean>'in route and 'get_num_threads'not in route
    assert 'shape[0]>=3 && shape[1]>=3'in route and 'if constexpr(sizeof(T)<=4)'in route
    driver=runpy.run_path(str(ROOT/'benchmarks/find_boundaries.py'))
    with pytest.raises(AssertionError)as error:
        driver['validate_source_closure'](json.loads((ROOT/DATA['benchmark_predecessor']['contract']).read_bytes()))
    assert error.value.args==('src/find_boundaries.cpp',)
