"""Source-only contract for the boundary layout component; no target imports."""
import ast
import hashlib
import json
from pathlib import Path

from tools.mark_lbp_fp_context import predecessor_bytes

ROOT=Path(__file__).resolve().parents[1]
DATA=json.loads((ROOT/'coverage/contracts/find-boundaries-f-row-optimization-v1.json').read_bytes())


def component_source_bytes(name):
    return predecessor_bytes(ROOT, name, (ROOT / name).read_bytes())


def historical_driver_bytes(raw):
    name = 'benchmarks/find_boundaries.py'
    for version, pin, key in (
        (5, '50755fc357ea074f77178f0d58d49aa9edb2a9434512a30c8b460f523d8a0454', 'f_row_successor'),
        (4, '8ad09d6c73f47d017a5da741d60e18049abe6425042725e29c6e233eab2e8ff7', 'layout_successor'),
    ):
        encoded = (ROOT / f'coverage/contracts/find-boundaries-benchmark-binding-v{version}.json').read_bytes()
        assert hashlib.sha256(encoded).hexdigest() == pin
        current = json.loads(encoded)
        assert hashlib.sha256(raw).hexdigest() == current['source_closure'][name]
        row = current[key]
        predecessor = (ROOT / row['contract']).read_bytes()
        assert hashlib.sha256(predecessor).hexdigest() == row['sha256']
        for fragment in row['driver_inverse']:
            after, before = fragment['current'].encode(), fragment['predecessor'].encode()
            assert after and raw.count(after) == 1
            raw = raw.replace(after, before, 1)
        assert hashlib.sha256(raw).hexdigest() == json.loads(predecessor)['source_closure'][name]
    return raw


def test_layout_source_closure_and_exact_shipping_inverse():
    for name, pin in DATA['source_files_raw_sha256'].items():
        assert hashlib.sha256(component_source_bytes(name)).hexdigest() == pin, name
    raw = component_source_bytes('src/find_boundaries.cpp').replace(b'\r\n', b'\n')
    for fragment in DATA['shipping_inverse_fragments']:
        current=fragment['current'].encode();assert raw.count(current)==1
        raw=raw.replace(current,fragment['shipping'].encode(),1)
    assert hashlib.sha256(raw).hexdigest()==DATA['shipping_leaf_canonical_sha256']=='9f910ba308ab90189c6a8c6386a1fbf40ffab9093437ad1f0dfad392e2c53885'


def test_frozen_public_cases_method_and_losses_are_preserved():
    old=ROOT/DATA['benchmark_predecessor']['contract'];raw=old.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==DATA['benchmark_predecessor']['raw_sha256']
    previous=json.loads(raw);assert DATA['cases']==previous['cases']and len(DATA['cases'])==8
    assert DATA['method_sha256']==previous['method_sha256']
    for name, pin in DATA['unchanged_benchmark_files_raw_sha256'].items():
        raw = component_source_bytes(name)
        if name == 'benchmarks/find_boundaries.py':
            raw = historical_driver_bytes(raw)
        assert hashlib.sha256(raw).hexdigest() == pin, name
    text=(ROOT/'tools/montage_sustained.py').read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='measure')
    assert hashlib.sha256((ast.get_source_segment(text,node)+'\n').encode()).hexdigest()==DATA['method_sha256']
    result=ROOT/DATA['historical_public_result']['path'];assert hashlib.sha256(result.read_bytes()).hexdigest()==DATA['historical_public_result']['raw_sha256']
    assert [r['case_id']for r in json.loads(result.read_bytes())['results']if r['speedup']<1]==DATA['historical_public_result']['losses']


def test_component_gate_and_unchanged_API_frontier():
    assert DATA['status']=='COMPONENT_ONLY_PENDING_INDEPENDENT_REVIEW_GENUINE_SOURCE_BINDING_AND_PERFORMANCE_GO'
    assert DATA['original_calls_authorized']==DATA['public_original_comparison_runs']==0
    header = component_source_bytes('include/skimagexx/find_boundaries.hpp')
    assert hashlib.sha256(header).hexdigest() == DATA['unchanged_header_raw_sha256']
    old=json.loads((ROOT/DATA['benchmark_predecessor']['contract']).read_bytes())
    assert old['source_closure']['src/find_boundaries.cpp']!=DATA['source_files_raw_sha256']['src/find_boundaries.cpp']
    assert DATA['active_benchmark_still_frozen_and_will_refuse_changed_source']is True
    assert DATA['assessed_domain_expansion']is False


def test_F_route_only_changes_metadata_and_frozen_driver_stays_closed(monkeypatch):
    import runpy
    import pytest
    raw = component_source_bytes('src/find_boundaries.cpp').replace(b'\r\n', b'\n').decode()
    fragment=DATA['shipping_inverse_fragments'][0]
    predecessor=raw.replace(fragment['current'],fragment['shipping'],1)
    assert hashlib.sha256(predecessor.encode()).hexdigest()==DATA['predecessor_leaf_canonical_sha256']
    route=fragment['current']
    assert 'return run<T,Boolean>'in route and 'get_num_threads'not in route
    assert 'shape[0]>=3 && shape[1]>=3'in route and 'if constexpr(sizeof(T)<=4)'in route
    driver = runpy.run_path(str(ROOT / 'benchmarks/find_boundaries.py'))
    active = json.loads(driver['CONTRACT'].read_bytes())
    driver['validate_source_closure'](active)

    leaf = ROOT / 'src/find_boundaries.cpp'
    read_bytes = Path.read_bytes

    def changed_leaf(path):
        raw = read_bytes(path)
        return raw + b'\n// unadmitted boundary source change\n' if path == leaf else raw

    monkeypatch.setattr(Path, 'read_bytes', changed_leaf)
    with pytest.raises(AssertionError) as error:
        driver['validate_source_closure'](active)
    assert error.value.args == ('src/find_boundaries.cpp',)
