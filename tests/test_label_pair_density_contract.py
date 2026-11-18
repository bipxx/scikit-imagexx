"""Source-only contract for the label-pair density component; no target imports."""
import ast
import hashlib
import json
from pathlib import Path

from tools.mark_lbp_fp_context import predecessor_bytes

ROOT=Path(__file__).resolve().parents[1]
DATA=json.loads((ROOT/'coverage/contracts/label-pair-density-optimization-v1.json').read_bytes())


def test_density_source_closure_and_exact_shipping_inverse():
    for name, pin in DATA['source_files_raw_sha256'].items():
        raw = (ROOT / name).read_bytes()
        if name in ('tests/test_label_pair_density_contract.py', 'src/execution.hpp', 'tests/cmake_consumer/label_pair_metrics_consumer.cpp', 'tests/contingency_preweighted_controls.hpp', 'tests/label_pair_metrics_native_controls.hpp', 'tests/test_label_pair_metrics_native.cpp', 'tests/test_label_pair_preweighted_native.cpp', 'tests/test_label_pair_rand_native.cpp'):
            raw = predecessor_bytes(ROOT, name, raw)
        assert hashlib.sha256(raw).hexdigest() == pin, name
    raw=(ROOT/'src/label_pair_metrics.cpp').read_bytes().replace(b'\r\n',b'\n')
    for fragment in DATA['shipping_inverse_fragments']:
        current=fragment['current'].encode();assert raw.count(current)==1
        raw=raw.replace(current,fragment['shipping'].encode(),1)
    assert hashlib.sha256(raw).hexdigest()==DATA['shipping_leaf_canonical_sha256']


def test_frozen_public_cases_method_and_losses_are_preserved():
    old=ROOT/DATA['benchmark_predecessor']['contract'];raw=old.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==DATA['benchmark_predecessor']['raw_sha256']
    previous=json.loads(raw);assert DATA['cases']==previous['cases']and len(DATA['cases'])==8
    assert DATA['method_sha256']==previous['method_sha256']
    for name, pin in DATA['unchanged_benchmark_files_raw_sha256'].items():
        raw = (ROOT / name).read_bytes()
        if name == 'tools/label_pair_benchmark_context.py':
            assert hashlib.sha256(raw).hexdigest() == '441f5f5e78ec566a7dad10854d05a65d6a9124c3819913af0e60355c2d32c4c0'
            # JOIN, density, and ABI integration added these exact source projections.
            for addition in (
                b"    import importlib.util\n"
                b"    helper_spec=importlib.util.spec_from_file_location('join_metric_source_context',Path(root)/'tools/join_source_context.py')\n"
                b"    helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)\n",
                b"        raw=helper.predecessor_bytes(root,path,raw)\n",
                b"        if path in ('src/label_pair_metrics.cpp','python/skimagexx/metrics.py'):raw=helper.predecessor_bytes(root,path,raw)\n",
            ):
                assert raw.count(addition) == 1
                raw = raw.replace(addition, b'', 1)
        assert hashlib.sha256(raw).hexdigest() == pin, name
    text=(ROOT/'tools/montage_sustained.py').read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='measure')
    assert hashlib.sha256((ast.get_source_segment(text,node)+'\n').encode()).hexdigest()==DATA['method_sha256']
    result=ROOT/DATA['historical_public_result']['path'];assert hashlib.sha256(result.read_bytes()).hexdigest()==DATA['historical_public_result']['raw_sha256']
    assert [r['case_id']for r in json.loads(result.read_bytes())['results']if r['speedup']<1]==DATA['historical_public_result']['losses']


def test_component_gate_and_unchanged_API_frontier():
    assert DATA['status']=='COMPONENT_ONLY_PENDING_INDEPENDENT_REVIEW_GENUINE_SOURCE_BINDING_AND_PERFORMANCE_GO'
    assert DATA['original_calls_authorized']==DATA['public_original_comparison_runs']==0
    header=(ROOT/'include/skimagexx/label_pair_metrics.hpp').read_bytes();assert hashlib.sha256(header).hexdigest()==DATA['unchanged_header_raw_sha256']
    old=json.loads((ROOT/DATA['benchmark_predecessor']['contract']).read_bytes())
    assert DATA['dense_gate']=={'maximum_cells':65536,'samples_per_cell_minimum':4,'weighted_requires_uniform_row_bits':True}
    assert DATA['active_benchmark_still_frozen_and_will_refuse_changed_source']is True
    assert DATA['assessed_domain_expansion']is False


def test_active_benchmark_refuses_changed_leaf_before_any_intake(monkeypatch):
    import runpy
    import pytest

    verifier = runpy.run_path(str(ROOT / 'tools/label_pair_benchmark_context.py'))['validate_source_closure']
    contract = json.loads((ROOT / DATA['benchmark_predecessor']['contract']).read_bytes())
    verifier(ROOT, contract)

    leaf = ROOT / 'src/label_pair_metrics.cpp'
    read_bytes = Path.read_bytes

    def changed_leaf(path):
        raw = read_bytes(path)
        return raw + b'\n// unadmitted density source change\n' if path == leaf else raw

    monkeypatch.setattr(Path, 'read_bytes', changed_leaf)
    with pytest.raises(AssertionError) as error:
        verifier(ROOT, contract)
    assert error.value.args == ('src/label_pair_metrics.cpp',)
