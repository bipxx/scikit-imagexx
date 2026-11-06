"""Source-only prospective join safety/interface guards; no scientific imports."""
import ast
import hashlib
import json
from pathlib import Path
import pytest
from join_segmentations_contract import API,SIGNATURE,cases,source_bounds,fixture
ROOT=Path(__file__).resolve().parents[1]
MATRIX=ROOT/'coverage/contracts/join-segmentations-v1.json'


def test_join_prospective_header_and_bounded_recipes():
    c=json.loads(MATRIX.read_bytes());rows=cases()
    assert c['state']=='SOURCE_TEST_FIRST_NO_ORIGINAL_NATIVE_OR_TIMING'and c['api']==API and c['signature']==SIGNATURE
    assert c['cases']==rows and len(rows)==len({r['id']for r in rows})==50
    h=c['reference_header'];assert h['original_calls_authorized']==0 and h['root_GO']is h['safety_peer']is None
    assert h['maximum_future_public_topcalls']==100 and h['recipes']==50
    assert h['native_calls']==h['timing_batches']==0
    assert sum(r['map_protocols']for r in rows)==1
    for row in rows:
        b=source_bounds(row);assert b['source_encoded_upper']==63 and b['max_dense_map_elements']==65
        assert b['input_elements_each']<=192 and b['map_key_value_lengths_equal']


def test_join_exact_upstream_source_signature_factor_and_order():
    c=json.loads(MATRIX.read_bytes());p=ROOT/c['source']['join_python_fixture'];raw=p.read_bytes()
    assert hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==c['source']['join_canonical_lf_sha256']
    tree=ast.parse(raw);join=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='join_segmentations')
    assert [a.arg for a in join.args.args]==['s1','s2','return_mapping']
    assert ast.unparse(join.args.args[-1].annotation)=='bool'and join.args.defaults[0].value is False
    factor=next(n for n in join.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='factor'for t in n.targets))
    assert ast.unparse(factor.value)=='s2.max() + np.uint8(1)'
    relabels=[n for n in ast.walk(join)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='relabel_sequential']
    assert len(relabels)==3
    text=raw.decode();assert text.index('if s1.shape != s2.shape:')<text.index('s1_relabeled, _, backward_map1')<text.index('factor = s2.max()')<text.index('if not return_mapping:')
    assert text.count('map_j_to_j_initial.in_values,')==2
    relabel=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='relabel_sequential')
    body=ast.get_source_segment(text,relabel)
    assert body.index('np.min(label_field)')<body.index('np.unique(label_field)')<body.index('input_type.kind not in "iu"')
    assert 'out_vals[-1] < np.iinfo(input_type).max'in body


def test_join_reused_source_dependency_pins_and_Cython_constraints():
    c=json.loads(MATRIX.read_bytes())
    for name,pin in c['reuse']['existing_paths'].items():assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin,name
    old=json.loads((ROOT/'coverage/contracts/map-array.json').read_bytes())
    assert c['source']['Cython_sources']==old['source']['primary']
    assert c['source']['map_array_python_sha256']=='aaca3e8ec753ab84d793a39e7aa71db991d5d39dd457cba96d11b4aba1f2cfe2'
    assert set(c['runtime_pins'])=={'1.24.4','2.5.4'}
    assert c['source']['Cython_sources']['_remap.pyx']['sha256']=='3fa759e1626f87a09c0efe53d52bdb0194ee7f84d772662951645554f46ca95c'
    assert 'internally' in c['reuse']['worker_caveat']and 'different' in c['reuse']['class_caveat']


def test_join_recipe_boundary_rejects_unsafe_values_before_fixture():
    bad=dict(cases()[0],left=[8,0,1,1,2,2,3,3])
    with pytest.raises(AssertionError):fixture(bad,object())
    bad=dict(cases()[0],shape=[10,10])
    with pytest.raises(AssertionError):source_bounds(bad)


def test_join_original_runtime_requires_explicit_root_GO_before_import():
    from join_segmentations_observer import reference_runtime
    with pytest.raises(AssertionError,match='NO_ROOT_GO'):reference_runtime({})


def test_join_helpers_have_no_implicit_target_import_or_call():
    for name in('join_segmentations_contract.py','join_segmentations_observer.py'):
        tree=ast.parse((ROOT/'tests'/name).read_bytes())
        for node in tree.body:
            if isinstance(node,ast.Import):assert all(a.name.split('.')[0]not in('numpy','scipy','skimage','skimagexx')for a in node.names)
            if isinstance(node,ast.ImportFrom):assert(node.module or'').split('.')[0]not in('numpy','scipy','skimage','skimagexx')
    c=json.loads(MATRIX.read_bytes());assert c['future_benchmark']['state']=='PROSPECTIVE_ONLY_NO_DRIVER_OR_CALLS'
    assert c['gates'][1]['state']=='PENDING'and c['gates'][2]['state']=='BLOCKED_NO_GO'
