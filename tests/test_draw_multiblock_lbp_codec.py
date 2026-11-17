"""LBP typed exception source closure; no scientific import or target call."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/draw-multiblock-lbp-codec-v2.json'


def test_exact_component_and_observer_inverse():
    c=json.loads(CONTRACT.read_bytes())
    for name,pin in c['source_raw_pins'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==pin,name
    for name,pin in c['old_source_raw_pins'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==pin,name
    raw=(ROOT/c['observer']).read_bytes()
    for part in c['observer_inverse']:
        a,b=part['current'].encode(),part['predecessor'].encode();assert raw.count(a)==1;raw=raw.replace(a,b,1)
    assert raw==(ROOT/'tests/draw_multiblock_lbp_observer.py').read_bytes()
    assert hashlib.sha256((ROOT/'tests/mark_boundaries_exception_codec.py').read_bytes()).hexdigest()==c['codec_component']['raw_sha256']


def test_exception_only_change_retains_one_call_and_qualname():
    c=json.loads(CONTRACT.read_bytes());tree=ast.parse((ROOT/c['observer']).read_text())
    observe=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='observe')
    assert len([n for n in ast.walk(observe)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='call'])==1
    source=ast.unparse(observe)
    assert 'encode_exception_args(exc.args, np)'in source and 'type(exc).__qualname__'in source
    assert "spec['policy']"in source and 'BaseException'not in source
    assert 'np_state'not in source and 'reference_runtime'not in source


def test_frozen94_and_source_exception_domains_are_not_outputs():
    c=json.loads(CONTRACT.read_bytes());legacy=json.loads((ROOT/'coverage/contracts/draw-multiblock-lbp-v1.json').read_bytes());new=json.loads((ROOT/'coverage/contracts/draw-multiblock-lbp-v2.json').read_bytes())
    assert len(legacy['cases'])==93 and new['case_count']==94 and len(legacy['benchmark_cases'])==4
    assert c['original_budget']==0 and c['root_GO']is None and c['source_only']is True
    assert {s['id'].split('/')[-1]for s in legacy['cases']} >= {'float-width','float-height','complex-image','object-image','alpha-none','bad-greater'}
    assert c['exception_domains']['casting']['arguments']==['ufunc','casting','from_dtype','to_dtype','index']
    assert c['unassessed']and c['codec_component']['review_sha256']=='0dbdf86486b1235e8ea39a04186a259213a06a7a415af811763cc37a7ea230f1'


def test_dtype_descriptor_frontier_is_explicit_and_never_applies_ufunc():
    c=json.loads(CONTRACT.read_bytes());source=(ROOT/'tests/mark_boundaries_exception_codec.py').read_text();tree=ast.parse(source)
    assert 'UNASSESSED_EXCEPTION_DTYPE'in source
    assert 'item.fields is None and item.subdtype is None and item.metadata is None'in source
    assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id in('item','value','ufunc')for n in ast.walk(tree))
    assert c['codec_component']['byte_exact_reuse']is True
