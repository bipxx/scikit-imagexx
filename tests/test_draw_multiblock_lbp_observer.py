"""Pure LBP observer/dispatch closure; no scientific imports or fixture execution."""
import ast,copy,hashlib,json
from pathlib import Path
import pytest
import draw_multiblock_lbp_observer as observer
ROOT=Path(__file__).resolve().parents[1]


def test_frozen_sources_and_one_supplied_observer_call():
    d=json.loads((ROOT/'coverage/contracts/draw-multiblock-lbp-capture-v1.json').read_bytes())
    assert d['original_budget']==0 and d['root_GO']is None
    assert(d['legacy_count'],d['added_count'],d['case_count'],d['prospective_two_profile_pool'])==(93,1,94,188)
    for name,pin in d['source_raw_pins'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==pin,name
    tree=ast.parse((ROOT/'tests/draw_multiblock_lbp_observer.py').read_text())
    method=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='observe')
    assert len([n for n in ast.walk(method)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='call'])==1
    assert 'BaseException'not in ast.unparse(method)and "spec['policy']"in ast.unparse(method)


def test_exact_93_legacy_and_one_added_dispatch_without_factories():
    rows=observer.cases();assert len(rows)==len({r['id']for r in rows})==94
    assert rows[:93]==observer.legacy.cases()and rows[-1]==observer.overlap.added_case()
    assert all(observer.select_fixture(s)is observer.legacy.fixture for s in rows[:93])
    assert observer.select_fixture(rows[-1])is observer.overlap.fixture
    with pytest.raises(AssertionError):observer.select_fixture(dict(rows[-1],alpha=0.5))
