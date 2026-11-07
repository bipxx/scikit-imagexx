"""Family authority/source guards; no scientific/runtime/helper body."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import flood_fill_capture_controller as C
from flood_fill_contract import recipes
ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('fault', ('missing','digest','schema','budget','GO','pool','profile-pool','controller'))
def test_entry_refusals_precede_git_science_helper(fault,tmp_path,monkeypatch):
    ids=[r['id']for r in recipes()]
    q=dict(schema='flood-fill-capture-v1',status='UNBOUND_CAPTURE_CAPABLE_ZERO_BUDGET',
           original_budget=0,native_image_budget=0,sampler_budget=0,observation_budget=0,
           root_original_GO=None,case_ids=ids,api_case_ids={'flood':ids[:45],'flood_fill':ids[45:]},
           profile_case_ids={'minimum':ids,'modern':ids},controller_raw_sha256=C.H(C.__file__))
    path=tmp_path/'binding.json'
    if fault=='schema':q['schema']='foreign'
    elif fault=='budget':q['original_budget']=1
    elif fault=='GO':q['root_original_GO']='invented'
    elif fault=='pool':q['case_ids']=ids[:-1]
    elif fault=='profile-pool':q['profile_case_ids']['modern']=ids[:-1]
    elif fault=='controller':q['controller_raw_sha256']='0'*64
    path.write_text(json.dumps(q))
    pin=C.H(path)
    if fault=='missing':path=tmp_path/'absent'
    elif fault=='digest':pin='0'*64
    monkeypatch.setattr(C.subprocess,'check_output',lambda *a,**k:pytest.fail('Git must not be requested'))
    with pytest.raises(AssertionError):C.validate(path,pin,'minimum')


def test_capture_has_no_implicit_authority():
    with pytest.raises(AssertionError,match='NO_ROOT_GO'):
        C.validate_go(SimpleNamespace(root_go=None,root_go_sha256=None),{})


def test_fixed_family_source_scope_and_no_percase_sourcehash():
    text=Path(C.__file__).read_text();tree=ast.parse(text)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    loop=next(n for n in ast.walk(main)if isinstance(n,ast.For)and ast.unparse(n.target)=='spec')
    assert 'source_files_raw_sha256'not in ast.unparse(loop)
    assert "active_api=='flood_fill' and frame.f_code is targets['flood'].__code__"in text
    assert 'RequestGuard(types.ModuleType)'in text
    assert "module.__class__=original_class"in text
    assert "getattr(aliases,a)is targets[a]"in text
    assert 'no_retry' in text and 'failure_writer_error' in text and 'last_case_diagnostics' in text
    assert 'len(ids)==len(set(ids))==96' in text
    assert "go['maximum_total_public_topcalls']==192"in text


def test_original_source_packet_preserved():
    contract=json.loads((ROOT/'coverage/contracts/flood-fill-v1.json').read_text())
    assert contract['recipe_counts']=={'flood':45,'flood_fill':51}
    assert contract['authority']['original_budget']==0
    assert hashlib.sha256((ROOT/contract['source_snapshot']).read_bytes()).hexdigest()==contract['source_snapshot_sha256']
