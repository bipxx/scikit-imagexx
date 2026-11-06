"""Pure source/data/runtime-admission guards; imports no scientific package."""
import ast,hashlib,json,re
from pathlib import Path
import pytest
from matrix_prescott_context import predecessor_bytes
ROOT=Path(__file__).resolve().parents[1]

def test_prescott_current_source_has_exact_qualified_predecessor():
    record=json.loads((ROOT/'coverage/contracts/matrix-prescott-v1.json').read_text())
    for path,pin in record['source_pins_canonical_lf'].items():
        assert hashlib.sha256((ROOT/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin
    old=predecessor_bytes(ROOT)
    cert=json.loads((ROOT/'coverage/evidence/matrix-numpy124-skylakex-certificate-v1.json').read_text())
    assert hashlib.sha256(old).hexdigest()==cert['source_after_canonical_lf_sha256']
    source=(ROOT/'src/matrix_transform.cpp').read_text()
    assert 'Copyright 2009, 2010 The University of Texas at Austin.'in source
    assert 'std::fma'in source # Oldprofilearithmeticremains; Prescottseparatetreeisnew.


def test_all48_saved_records_and_fixture_words_are_complete():
    gold=json.loads((ROOT/'coverage/contracts/matrix-prescott-reference-v1.json').read_text())
    assert len(gold['cases'])==48 and len({s['spec']['id']for s in gold['cases']})==48
    assert sum(len(s['expected'])for s in gold['cases'])==432
    raw=(ROOT/'tests/fixtures/matrix_prescott_native.hpp').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==gold['fixture_header_raw_sha256']
    lines=[s for s in raw.decode().splitlines()if s.startswith('{"')]
    assert len(lines)==48
    for case,line in zip(gold['cases'],lines):
        assert line.startswith('{'+json.dumps(case['spec']['id'])+',')
        words=re.findall(r'UINT64_C\(0x([0-9a-f]{16})\)',line)
        expected=case['inputs']['A']+case['inputs'].get('B',[])+case['inputs'].get('x',[])+case['expected']
        assert words==expected
    assert gold['root_admission']['ordinals_unique_exact_1_to_96']is True


@pytest.mark.parametrize('capability',(False,True))
def test_prescott_runtime_needs_exact_pinned_version_and_owned_v3(capability):
    source=(ROOT/'python/skimagexx/_matrix_transform.py').read_text();tree=ast.parse(source)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_runtime_profile' or isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='_RUNTIME_CORES'for t in n.targets)]
    env={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROOT/'python/skimagexx/_matrix_transform.py'),'exec'),env)
    env.update(_unforced=lambda:True,_authenticate_runtime=lambda:('1.24.4','554bde1d8a0c71d8dc21ae74de05c44da4fff5dbc6791a819f6acf5adfe90bd9',lambda:b'Prescott',lambda:1,None,()),_checked_entry=lambda profile:object()if capability else None)
    assert env['_runtime_profile']()==('1.24.4'if capability else None)
    if not capability:assert env['_RUNTIME_CORENAME']is None and env['_THREAD_QUERY']is None
