"""Frozen reference safety and observations, without original replay."""
import base64,hashlib,json,zlib
from pathlib import Path
import pytest
if __package__:
    from .line_aa_contract import API,MAX_OUTPUT_ELEMENTS,UNASSESSED,cases,fixture,source_safe,benchmark_cases
    from .line_aa_observer import encode,scientific_profile,SOURCE_SHA,SOURCE_PYX_SHA,BINARY_SHA,SIGNATURE
    from .test_draw_coordinates_public import without_draw_activity
else:
    from line_aa_contract import API,MAX_OUTPUT_ELEMENTS,UNASSESSED,cases,fixture,source_safe,benchmark_cases
    from line_aa_observer import encode,scientific_profile,SOURCE_SHA,SOURCE_PYX_SHA,BINARY_SHA,SIGNATURE
    from test_draw_coordinates_public import without_draw_activity
ROOT=Path(__file__).parents[1]
MATRIX=ROOT/'coverage/contracts/line-aa.json'
CASES=cases()


def test_line_aa_prospective_matrix_and_source_safety():
    matrix=json.loads(MATRIX.read_bytes())
    assert matrix['api']==API and matrix['cases']==CASES
    assert len(CASES)==48 and len({s['id']for s in CASES})==48
    assert matrix['signature']==SIGNATURE and matrix['original_python_sha256']==SOURCE_SHA and matrix['original_pyx_sha256']==SOURCE_PYX_SHA and matrix['original_binary_sha256']==BINARY_SHA
    assert matrix['unassessed']==list(UNASSESSED)and matrix['benchmark_cases']==benchmark_cases()
    assert matrix['benchmark_max_output_elements']==1_000_000
    for case in benchmark_cases():
        r0,c0,r1,c1=case['args'];dr,dc=abs(r1-r0),abs(c1-c0)
        assert all(-(1<<31)+2<=v<=(1<<31)-2 for v in case['args'])
        assert dr*dr+dc*dc<=(1<<31)-1 and 9*(dr+dc+1)<=matrix['benchmark_max_output_elements']
    assert matrix['weight_policy']==dict(mode='bitwise-float64-ordinary',max_ulp=0,rtol=0,atol=0,association='literal sqrt(dc*dc+dr*dr); fabs ordered additions/division; then1.-storedfloat64distance',hardware_floating_environment='not claimed outside ordinary default rounding')
    for spec in CASES:source_safe(spec)
    for name,pin in matrix['source_freeze'].items():assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin


@pytest.mark.parametrize('spec',CASES,ids=[s['id']for s in CASES])
def test_line_aa_fixture_without_original_activity(spec):
    np=scientific_profile()
    with without_draw_activity():
        args,options,events,buffers=fixture(spec,np)
        assert not events and sum(a.size for a in buffers.values())<=1
        json.dumps(encode((args,options,buffers),np),allow_nan=False)


@pytest.mark.parametrize('profile',('1.24.4','2.5.4'))
def test_line_aa_saved_original_rows_without_replay(profile):
    matrix=json.loads(MATRIX.read_bytes());rows=matrix['original_observations']
    if not rows:pytest.skip('Prospective only; original capture pending')
    rows=rows[profile];assert len(rows)==len(CASES)and[r['id']for r in rows]==[s['id']for s in CASES]
    for row in rows:
        assert all(row['checks'].values());pending=[row]
        while pending:
            value=pending.pop()
            if isinstance(value,dict):
                if 'raw_zlib_base64'in value:assert hashlib.sha256(zlib.decompress(base64.b64decode(value['raw_zlib_base64']))).hexdigest()==value['raw_sha256']
                pending.extend(value.values())
            elif isinstance(value,list):pending.extend(value)
