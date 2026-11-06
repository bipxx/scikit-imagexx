"""Saved crop observations only; no NumPy, upstream, or native imports/calls."""
import base64
import copy
import hashlib
import json
import math
from pathlib import Path
import zlib
import pytest
from crop_contract import cases

ROOT=Path(__file__).parents[1]
FIXTURE=ROOT/'tests/fixtures/crop_originals_v1.json'
FIXTURE_SHA256='cad3b7f4b5bd21573c30af67758912aecf379c710136f1baec5118dc2552b8f3'
DATA=json.loads(FIXTURE.read_bytes())
ROWS=[(profile,row)for profile,saved in DATA['profiles'].items()for row in saved['rows']]


def check_encoded(value):
    if isinstance(value,list):
        for item in value:check_encoded(item)
    elif isinstance(value,dict):
        if 'raw_zlib_base64'in value:
            raw=zlib.decompress(base64.b64decode(value['raw_zlib_base64'],validate=True))
            assert hashlib.sha256(raw).hexdigest()==value['raw_sha256']
        if 'raw'in value:assert type(value['raw'])is str and len(value['raw'])%2==0 and bytes.fromhex(value['raw'])is not None
        for key in ('own','writeable','aligned','c','f'):
            if key in value:assert type(value[key])is bool
        for item in value.values():check_encoded(item)


def check_distinct_payload_identity(row):
    relation=row['object_payload_identity']
    if row['id']=='crop/object-distinct-copy':
        assert relation['kind']=='array'and relation['shape']==[2,2]
        assert relation['slots']==[[name,i]for name in ('owner','input_array')for i in range(4)]
        assert relation['is_retained']==[[j%4==i for j in range(8)]for i in range(4)]
        assert all(not alias['shares_memory']for alias in row['aliases'].values())
    elif row['id']=='crop/zeroD-distinct-object':
        assert row['result_type']==dict(module='builtins',name='str')
        assert relation==dict(slots=[['owner',0],['input_array',0]],kind='scalar',is_retained=[True,True])


def test_crop_saved_provenance_and_immutable_sources():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest()==FIXTURE_SHA256
    assert DATA['schema']=='crop-reference-capture-v1'
    assert DATA['capture_source_revision']=='182a1d8d6c169a678078a9ecf874d5165058b7f7'
    assert DATA['root_GO_sha256']=='a20a534ad80786e7290bee3a78b942da498b57c70f6c571764f8bf0b82641377'
    assert DATA['cases']==cases()and len(cases())==108
    assert DATA['actual_public_topcalls']==216
    assert DATA['original_replays']==DATA['native_calls']==DATA['builds']==DATA['timed_batches']==0
    assert set(DATA['profiles'])=={'minimum','modern'}
    for name,pin in DATA['source_files_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==pin
    for profile,np_version,scipy_version in (('minimum','1.24.4','1.15.3'),('modern','2.5.4','1.18.1')):
        saved=DATA['profiles'][profile]
        assert saved['runtime']['numpy']==np_version and saved['runtime']['scipy']==scipy_version
        assert saved['runtime']['skimage']=='0.26.0'
        assert saved['runtime']['upstream_canonical_lf_sha256']==DATA['original_source_canonical_lf_sha256']
        assert [row['id']for row in saved['rows']]==[spec['id']for spec in cases()]
        assert saved['terminal']['counts']==dict(top=108,refused=0)
        for name in ('journal_raw_sha256','observations_raw_sha256','precall_freeze_raw_sha256'):
            assert len(bytes.fromhex(saved[name]))==32


@pytest.mark.parametrize('profile,row',ROWS,ids=[profile+'/'+row['id']for profile,row in ROWS])
def test_crop_saved_row_state_aliases_payload_and_raw_words(profile,row):
    assert row['before']==row['after']and all(row['checks'].values())
    assert all(row['controller_state_restored'].values())and row['original_body_budget']==1
    assert row['warnings']==[]and row['FP_events']==[]
    check_encoded(row['before']);check_encoded(row['result'])
    assert type(row['result_is_argument'])is bool
    for alias in row['aliases'].values():
        assert set(alias)=={'identity','base_is_input','shares_memory','pointer_delta'}
        assert all(type(alias[key])is bool for key in ('identity','base_is_input','shares_memory'))
        # Nonsharing allocation deltas are retained raw observations, not
        # deterministic cross-process addresses or a candidate parity rule.
        assert alias['pointer_delta']is None or type(alias['pointer_delta'])is int
    relation=row['object_payload_identity']
    if relation is not None:
        assert len(relation['slots'])<=512
        assert all(type(name)is str and type(index)is int and 0<=index<256 for name,index in relation['slots'])
        if relation['kind']=='array':
            assert len(relation['is_retained'])==math.prod(relation['shape'])<=256
            assert all(len(line)==len(relation['slots'])and all(type(x)is bool for x in line)for line in relation['is_retained'])
        elif relation['kind']=='scalar':assert len(relation['is_retained'])==len(relation['slots'])and all(type(x)is bool for x in relation['is_retained'])
        else:assert relation['kind']=='error'and row['error']is not None
    if row['error']is not None:
        assert set(row['error'])=={'module','type','message'}and row['result']is None
        assert row['result_type']==dict(module='builtins',name='NoneType')
    fenv=row['controller_fenv']
    assert len(bytes.fromhex(fenv['before']))==len(bytes.fromhex(fenv['after']))==32
    check_distinct_payload_identity(row)


@pytest.mark.parametrize('profile',('minimum','modern'))
def test_crop_saved_coercion_protocol_and_scalar_boundaries(profile):
    rows={r['id']:r for r in DATA['profiles'][profile]['rows']}
    expected_errors=15 if profile=='minimum'else 18
    assert sum(r['error']is not None for r in rows.values())==expected_errors
    for name in ('input-list','input-tuple','input-python_scalar','input-numpy_scalar'):
        error=rows['crop/'+name]['error']
        assert error is None if profile=='minimum'else error['type']=='ValueError'
    assert (rows['crop/width-numpy-uint64']['error']is not None)==(profile=='minimum')
    copy_value=None if profile=='minimum'else False
    event=dict(event='array.__array__',dtype=None,copy=copy_value)
    assert rows['crop/input-array_provider']['events']==[event]
    assert rows['crop/coercion-error-before-width']['events']==[event]
    assert rows['crop/coercion-error-before-width']['error']==dict(module='builtins',type='RuntimeError',message='crop array provider refused')
    assert rows['crop/width-index-probe']['events']==[dict(event='index',value=1)]*2
    assert rows['crop/zeroD-packed-view']['result_type']==dict(module='numpy',name='void')
    assert rows['crop/zeroD-packed-view']['result']['dtype']=='|V6'
    assert len(bytes.fromhex(rows['crop/zeroD-packed-view']['result']['raw']))==6
    assert rows['crop/zeroD-object-view']['result_type']==dict(module='builtins',name='str')


@pytest.mark.parametrize('profile',('minimum','modern'))
@pytest.mark.parametrize('name',('crop/object-distinct-copy','crop/zeroD-distinct-object'))
def test_crop_saved_identity_guard_rejects_equal_payload_reconstruction(profile,name):
    row=copy.deepcopy(next(r for r in DATA['profiles'][profile]['rows']if r['id']==name))
    relation=row['object_payload_identity']
    if relation['kind']=='array':relation['is_retained']=[[False]*len(line)for line in relation['is_retained']]
    else:relation['is_retained']=[False]*len(relation['is_retained'])
    with pytest.raises(AssertionError):check_distinct_payload_identity(row)
