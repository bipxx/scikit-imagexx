"""Authenticate unchanged admitted original rows; no scientific imports/calls."""
import collections
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'tests/fixtures/regular_grid_originals_v1.json'


def saved():
    raw=PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='e29d10edf77af031674bb2118281976512a08ba95496346e650ed96ddadfd80a'
    return json.loads(raw)


def test_frozen_packet_identity():
    data=saved()
    assert data['source_revision']=='5750481bd1427311b92abda5c6943c6f3cc41c82'
    assert data['actual_public_topcalls']==90
    assert data['original_replays']==data['native_calls']==data['timed_batches']==0
    assert data['root_GO_sha256']=='a54eb4380e24ebd504aa8dd932946c20dbcd5c7f16f1fb0258f2c6056e1db2e4'
    assert data['original_source_canonical_lf_sha256']=='64fcce3556f0cbf8cafd05a54ea3545b65b18bdcabbee25b953a21ba06875581'


def test_saved_rows_state_and_types():
    from regular_grid_contract import cases
    ids=[row['id'] for row in cases()]
    for profile in saved()['profiles'].values():
        rows=profile['raw_rows'];assert len(rows)==45 and [r['id'] for r in rows]==ids
        assert profile['terminal']['counts']==dict(top=45,refused=0)
        for row in rows:
            assert row['before']==row['after']
            assert all(row['checks'].values()) and all(row['controller_state_restored'].values())
            assert row['state_during']['handler_unchanged']
            if row['error']:
                assert row['result'] is None and set(row['error'])=={'type','message'}
                continue
            assert row['result']['type']==['builtins','tuple']
            for item in row['result']['items']:
                assert item['type']==['builtins','slice']
                assert item['stop']==dict(type=['builtins','NoneType'],value=None)
                for field in ('start','step'):
                    scalar=item[field]
                    if scalar['type']==['builtins','NoneType']:assert scalar['value'] is None
                    else:
                        assert scalar['type']==['numpy','int64'] and scalar['dtype']=='<i8'
                        assert len(bytes.fromhex(scalar['raw']))==8


def test_exact_retained_partitions():
    for profile in saved()['profiles'].values():
        rows=profile['raw_rows']
        assert sum(row['error'] is None for row in rows)==36
        assert collections.Counter(row['error']['type'] for row in rows if row['error'])==dict(ZeroDivisionError=5,TypeError=2,ValueError=1,FloatingPointError=1)
        assert sum(len(row['warnings']) for row in rows)==3
        assert sum(len(row['events']) for row in rows)==3
        assert [r['id'] for r in rows if r['fenv_before']!=r['fenv_after']]==['numpy-zero-warn','numpy-zero-raise','numpy-zero-call']


def test_profile_scope_is_not_platform_identity():
    data=saved()
    for profile in data['profiles'].values():
        assert profile['runtime']['default_int_dtype']==profile['runtime']['intp_dtype']=='<i8'
    one,two=[p['raw_rows'] for p in data['profiles'].values()]
    for a,b in zip(one,two):
        assert all(a[k]==b[k] for k in ('before','after','result','error','warnings','events'))
