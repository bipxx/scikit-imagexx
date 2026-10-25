"""Prospective original-only terrain observations; no controller, CPP or clocks."""
import ast
import base64
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
import warnings
import zlib
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
_spec=importlib.util.spec_from_file_location('reconstruction_terrain_cases',ROOT/'benchmarks/reconstruction_terrain_cases.py')
CASES=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(CASES)
MATRIX=ROOT/'coverage/contracts/reconstruction-terrain.json'
OBSERVER=ROOT/'tests/test_reconstruction_benchmark_contracts.py'
OBSERVER_SHA='b58401734e9b8492c631f2dadc0e169981a6ae3b65ceb4a0ec217227a2ef36cb'
assert hashlib.sha256(OBSERVER.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==OBSERVER_SHA
# Separate adapter namespace, exact helper ASTs; no historic module mutation.
HELPERS={'profile','digest','packed','array','source_auth','prepared_certificate','observe'}
_nodes=[n for n in ast.parse(OBSERVER.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name in HELPERS]
assert len(_nodes)==len(HELPERS)
exec(compile(ast.Module(body=_nodes,type_ignores=[]),'<exact-frozen48-observer-helpers>','exec'),globals())


def matrix():return json.loads(MATRIX.read_bytes())


def unpack_array(record,assets):
    item=record['data'];packed=assets[item['raw_sha256']]
    raw=zlib.decompress(base64.b64decode(packed['payload'],validate=True))
    assert len(raw)==item['raw_bytes']==packed['raw_bytes'] and digest(raw)==item['raw_sha256']
    return np.frombuffer(raw,dtype=record['dtype']).reshape(record['shape'])


def assert_terrain_outcome(row,observation,assets):
    assert observation['error'] is None,row['id']
    assert len(observation['source_backend'])==1 and observation['source_backend'][0]['completed']
    assert observation['before']==observation['after'] and not any(observation['aliases'].values())
    seed,mask,kw=CASES.fixture(row);terrain=CASES.terrain_certificate(row,seed,mask)
    assert observation['changed_elements']==terrain['expected_changed_elements']>0
    result=unpack_array(observation['result'],assets)
    assert result.shape==seed.shape and result.dtype==np.dtype('f4' if row['dtype']=='f4' else 'f8')
    assert np.any(result!=mask),row['id']
    witness=terrain['nontrivial_output_witness']
    if isinstance(witness,dict):assert result[tuple(witness['point'])]==witness['expected_output']
    else:
        if row['method']=='dilation':assert result.max()<=seed.max()<mask.max()
        else:assert result.min()>=seed.min()>mask.min()


def record(primary,output):
    data=matrix();assert data['phase']=='prospective-original-only' and data['profiles']=={}
    assert data['specifications']==CASES.specifications() and data['case_manifest']==CASES.case_manifest()
    for name,want in data['source_text_sha256'].items():assert digest((ROOT/name).read_bytes().replace(b'\r\n',b'\n'))==want
    source,auth=source_auth(primary);assets={};observations={}
    for row in CASES.specifications():
        observation=observe(row,source,assets);observations[row['id']]=observation
        # First raw failure retained; never repair/remove a prospective case.
        report=dict(schema='reconstruction-terrain-reference-profile-v1',profile=profile(),source=auth,
            observations=observations,assets=assets,native_acceptance=False,operation_timings=0)
        output.write_bytes((json.dumps(report,indent=2)+'\n').encode())
        assert_terrain_outcome(row,observation,assets)
    return report


def test_prospective_specs_and_original48_immutable():
    data=matrix();assert data['specifications']==CASES.specifications() and data['case_manifest']==CASES.case_manifest()
    assert data['case_contract_sha256']==CASES.contract_hash(data['case_manifest'])
    assert len(data['specifications'])==16 and data['native_acceptance'] is False and data['accepted_product_pins'] is None
    assert {s['method'] for s in CASES.specifications()}=={'dilation','erosion'}
    assert sum(s['group']=='high-cardinality' for s in CASES.specifications())==8
    for name,want in data['preserved_original48_raw_sha256'].items():assert digest((ROOT/name).read_bytes())==want
    for name,want in data['preserved_original48_source_text_sha256'].items():assert digest((ROOT/name).read_bytes().replace(b'\r\n',b'\n'))==want
    for name,want in data['source_text_sha256'].items():assert digest((ROOT/name).read_bytes().replace(b'\r\n',b'\n'))==want


def test_helpers_and_bounds_are_exact_readonly_interfaces():
    old={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(OBSERVER.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name in HELPERS}
    assert {n.name:ast.dump(n,include_attributes=False) for n in _nodes}==old
    original=next(n for n in ast.parse(CASES.BASE_PATH.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='certificate')
    assert ast.dump(CASES._node,include_attributes=False)==ast.dump(original,include_attributes=False)
    for row in CASES.case_manifest():
        assert row['certificate']['workspace_bound_bytes']<=256*1024*1024
        assert row['terrain']['marker_count']>=2 and row['terrain']['expected_changed_elements']>0
        if row['spec']['group']=='high-cardinality':assert row['terrain']['mask_unique_values']==math.prod(row['spec']['shape'])


def test_native_pending_before_fixture(monkeypatch):
    monkeypatch.setattr(CASES,'fixture',lambda *a,**k:pytest.fail('fixture before native admission'))
    with pytest.raises(RuntimeError,match='pending genuine qualified public200'):CASES.native_admission()


@pytest.mark.parametrize('fault',['spec','shape','order','nonfinite','footprint'])
def test_unsafe_entries_reject_before_Cython(fault):
    row=next(s for s in CASES.specifications() if s['dtype']=='f4');seed,mask,kw=CASES.fixture(row)
    if fault=='spec':row=dict(row,radius=2)
    elif fault=='shape':seed=seed[:0];mask=mask[:0]
    elif fault=='order':seed=seed.copy();seed.flat[0]=2
    elif fault=='nonfinite':seed=seed.copy();seed.flat[0]=np.nan
    else:kw['footprint']=np.ones((2,2),bool)
    with pytest.raises(AssertionError):CASES.certificate(row,seed,mask,kw)


@pytest.mark.parametrize('row',CASES.specifications(),ids=lambda s:s['id'])
def test_original_frozen_parity(row):
    data=matrix();assert set(data['profiles'])=={'windows','modern','minimum'},'fresh originals pending; not a native admission'
    source,_=source_auth(Path(os.environ['SKIMAGEXX_RECONSTRUCTION_PRIMARY_SOURCE']));assets={}
    actual=observe(row,source,assets);assert_terrain_outcome(row,actual,assets)
    assert actual==data['profiles'][profile()]['observations'][row['id']]


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--primary-source',type=Path,required=True);parser.add_argument('--record',type=Path,required=True)
    args=parser.parse_args();assert not args.record.exists();record(args.primary_source,args.record)
