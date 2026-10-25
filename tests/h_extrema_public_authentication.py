"""Closed metadata-only H authentication through one reconstruction successor.

The two runtime authentication bodies below are mechanically derived from the
exact 448 helpers. Every distribution/version/source/warning check is preserved.
Only frozen-file reads may obtain the exact historical reconstruction bytes,
after proving the actual f82 file differs in one named metadata body only.
This module never imports skimage, NumPy, skimagexx, or scientific test modules.
"""
import ast
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/h-extrema-plan.json'
PLAN=ROOT/'coverage/contracts/h-extrema-floating-plan.json'
RECORD='coverage/evidence/h-extrema-reconstruction-reference-integration.json'
PATH='tests/test_reconstruction_contracts.py'
BODY='test_source_signature_and_fixed_scope'
OLD='ee2db6cd6d5bd694ed06e3b1133445b680b205f8a802ea1fc9e0ae1d9f624868'
CURRENT='b18f5ab2777a4dda3a1fe244cb7edc75627d6fd9ccb59fcc3c08343c8c4189cb'
OLD_LF='b66e947122231f7edad482cc20b58ce3d85da9fffa3c59ee93499bb8713f6a80'
CURRENT_LF='16a5009b2dee9dfb154cddcf5eadb805d300c1c532451c0718c4d0f0ee35be68'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def body(raw):
    nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name==BODY]
    assert len(nodes)==1
    node=nodes[0]
    return b''.join(raw.splitlines(keepends=True)[node.body[0].lineno-1:node.body[-1].end_lineno])


def verify_reconstruction(record,current,migration_raw):
    assert record['schema']=='h-reconstruction-exact-reference-metadata-integration-v1'
    assert record['path']==PATH and record['allowed_body']==BODY
    assert record['historical_source']=='3f4ed95e9fb322985374f2af71522398dbd8db7f'
    assert record['current_source']=='f82c0c26433fb60c233bfc888a5e8e1a0a925bb6'
    old=record['historical_source_utf8'].encode('utf-8')
    assert sha(old)==record['historical_raw_sha256']==OLD
    assert sha(current)==record['current_raw_sha256']==CURRENT
    assert sha(old.replace(b'\r\n',b'\n'))==record['historical_canonical_lf_sha256']==OLD_LF
    assert sha(current.replace(b'\r\n',b'\n'))==record['current_canonical_lf_sha256']==CURRENT_LF
    before,after=record['historical_body_utf8'].encode(),record['current_body_utf8'].encode()
    assert before==body(old) and after==body(current) and old.count(before)==current.count(after)==1
    assert current.replace(after,before,1)==old
    a,b=ast.parse(old),ast.parse(current)
    for node in b.body:
        if isinstance(node,ast.FunctionDef) and node.name==BODY:
            node.body=copy.deepcopy(next(n.body for n in a.body if isinstance(n,ast.FunctionDef) and n.name==BODY))
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    assert sha(migration_raw)==record['existing_f82_migration_raw_sha256']
    migration=json.loads(migration_raw)
    assert migration['current_raw_sha256']==CURRENT and migration['current_canonical_lf_sha256']==CURRENT_LF
    original=migration['original_source_UTF8'].encode()
    assert sha(original)==migration['original_raw_sha256']=='9a506a465f56e295edeb21a83b7e5988a7cbf8a25288513b935e59f1e85e05de'
    assert sha(original.replace(b'\r\n',b'\n'))==migration['original_canonical_lf_sha256']==migration['original_normalized_lf_sha256']=='8160f83f1f18049b0df90f882a21d3fdf8b42feb435df85a5080e69567df744c'
    return old


def validate_reconstruction(read_bytes):
    return verify_reconstruction(json.loads(read_bytes(RECORD)),read_bytes(PATH),
                                 read_bytes('coverage/evidence/reconstruction-public-assertion-migration.json'))


def source_bytes(name,expected):
    if name==PATH:
        assert expected==OLD, 'unknown historical reconstruction pin'
        return validate_reconstruction(lambda n:(ROOT/n).read_bytes())
    return (ROOT/name).read_bytes()


def load(name,filename):
    # The copied metadata bodies only load the exact four-file provenance reader.
    assert filename=='h_extrema_source_provenance.py'
    spec=importlib.util.spec_from_file_location(name,ROOT/'tests'/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def authenticate_legacy(profile):
    contract = json.loads(CONTRACT.read_bytes())
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-image')}
    assert versions == contract['source_audit'][profile]['versions']
    records = []
    for relative, expected in contract['source_audit'][profile]['files'].items():
        distribution = 'scikit-image' if relative.startswith('skimage/') else 'numpy'
        path = Path(importlib.metadata.distribution(distribution).locate_file(relative))
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected['sha256'], str(path)
        records.append(dict(path=str(path), bytes=len(raw), sha256=expected['sha256']))
    source_evolution=[]
    for name, expected in contract['frozen_files'].items():
        raw = source_bytes(name, expected)
        if hashlib.sha256(raw).hexdigest() != expected:
            provenance=load('h_reference_source_provenance','h_extrema_source_provenance.py')
            source_evolution.append(provenance.authenticate_source(ROOT,name,expected))
        else:
            assert hashlib.sha256(raw).hexdigest() == expected, name
    return contract, dict(profile=profile, versions=versions, files=records, python=sys.version, executable=sys.executable, source_evolution=source_evolution)


def authenticate_floating(profile):
    plan = json.loads(PLAN.read_bytes())
    _, authentication = authenticate_legacy(profile)
    for name, expected in {**plan['preserved_files'], **plan['reference_files']}.items():
        if hashlib.sha256(source_bytes(name, expected)).hexdigest() != expected:
            provenance=load('h_floating_source_provenance','h_extrema_source_provenance.py')
            evolution=provenance.authenticate_source(ROOT,name,expected)
            authentication.setdefault('floating_source_evolution',[]).append(evolution)
        else:
            assert hashlib.sha256(source_bytes(name, expected)).hexdigest() == expected, name
    helper = plan['warning_helpers'][profile]
    path = Path(importlib.metadata.distribution('scikit-image').locate_file(helper['relative']))
    raw = path.read_bytes()
    assert len(raw) == helper['bytes'] and hashlib.sha256(raw).hexdigest() == helper['sha256']
    authentication['warning_helper'] = dict(path=str(path), bytes=len(raw), sha256=helper['sha256'])
    return plan, authentication
