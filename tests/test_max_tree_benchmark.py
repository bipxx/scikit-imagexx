"""Phase1 harness guards only: no native product or timing acceptance."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import zlib

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('max_tree_benchmark_tooling',ROOT/'benchmarks/max_tree.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)


@pytest.mark.parametrize('argv',[[],['--check-only'],['--workers','4','--check-only'],['--record-reference','--check-only']])
def test_native_rejects_before_manifest_capture_fixture_or_original(argv,monkeypatch):
    def forbidden(*a,**k):raise AssertionError('reference/fixture reached before admission')
    monkeypatch.setattr(b,'load_reference',forbidden);monkeypatch.setattr(b,'fixture',forbidden)
    monkeypatch.setattr(b,'ACCEPTED_PRODUCT_PINS',{'pretend':'qualified'})
    with pytest.raises(b.AdmissionPending,match='genuine qualified public199/SDK20'):b.main(argv)


def test_reference_cannot_time(monkeypatch):
    monkeypatch.setattr(b,'load_reference',lambda *a:pytest.fail('operation before check-only guard'))
    with pytest.raises(b.AdmissionPending,match='no operation timing'):b.main(['--reference-only'])


def test_no_timer_or_fake_native_import():
    for node in ast.walk(ast.parse(Path(b.__file__).read_text())):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names=[n.name for n in node.names] if isinstance(node,ast.Import) else [node.module or '']
            assert not any(n in ('time','timeit','benchmarks.run','skimagexx') or n.startswith('skimagexx.') for n in names)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            assert node.func.attr not in ('perf_counter','perf_counter_ns','process_time','time_ns')


def test_complete_fixed_cohort_and_source_matrix():
    matrix=json.loads(b.MATRIX.read_text());assert b.pin(b.MATRIX)['sha256']==b.MATRIX_SHA
    assert len(matrix['cases'])==351 and sum(c['mode']=='numeric' for c in matrix['cases'])==233
    assert len(b.CASES)==252 and len({c['id'] for c in b.CASES})==252
    assert {c['function'] for c in b.CASES}==set(b.NAMES)
    assert {c['dtype'] for c in b.CASES}==set(b.DTYPES)
    assert all(b.validate_case(c)<=b.MEMORY_LIMIT for c in b.CASES)
    for name in b.NAMES:
        rows=[c for c in b.CASES if c['function']==name]
        for dtype in ('u1','u2','f4','f8'):
            for shape in ([256,256],[1024,1024],[64,64,64]):
                assert any(c['dtype']==dtype and c['shape']==shape for c in rows)
        assert any(c['pattern']=='unique' and c['dtype']=='u2' and c['shape']==[256,256] for c in rows)
        if name!='max_tree':assert any(c['provided'] and len(c['shape'])==3 for c in rows)


@pytest.mark.parametrize('change',[
    {'shape':[0,5]},{'shape':[3,5]},{'shape':[4]},{'shape':[4,5,4,5]},
    {'shape':[1024,1024,4]},{'shape':[4,True]},{'dtype':'f2'},{'dtype':'>f4'},
    {'layout':'unaligned'},{'layout':'readonly-C'},{'pattern':'nan'},
    {'connectivity':0},{'connectivity':3},{'connectivity':True},
    {'threshold':0},{'threshold':float('inf')},{'threshold':True},
    {'provided':1},{'provided':True},{'function':'invented'},{'extra':1},
    {'pattern':'unique'},{'shape':[65536,65536]},
])
def test_unsafe_spec_rejected_before_allocation(change):
    case=copy.deepcopy(b.CASES[0]);case.update(change)
    class NoArray:
        def __getattr__(self,name):raise AssertionError('allocated before certificate')
    with pytest.raises(b.SafetyError):b.fixture(case,NoArray())


def test_unique_stress_is_not_modulo_ramp():
    case=next(c for c in b.CASES if c['pattern']=='unique');image,_=b.fixture(case,np)
    assert image.dtype==np.uint16 and image.shape==(256,256)
    assert np.array_equal(image.ravel(),np.arange(65536,dtype=np.uint16))


@pytest.mark.parametrize('layout',['C','F','reverse','step'])
def test_layout_exact_values_metadata_and_compressed_raw_blob(layout,tmp_path):
    case=copy.deepcopy(b.CASES[0]);case.update(layout=layout,pattern='multilevel')
    image,owner=b.fixture(case,np);record=b.array_record(image,np,tmp_path)
    assert zlib.decompress((tmp_path/(record['sha256']+'.bin.z')).read_bytes())==image.tobytes()
    assert record['bytes']==image.nbytes and record['sha256']==hashlib.sha256(image.tobytes()).hexdigest()
    assert np.shares_memory(image,owner)
    if layout=='F':assert image.flags.f_contiguous
    if layout=='reverse':assert any(s<0 for s in image.strides)
    if layout=='step':assert not image.flags.c_contiguous


def valid_builder():
    image=np.arange(20,dtype=np.uint16)
    mask=np.zeros((4,5),dtype=np.uint8);mask[1:-1,1:-1]=1
    return [image,mask.ravel(),np.array([-5,-1,1,5],np.int32),np.ones(2,np.int32),
            np.array([4,5],np.int32),np.zeros(20,np.int64),np.arange(20,dtype=np.int64)]


@pytest.mark.parametrize('fault',['edge-mask','neighbor','dimensions','offset','permutation','unstable-tie','parent-type','alias','nan','endian','readonly'])
def test_builder_certificate_refuses_unsafe_entry(fault):
    args=valid_builder()
    if fault=='edge-mask':args[1][0]=1
    elif fault=='neighbor':args[2][0]=-100
    elif fault=='dimensions':args[4][0]=3
    elif fault=='offset':args[3][0]=2
    elif fault=='permutation':args[6][0]=1
    elif fault=='unstable-tie':args[0][:]=0;args[6][[0,1]]=args[6][[1,0]]
    elif fault=='parent-type':args[5]=args[5].astype(np.int32)
    elif fault=='alias':args[5]=args[6]
    elif fault=='nan':args[0]=args[0].astype(np.float64);args[0][0]=np.nan
    elif fault=='endian':args[0]=args[0].astype('>u2')
    else:args[0].flags.writeable=False
    with pytest.raises(b.SafetyError):b.certify_entry('_max_tree',args,(4,5),np)


def test_valid_boundary_certificate_does_not_execute_original():
    b.certify_entry('_max_tree',valid_builder(),(4,5),np)
    image=np.arange(20,dtype=np.uint16);parent=np.arange(20,dtype=np.int64)-1;parent[0]=0
    traversal=np.arange(20,dtype=np.int64)
    b.certify_tree(image,parent,traversal,np)
    b.certify_entry('_direct_filter',[image,np.zeros_like(image),parent,traversal,np.ones(20),3],(4,5),np)


@pytest.mark.parametrize('fault',['bounds','cycle','order','dtype','overlap'])
def test_provided_tree_certificate_blocks_bad_indices(fault):
    image=np.arange(20,dtype=np.uint16);parent=np.arange(20,dtype=np.int64)-1;parent[0]=0
    traversal=np.arange(20,dtype=np.int64)
    if fault=='bounds':parent[2]=20
    elif fault=='cycle':parent[2]=3
    elif fault=='order':traversal[[0,1]]=traversal[[1,0]]
    elif fault=='dtype':parent=parent.astype(np.int32)
    else:parent=traversal
    with pytest.raises(b.SafetyError):b.certify_tree(image,parent,traversal,np)


@pytest.fixture(scope='module')
def frozen():
    return json.loads(b.MANIFEST.read_text())


def test_frozen_source_and_raw_capture_authentication(frozen):
    b.verify_manifest(frozen)
    assert b.canonical_sha(frozen['reference_capture']['utf8'].encode())==b.CAPTURE_SHA
    assert frozen['reference_capture']['git_object']==b.CAPTURE_GIT
    assert len(frozen['profiles'])==3 and frozen['native_checks']==0 and frozen['accepted_product_pins'] is None
    assert frozen['counts']=={'reference':351,'reference_numeric':233,'reference_metadata':118,'timing_candidates':252}


@pytest.mark.parametrize('field',['schema','native_checks','operation_timings','accepted_product_pins','case_spec_sha256','matrix_sha256','harness_canonical_lf_sha256'])
def test_manifest_tampering_rejected(frozen,field):
    changed=copy.deepcopy(frozen);changed[field]='tampered'
    with pytest.raises(b.SafetyError):b.verify_manifest(changed)


@pytest.mark.parametrize('fault',['missing','duplicate','spec','result','runtime','native','capture','ast'])
def test_partial_or_promoted_receipt_rejected(frozen,fault):
    changed=copy.deepcopy(frozen);profile=next(iter(changed['profiles']));report=copy.deepcopy(changed['profiles'][profile])
    report.update(profile=profile,native_checks=0,operation_timings=0,accepted_product_pins=None)
    if fault=='missing':report['cases'].pop()
    elif fault=='duplicate':report['cases'][1]=report['cases'][0]
    elif fault=='spec':report['cases'][0]['spec']['threshold']=100
    elif fault=='result':report['cases'][0]['observation']['result']=None
    elif fault=='runtime':report['runtime']['numpy']='invented'
    elif fault=='native':report['native_checks']=1
    elif fault=='capture':changed['reference_capture']['utf8']+='\n'
    else:changed['scientific_ast']['fixture']='0'*64
    with pytest.raises(b.SafetyError):b.verify_report(report,changed)


def test_full_recorded_reports_self_consistent(frozen):
    for profile,rows in frozen['profiles'].items():
        report=dict(rows,profile=profile,native_checks=0,operation_timings=0,accepted_product_pins=None)
        b.verify_report(report,frozen)


@pytest.mark.parametrize('fault',['schema','function','field'])
def test_each_complete_python_ast_schema_is_required(frozen,fault):
    changed=copy.deepcopy(frozen)
    minor=str(b.sys.version_info.major)+'.'+str(b.sys.version_info.minor)
    if fault=='schema':changed['scientific_ast'].pop('3.11')
    elif fault=='function':changed['scientific_ast'][minor].pop('fixture')
    else:changed['scientific_ast'][minor]['fixture']='0'*64
    with pytest.raises(b.SafetyError):b.verify_manifest(changed)


def test_unknown_ast_schema_is_not_normalized_or_silently_admitted(frozen,monkeypatch):
    class OtherPython:
        version_info=type('Version',(),{'major':3,'minor':14})()
    monkeypatch.setattr(b,'sys',OtherPython())
    with pytest.raises(b.SafetyError,match='unassessed Python AST schema'):b.verify_manifest(frozen)
