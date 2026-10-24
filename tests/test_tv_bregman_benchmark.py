"""Reference-tooling controls only; these tests never admit a native product."""
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import zlib

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('bregman_benchmark_tooling',ROOT/'benchmarks/tv_bregman.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)


@pytest.mark.parametrize('arguments',[[],['--check-only'],['--record-reference','--check-only'],['--workers','4','--check-only']])
def test_native_entry_rejects_before_fixture_or_reference_import(arguments,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('science reached before native admission')
    monkeypatch.setattr(b,'load_reference',forbidden)
    monkeypatch.setattr(b,'fixture',forbidden)
    monkeypatch.setattr(b,'ACCEPTED_PRODUCT_PINS',{'pretend':'qualified'})
    with pytest.raises(b.AdmissionPending,match='qualified whole public product'):b.main(arguments)


def test_reference_without_check_cannot_measure(monkeypatch):
    monkeypatch.setattr(b,'load_reference',lambda:pytest.fail('reference imported for timing'))
    with pytest.raises(b.AdmissionPending,match='no operation timing'):b.main(['--reference-only'])


def test_no_operation_clock_or_native_surrogate():
    tree=ast.parse(Path(b.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or '']
            assert not any(n in ('time','timeit','benchmarks.run','skimagexx') or n.startswith('skimagexx.') for n in names)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            assert node.func.attr not in ('perf_counter','perf_counter_ns','process_time','time_ns')
    assert b.ACCEPTED_PRODUCT_PINS is None


def test_original_matrix_and_safe_scope_stay_exact():
    assert b.pin(b.REFERENCE_MATRIX)['sha256']==b.MATRIX_SHA256
    assert b.canonical_hash(b.REFERENCE_HELPERS.read_bytes())==b.HELPERS_CANONICAL_SHA256
    matrix=json.loads(b.REFERENCE_MATRIX.read_text())
    assert len(matrix['cases'])==135
    assert sum(c['mode']=='compiled' for c in matrix['cases'])==112
    assert len(b.TIMING_CASES)==110
    assert len({c['id'] for c in b.TIMING_CASES})==110
    assert len(b.UNSAFE_INTAKES)==4
    assert {c['dtype'] for c in b.TIMING_CASES}=={'u1','u2','>u2','f4','f8'}
    assert {c['max_num_iter'] for c in b.TIMING_CASES}=={1,2,5,12,64}
    assert all(b.validate_case(c)<=b.MEMORY_LIMIT for c in b.TIMING_CASES)


@pytest.mark.parametrize('change',[
    {'shape':[0,5]},{'shape':[1,5]},{'shape':[4]},{'shape':[4,5,4]},
    {'shape':[1000000,1000000]},{'shape':[4,5,True]},
    {'dtype':'f2'},{'dtype':'>f4'},{'layout':'unaligned'},{'dtype':'f4','layout':'offset'},
    {'dtype':'u2','layout':'unaligned-step'},{'layout':'unknown'},
    {'max_num_iter':0},{'max_num_iter':-1},{'max_num_iter':65},{'max_num_iter':True},
    {'weight':0},{'weight':float('nan')},{'weight':float('inf')},{'weight':True},
    {'eps':float('nan')},{'eps':-1},{'eps':True},{'isotropic':1},
    {'channel_axis':0},{'channel_axis':2},{'pattern':'unsafe'},{'extra':1},
])
def test_unsafe_fixture_rejected_before_allocation(change):
    row=copy.deepcopy(b.TIMING_CASES[0]);row.update(change)
    class NoArrayModule:
        def __getattr__(self,name):raise AssertionError('allocation before safe case certificate')
    with pytest.raises(b.SafetyError):b.fixture(row,NoArrayModule())


@pytest.mark.parametrize('which',['small','nan','readonly','endian','unaligned','strided','alias','output_shape','output_nan','max','weight','eps','iso'])
def test_prepared_entry_certificate_blocks_unsafe_cython(which):
    image=np.zeros((4,5,1),dtype=np.float32);out=np.zeros((6,7,1),dtype=np.float32)
    args=[image,np.float32(.5),2,1e-6,True,out]
    if which=='small':args[0]=np.zeros((1,5,1),dtype=np.float32)
    elif which=='nan':image[0,0]=np.nan
    elif which=='readonly':image.flags.writeable=False
    elif which=='endian':args[0]=image.astype('>f4')
    elif which=='unaligned':args[0]=np.ndarray(image.shape,dtype='f4',buffer=np.zeros(image.nbytes+1,dtype='u1'),offset=1)
    elif which=='strided':args[0]=image[::-1]
    elif which=='alias':args[5]=image
    elif which=='output_shape':args[5]=out[:4]
    elif which=='output_nan':out[0,0]=np.inf
    elif which=='max':args[2]=True
    elif which=='weight':args[1]=np.float32(0)
    elif which=='eps':args[3]=float('nan')
    elif which=='iso':args[4]=1
    with pytest.raises(b.SafetyError):b.certify_entry(np,*args)


def test_full_raw_values_and_layout_are_retained(tmp_path):
    image=np.arange(40,dtype=np.float32).reshape(5,8)[::-1,::2]
    row=b.array_record(image,np,tmp_path)
    raw=zlib.decompress((tmp_path/(row['sha256']+'.bin.z')).read_bytes())
    assert raw==image.tobytes()
    assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    assert row['strides']==list(image.strides) and not row['C'] and not row['own']


def test_frozen_manifest_integrity_and_native_zero():
    manifest=json.loads(b.MANIFEST.read_text());b.verify_manifest_metadata(manifest)
    assert set(manifest['profiles'])=={'windows','modern','minimum'}
    for profile,row in manifest['profiles'].items():
        assert len(row['reference135'])==135
        assert len(row['timing_observation_sha256'])==110
        assert len(row['blocked_nonfinite'])==4
        assert row['runtime']['helper_binary']['sha256']==b.REFERENCE_BINARY_SHA256[profile]


@pytest.mark.parametrize('change',[
    {'accepted_product_pins':{}},{'native_checks':1},{'native_checks':False},
    {'operation_timings':1},{'operation_timings':False},{'timing_cases':[]},
    {'timing_spec_sha256':'0'*64},{'raw_matrix_sha256':'0'*64},
    {'helper_source_canonical_lf_sha256':'0'*64},{'harness_source_canonical_lf_sha256':'0'*64},
    {'scientific_function_sha256':{}},
])
def test_manifest_tampering_is_rejected(change):
    manifest=json.loads(b.MANIFEST.read_text());manifest.update(change)
    with pytest.raises(b.SafetyError):b.verify_manifest_metadata(manifest)


@pytest.mark.parametrize('change',['native','case','value','warning','source','runtime','missing','duplicate'])
def test_reference_receipt_tampering_is_rejected(change):
    manifest=json.loads(b.MANIFEST.read_text());expected=manifest['profiles']['windows']
    report={'profile':'windows','native_checks':0,'operation_timings':0,'accepted_product_pins':None,
            'reference135':copy.deepcopy(expected['reference135']),
            'blocked_nonfinite':copy.deepcopy(expected['blocked_nonfinite']),
            'timing_cases':copy.deepcopy(expected['timing_cases']),
            'runtime':copy.deepcopy(expected['runtime'])}
    if change=='native':report['native_checks']=1
    elif change=='case':report['reference135'][0]['id']='different'
    elif change=='value':report['timing_cases'][0]['digest']='0'*64
    elif change=='warning':report['blocked_nonfinite'][0]['observation']['error']['message']='different'
    elif change=='source':report['runtime']['public_source_sha256']='0'*64
    elif change=='runtime':report['runtime']['numpy']='wrong'
    elif change=='missing':report['timing_cases'].pop()
    elif change=='duplicate':report['timing_cases'].append(copy.deepcopy(report['timing_cases'][0]))
    with pytest.raises(b.SafetyError):b.verify_report(report,manifest)


def test_safe_conversion_layout_and_prepared_channel_semantics():
    # Actual original numerical observations live in the separately frozen three
    # runtime reference reports. This test checks owned fixture setup only.
    for row in b.TIMING_CASES:
        image,owner,kwargs=b.fixture(row,np)
        assert np.isfinite(image).all() and image.dtype==np.dtype(row['dtype'])
        if row['layout']=='unaligned-step':assert not image.flags.aligned and not image.flags.c_contiguous
        if row['layout']=='unaligned':assert image.dtype.kind=='u' and not image.flags.aligned
        if row['channel_axis'] is not None:assert image.shape[row['channel_axis']]==row['shape'][-1]
        assert kwargs['max_num_iter']>0
