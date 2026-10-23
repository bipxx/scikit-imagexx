"""Untimed harness guards; scoped stubs never establish public parity/speed."""
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import time
import zipfile
import numpy as np
import pytest
from benchmarks import transform_class_performance as harness


def no_timer(*args,**kwargs):raise AssertionError('untimed guard reached timer')


def tiny(monkeypatch,cases=None):
    cases=cases or [harness.contract.Case('ProjectiveTransform','apply','unit-only',count=3)]
    monkeypatch.setattr(harness.contract,'corpus',lambda:iter(cases))
    monkeypatch.setattr(harness,'CORPUS_SHA256',hashlib.sha256(json.dumps([asdict(c) for c in cases],sort_keys=True).encode()).hexdigest())
    return cases


def test_reference_only_never_times_or_calls_native(monkeypatch,tmp_path):
    tiny(monkeypatch);monkeypatch.setattr(harness,'measure',no_timer);monkeypatch.setattr(time,'perf_counter_ns',no_timer)
    monkeypatch.setattr(harness.xx.transform,'ProjectiveTransform',no_timer)
    out=tmp_path/'reference.json'
    assert harness.main(['--check-only','--reference-only','--output',str(out)])==0
    report=json.loads(out.read_text());assert report['status']=='complete-reference-check' and not report['results']


def test_actual_public_check_never_times_and_restores_policy(monkeypatch,tmp_path):
    tiny(monkeypatch);monkeypatch.setattr(harness,'measure',no_timer);monkeypatch.setattr(time,'perf_counter_ns',no_timer)
    workers=harness.xx.get_num_threads();previous=harness.xx.set_fallback_enabled(True)
    try:
        out=tmp_path/'check.json'
        assert harness.main(['--check-only','--threads','1','4','--output',str(out)])==0
        report=json.loads(out.read_text());assert report['status']=='complete-parity-check' and not report['results']
        assert harness.xx.get_num_threads()==workers and harness.xx.set_fallback_enabled(False) is True
    finally:harness.xx.set_fallback_enabled(previous)


def test_missing_registration_and_forwarded_class_are_rejected(monkeypatch,tmp_path):
    tiny(monkeypatch);monkeypatch.setattr(harness,'measure',no_timer)
    with monkeypatch.context() as scope:
        scope.setattr(harness,'native_classes',lambda:())
        out=tmp_path/'missing.json';assert harness.main(['--check-only','--output',str(out)])==1
        assert 'not registered' in json.loads(out.read_text())['failures'][0]['message']
    monkeypatch.setattr(harness.xx.transform,'ProjectiveTransform',harness.contract.original.ProjectiveTransform)
    assert harness.main(['--check-only','--output',str(tmp_path/'forwarded.json')])==1


def test_late_corpus_failure_blocks_every_timing(monkeypatch,tmp_path):
    cases=tiny(monkeypatch,[harness.contract.Case('ProjectiveTransform','apply','first',count=3),harness.contract.Case('AffineTransform','apply','late',count=3)])
    monkeypatch.setenv('GITHUB_ACTIONS','true');monkeypatch.setattr(harness,'measure',no_timer)
    monkeypatch.setattr(harness,'verify_prepared',lambda *a:{'unit_stub':True});monkeypatch.setattr(harness,'verify_sealed',lambda *a:{'unit_stub':True})
    original=harness.check_case
    def check(case,*args):
        if case.label=='late':raise ValueError('late unit parity failure')
        return original(case,*args)
    monkeypatch.setattr(harness,'check_case',check)
    out=tmp_path/'late.json'
    assert harness.main(['--prepared-source',str(tmp_path/'unit-stub'),'--output',str(out)])==1
    report=json.loads(out.read_text());assert report['status']=='failed-parity' and len(report['checks'])==2
    assert report['checks'][0]['status']=='passed' and report['checks'][1]['status']=='failed' and not report['results']


def test_expected_errors_are_retained_untimed(monkeypatch,tmp_path):
    tiny(monkeypatch,[harness.contract.Case('ProjectiveTransform','bad-matrix','expected-error',expected_error=True)])
    monkeypatch.setattr(harness,'measure',no_timer)
    out=tmp_path/'error.json';assert harness.main(['--check-only','--output',str(out)])==0
    report=json.loads(out.read_text());assert len(report['untimed_cases'])==1 and not report['results']


def test_estimate_uses_fresh_target_and_charges_constructor_reset():
    case=harness.contract.Case('AffineTransform','estimate','fit')
    call=harness.prepared_action(harness.contract.original,case)
    with pytest.warns(FutureWarning):success,first=call()
    with pytest.warns(FutureWarning):again,second=call()
    assert success and again and first is not second and not np.shares_memory(first.params,second.params)
    captures=inspect.getclosurevars(call).nonlocals
    assert not captures['src'].flags.writeable and not captures['dst'].flags.writeable
    assert 'constructor-plus-estimate' in harness.timed_scope(case)


def test_prepared_composition_reuses_both_immutable_operands():
    case=harness.contract.Case('ProjectiveTransform','compose','nominal')
    call=harness.prepared_action(harness.contract.original,case);captures=inspect.getclosurevars(call).nonlocals
    assert set(captures)=={'model','right'}
    assert not captures['model'].params.flags.writeable and not captures['right'].params.flags.writeable
    first,second=call(),call();assert first is not second
    np.testing.assert_array_equal(first.params,second.params)


def test_frozen_case_ids_hash_and_work_kinds_remain_exact():
    cases=list(harness.contract.corpus());specs=[asdict(case) for case in cases]
    assert len(cases)==296
    assert hashlib.sha256(json.dumps(specs,sort_keys=True).encode()).hexdigest()==harness.CORPUS_SHA256
    assert {harness.work_kind(c) for c in cases}=={'public-coordinate-call','runtime-fitting-or-matrix-provider','state-metadata-or-serialization'}
    assert all(harness.work_kind(c)=='state-metadata-or-serialization' for c in cases if c.operation=='array')
    assert sum(c.expected_error for c in cases)==20


def test_local_measurements_and_nonprimary_workers_rejected(monkeypatch,tmp_path):
    monkeypatch.delenv('GITHUB_ACTIONS',raising=False);monkeypatch.setattr(harness,'measure',no_timer)
    for args in [[],['--reference-only'],['--check-only','--threads','4'],['--check-only','--seal-wheel','fake.whl']]:
        with pytest.raises(SystemExit) as error:harness.main(args+['--output',str(tmp_path/'out.json')])
        assert error.value.code==2


def test_wheel_sdk_harness_seal_rejects_tampering(monkeypatch,tmp_path):
    site=tmp_path/'site';package=site/'skimagexx';prepared=tmp_path/'artifacts/exact-package/baseline-source.json'
    prepared.parent.mkdir(parents=True);prepared.write_text('{}')
    wheel=prepared.parent.parent/'wheels/unit-stub.whl';wheel.parent.mkdir()
    members={'skimagexx/__init__.py':b'#unit','skimagexx/_core.so':b'fakecore','include/skimagexx/image.hpp':b'header','lib/libskimagexx.a':b'archive','lib/cmake/skimagexx/skimagexxConfig.cmake':b'cmake','share/licenses/scikit-imagexx/LICENSE':b'license'}
    with zipfile.ZipFile(wheel,'w') as z:
        for name,data in members.items():
            z.writestr(name,data);path=site/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    monkeypatch.setattr(harness.xx,'__file__',str(package/'__init__.py'))
    monkeypatch.setattr(harness,'harness_hashes',lambda root:{'unit-only':'hash'})
    sealed=harness.seal_artifacts(prepared,wheel,site,'unit-corpus');assert sealed['corpus_sha256']=='unit-corpus'
    header=site/'include/skimagexx/image.hpp';header.write_bytes(b'changed')
    with pytest.raises(AssertionError):harness.verify_sealed(prepared,site,'unit-corpus')
    header.write_bytes(b'header');wheel.write_bytes(wheel.read_bytes()+b'changed')
    with pytest.raises(AssertionError):harness.verify_sealed(prepared,site,'unit-corpus')


def test_workflow_measurement_hosts_never_build_or_test():
    path=Path(__file__).parents[1]/'.github/workflows/transform-classes-benchmark.yml';text=path.read_text()
    preparation,measurement=text.split('\n  measure:',1)
    commands='\n'.join(line for line in measurement.splitlines() if not line.lstrip().startswith('#'))
    assert 'needs: prepare' in commands and 'pytest' not in commands and 'pip wheel' not in commands
    assert '--no-deps' in measurement and '--only-binary=:all:' in measurement
    assert '--threads 1 4' in preparation and 'SKIMAGEXX_DISABLE_SIMD' in preparation
