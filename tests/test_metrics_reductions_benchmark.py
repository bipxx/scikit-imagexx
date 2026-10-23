"""Metric measurements must never precede complete strict scalar parity."""
import json
import time

import numpy as np
import pytest
from benchmarks import metrics_reductions as harness


def no_timing(*args, **kwargs):
    raise AssertionError('untimed/failing metric run reached timing')


def tiny_corpus():
    yield harness.Case(harness.MSE, 'tiny-test', np.ones((3, 5)), np.zeros((3, 5)), {})


def test_reference_check_never_calls_native_or_timer(monkeypatch, tmp_path):
    monkeypatch.setattr(harness, 'corpus', tiny_corpus)
    monkeypatch.setattr(harness, 'measure', no_timing)
    monkeypatch.setattr(time, 'perf_counter_ns', no_timing)
    monkeypatch.setattr(harness.xx.metrics, harness.MSE, no_timing, raising=False)
    output=tmp_path/'reference.json'
    assert harness.main(['--reference-only','--check-only','--output',str(output)])==0
    report=json.loads(output.read_text(encoding='utf-8'))
    assert report['status']=='complete-reference-check'
    assert report['metadata']['validation_scope']=='reference-only; native not exercised'
    assert report['checks'][0]['expected_scalar']['type']=='numpy.float64'
    assert not report['results'] and not report['failures']


def test_registry_gate_retains_missing_native_api_failure(monkeypatch,tmp_path):
    monkeypatch.setattr(harness,'native_functions',lambda:())
    monkeypatch.setattr(harness,'measure',no_timing)
    output=tmp_path/'missing.json'
    assert harness.main(['--check-only','--output',str(output)])==1
    report=json.loads(output.read_text(encoding='utf-8'))
    assert report['failures'][0]['phase']=='preflight'
    assert 'public native APIs are not registered' in report['failures'][0]['message']
    assert not report['results']


def test_native_parity_check_restores_policy_and_never_times(monkeypatch,tmp_path):
    monkeypatch.setattr(harness,'corpus',tiny_corpus)
    monkeypatch.setattr(harness,'native_functions',lambda:harness.APIS)
    monkeypatch.setattr(harness.xx.metrics,harness.MSE,harness.reference.mean_squared_error,raising=False)
    monkeypatch.setattr(harness,'measure',no_timing)
    monkeypatch.setattr(time,'perf_counter_ns',no_timing)
    old_workers=harness.xx.get_num_threads();old_fallback=harness.xx.set_fallback_enabled(True)
    try:
        output=tmp_path/'parity.json'
        assert harness.main(['--check-only','--threads','1','4','--output',str(output)])==0
        report=json.loads(output.read_text(encoding='utf-8'))
        assert report['status']=='complete-parity-check'
        assert set(report['checks'][0]['error_by_workers'])=={'1','4'}
        assert not report['results']
        assert harness.xx.get_num_threads()==old_workers
        assert harness.xx.set_fallback_enabled(False) is True
    finally:
        harness.xx.set_fallback_enabled(old_fallback)


def test_scalar_failure_blocks_all_timings(monkeypatch,tmp_path):
    monkeypatch.setattr(harness,'corpus',tiny_corpus)
    monkeypatch.setattr(harness,'native_functions',lambda:harness.APIS)
    monkeypatch.setattr(harness.xx.metrics,harness.MSE,lambda *args:np.float64(0),raising=False)
    monkeypatch.setattr(harness,'measure',no_timing)
    monkeypatch.setattr(harness,'verify_prepared',lambda *args:{'unit_test_stub':True})
    monkeypatch.setenv('GITHUB_ACTIONS','true')
    output=tmp_path/'failure.json'
    assert harness.main(['--prepared-source',str(tmp_path/'unused.json'),'--output',str(output)])==1
    report=json.loads(output.read_text(encoding='utf-8'))
    assert report['status']=='failed-parity' and report['failures'][0]['case']=='tiny-test'
    assert not report['results']


def test_expected_errors_are_verified_and_not_timed(monkeypatch,tmp_path):
    def corpus():
        yield harness.Case(harness.NRMSE,'invalid-norm',np.ones((3,5)),np.zeros((3,5)),dict(normalization='invalid'),True)
    monkeypatch.setattr(harness,'corpus',corpus)
    monkeypatch.setattr(harness,'native_functions',lambda:harness.APIS)
    monkeypatch.setattr(harness.xx.metrics,harness.NRMSE,harness.reference.normalized_root_mse,raising=False)
    monkeypatch.setattr(harness,'measure',no_timing)
    output=tmp_path/'error.json'
    assert harness.main(['--check-only','--output',str(output)])==0
    report=json.loads(output.read_text(encoding='utf-8'))
    assert report['checks'][0]['expected_exception']['type']=='ValueError'
    assert report['checks'][0]['error_parity_workers']==[1]
    assert report['metadata']['timing_eligible_case_count']==0
    assert len(report['untimed_cases'])==1 and not report['results']


def test_scalar_precision_contract_uses_work_dtype_and_exact_zero_sign():
    pair32=harness.Case(harness.MSE,'float32',np.ones(3,np.float32),np.zeros(3,np.float16),{})
    assert harness.assert_result(np.float64(1+1e-6),np.float64(1),pair32)['rtol']==3e-6
    pair64=harness.Case(harness.MSE,'float64',np.ones(3,np.float64),np.zeros(3,np.float16),{})
    with pytest.raises(AssertionError):
        harness.assert_result(np.float64(1+1e-6),np.float64(1),pair64)
    with pytest.raises(AssertionError,match='zero sign'):
        harness.assert_result(np.float64(-0.),np.float64(0.),pair64)
    with pytest.raises(AssertionError):
        harness.assert_result(np.float64(0.),np.float64(1e-20),pair32)


def test_corpus_covers_promotions_layouts_normalizations_and_ranges():
    cases=list(harness.corpus())
    pairs={(a,b) for a in harness.TYPES for b in harness.TYPES}
    for api in (harness.MSE,harness.NRMSE,harness.PSNR):
        assert {(c.first.dtype.type,c.second.dtype.type) for c in cases if c.api==api and c.label=='promotion'}==pairs
    labels={c.label for c in cases}
    assert {'large-c','fortran','mixed-c-f','reversed','stepped','endian','unaligned','broadcast','readonly','tagged',
            'scalar','volume','tiny','identical','zero-reference','megapixel','nan','infinity','all-nan','finite-extreme','one-ulp-defect'} <= labels
    assert {'euclidean','min-max','mean','EuClIdEaN'} <= {c.options.get('normalization') for c in cases}
    assert any(c.api==harness.PSNR and 'data_range' in c.options for c in cases)
    specs=[harness.spec(case) for case in cases]
    keys={(c['api'],c['first']['dtype'],c['second']['dtype'],c['case']) for c in specs}
    assert len(keys)==len(specs),'case identifiers must be stable and unique'
    json.dumps(specs,allow_nan=False)


def test_local_timing_is_rejected(monkeypatch,tmp_path):
    monkeypatch.delenv('GITHUB_ACTIONS',raising=False)
    monkeypatch.setattr(harness,'measure',no_timing)
    with pytest.raises(SystemExit) as caught:
        harness.main(['--prepared-source',str(tmp_path/'unused.json'),'--output',str(tmp_path/'unused-result.json')])
    assert caught.value.code==2
