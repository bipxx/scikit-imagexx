"""Untimed class corpus guards; unit stubs are never public parity evidence."""
import json
import time
import numpy as np
import pytest
from benchmarks import transform_classes as harness


def no_clock(*args,**kwargs):raise AssertionError('untimed class corpus used a timer')


def tiny():yield harness.Case('ProjectiveTransform','apply','unit-only',count=3)


def test_reference_check_never_calls_public_class_or_timer(monkeypatch,tmp_path):
    monkeypatch.setattr(harness,'corpus',tiny)
    monkeypatch.setattr(time,'perf_counter_ns',no_clock)
    monkeypatch.setattr(time,'perf_counter',no_clock)
    monkeypatch.setattr(harness.xx.transform,'ProjectiveTransform',no_clock)
    path=tmp_path/'reference.json'
    assert harness.main(['--check-only','--reference-only','--output',str(path)])==0
    report=json.loads(path.read_text());assert report['status']=='complete-reference-check'
    assert not report['results'] and not report['failures']


def test_class_registry_gate_is_distinct_from_image_functions(monkeypatch,tmp_path):
    monkeypatch.setattr(harness,'native_classes',lambda:())
    monkeypatch.setattr(time,'perf_counter_ns',no_clock)
    path=tmp_path/'missing.json'
    assert harness.main(['--check-only','--output',str(path)])==1
    report=json.loads(path.read_text());assert 'public classes not registered' in report['failures'][0]['message']
    assert not report['checks'] and not report['results']


def test_actual_public_check_is_untimed_and_restores_global_policy(monkeypatch,tmp_path):
    monkeypatch.setattr(harness,'corpus',tiny)
    monkeypatch.setattr(time,'perf_counter_ns',no_clock)
    workers=harness.xx.get_num_threads();previous=harness.xx.set_fallback_enabled(True)
    try:
        path=tmp_path/'public.json'
        assert harness.main(['--check-only','--threads','1','4','--output',str(path)])==0
        report=json.loads(path.read_text());assert report['status']=='complete-public-class-check'
        assert not report['results'] and harness.xx.get_num_threads()==workers
        assert harness.xx.set_fallback_enabled(False) is True
    finally:harness.xx.set_fallback_enabled(previous)


def test_failure_retains_case_and_never_claims_complete(monkeypatch,tmp_path):
    monkeypatch.setattr(harness,'corpus',tiny)
    monkeypatch.setattr(harness.xx.transform,'ProjectiveTransform',lambda *a,**k:(_ for _ in ()).throw(ValueError('unit failure')))
    path=tmp_path/'failed.json'
    assert harness.main(['--check-only','--output',str(path)])==1
    report=json.loads(path.read_text());assert report['status']=='failed-parity'
    assert len(report['checks'])==len(report['failures'])==1 and not report['results']


def test_tuple_state_and_layout_comparison_is_exact():
    values=np.ones((3,2))
    with pytest.raises(AssertionError):harness.compare(values.T.copy().T,values)
    with pytest.raises(AssertionError):harness.compare((True,values),(False,values))


def test_fixed_cases_cover_classes_operations_and_large_strides():
    cases=list(harness.corpus());assert len(cases)==296
    assert {case.model for case in cases}==set(harness.NAMES)
    assert {'apply','construct','estimate','from_estimate','inverse','compose','residuals','properties','array','pickle','failed-pickle','failed-access'}<={c.operation for c in cases}
    assert {'c','f','reverse','broadcast'}<={c.layout for c in cases}
    assert any(c.count>131072 and c.dimensions==3 for c in cases)
    assert len({tuple(harness.asdict(c).values()) for c in cases})==len(cases)


def test_cli_has_no_timing_mode_and_requires_one_worker(tmp_path):
    for args in [[],['--check-only','--threads','4']]:
        with pytest.raises(SystemExit) as result:harness.main(args+['--output',str(tmp_path/'unused.json')])
        assert result.value.code==2
