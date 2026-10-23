"""The resize harness must remain untimed until all strict parity gates pass."""
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest
from benchmarks import resize as harness


def no_timing(*args, **kwargs):
    raise AssertionError('timing was called by an untimed or failing run')


def tiny_corpus():
    yield harness.Case('resize', 'tiny-test', np.ones((3, 5)), (5, 7), {})


def test_reference_only_never_times(monkeypatch, tmp_path):
    monkeypatch.setattr(harness, 'corpus', tiny_corpus)
    monkeypatch.setattr(harness, 'measure', no_timing)
    monkeypatch.setattr(time, 'perf_counter_ns', no_timing)
    monkeypatch.setattr(harness.xx.transform, 'resize', no_timing, raising=False)
    output = tmp_path/'reference.json'
    assert harness.main(['--reference-only', '--check-only', '--output', str(output)]) == 0
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['status'] == 'complete-reference-check'
    assert report['metadata']['reference_only'] is True
    assert report['metadata']['validation_scope'] == 'reference-only; native not exercised'
    assert not report['results'] and not report['failures']
    assert 'speedup' not in report['checks'][0]


def test_missing_registry_is_a_retained_preflight_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(harness, 'native_functions', lambda: ())
    monkeypatch.setattr(harness, 'measure', no_timing)
    output = tmp_path/'pending.json'
    assert harness.main(['--check-only', '--output', str(output)]) == 1
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['status'] == 'failed'
    assert report['failures'][0]['phase'] == 'preflight'
    assert 'public native APIs are not registered' in report['failures'][0]['message']
    assert not report['results']


def test_native_check_restores_global_policy_and_never_times(monkeypatch, tmp_path):
    monkeypatch.setattr(harness, 'corpus', tiny_corpus)
    monkeypatch.setattr(harness, 'native_functions', lambda: harness.APIS)
    monkeypatch.setattr(harness.xx.transform, 'resize', harness.reference.resize, raising=False)
    monkeypatch.setattr(harness, 'measure', no_timing)
    monkeypatch.setattr(time, 'perf_counter_ns', no_timing)
    old_workers = harness.xx.get_num_threads()
    old_fallback = harness.xx.set_fallback_enabled(True)
    try:
        output = tmp_path/'parity.json'
        assert harness.main(['--check-only', '--threads', '1', '4', '--output', str(output)]) == 0
        report = json.loads(output.read_text(encoding='utf-8'))
        assert report['status'] == 'complete-parity-check' and not report['results']
        assert set(report['checks'][0]['max_abs_error_by_workers']) == {'1', '4'}
        assert harness.xx.get_num_threads() == old_workers
        assert harness.xx.set_fallback_enabled(False) is True
    finally:
        harness.xx.set_fallback_enabled(old_fallback)


def test_any_parity_failure_blocks_all_measurements(monkeypatch, tmp_path):
    monkeypatch.setattr(harness, 'corpus', tiny_corpus)
    monkeypatch.setattr(harness, 'native_functions', lambda: harness.APIS)
    monkeypatch.setattr(harness.xx.transform, 'resize', lambda a, target, **kw: np.zeros(target), raising=False)
    monkeypatch.setattr(harness, 'measure', no_timing)
    monkeypatch.setattr(harness, 'verify_prepared', lambda *args: {'test_stub': True})
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    output = tmp_path/'failure.json'
    assert harness.main(['--prepared-source', str(tmp_path/'unused.json'), '--output', str(output)]) == 1
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['status'] == 'failed-parity'
    assert report['failures'][0]['case'] == 'tiny-test'
    assert report['checks'][0]['status'] == 'failed'
    assert not report['results']


def test_corpus_covers_the_public_contract_dimensions():
    cases = list(harness.corpus())
    assert {case.api for case in cases} == {'resize', 'rescale'}
    assert {case.image.dtype.type for case in cases} == set(harness.TYPES)
    assert set(range(6)) <= {case.options.get('order') for case in cases}
    assert set(harness.MODES) <= {case.options.get('mode') for case in cases}
    assert {0, 1, -1} <= {case.options.get('channel_axis') for case in cases}
    assert {True, False} <= {case.options.get('anti_aliasing') for case in cases}
    labels = {case.label for case in cases}
    assert {'fortran', 'reversed', 'stepped', 'endian', 'unaligned', 'broadcast',
            'readonly', 'volume', 'mixed-zero-identity', 'nan-identity', 'zero', 'identity'} <= labels
    specs = [harness.spec(case) for case in cases]
    identifiers = {(item['api'], item['dtype'], item['case']) for item in specs}
    assert len(identifiers) == len(specs), 'case identifiers must be stable and unique'
    assert all(len(item['input_sha256']) == 64 for item in specs)


def test_timing_is_rejected_outside_actions(monkeypatch, tmp_path):
    monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
    monkeypatch.setattr(harness, 'measure', no_timing)
    with pytest.raises(SystemExit) as caught:
        harness.main(['--prepared-source', str(tmp_path/'unused.json'), '--output', str(tmp_path/'unused-result.json')])
    assert caught.value.code == 2


def test_expected_error_is_checked_and_reported_as_untimed(monkeypatch, tmp_path):
    def cases():
        yield harness.Case('resize', 'endian-half-contract', np.ones((3, 5), dtype='>f2'), (2, 3), {}, True)
    monkeypatch.setattr(harness, 'corpus', cases)
    monkeypatch.setattr(harness, 'native_functions', lambda: harness.APIS)
    monkeypatch.setattr(harness.xx.transform, 'resize', harness.reference.resize, raising=False)
    monkeypatch.setattr(harness, 'measure', no_timing)
    output = tmp_path/'expected-error.json'
    assert harness.main(['--check-only', '--output', str(output)]) == 0
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['checks'][0]['expected_exception']['type'] == 'RuntimeError'
    assert report['untimed_cases'][0]['case_index'] == 0
    assert report['metadata']['timing_eligible_case_count'] == 0
    assert not report['results'] and not report['failures']


def test_summary_keeps_slow_cases_and_rejects_partial_corpora():
    rows = [dict(case_index=i, threads=1, speedup=value) for i, value in enumerate((.4, 2., 2.))]
    summary = harness.summarize(rows, [1], 3)['1']
    assert summary['case_count'] == 3 and summary['primary'] is True
    assert summary['slower_cases'] == [dict(case_index=0, speedup=.4)]
    assert math.isclose(summary['geometric_mean_speedup'], 1.6**(1/3))
    with pytest.raises(AssertionError, match='incomplete measurement corpus'):
        harness.summarize(rows, [1], 4)
