"""Metadata/control-flow tests; these do not import scientific packages or time APIs."""
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from benchmarks import phase3_public as harness


def test_fixed_workloads_preserve_existing_coloc_specs_and_cover_all_eight_apis():
    rows = harness.case_specs()
    assert len(rows) == len({r['id'] for r in rows}) == 73
    assert {r['api'] for r in rows} == set(harness.APIS)
    coloc = [r['fixture'] for r in rows if r['family'] == 'colocalization']
    original = harness.COLOC_FACTORY.benchmark_cases()
    assert coloc[:18] == original
    assert coloc[18:] == [dict(r, id=r['id'] + '-F', layout='F')
                          for r in original if r['name'] == 'cache']
    assert all(math.prod(r['shape']) <= 1024**2 for r in coloc)
    footprint = [r for r in rows if r['family'] == 'footprint']
    for api in harness.FOOTPRINTS:
        selected = [r for r in footprint if r['api'] == 'morphology.' + api]
        assert {r['tier'] for r in selected} == {'tiny', 'cache', 'large'}
        if api != 'footprint_from_sequence':
            assert {r['dtype'] for r in selected} == {'uint8', 'float64'}
    assert {r['layout'] for r in footprint if r['work_kind'] == 'sequence_expansion'} == {'C', 'F', 'reverse'}
    for row in footprint:
        name, values = row['fixture']['api'], row['fixture']['args']
        if name == 'ellipse': elements = math.prod(2*x+1 for x in values)
        elif name == 'octagon': elements = (values[0]+2*values[1])**2
        elif name == 'octahedron': elements = (2*values[0]+1)**3
        elif name == 'star': elements = (2*values[0]+1+2*(values[0]//2))**2
        else: elements = (2*row['fixture']['kwargs']['future_repetitions']+1)**2
        assert elements <= 1024**2
    assert {r['fixture']['kwargs'].get('decomposition') for r in footprint} >= {'crosses', 'sequence', None}


def fake_runtime(monkeypatch, fail_index=None):
    """Only exercise report/barrier control flow, never simulate native parity."""
    state = dict(workers=4, fallback=True, checked=[], measured=[])
    compat = SimpleNamespace(_fallback_enabled=True)
    def workers(value): state['workers'] = value
    def fallback(value):
        previous = compat._fallback_enabled
        compat._fallback_enabled = value
        state['fallback'] = value
        return previous
    xx = SimpleNamespace(get_num_threads=lambda: state['workers'], set_num_threads=workers,
                         set_fallback_enabled=fallback, has_avx2=lambda: False)
    np = SimpleNamespace(__version__='metadata-stub', geterr=lambda: dict(harness.DEFAULT_POLICY),
                         geterrcall=lambda: None, getbufsize=lambda: 8192)
    helpers = SimpleNamespace(cpu_metadata=lambda: {}, source_revision=lambda _: 'metadata-source',
        binary_hashes=lambda: {}, python_hashes=lambda: {},
        verify_prepared=lambda *args: {'metadata_fixture': True})
    def measure(reference, native, rounds, target):
        assert len(state['checked']) == 73, 'timer entered before complete parity barrier'
        state['measured'].append((rounds, target))
        return dict(reference_ms=1.0, cpp_ms=2.0, speedup=0.5, samples_ms=[[1.0]*7, [2.0]*7],
                    loops=[1, 1], sample_labels=['scikit-image', 'current'])
    runtime = SimpleNamespace(xx=xx, compat=compat, np=np, scipy=SimpleNamespace(__version__='metadata-stub'),
        skimage=SimpleNamespace(__version__='0.26.0'), helpers=helpers, measure=measure,
        pool_limits=lambda **kwargs: nullcontext(), pool_info=lambda: [dict(num_threads=1)],
        functions={api: (lambda: None, lambda: None) for api in harness.APIS})
    def check(runtime, spec, row):
        state['checked'].append(spec['id'])
        row['input'] = 'metadata-fixture'
        row['actual_retained_before_failure'] = True
        if len(state['checked']) - 1 == fail_index:
            raise AssertionError('retained parity failure')
    monkeypatch.setattr(harness, '_runtime', lambda: runtime)
    monkeypatch.setattr(harness, '_check_case', check)
    monkeypatch.setattr(harness, '_fixture', lambda *args: ((), {}))
    monkeypatch.setattr(harness, '_snapshot', lambda *args: 'metadata-fixture')
    monkeypatch.setattr(harness, '_verify_wheel', lambda *args: {'metadata_fixture': True})
    monkeypatch.setattr(harness, 'source_fingerprint', lambda *args: {})
    return runtime, state


def test_check_only_visits_complete_cohort_without_timing_and_restores_policy(monkeypatch, tmp_path):
    _, state = fake_runtime(monkeypatch)
    output = tmp_path/'check.json'
    assert harness.main(['--check-only', '--output', str(output)]) == 0
    report = json.loads(output.read_bytes())
    assert report['status'] == 'complete-parity-check'
    assert len(report['checks']) == len(state['checked']) == 73
    assert not report['results'] and not state['measured']
    assert state['workers'] == 4 and state['fallback'] is True


def test_even_last_parity_failure_blocks_every_timer_and_retains_observation(monkeypatch, tmp_path):
    _, state = fake_runtime(monkeypatch, fail_index=72)
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    output = tmp_path/'failed.json'
    assert harness.main(['--prepared-source', 'stub.json', '--wheel', 'stub.whl', '--output', str(output)]) == 1
    report = json.loads(output.read_bytes())
    assert report['status'] == 'failed-parity' and len(report['checks']) == 73
    assert report['failures'][0]['actual_retained_before_failure'] is True
    assert report['failures'][0]['message'] == 'retained parity failure'
    assert not state['measured'] and not report['results']
    assert state['workers'] == 4 and state['fallback'] is True


def test_all_raw_paired_samples_and_slower_absolute_outcomes_are_retained(monkeypatch, tmp_path):
    _, state = fake_runtime(monkeypatch)
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    output = tmp_path/'timing-control.json'
    assert harness.main(['--prepared-source', 'stub.json', '--wheel', 'stub.whl', '--output', str(output)]) == 0
    report = json.loads(output.read_bytes())
    assert report['status'] == 'complete-measurement'
    assert state['measured'] == [(7, 15)]*73
    assert len(report['results']) == len(report['slower_cases_by_absolute_overhead']) == 73
    assert len(report['metadata']['paired_sample_order']) == 7
    for row in report['results']:
        assert row['samples_ms'] == [[1.0]*7, [2.0]*7]
        assert row['absolute_overhead_ms'] == 1.0 and row['slower_than_original']
    assert state['workers'] == 4 and state['fallback'] is True


@pytest.mark.parametrize('fault', ['pool', 'worker', 'fallback', 'numpy_policy'])
def test_actual_policy_getters_reject_environment_only_claims(monkeypatch, fault):
    runtime, state = fake_runtime(monkeypatch)
    state['workers'] = 1
    runtime.compat._fallback_enabled = False
    if fault == 'pool': runtime.pool_info = lambda: [dict(num_threads=2)]
    elif fault == 'worker': state['workers'] = 4
    elif fault == 'fallback': runtime.compat._fallback_enabled = True
    else: runtime.np.geterr = lambda: dict(harness.DEFAULT_POLICY, under='warn')
    with pytest.raises(RuntimeError): harness._pools(runtime)


def test_wheel_check_requires_complete_installed_package_identity(tmp_path):
    wheel = tmp_path/'exact.whl'
    contents = {'__init__.py': b'original python', '_core.so': b'exact binary'}
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name, data in contents.items(): archive.writestr('skimagexx/' + name, data)
    installed = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    runtime = SimpleNamespace(xx=SimpleNamespace(__file__=str(tmp_path/'skimagexx/__init__.py')),
                              package_files=lambda _: dict(installed))
    assert harness._verify_wheel(runtime, wheel)['package_files'] == 2
    for changed in ('missing', 'extra', 'tampered'):
        bad = dict(installed)
        if changed == 'missing': del bad['_core.so']
        elif changed == 'extra': bad['unlisted.py'] = '0'*64
        else: bad['_core.so'] = '0'*64
        runtime.package_files = lambda _, values=bad: values
        with pytest.raises(RuntimeError, match='complete package'):
            harness._verify_wheel(runtime, wheel)


@pytest.mark.parametrize('arguments', [[], ['--prepared-source', 'only.json'],
                                    ['--wheel', 'only.whl']])
def test_local_or_unpaired_timing_inputs_fail_before_runtime(monkeypatch, tmp_path, arguments):
    monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
    def forbidden(): raise AssertionError('runtime was imported')
    monkeypatch.setattr(harness, '_runtime', forbidden)
    with pytest.raises(SystemExit) as caught:
        harness.main(arguments + ['--output', str(tmp_path/'unused.json')])
    assert caught.value.code == 2


def test_prior_output_is_never_replaced(monkeypatch, tmp_path):
    output = tmp_path/'prior.json'
    output.write_bytes(b'original evidence')
    with pytest.raises(SystemExit): harness.main(['--check-only', '--output', str(output)])
    assert output.read_bytes() == b'original evidence'


def test_subdivide_combined_selector_scope_and_exact_current_counts():
    root = Path(__file__).resolve().parents[1]
    text = (root / '.github/workflows/phase3-public-benchmark.yml').read_text()
    assert text.count('tests/test_subdivide_polygon_integration.py') == 2
    assert text.count('tests/test_subdivide_polygon_pipeline.py') == 2
    assert 'tests/test_subdivide_polygon_public.py' not in text
    assert 'assert len(cases) == 5472' in text
    assert 'len(native) == 250' in text and 'assert expected == 1447' in text
