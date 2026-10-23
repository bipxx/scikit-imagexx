"""Controlled-harness unit stubs are tooling tests, never native/performance proof."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
from benchmarks import resize_controlled as controlled

helper = controlled.artifacts


def forbidden(*args, **kwargs):
    raise AssertionError('untimed/failing controlled run reached forbidden work')


def unit_cases():
    for index in range(372):
        yield controlled.h.Case('resize', 'unit-only-' + str(index), np.ones((2, 3)), (3, 4), {})
    yield controlled.h.Case('resize', 'unit-error-only', np.ones((2, 3), bool), (3, 4), {'anti_aliasing': True}, True)


def install_unit_stubs(monkeypatch, bad_index=None):
    """Original-delegating unit stubs; explicitly bypass native authentication."""
    state = {'workers': 1, 'fallback': True, 'calls': 0}
    def old_function(*args, **kwargs):
        state['calls'] += 1
        value = controlled.h.reference.resize(*args, **kwargs)
        if bad_index is not None and state['calls'] == bad_index:
            value.flat[0] += 1
        return value
    def fallback(value):
        previous = state['fallback']; state['fallback'] = bool(value); return previous
    old = SimpleNamespace(__name__='unit_stub_old', get_num_threads=lambda: state['workers'],
        set_num_threads=lambda value: state.update(workers=value), set_fallback_enabled=fallback,
        transform=SimpleNamespace(resize=old_function), policy=SimpleNamespace())
    real_policies = controlled.policies
    monkeypatch.setattr(controlled, 'policies', lambda m: SimpleNamespace(_fallback_enabled=state['fallback']) if m is old else real_policies(m))
    monkeypatch.setattr(controlled, 'verify_backends', lambda *args: {'unit_stub_only': True})
    monkeypatch.setattr(controlled.h, 'corpus', unit_cases)
    monkeypatch.setattr(controlled.h, 'native_functions', lambda: controlled.h.APIS)
    monkeypatch.setattr(controlled.h.xx.transform, 'resize', controlled.h.reference.resize)
    specs = [controlled.h.spec(c) for c in unit_cases()]
    contracts = [{k: v for k, v in item.items() if k != 'input_sha256'} for item in specs]
    initial = {'case_contract_sha256': hashlib.sha256(json.dumps(contracts, sort_keys=True).encode()).hexdigest(),
        'case_spec_sha256': 'unit-only', 'python': 'unit-only', 'numpy': 'unit-only', 'scipy': 'unit-only', 'scikit_image': '0.26.0'}
    seal = {'unit_stub_only': True, 'initial': {'initial_measurement': initial, 'manifest': {'unit_stub_only': True},
                                             'authenticated_sha256': {'unit_stub_only': True}}}
    monkeypatch.setattr(helper, 'verify', lambda *a: seal)
    monkeypatch.setattr(helper, 'load_baseline', lambda *a: old)
    return old, state


def test_reference_only_does_not_load_old_or_time(monkeypatch, tmp_path):
    monkeypatch.setattr(helper, 'verify', forbidden); monkeypatch.setattr(helper, 'load_baseline', forbidden)
    monkeypatch.setattr(controlled.h, 'measure', forbidden); monkeypatch.setattr(time, 'perf_counter_ns', forbidden)
    monkeypatch.setattr(controlled.h, 'corpus', lambda: iter([next(unit_cases())]))
    output = tmp_path / 'reference.json'
    assert controlled.main(['--reference-only', '--check-only', '--output', str(output)]) == 0
    report = json.loads(output.read_text())
    assert report['status'] == 'complete-reference-check' and not report['results']


def test_missing_registry_precedes_capsule_and_fixtures(monkeypatch, tmp_path):
    monkeypatch.setattr(controlled.h, 'native_functions', lambda: ())
    monkeypatch.setattr(helper, 'verify', forbidden); monkeypatch.setattr(controlled.h, 'corpus', forbidden)
    monkeypatch.setattr(controlled.h, 'measure', forbidden)
    output = tmp_path / 'missing.json'
    assert controlled.main(['--capsule', str(tmp_path), '--check-only', '--output', str(output)]) == 1
    report = json.loads(output.read_text()); assert not report['checks'] and not report['results']
    assert 'not registered' in report['failures'][0]['message']


def test_capsule_rejection_precedes_fixtures(monkeypatch, tmp_path):
    monkeypatch.setattr(controlled.h, 'native_functions', lambda: controlled.h.APIS)
    def reject(*args): raise AssertionError('unit tampered capsule')
    monkeypatch.setattr(helper, 'verify', reject); monkeypatch.setattr(controlled.h, 'corpus', forbidden)
    monkeypatch.setattr(controlled.h, 'measure', forbidden)
    assert controlled.main(['--capsule', str(tmp_path), '--check-only']) == 1


def test_unit_check_checks_both_backends_and_restores_policies(monkeypatch, tmp_path):
    old, state = install_unit_stubs(monkeypatch)
    monkeypatch.setattr(controlled.h, 'measure', forbidden); monkeypatch.setattr(time, 'perf_counter_ns', forbidden)
    current_workers = controlled.h.xx.get_num_threads()
    current_fallback = controlled.policies(controlled.h.xx)._fallback_enabled
    output = tmp_path / 'unit-check.json'
    assert controlled.main(['--capsule', str(tmp_path), '--check-only', '--threads', '1', '4', '--output', str(output)]) == 0
    r = json.loads(output.read_text()); assert len(r['checks']) == 373 and not r['results']
    assert r['metadata']['backend_identity']['unit_stub_only'] is True
    assert set(r['checks'][0]['backends']) == {'current', 'baseline'}
    assert all(set(v) == {'1', '4'} for v in r['checks'][0]['backends'].values())
    assert state['workers'] == 1 and state['fallback'] is True
    assert controlled.h.xx.get_num_threads() == current_workers
    assert controlled.policies(controlled.h.xx)._fallback_enabled == current_fallback


def test_late_old_failure_blocks_every_timer(monkeypatch, tmp_path):
    install_unit_stubs(monkeypatch, bad_index=371)
    monkeypatch.setattr(controlled.h, 'measure', forbidden); monkeypatch.setattr(time, 'perf_counter_ns', forbidden)
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    output = tmp_path / 'late-bad-unit.json'
    assert controlled.main(['--capsule', str(tmp_path), '--output', str(output)]) == 1
    r = json.loads(output.read_text())
    assert r['status'] == 'failed-parity' and len(r['checks']) == 373 and not r['results']


def test_unit_measure_dispatches_three_callbacks_and_keeps_raw_vectors(monkeypatch, tmp_path):
    old, state = install_unit_stubs(monkeypatch)
    monkeypatch.setenv('GITHUB_ACTIONS', 'true'); monkeypatch.setattr(time, 'perf_counter_ns', forbidden)
    calls = []
    def unit_measure(reference, current, repeat, target_ms, baseline=None):
        assert baseline is not None and state['fallback'] is False
        np.testing.assert_array_equal(reference(), current()); np.testing.assert_array_equal(current(), baseline())
        calls.append(state['workers'])
        return dict(reference_ms=3., cpp_ms=2., baseline_ms=1., speedup=1.5, improvement=.5,
                    samples_ms=[[3.] * repeat, [2.] * repeat, [1.] * repeat], loops=[1, 1, 1],
                    sample_labels=['scikit-image', 'current', 'baseline'])
    monkeypatch.setattr(controlled.h, 'measure', unit_measure)
    output = tmp_path / 'unit-measure-only.json'
    assert controlled.main(['--capsule', str(tmp_path), '--repeat', '2', '--output', str(output)]) == 0
    r = json.loads(output.read_text()); assert len(calls) == len(r['results']) == 372
    assert r['metadata']['backend_identity']['unit_stub_only'] is True
    assert r['results'][0]['samples_ms'] == [[3., 3.], [2., 2.], [1., 1.]]
    assert r['results'][0]['baseline_speedup'] == 3. and r['results'][0]['regression_vs_initial'] is True
    assert len(r['summary_by_workers']['1']['regressions']) == 372
    assert all(set(order) == {'scikit-image', 'current', 'baseline'} for order in r['metadata']['paired_sample_order'])


@pytest.mark.parametrize('args', [[], ['--capsule', 'unused'], ['--reference-only'],
    ['--reference-only', '--check-only', '--capsule', 'unused'], ['--capsule', 'unused', '--check-only', '--threads', '4'],
    ['--capsule', 'unused', '--check-only', '--threads', '1', '1']])
def test_invalid_or_local_timing_configuration_rejected(monkeypatch, args):
    monkeypatch.delenv('GITHUB_ACTIONS', raising=False); monkeypatch.setattr(controlled.h, 'measure', forbidden)
    with pytest.raises(SystemExit): controlled.main(args)


@pytest.fixture
def initial_fixture(tmp_path, monkeypatch):
    initial = tmp_path / 'initial'; source = tmp_path / 'old-source'
    for directory in ('src', 'include/skimagexx', 'python/skimagexx', 'benchmarks'):
        (source / directory).mkdir(parents=True, exist_ok=True)
    (source / 'CMakeLists.txt').write_text('# unit source\n'); (source / 'pyproject.toml').write_text('# unit project\n')
    (source / 'python/skimagexx/__init__.py').write_text('# unit old Python\n')
    shutil.copyfile(controlled.ROOT / 'benchmarks/resize.py', source / 'benchmarks/resize.py')
    wheel = initial / 'prepared/wheels/unit.whl'; wheel.parent.mkdir(parents=True)
    package = initial / 'prepared/exact-package/_baseline'; package.mkdir(parents=True)
    members = {'__init__.py': b'# unit old Python\n', '_core.so': b'unit-only binary'}
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name, value in members.items(): archive.writestr('skimagexx/' + name, value); (package / name).write_bytes(value)
    fingerprint = helper.source_fingerprint(source)
    monkeypatch.setattr(helper, 'INITIAL_SOURCE', fingerprint['source_sha256'])
    binary = {'_core.so': helper.sha(package / '_core.so')}
    manifest = initial / 'prepared/exact-package/baseline-source.json'
    manifest.write_text(json.dumps(dict(revision=helper.INITIAL_REVISION, binary_sha256=binary, **fingerprint)))
    profile = 'windows-modern'
    data = dict(github_run_id=helper.INITIAL_RUN, source_revision=helper.INITIAL_REVISION, **fingerprint,
        case_count=373, timing_eligible_case_count=372, requested_workers=[1], current_binary_sha256=binary,
        current_python_sha256={'__init__.py': helper.sha(package / '__init__.py')}, prepared_source={'manifest_sha256': helper.sha(manifest)},
        platform='Windows-unit-only', python='3.13.unit-only', numpy='2.5.4', scipy='1.18.1', scikit_image='0.26.0')
    measurement = initial / 'results/resize-performance.json'; measurement.parent.mkdir()
    measurement.write_text(json.dumps(dict(status='complete-measurement', failures=[], metadata=data,
        checks=[{'status': 'passed'}] * 373, results=[dict(case_index=i, threads=1) for i in range(372)])))
    monkeypatch.setitem(helper.PINS, profile, {'manifest': helper.sha(manifest), 'measurement': helper.sha(measurement), 'wheel': helper.sha(wheel)})
    return initial, source, profile


def test_initial_identity_binds_all_python_binary_source_and_measurement(initial_fixture):
    initial, source, profile = initial_fixture
    result = helper.baseline_inputs(initial, source, profile, check_runtime=False)
    assert result['binary_sha256'] and result['manifest']['revision'] == helper.INITIAL_REVISION


@pytest.mark.parametrize('target', ['manifest', 'measurement', 'wheel', 'python', 'binary', 'extra-python', 'source', 'corpus'])
def test_tampered_initial_inputs_rejected(initial_fixture, target):
    initial, source, profile = initial_fixture
    paths = {'manifest': initial / 'prepared/exact-package/baseline-source.json', 'measurement': initial / 'results/resize-performance.json',
        'wheel': initial / 'prepared/wheels/unit.whl', 'python': initial / 'prepared/exact-package/_baseline/__init__.py',
        'binary': initial / 'prepared/exact-package/_baseline/_core.so', 'extra-python': initial / 'prepared/exact-package/_baseline/extra.py',
        'source': source / 'CMakeLists.txt', 'corpus': source / 'benchmarks/resize.py'}
    path = paths[target]
    if target == 'corpus': path.write_text(path.read_text().replace('SEED = 5781', 'SEED = 5782'))
    else: path.write_bytes((path.read_bytes() if path.exists() else b'') + b'changed')
    with pytest.raises(AssertionError): helper.baseline_inputs(initial, source, profile, check_runtime=False)


def test_wrong_runtime_rejected_before_old_module_load(initial_fixture, monkeypatch):
    monkeypatch.setattr(helper.platform, 'system', lambda: 'wrong-OS')
    with pytest.raises(AssertionError, match='ABI'):
        helper.baseline_inputs(*initial_fixture)


def test_baseline_loader_rejects_current_module_alias(monkeypatch):
    with pytest.raises(AssertionError, match='distinct'):
        controlled.verify_backends(controlled.h.xx, controlled.h.xx)


@pytest.fixture
def backend_modules(tmp_path, monkeypatch):
    modules = []
    for index, name in enumerate(('unit_current', 'unit_initial')):
        state = {'workers': 1}
        policy = SimpleNamespace(_fallback_enabled=True)
        path = tmp_path / (name + '.so'); path.write_bytes(bytes([index]))
        core = SimpleNamespace(__file__=str(path))
        core.get_num_threads = lambda state=state: state['workers']
        core.set_num_threads = lambda value, state=state: state.update(workers=value)
        def fallback(value, policy=policy):
            old = policy._fallback_enabled; policy._fallback_enabled = bool(value); return old
        functions = {}
        for function_name in ('resize', 'rescale'):
            function = eval('lambda *args, **kwargs: None', {'_core': core})
            function.__module__ = name + '.transform'; functions[function_name] = function
        module = SimpleNamespace(__name__=name, _core=core, transform=SimpleNamespace(**functions),
            get_num_threads=core.get_num_threads, set_num_threads=core.set_num_threads,
            set_fallback_enabled=fallback, unit_policy=policy)
        monkeypatch.setitem(sys.modules, name + '.coverage', SimpleNamespace(native_functions=lambda: controlled.h.APIS))
        monkeypatch.setitem(sys.modules, name + '._compat', policy)
        modules.append(module)
    return modules


def test_independence_probe_restores_unit_module_states(backend_modules):
    proof = controlled.verify_backends(*backend_modules)
    assert proof['independent_workers'] and proof['independent_fallback']
    assert all(m.get_num_threads() == 1 and m.unit_policy._fallback_enabled for m in backend_modules)


@pytest.mark.parametrize('bad', ['forwarded', 'wrong-core', 'shared-workers', 'shared-fallback', 'wrong-default'])
def test_bad_backend_identity_or_shared_state_rejected(backend_modules, monkeypatch, bad):
    current, old = backend_modules
    if bad == 'forwarded': old.transform.resize = controlled.h.reference.resize
    elif bad == 'wrong-core': old.transform.resize.__globals__['_core'] = current._core
    elif bad == 'wrong-default': old.set_num_threads(4)
    elif bad == 'shared-workers':
        old.get_num_threads = old._core.get_num_threads = current.get_num_threads
        old.set_num_threads = old._core.set_num_threads = current.set_num_threads
    elif bad == 'shared-fallback':
        old.set_fallback_enabled = current.set_fallback_enabled
        monkeypatch.setitem(sys.modules, old.__name__ + '._compat', current.unit_policy)
    with pytest.raises(AssertionError): controlled.verify_backends(current, old)


def test_current_source_requires_complete_python_and_matching_sdk(tmp_path):
    root = tmp_path / 'source'
    for name in ('src', 'include/skimagexx', 'python/skimagexx'):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / 'CMakeLists.txt').write_text('# fixture'); (root / 'pyproject.toml').write_text('# fixture')
    (root / 'include/skimagexx/image.hpp').write_bytes(b'header')
    (root / 'python/skimagexx/__init__.py').write_bytes(b'python')
    capsule = tmp_path / 'capsule'; (capsule / 'wheels').mkdir(parents=True)
    with zipfile.ZipFile(capsule / 'wheels/unit.whl', 'w') as z:
        z.writestr('skimagexx/__init__.py', b'python'); z.writestr('include/skimagexx/image.hpp', b'header')
    helper.current_source(capsule, root)
    (root / 'python/skimagexx/extra.py').write_bytes(b'extra')
    with pytest.raises(AssertionError, match='Python'):
        helper.current_source(capsule, root)


def test_omission_from_wheel_and_install_rejected_before_fixtures_or_timers(tmp_path, monkeypatch):
    root = tmp_path / 'source'
    for directory in ('src', 'include/skimagexx', 'python/skimagexx'):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / 'CMakeLists.txt').write_text('# fixture'); (root / 'pyproject.toml').write_text('# fixture')
    (root / 'python/skimagexx/__init__.py').write_bytes(b'python')
    (root / 'include/skimagexx/image.hpp').write_bytes(b'image header')
    (root / 'include/skimagexx/omitted.hpp').write_bytes(b'must be installed')
    capsule = tmp_path / 'capsule'; (capsule / 'wheels').mkdir(parents=True)
    installed = tmp_path / 'installed/include/skimagexx'; installed.mkdir(parents=True)
    (installed / 'image.hpp').write_bytes(b'image header')
    with zipfile.ZipFile(capsule / 'wheels/unit.whl', 'w') as archive:
        archive.writestr('skimagexx/__init__.py', b'python')
        archive.writestr('include/skimagexx/image.hpp', b'image header')
    # Installed-vs-wheel checks alone accept this shared omission; source must not.
    assert {p.name for p in installed.iterdir()} == {'image.hpp'}
    (capsule / 'consumer-unused').write_bytes(b'fixture')
    (capsule / 'current-runtime.json').write_text('{}')
    manifest = dict(schema='resize-controlled-capsule-v1', profile='windows-modern',
        initial_sha256=helper.PINS['windows-modern'], current_source={}, files_sha256=helper.tree(capsule))
    (capsule / 'controlled-capsule.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(helper.artifacts, 'verify', lambda *args: None)
    current_source = helper.current_source
    monkeypatch.setattr(helper, 'current_source', lambda c, r: current_source(c, root))
    monkeypatch.setattr(controlled.h, 'native_functions', lambda: controlled.h.APIS)
    monkeypatch.setattr(controlled.h, 'corpus', forbidden)
    monkeypatch.setattr(controlled.h, 'measure', forbidden)
    monkeypatch.setattr(time, 'perf_counter_ns', forbidden)
    output = tmp_path / 'omission.json'
    assert controlled.main(['--capsule', str(capsule), '--check-only', '--output', str(output)]) == 1
    report = json.loads(output.read_text())
    assert not report['checks'] and not report['results']
    assert 'complete current SDK header set' in report['failures'][0]['message']


def test_export_rejects_source_path_escape(tmp_path):
    archive = tmp_path / 'escape.zip'
    with zipfile.ZipFile(archive, 'w') as z: z.writestr('../outside.py', '# unsafe')
    with pytest.raises(AssertionError, match='escapes'):
        helper.export_source(archive, tmp_path / 'output')
    assert not (tmp_path / 'outside.py').exists()


def test_capsule_exact_file_set_is_sealed(tmp_path, monkeypatch):
    capsule = tmp_path / 'capsule'; capsule.mkdir(); (capsule / 'input').write_bytes(b'frozen')
    monkeypatch.setattr(helper, 'baseline_inputs', lambda *a: {'authenticated_sha256': helper.PINS['windows-modern']})
    monkeypatch.setattr(helper, 'current_source', lambda *a: {'unit_stub_only': True})
    monkeypatch.setattr(helper.artifacts, 'freeze', lambda root, wheels, path: path.write_text('{}'))
    monkeypatch.setattr(helper.artifacts, 'verify', lambda *a: None)
    helper.freeze(capsule, 'windows-modern')
    (capsule / 'extra').write_bytes(b'unexpected')
    with pytest.raises(AssertionError, match='file set'):
        helper.verify(capsule)


def test_workflow_fresh_hosts_offline_artifacts_and_fixed_old_run():
    text = (controlled.ROOT / '.github/workflows/resize-controlled.yml').read_text()
    prepare, measure = text.split('\n  measure:', 1)
    assert 'needs: prepare' in measure and 'run-id: 39957875987' in prepare
    assert helper.INITIAL_REVISION in prepare
    assert 'resize-controlled' in text and 'include-hidden-files: true' in text
    commands = '\n'.join(line for line in measure.splitlines() if not line.lstrip().startswith('#'))
    assert 'pip wheel' not in commands and 'pytest' not in commands and 'cmake' not in commands
    assert '--no-index --no-deps --require-hashes' in measure
    assert 'controlled-capsule.json' in prepare and '--check-only' in prepare
