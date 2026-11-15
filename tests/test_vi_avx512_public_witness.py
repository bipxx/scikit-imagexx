"""Inert public-witness controls: no scientific imports, F1 loads or targets."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'tools/vi_avx512_public_witness.py'
CONTEXT = ROOT / 'coverage/contracts/vi-avx512-public-witness-v2.json'
WORKFLOW = ROOT / '.github/workflows/vi-avx512-public-witness.yml'
spec = importlib.util.spec_from_file_location('_inert_vi_public_witness', PATH)
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)
C = json.loads(CONTEXT.read_bytes())
H = lambda b: hashlib.sha256(b).hexdigest()


def blocks(path):
    result = []
    current = None
    for line in path.read_text().splitlines():
        if line.rstrip().endswith("<<'PY'"):
            assert current is None
            current = []
        elif current is not None:
            if line == '          PY':
                result.append('\n'.join(current) + '\n')
                current = None
            else:
                assert line.startswith('          ')
                current.append(line[10:])
    assert current is None
    return result


def calls():
    return [dict(event=event, role=role, counts=dict(original=1, native=int(index >= 2)))
            for index, (event, role) in enumerate((('before', 'original'), ('after', 'original'),
                                                   ('before', 'native'), ('after', 'native')))]


def capture(same=False):
    def output(raw):
        return dict(type=['numpy', 'ndarray'], dtype='<f8', dtype_descr=[['', '<f8']],
                    dtype_metadata=None, shape=[2], strides=[8], nbytes=16,
                    raw_hex=raw.hex(), payload_sha256=H(raw), base_type=None,
                    flags=dict(owndata=True, writeable=True, c_contiguous=True, f_contiguous=True))
    expected = output(bytes(16))
    actual = output(bytes(16) if same else bytes(15) + b'\1')
    before, after = [], []
    for pin in C['input_payload_sha256']:
        before.append(dict(type=['numpy', 'ndarray'], base_type=['numpy', 'ndarray'],
                           payload=pin, dtype='<i8', shape=[128, 128], strides=[1024, 8],
                           own=False, writeable=True, c=True, f=False))
        after.append(dict(type=['numpy', 'ndarray'], base_type=['numpy', 'ndarray'],
                          payload_sha256=pin, dtype='<i8', shape=[128, 128], strides=[1024, 8],
                          nbytes=131072, flags=dict(owndata=False, writeable=True,
                                                   c_contiguous=True, f_contiguous=False)))
    fp = dict(control=895, status=32, mxcsr=8098)
    requirements = C['native_requirements']
    return dict(value_frame_found=True, secondary_errors=[], probe_restored=True,
        own_full32_restored=True, own_before_raw32='00' * 32, own_restored_raw32='00' * 32,
        after_old_probe_raw32='01' * 32, driver_caller_raw32='02' * 32,
        probe_counts=dict(original=1, native=1), case=C['first_case'],
        strict_trace=dict(reference_before=fp, native_before=fp, reference_FP=fp, native_FP=fp,
            accepted=[True], suffix=[True], requirements=requirements,
            returns=[name for name, count in requirements.items() for _ in range(count)]),
        reference_notices=[], native_notices=[], expected=expected, actual=actual,
        input_before=before, input_after=after,
        execute_binding_identity=dict(source_revision=C['qualified_source'],
            runtime_sha256=C['runtime_sha256'], core_sha256=C['core_sha256']),
        status='PASS_OLD_PROBE_SAMPLER_STOP_NO_TIMING' if same else 'FAILED_OLD_PROBE_NO_RETRY',
        primary_type='AssertionError',
        terminal=dict(sampler_body_calls=0, timed_batches=0, poststop_source_authority_unchanged=True))


def late_capture():
    return dict(role='default', boundary_seen=True, writer_visits=1, complete=True,
        site=C['expected_failure_site'], aliases=dict(expected_input0=False, actual_input0=False, actual_input1=False),
        policy=dict(errors_equal=True, handler_same_object=True, buffer_equal=True), workers=1,
        independent_late_predicates_passed=True)


def result(status='EXPECTED_WORD_MISMATCH', exit_code=1):
    return dict(exit=exit_code, source_helper_verified=True, postidentity_exit=0,
                child=dict(status=status, complete=True, late_gate=late_capture()), value=capture(status == 'PARITY_PASS'))


def test_exact_reused_sources_cases_and_source_budget():
    assert C['runner_sha256'] == H(PATH.read_bytes())
    assert C['limits'] == D.LIMITS and C['execution_authority'] is None
    assert C['controller_base'] == '1df4fc0c42159b2b2604ed5537da6d65c9040dbe'
    for name, pin in C['controller_dependencies'].items():
        assert H((ROOT / name).read_bytes()) == pin
    value = json.loads((ROOT / C['value_context']).read_bytes())
    assert C['first_case'] == value['first_case']
    assert C['prepared_dependencies'] == value['prepared_dependencies']
    assert C['controller_dependencies'][C['wrapper']] == '3f5c025b6d7d91b9b136c535f95e4f386f3fddfa0ae7791f7f5d746bb56d0c07'
    assert C['controller_dependencies'][C['probe']] == '4346f3663fceced18c90167f2a37f3fbb76998baf31179c58bd8edc22bae5948'
    assert (D.LIMITS['FP_save_source_ceiling'], D.LIMITS['FP_restore_source_ceiling']) == (42, 14)
    assert (D.LIMITS['original_public_max'], D.LIMITS['native_public_max']) == (2, 2)


@pytest.mark.parametrize('same', (False, True))
def test_complete_existing_capture_classification(same):
    assert D.classify(capture(same), calls(), None if same else C['expected_failure_site'], C, late_capture()) == (
        'PARITY_PASS' if same else 'EXPECTED_WORD_MISMATCH')


@pytest.mark.parametrize('fault', ('dtype', 'FP', 'input', 'calls', 'restore', 'codec', 'writer',
                                  'warnings', 'ownership', 'native-count', 'wrong-line'))
def test_every_other_failure_refuses_word_failure_classification(fault):
    value, journal, site = capture(), calls(), dict(C['expected_failure_site'])
    if fault == 'dtype': value['actual']['dtype'] = '<f4'
    elif fault == 'FP': value['strict_trace']['native_FP'] = {}
    elif fault == 'input': value['input_after'][0]['payload_sha256'] = '0' * 64
    elif fault == 'calls': journal.append(journal[-1])
    elif fault == 'restore': value['own_full32_restored'] = False
    elif fault == 'codec': value['secondary_errors'] = ['inert-codec-failure']
    elif fault == 'writer': value['writer_error'] = {}
    elif fault == 'warnings': value['native_notices'] = ['warning']
    elif fault == 'ownership': value['actual']['flags']['owndata'] = False
    elif fault == 'native-count': value['strict_trace']['returns'] = []
    else: site['line'] += 1
    with pytest.raises(AssertionError): D.classify(value, journal, site, C, late_capture())


@pytest.mark.parametrize('fault', ('none', 'no-authority', 'identity', 'source', 'incomplete', 'unexpected', 'missing', 'signal', 'late-missing', 'late-failed'))
def test_actual_campaign_one_pair_or_conditional_two_and_primary_exit(tmp_path, monkeypatch, fault):
    first = result()
    if fault == 'identity': first['postidentity_exit'] = 1
    elif fault == 'source': first['source_helper_verified'] = False
    elif fault == 'incomplete': first['child']['complete'] = False
    elif fault == 'unexpected': first['child']['status'] = 'UNEXPECTED_FAILURE'
    elif fault == 'missing': first['child'] = None
    elif fault == 'signal': first['exit'] = 137
    elif fault == 'late-missing': first['child'].pop('late_gate')
    elif fault == 'late-failed': first['child']['late_gate']['independent_late_predicates_passed'] = False
    observed = []
    def run(go, context, role, *unused):
        observed.append(role)
        return first if role == 'default' else result('PARITY_PASS', 0)
    monkeypatch.setattr(D, 'run_role', run)
    go = dict(output_root=str(tmp_path / 'fresh'), allow_disabled_after_expected_word_mismatch=fault != 'no-authority')
    assert D.campaign(go, C, 'inert-go', 'inert-pin') == (137 if fault == 'signal' else 1)
    assert observed == (['default', 'disabled'] if fault == 'none' else ['default'])
    summary = json.loads((tmp_path / 'fresh/summary.json').read_bytes())
    assert summary['default']['exit'] == first['exit'] and not summary['old_9V45_causal_credit']


def test_default_pass_stops_without_disabled_or_changed_workload(tmp_path, monkeypatch):
    roles = []
    monkeypatch.setattr(D, 'run_role', lambda go, context, role, *a: roles.append(role) or result('PARITY_PASS', 0))
    assert D.campaign(dict(output_root=str(tmp_path / 'fresh'), allow_disabled_after_expected_word_mismatch=True), C, 'go', 'pin') == 0
    assert roles == ['default']


@pytest.mark.parametrize('role,exit_code,identity_exit,source_pass', (
    ('default', 1, 0, True), ('default', 1, 8, True), ('default', 0, 0, True),
    ('default', 1, 0, False), ('disabled', 0, 0, True), ('disabled', 1, 0, True)))
def test_actual_role_postchecks_identity_and_only_fresh_child_environment(tmp_path, monkeypatch, role, exit_code, identity_exit, source_pass):
    folder = tmp_path / role; folder.mkdir()
    (folder / 'authority.json').write_text('{}')
    binding, cohort = folder / 'binding.json', folder / 'go.json'
    binding.write_text('{"kind":"binding"}'); cohort.write_text('{"kind":"go"}')
    order, environments = [], []
    def source(b, m):
        order.append('source')
        assert b['kind'] == 'binding' and m == {'manifest': True}
        if not source_pass: raise AssertionError('INERT_CHANGED_SOURCE')
    def helper(b, g):
        order.append('helper'); assert g['kind'] == 'go'
    probe = SimpleNamespace(verify_source_state=source, verify_helper_state=helper)
    monkeypatch.setattr(D, 'prepare_role', lambda *a: (folder, dict(binding=str(binding), go=str(cohort)), probe, {'manifest': True}))
    monkeypatch.delenv('NPY_DISABLE_CPU_FEATURES', raising=False)
    targets = ['AVX512F', 'AVX512_SKX']
    def run(argv, **options):
        environments.append(options['env'])
        if '--child-role' in argv:
            order.append('child'); assert argv[argv.index('--child-role') + 1] == role
            D.write_json(folder / 'value.json', capture(exit_code == 0))
            D.write_json(folder / 'child.json', dict(complete=True,
                status='PARITY_PASS' if exit_code == 0 else 'EXPECTED_WORD_MISMATCH',
                value_path=str(folder / 'value.json'), value_sha256=D.sha(folder / 'value.json')))
            return SimpleNamespace(returncode=exit_code)
        order.append('identity')
        assert argv[2:] == ['verify', '--wheel-dir', str(tmp_path / 'stage/wheels'),
                            '--artifact', str(tmp_path / 'stage/runtime.json')]
        return SimpleNamespace(returncode=identity_exit)
    monkeypatch.setattr(D.subprocess, 'run', run)
    value = D.run_role(dict(interpreter='INERT', source_root=str(tmp_path / 'source'),
        runtime_path=str(tmp_path / 'stage/runtime.json'), disabled_targets=targets), C, role, 'go', 'pin')
    assert order == (['child', 'source', 'helper', 'identity'] if source_pass else ['child', 'source', 'identity'])
    assert value['exit'] == exit_code and value['postidentity_exit'] == identity_exit
    assert value['source_helper_verified'] == source_pass
    assert environments[0] == environments[1]
    assert {k: v for k, v in environments[0].items() if k not in os.environ or os.environ[k] != v} == (
        {'NPY_DISABLE_CPU_FEATURES': ','.join(targets)} if role == 'disabled' else {})


def test_cross_role_FP_status_change_is_not_comparable():
    first, second = result(), result('PARITY_PASS', 0)
    second['value']['strict_trace']['reference_before'] = dict(control=895, status=0, mxcsr=8064)
    assert D.compare_control(first, second) == 'NOT_COMPARABLE_TARGET_FP'
    second = result('PARITY_PASS', 0)
    # A provider-only control changes reference words and preserves native words.
    second['value']['expected'] = copy.deepcopy(first['value']['actual'])
    second['value']['actual'] = copy.deepcopy(first['value']['actual'])
    assert D.compare_control(first, second) == 'OBSERVED_CURRENT_PUBLIC_FEATURE_SENSITIVE_WORD_MISMATCH'


@pytest.mark.parametrize('status', ('UNBOUND', 'ROOT_AUTHORIZED_ONE_VI_VALUE_DIAGNOSTIC',
                                   'ROOT_AUTHORIZED_ONE_LOG2_PROVIDER_EXPERIMENT'))
def test_wrong_purpose_refuses_before_runtime_imports(tmp_path, status):
    p = tmp_path / 'go.json'; p.write_text(json.dumps(dict(status=status)))
    with pytest.raises(AssertionError): D.check_go(p, D.sha(p))


def test_disabled_GO_requires_explicit_permission_before_runtime_or_libs(tmp_path):
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    go = dict(status=D.AUTHORITY, purpose=C['purpose'], limits=D.LIMITS,
              runner_sha256=H(PATH.read_bytes()), context_sha256=H(CONTEXT.read_bytes()),
              created_utc=(now - timedelta(seconds=1)).isoformat(),
              deadline_utc=(now + timedelta(minutes=1)).isoformat(),
              allow_disabled_after_expected_word_mismatch=False)
    p = tmp_path / 'go.json'; D.write_json(p, go)
    with pytest.raises(AssertionError): D.check_go(p, D.sha(p), 'disabled')


def test_feature_gate_unbound_refuses_before_contract_or_numpy(tmp_path, monkeypatch):
    monkeypatch.delenv('DIAGNOSTIC_AUTHORITY', raising=False)
    monkeypatch.setattr(D, 'contract', lambda: pytest.fail('UNBOUND reached contract'))
    with pytest.raises(AssertionError): D.feature_gate(tmp_path / 'features.json')


@pytest.mark.parametrize('sticky', (0, 1, 32, 63))
@pytest.mark.parametrize('bad_control', (False, True))
@pytest.mark.parametrize('outcome', ('pass', 'word', 'word-note'))
def test_actual_child_FP_gate_records_sticky_without_requiring_it(tmp_path, monkeypatch, sticky, bad_control, outcome):
    folder = tmp_path / 'default'; folder.mkdir()
    a = {key: str(folder / relative) for key, relative in (
        ('binding', 'authority-vi/binding.json'), ('go', 'authority-vi/root-go.json'),
        ('diagnostic_go', 'authority-vi/diagnostic-go.json'), ('output', 'vi-value-diagnostic.json'))}
    a['diagnostic_go_sha256'] = 'inert'; D.write_json(folder / 'authority.json', a)
    go = dict(output_root=str(tmp_path), source_root=str(tmp_path / 'source'), cpu={'model': 'inert'}, base_features={})
    raw = bytearray(32); raw[0:2] = (0x037e if bad_control else 0x037f).to_bytes(2, 'little')
    raw[4:6] = sticky.to_bytes(2, 'little'); raw[28:32] = (0x1f80 | sticky).to_bytes(4, 'little')
    (tmp_path / 'source/coverage/contracts').mkdir(parents=True)
    (tmp_path / 'source' / C['driver_context']).write_text('{}')
    invoked = []
    primary = AssertionError('INERT_WORD_MISMATCH')
    namespace = {'failure': primary}
    exec(compile('\n' * 134 + 'def execute():\n    raise failure\n',
                 str(Path(go['source_root']) / C['expected_failure_site']['path']), 'exec'), namespace)
    def wrapper_main():
        invoked.append('one-wrapper')
        D.write_json(a['output'], capture(outcome == 'pass'))
        path = folder / 'first-cohort/vi'; path.mkdir(parents=True)
        (path / 'calls.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in calls()))
        if outcome != 'pass':
            if outcome == 'word-note': primary.add_note('Final diagnostic FP restoration failed: INERT')
            namespace['execute']()
    fake_common = SimpleNamespace(fp_controller=lambda *a: (lambda: raw, None, None))
    monkeypatch.setattr(D, 'module', lambda p, name: fake_common if p.name == 'next_util_execution.py' else SimpleNamespace(main=wrapper_main))
    # This control isolates the unchanged pretarget gate; the real bridge is exercised below.
    monkeypatch.setattr(D, 'run_with_late_gate', lambda wrapper, go, c, late: (late.update(late_capture()), wrapper.main())[1])
    import builtins
    old = builtins.__import__
    monkeypatch.setattr(builtins, '__import__', lambda name, *a, **k: SimpleNamespace() if name == 'numpy' else old(name, *a, **k))
    provider = SimpleNamespace(feature_map=lambda np: {}, cpu_record=lambda: go['cpu'])
    if bad_control:
        with pytest.raises(AssertionError): D.child(go, C, provider, 'default', D.sha(folder / 'authority.json'))
        assert not invoked
    elif outcome == 'pass':
        assert D.child(go, C, provider, 'default', D.sha(folder / 'authority.json'))['complete']
        assert invoked == ['one-wrapper']
    else:
        with pytest.raises(AssertionError) as error:
            D.child(go, C, provider, 'default', D.sha(folder / 'authority.json'))
        assert error.value is primary and invoked == ['one-wrapper']
    record = json.loads((folder / 'child.json').read_bytes())
    assert record['pretarget_FP']['status'] == sticky
    assert record['pretarget_FP']['mxcsr'] & 63 == sticky
    if not bad_control:
        assert record['complete'] == (outcome != 'word-note')
        assert record['status'] == {'pass': 'PARITY_PASS', 'word': 'EXPECTED_WORD_MISMATCH',
                                    'word-note': 'UNEXPECTED_FAILURE'}[outcome]


def test_disabled_child_cannot_bypass_baseline_receipt(tmp_path):
    folder = tmp_path / 'disabled'; folder.mkdir()
    a = {key: str(folder / relative) for key, relative in (
        ('binding', 'authority-vi/binding.json'), ('go', 'authority-vi/root-go.json'),
        ('diagnostic_go', 'authority-vi/diagnostic-go.json'), ('output', 'vi-value-diagnostic.json'))}
    a['baseline_terminal_sha256'] = '0' * 64
    D.write_json(folder / 'authority.json', a)
    with pytest.raises(FileNotFoundError):
        D.child(dict(output_root=str(tmp_path)), C, None, 'disabled', D.sha(folder / 'authority.json'))
    assert not (folder / 'child-entry.claim.json').exists()


def test_per_role_helper_copy_only_and_containment(tmp_path, monkeypatch):
    helper = tmp_path / 'shared-helper-binding.json'; helper.write_bytes(b'{"env":{}}')
    seen = []
    def prepare(controller, source, folder, manifest, python, site, deadline):
        assert (folder / 'helper-binding.json').read_bytes() == helper.read_bytes()
        seen.append(folder)
        directory = folder / 'authority-vi'; directory.mkdir()
        for name in ('binding.json', 'root-go.json'): (directory / name).write_text('{}')
        return {'vi': dict(binding=str(folder / 'authority-vi/binding.json'), go=str(folder / 'authority-vi/root-go.json'))}
    probe = SimpleNamespace(read_manifest=lambda root: ('path', {}), prepare_cohorts=prepare)
    monkeypatch.setattr(D, 'module', lambda *a: probe)
    go = dict(output_root=str(tmp_path), helper_binding_path=str(helper), source_root=str(tmp_path / 'source'),
              source_revision=C['qualified_source'], interpreter='inert', installed_site='inert',
              deadline_utc='inert', controller_revision='inert')
    folder, authority, _, _ = D.prepare_role(go, C, 'default')
    assert seen == [folder] and Path(authority['output']).parent == folder
    assert not any(p.suffix in ('.so', '.dll', '.zip') for p in tmp_path.rglob('*'))


@pytest.mark.parametrize('fault', ('none', 'baseline-forced', 'disabled-missing', 'disabled-exact',
                                  'enable', 'preload', 'loader-extra', 'threads'))
def test_explicit_new_environment_policy_without_old_guard_changes(monkeypatch, fault):
    monkeypatch.setattr(D, 'sys', SimpleNamespace(flags=SimpleNamespace(optimize=0),
        platform='linux', byteorder='little', version=C['genuine_python'],
        base_prefix=C['hosted_loader_environment']['base_prefix']))
    for key in ('PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD', 'LD_AUDIT', 'OPENBLAS_CORETYPE',
                'SKIMAGEXX_DISABLE_SIMD', 'NPY_ENABLE_CPU_FEATURES', 'NPY_DISABLE_CPU_FEATURES', 'GIT_DIR', 'GIT_WORK_TREE'):
        monkeypatch.delenv(key, raising=False)
    for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS'):
        monkeypatch.setenv(key, '1')
    monkeypatch.setenv('LD_LIBRARY_PATH', C['hosted_loader_environment']['LD_LIBRARY_PATH'])
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE', '1')
    targets = ['AVX512F', 'AVX512_SKX'] if fault.startswith('disabled') else ()
    if fault in ('baseline-forced', 'disabled-exact'): monkeypatch.setenv('NPY_DISABLE_CPU_FEATURES', 'AVX512F,AVX512_SKX')
    elif fault == 'enable': monkeypatch.setenv('NPY_ENABLE_CPU_FEATURES', 'AVX512_SKX')
    elif fault == 'preload': monkeypatch.setenv('LD_PRELOAD', 'inert-injection')
    elif fault == 'loader-extra': monkeypatch.setenv('LD_LIBRARY_PATH', C['hosted_loader_environment']['LD_LIBRARY_PATH'] + ':extra')
    elif fault == 'threads': monkeypatch.setenv('OMP_NUM_THREADS', '2')
    if fault in ('none', 'disabled-exact'): D.environment(C, targets)
    else:
        with pytest.raises(AssertionError): D.environment(C, targets)


def test_actual_workflow_blocks_compile_and_preserve_preparation():
    actual = blocks(WORKFLOW)
    previous = blocks(ROOT / '.github/workflows/vi-log2-provider-diagnostic.yml')
    for index, text in enumerate(actual): compile(text, '<actual public workflow ' + str(index) + '>', 'exec')
    assert actual[0] == previous[0]  # capacity, before checkout/transport
    assert actual[2] == previous[2]  # whole producer/source/runtime/SDK/offline extraction
    assert actual[4] == previous[4]  # existing one actual-host F1preparation
    text = WORKFLOW.read_text()
    assert text.index('ROOT_PREPARATION_CAPACITY_BEGIN') < text.index('actions/checkout') < text.index('prepared.zip')
    assert text.index('ROOT_CURRENT_PRODUCT_HOST_GATE_BEGIN') < text.index('prepared.zip')
    assert text.index('Require compiled and enabled') < text.index('Prepare the authenticated small F1')
    assert 'default: UNBOUND' in text and 'default: false' in text and 'if: always()' in text
    assert C['runner_sha256'] in text and H(CONTEXT.read_bytes()) in text
    assert 'run_measurements' not in text and 'scalar_provider(' not in text


def test_actual_workflow_unbound_stops_before_setup_or_API(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('DIAGNOSTIC_AUTHORITY', 'UNBOUND')
    with pytest.raises(AssertionError, match='UNBOUND'):
        exec(compile(blocks(WORKFLOW)[1], '<actual unbound public workflow>', 'exec'), {})


@pytest.mark.parametrize('workspace_gib,runtime_gib,passed', ((32, 32, True), (29, 40, False), (40, 29, False)))
def test_actual_workflow_capacity_gate(tmp_path, monkeypatch, workspace_gib, runtime_gib, passed):
    import shutil
    monkeypatch.chdir(tmp_path); (tmp_path / 'results').mkdir(); runtime = tmp_path / 'runtime'; runtime.mkdir()
    monkeypatch.setenv('RUNNER_TEMP', str(runtime))
    monkeypatch.setattr(shutil, 'disk_usage', lambda p: SimpleNamespace(free=(runtime_gib if Path(p) == runtime else workspace_gib) * 1024**3))
    if passed: exec(compile(blocks(WORKFLOW)[0], '<actual capacity>', 'exec'), {})
    else:
        with pytest.raises(AssertionError): exec(compile(blocks(WORKFLOW)[0], '<actual capacity>', 'exec'), {})
    assert json.loads((tmp_path / 'results/preparation-capacity.json').read_bytes())['passed'] == passed


def test_v2_exact_predecessor_inverse_and_frozen_dependencies():
    import ast
    old = ROOT / 'coverage/contracts/vi-avx512-public-witness-v1.json'
    assert H(old.read_bytes()) == '99d598856039aff8b843878c23513210fe8aa89034d19098991719350cb5d996'
    source = PATH.read_text(encoding='utf-8')
    start, end = source.index('\ndef late_snapshot('), source.index('\ndef child(')
    source = source[:start] + source[end:]
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'classify')
    source = source.replace(ast.get_source_segment(source, node), C['predecessor_classifier_source'], 1)
    for new, old in C['runner_inverse_replacements']:
        assert source.count(new) == 1
        source = source.replace(new, old, 1)
    assert H(source.encode('utf-8')) == '6f1de619c5228be5011cce02d709a2112e8e006f2bb5c23f818335060a574687'
    previous = json.loads((ROOT / 'coverage/contracts/vi-avx512-public-witness-v1.json').read_bytes())
    for key in ('prepared_dependencies', 'controller_dependencies', 'first_case', 'native_requirements',
                'qualified_source', 'runtime_sha256', 'core_sha256', 'FP_policy', 'host_eligibility'):
        assert C[key] == previous[key]
    assert C['source_path_bounds']['word_failure'] == dict(FP_save=18, FP_restore=6, late_worker_getters=1)
    assert C['source_path_bounds']['default_word_failure_control_pass'] == dict(FP_save=39, FP_restore=13, late_worker_getters=1)
    workflow = WORKFLOW.read_text(encoding='utf-8')
    workflow = workflow.replace('vi-avx512-public-witness-v2.json', 'vi-avx512-public-witness-v1.json')
    workflow = workflow.replace(C['runner_sha256'], '6f1de619c5228be5011cce02d709a2112e8e006f2bb5c23f818335060a574687')
    workflow = workflow.replace(H(CONTEXT.read_bytes()), '99d598856039aff8b843878c23513210fe8aa89034d19098991719350cb5d996')
    workflow = workflow.replace('Preauthorize control only after strict word failure and complete pre-cleanup late predicates',
                                'Preauthorize control only after fully captured strict baseline word failure')
    assert H(workflow.encode('utf-8')) == '41027e30e0d2c2f2408efa2e8827b293e972fa12c77509f726d2fefb367d7701'


@pytest.mark.parametrize('fault', ('type-before', 'type-after', 'base-before', 'base-after', 'descriptor'))
def test_all_input_schema_fields_and_structural_output_drift_refuse(fault):
    value, late = capture(), late_capture()
    if fault == 'type-before': value['input_before'][0]['type'] = ['inert', 'wrong']
    elif fault == 'type-after': value['input_after'][0]['type'] = ['inert', 'wrong']
    elif fault == 'base-before': value['input_before'][0]['base_type'] = ['inert', 'wrong']
    elif fault == 'base-after': value['input_after'][0]['base_type'] = None
    else: value['actual']['strides'] = [16]
    with pytest.raises(AssertionError): D.classify(value, calls(), C['expected_failure_site'], C, late)
    if fault == 'descriptor': assert late['structural_metadata_equal'] is False


def test_none_base_mapping_is_explicit_and_payload137_stays_false():
    value, late = capture(), late_capture()
    value['input_before'][0]['base_type'] = ['builtins', 'NoneType']
    value['input_after'][0]['base_type'] = None
    assert D.classify(value, calls(), C['expected_failure_site'], C, late) == 'EXPECTED_WORD_MISMATCH'
    assert late['input_schema_mapped_equal'] and late['structural_metadata_equal']
    assert late['full137_array_state_equality'] is False and late['original137_executed'] is False


def test_unexpected139_with_matching_words_refuses_without_inventing137_state():
    value, late = capture(same=True), late_capture()
    value['status'] = 'FAILED_OLD_PROBE_NO_RETRY'
    site = dict(C['expected_failure_site'], line=139)
    with pytest.raises(AssertionError): D.classify(value, calls(), site, C, late)
    assert not {'full137_array_state_equality', 'original137_executed', 'structural_metadata_equal'}.intersection(late)


@pytest.mark.parametrize('fault', ('none', 'policy', 'handler', 'buffer', 'workers',
    'expected-alias', 'actual0-alias', 'actual1-alias', 'missing-hook', 'wrong-path',
    'wrong-hook-line', 'wrong-site', 'getter-failure', 'capture-failure', 'writer-failure'))
def test_real_bridge_observes_before_cleanup_preserves_primary_writer_and_references(tmp_path, monkeypatch, fault):
    from types import ModuleType
    sequence, queries, records = [], [], []
    primary = AssertionError('INERT_136_PRIMARY')
    writer_error = OSError('INERT_WRITER_FAILURE')
    initial = dict(errors={'divide': 'warn'}, handler=None, buffer=8192, workers=1)
    state = dict(initial)
    if fault == 'policy': state['errors'] = {'divide': 'ignore'}
    elif fault == 'handler': state['handler'] = object()
    elif fault == 'buffer': state['buffer'] = 4096
    elif fault == 'workers': state['workers'] = 2
    expected, actual, input0, input1 = (object() for _ in range(4))
    pairs = {(expected, input0): 'expected-alias', (actual, input0): 'actual0-alias', (actual, input1): 'actual1-alias'}
    def sharing(left, right):
        queries.append(pairs[(left, right)]); return fault == pairs[(left, right)]
    def getter(key):
        queries.append(key)
        if fault == 'getter-failure' and key == 'workers': raise RuntimeError('INERT_GETTER_FAILURE')
        return state[key]
    np = SimpleNamespace(shares_memory=sharing, geterr=lambda: getter('errors'),
        geterrcall=lambda: getter('handler'), getbufsize=lambda: getter('buffer'))
    xx = SimpleNamespace(get_num_threads=lambda: getter('workers'))
    source = tmp_path / 'prepared'
    execute_name = str(source / C['expected_failure_site']['path'])
    execute_globals = dict(primary=primary, sequence=sequence)
    execute_text = '\n' * (134 if fault == 'wrong-site' else 133)
    execute_text += 'def execute(np, xx, expected, actual, args, policy):\n    try:\n        raise primary\n    finally:\n        sequence.append("driver-restored")\n'
    exec(compile(execute_text, execute_name, 'exec'), execute_globals)
    probe = ModuleType('_inert_signed_probe'); probe.__file__ = str(ROOT / C['probe'])
    failure_path = tmp_path / 'default/first-cohort/vi/failure.json'
    actual_path = failure_path.with_name('other.json') if fault == 'wrong-path' else failure_path
    def writer(path, record):
        sequence.append('old-writer'); records.append((path, record, dict(record)))
        if fault == 'writer-failure': raise writer_error
    probe.__dict__.update(execute=execute_globals['execute'], np=np, xx=xx, expected=expected,
        actual=actual, args=(input0, input1), policy=(initial['errors'], initial['handler'], initial['buffer']),
        state=state, initial=initial, sequence=sequence, write_json=writer, failure_path=actual_path)
    probe_text = '\n' * (234 if fault == 'wrong-hook-line' else 233)
    probe_text += 'def main():\n    try:\n        execute(np, xx, expected, actual, args, policy)\n    except BaseException as error:\n        record = dict(type=type(error).__name__, error=str(error), counts=dict(original=1,native=1), traceback="inert", retry=False)\n        sequence.append("failure-writer")\n        write_json(failure_path, record); raise\n    finally:\n        state.update(initial)\n        sequence.append("probe-restored")\n'
    exec(compile(probe_text, probe.__file__, 'exec'), probe.__dict__)
    wrapper = SimpleNamespace()
    entered = []
    def run_diagnostic(p):
        entered.append('one-old-wrapper-call'); return p.main()
    wrapper.run_diagnostic = run_diagnostic
    wrapper.main = (lambda: probe.main()) if fault == 'missing-hook' else (lambda: wrapper.run_diagnostic(probe))
    if fault == 'capture-failure':
        def fail_capture(values): raise RuntimeError('INERT_CAPTURE_FAILURE')
        monkeypatch.setattr(D, 'late_snapshot', fail_capture)
    late = dict(role='default', boundary_seen=False, writer_visits=0, complete=False)
    with pytest.raises((AssertionError, OSError)) as error:
        D.run_with_late_gate(wrapper, dict(output_root=str(tmp_path), source_root=str(source)), C, late)
    assert error.value is (writer_error if fault == 'writer-failure' else primary)
    assert wrapper.run_diagnostic is run_diagnostic and probe.write_json is writer
    assert len(records) == 1 and records[0][1] == records[0][2]
    assert records[0][1]['counts'] == dict(original=1, native=1)
    assert sequence == ['driver-restored', 'failure-writer', 'old-writer', 'probe-restored']
    assert state == initial  # Post-wrapper getters would falsely pass the four mutation cases.
    assert entered == ([] if fault == 'missing-hook' else ['one-old-wrapper-call'])
    if fault in ('missing-hook', 'wrong-path', 'wrong-hook-line', 'wrong-site', 'capture-failure'):
        assert queries == []
    else:
        assert queries == ['expected-alias', 'actual0-alias', 'actual1-alias', 'errors', 'handler', 'buffer', 'workers']
    if fault == 'none':
        assert D.classify(capture(), calls(), C['expected_failure_site'], C, late) == 'EXPECTED_WORD_MISMATCH'
        assert late['independent_late_predicates_passed'] and late['full137_array_state_equality'] is False
    elif fault != 'writer-failure':
        with pytest.raises(AssertionError): D.classify(capture(), calls(), C['expected_failure_site'], C, late)


def test_success_bridge_performs_no_late_science_and_restores_reference(monkeypatch):
    def forbidden(*unused): pytest.fail('Success performed a failure-only observation')
    monkeypatch.setattr(D, 'late_snapshot', forbidden)
    original = lambda *args: 'inert-success'
    wrapper = SimpleNamespace(run_diagnostic=original)
    wrapper.main = lambda: 'inert-success'
    late = dict(role='default', boundary_seen=False, writer_visits=0, complete=False)
    assert D.run_with_late_gate(wrapper, dict(output_root='/inert'), C, late) == 'inert-success'
    assert wrapper.run_diagnostic is original and not late['boundary_seen']
