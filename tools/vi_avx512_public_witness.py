"""Root-authorized public VI words; two conditional children, never timing.

Reuse the admitted value wrapper and real first-cohort probe unchanged. Import
is inert. A captured baseline failure remains a failure after the control.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ROOT / 'coverage/contracts/vi-avx512-public-witness-v2.json'
AUTHORITY = 'ROOT_AUTHORIZED_CONTROLLED_VI_PUBLIC_FEATURE_WITNESS'
LIMITS = dict(original_public_max=2, native_public_max=2, target_children_max=2,
              fixtures_max=2, sampler_bodies=0, timed_batches=0, retries=0,
              FP_save_source_ceiling=42, FP_restore_source_ceiling=14,
              ABI_queries_source_ceiling=8, glibc_queries_source_ceiling=8,
              postidentity_max=2, F1_preparations_max=1,
              late_alias_queries_source_ceiling=6,
              late_software_getters_source_ceiling_each=2,
              late_worker_getters_source_ceiling=2)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def contract():
    value = json.loads(CONTEXT.read_bytes())
    assert value['schema'] == 'vi-avx512-public-witness-v2'
    assert value['purpose'] == 'CURRENT16_GENUINE_PUBLIC_VI_AVX512_INTERVENTION'
    assert value['limits'] == LIMITS and value['execution_authority'] is None
    assert sha(__file__) == value['runner_sha256']
    for name, pin in value['controller_dependencies'].items():
        assert sha(ROOT / name) == pin, name
    return value


def environment(context, targets=()):
    assert not sys.flags.optimize
    assert sys.platform == 'linux' and sys.byteorder == 'little'
    assert sys.version == context['genuine_python']
    loader = context['hosted_loader_environment']
    assert sys.base_prefix == loader['base_prefix']
    assert os.environ.get('LD_LIBRARY_PATH') == loader['LD_LIBRARY_PATH'] == sys.base_prefix + '/lib'
    assert all(os.environ.get(k) == '1' for k in
               ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS'))
    assert os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
    assert all(not os.environ.get(k) for k in
               ('PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD', 'LD_AUDIT', 'OPENBLAS_CORETYPE',
                'SKIMAGEXX_DISABLE_SIMD', 'NPY_ENABLE_CPU_FEATURES', 'GIT_DIR', 'GIT_WORK_TREE'))
    assert os.environ.get('NPY_DISABLE_CPU_FEATURES', '') == ','.join(targets)


def check_go(path, pin, role=None):
    assert sha(path) == pin
    go = json.loads(Path(path).read_bytes())
    assert go['status'] == AUTHORITY, 'UNBOUND public witness'
    c = contract()
    assert go['purpose'] == c['purpose'] and go['limits'] == LIMITS
    assert go['runner_sha256'] == sha(__file__) == c['runner_sha256']
    assert go['context_sha256'] == sha(CONTEXT)
    created, deadline = (datetime.fromisoformat(go[k]) for k in ('created_utc', 'deadline_utc'))
    assert created.utcoffset() is not None and deadline.utcoffset() is not None
    assert created.utcoffset().total_seconds() == deadline.utcoffset().total_seconds() == 0
    assert 0 < (deadline - created).total_seconds() <= 7200
    assert created <= datetime.now(timezone.utc) < deadline
    assert type(go['allow_disabled_after_expected_word_mismatch']) is bool
    assert role != 'disabled' or go['allow_disabled_after_expected_word_mismatch']
    assert go['source_revision'] == c['qualified_source']
    assert go['runtime_sha256'] == c['runtime_sha256'] and sha(go['runtime_path']) == c['runtime_sha256']
    runtime = json.loads(Path(go['runtime_path']).read_bytes())
    assert runtime['source']['revision'] == c['qualified_source']
    assert runtime['runtime']['python'] == sys.version
    assert len(runtime['runtime']['sdk_sha256']) == 108
    assert go['core_sha256'] == c['core_sha256']
    assert runtime['runtime']['packages_sha256']['skimagexx'][Path(c['core_member']).name] == c['core_sha256']
    assert Path(go['interpreter']).resolve() == Path(sys.executable).resolve()
    assert Path(go['source_root']).is_absolute() and Path(go['output_root']).is_absolute()
    assert sha(go['helper_binding_path']) == go['helper_binding_sha256']
    helper = json.loads(Path(go['helper_binding_path']).read_bytes())
    assert set(helper['env']) == {'UTIL_VIEWS_FENV_MODE', 'UTIL_VIEWS_FENV_RECEIPT', 'UTIL_VIEWS_FENV_RECEIPT_SHA256'}
    assert helper['env']['UTIL_VIEWS_FENV_MODE'] == 'source-built-v1'
    assert not os.environ.get('UTIL_VIEWS_FENV_HELPER')
    assert all(os.environ.get(k) == v for k, v in helper['env'].items())
    for name, expected in c['prepared_dependencies'].items():
        assert sha(Path(go['source_root']) / name) == expected, name
    git = ['git', '-C', str(ROOT)]
    assert subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True).strip() == go['controller_revision']
    assert not subprocess.check_output([*git, 'status', '--porcelain'], text=True).strip()
    provider = module(ROOT / c['provider_metadata'], '_vi_public_features')
    assert provider.disabled_targets(go['base_features']) == go['disabled_targets']
    targets = go['disabled_targets'] if role == 'disabled' else ()
    environment(c, targets)
    return go, c, provider


def feature_gate(output):
    assert os.environ.get('DIAGNOSTIC_AUTHORITY') == AUTHORITY, 'UNBOUND feature metadata'
    c = contract()
    environment(c)
    assert not any(n in sys.modules for n in ('numpy', 'scipy', 'skimage', 'skimagexx'))
    assert sha(Path('stage/runtime.json')) == c['runtime_sha256']
    provider = module(ROOT / c['provider_metadata'], '_vi_public_gate_features')
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        import numpy as np
        features = provider.feature_map(np)
    eligible = not notices and 'AVX512_SKX' in features['dispatch'] and features['enabled'].get('AVX512_SKX') is True
    targets = provider.disabled_targets(features) if eligible else []
    write_json(output, dict(schema='vi-public-feature-gate-v1', eligible=eligible,
        features=features, cpu=provider.cpu_record(), disabled_targets=targets,
        warnings=[str(w.message) for w in notices], original_calls=0, native_calls=0,
        retry=False, prior_failed_host_causal_credit=False))


def capture_complete(value, calls, c):
    """Only already-collected JSON/words: no arrays, targets or FP queries."""
    assert value['value_frame_found'] and not value['secondary_errors']
    assert 'writer_error' not in value and value['probe_restored'] is True
    assert value['own_full32_restored'] is True
    assert value['own_before_raw32'] == value['own_restored_raw32']
    for key in ('own_before_raw32', 'own_restored_raw32', 'after_old_probe_raw32', 'driver_caller_raw32'):
        assert len(bytes.fromhex(value[key])) == 32
    assert value['probe_counts'] == dict(original=1, native=1)
    assert [(r['event'], r['role']) for r in calls] == [
        ('before', 'original'), ('after', 'original'), ('before', 'native'), ('after', 'native')]
    assert calls[-1]['counts'] == dict(original=1, native=1)
    assert [r['counts'] for r in calls] == [dict(original=1, native=0)] * 2 + [dict(original=1, native=1)] * 2
    assert value['case'] == c['first_case']
    trace = value['strict_trace']
    assert trace['reference_before'] == trace['native_before']
    assert trace['reference_FP'] == trace['native_FP']
    assert trace['accepted'] == trace['suffix'] == [True]
    assert trace['requirements'] == c['native_requirements']
    assert {k: trace['returns'].count(k) for k in trace['requirements']} == c['native_requirements']
    assert len(trace['returns']) == sum(c['native_requirements'].values())
    assert not value['reference_notices'] and not value['native_notices']
    for key in ('expected', 'actual'):
        row = value[key]
        assert row['type'] == ['numpy', 'ndarray'] and row['dtype'] == '<f8'
        assert row['shape'] == [2] and row['nbytes'] == 16
        raw = bytes.fromhex(row['raw_hex'])
        assert len(raw) == 16 and hashlib.sha256(raw).hexdigest() == row['payload_sha256']
    for key in ('dtype_descr', 'dtype_metadata', 'strides', 'flags', 'base_type'):
        assert value['expected'][key] == value['actual'][key]
    assert value['actual']['flags']['owndata'] is True
    assert len(value['input_before']) == len(value['input_after']) == 2
    for before, after, pin in zip(value['input_before'], value['input_after'], c['input_payload_sha256']):
        assert before['type'] == after['type'] == ['numpy', 'ndarray']
        base = ['builtins', 'NoneType'] if after['base_type'] is None else after['base_type']
        assert before['base_type'] == base
        assert before['payload'] == after['payload_sha256'] == pin
        assert before['dtype'] == after['dtype'] == '<i8'
        assert before['shape'] == after['shape'] == [128, 128]
        assert before['strides'] == after['strides'] == [1024, 8]
        assert after['nbytes'] == 131072
        for old, new in (('own', 'owndata'), ('writeable', 'writeable'), ('c', 'c_contiguous'), ('f', 'f_contiguous')):
            assert before[old] == after['flags'][new]
    binding = value['execute_binding_identity']
    assert binding['source_revision'] == c['qualified_source']
    assert binding['runtime_sha256'] == c['runtime_sha256'] and binding['core_sha256'] == c['core_sha256']


def classify(value, calls, site, c, late=None):
    if site == c['expected_failure_site'] and late is not None:
        keys = ('type', 'dtype', 'dtype_descr', 'dtype_metadata', 'shape', 'strides',
                'nbytes', 'flags', 'base_type')
        late['structural_metadata_equal'] = all(value['expected'][k] == value['actual'][k] for k in keys)
        # Original137 includes the unequal payload. It was skipped, never passed.
        late['full137_array_state_equality'] = False
        late['original137_executed'] = False
    capture_complete(value, calls, c)
    same = value['expected']['raw_hex'] == value['actual']['raw_hex']
    if site is None:
        assert same and value['status'] == 'PASS_OLD_PROBE_SAMPLER_STOP_NO_TIMING'
        terminal = value['terminal']
        assert terminal['sampler_body_calls'] == terminal['timed_batches'] == 0
        assert terminal['poststop_source_authority_unchanged'] is True
        return 'PARITY_PASS'
    assert not same and value['status'] == 'FAILED_OLD_PROBE_NO_RETRY'
    assert value['primary_type'] == 'AssertionError' and site == c['expected_failure_site']
    assert late is not None and late['boundary_seen'] and late['writer_visits'] == 1
    assert late['complete'] and 'capture_error' not in late
    assert late['site'] == c['expected_failure_site'] and late['structural_metadata_equal']
    assert late['aliases'] == dict(expected_input0=False, actual_input0=False, actual_input1=False)
    assert late['policy']['errors_equal'] and late['policy']['handler_same_object']
    assert late['policy']['buffer_equal'] and type(late['workers']) is int and late['workers'] == 1
    late['input_schema_mapped_equal'] = True
    late['independent_late_predicates_passed'] = True
    return 'EXPECTED_WORD_MISMATCH'


def late_snapshot(values):
    """Observe metadata before old probe cleanup; never call targets or F1."""
    np, xx = values['np'], values['xx']
    expected, actual, args = values['expected'], values['actual'], values['args']
    assert len(args) == 2
    aliases = dict(expected_input0=bool(np.shares_memory(expected, args[0])),
                   actual_input0=bool(np.shares_memory(actual, args[0])),
                   actual_input1=bool(np.shares_memory(actual, args[1])))
    errors, handler, buffer = np.geterr(), np.geterrcall(), np.getbufsize()
    workers = xx.get_num_threads()  # Separate metadata getter, outside the public pair.
    policy = values['policy']
    return dict(complete=True, aliases=aliases, workers=workers,
        policy=dict(expected_errors=policy[0], observed_errors=errors,
                    errors_equal=errors == policy[0], handler_same_object=handler is policy[1],
                    expected_handler_type=[type(policy[1]).__module__, type(policy[1]).__name__],
                    observed_handler_type=[type(handler).__module__, type(handler).__name__],
                    expected_buffer=policy[2], observed_buffer=buffer, buffer_equal=buffer == policy[2]))


def run_with_late_gate(wrapper, go, c, late):
    """Bridge this fresh wrapper/probe only; delegate its failure writer unchanged."""
    original_run = wrapper.run_diagnostic
    failure_path = Path(go['output_root']) / late['role'] / 'first-cohort/vi/failure.json'

    def bridge(probe, *args, **kwargs):
        original_write = probe.write_json

        def observe(path, record):
            if Path(path).resolve() == failure_path.resolve():
                late['boundary_seen'] = True
                late['writer_visits'] += 1
                try:
                    caller = sys._getframe(1)
                    assert late['writer_visits'] == 1
                    assert Path(probe.__file__).resolve() == ROOT / c['probe']
                    assert caller.f_code is probe.main.__code__
                    assert caller.f_lineno == c['late_gate']['probe_failure_writer_line']
                    error = caller.f_locals['error']
                    assert type(error) is AssertionError
                    assert set(record) == {'type', 'error', 'counts', 'traceback', 'retry'}
                    assert record['type'] == 'AssertionError' and record['error'] == str(error)
                    assert record['counts'] == dict(original=1, native=1) and record['retry'] is False
                    leaf = error.__traceback__
                    while leaf.tb_next is not None:
                        leaf = leaf.tb_next
                    site = dict(path=c['expected_failure_site']['path'], line=leaf.tb_lineno,
                                function=leaf.tb_frame.f_code.co_name, type=type(error).__name__)
                    assert Path(leaf.tb_frame.f_code.co_filename).resolve() == Path(go['source_root']) / site['path']
                    assert site == c['expected_failure_site']
                    late.update(late_snapshot(leaf.tb_frame.f_locals), site=site)
                except BaseException as error:
                    # Preserve the original writer and primary. Incomplete metadata forbids control.
                    late.update(complete=False, capture_error=dict(type=type(error).__name__, error=str(error)))
            return original_write(path, record)

        probe.write_json = observe
        try:
            return original_run(probe, *args, **kwargs)
        finally:
            probe.write_json = original_write

    wrapper.run_diagnostic = bridge
    try:
        return wrapper.main()
    finally:
        wrapper.run_diagnostic = original_run


def child(go, c, provider, role, authority_pin):
    folder = Path(go['output_root']) / role
    authority_path = folder / 'authority.json'
    assert sha(authority_path) == authority_pin
    authority = json.loads(authority_path.read_bytes())
    for key, relative in (('binding', 'authority-vi/binding.json'), ('go', 'authority-vi/root-go.json'),
                          ('diagnostic_go', 'authority-vi/diagnostic-go.json'), ('output', 'vi-value-diagnostic.json')):
        assert Path(authority[key]).resolve() == (folder / relative).resolve()
    if role == 'disabled':
        baseline = folder.parent / 'default/terminal.json'
        assert sha(baseline) == authority['baseline_terminal_sha256']
        result = json.loads(baseline.read_bytes())
        assert result['exit'] == 1 and usable(result, 'EXPECTED_WORD_MISMATCH')
    write_json(folder / 'child-entry.claim.json', dict(role=role, retry=False))
    record = dict(role=role, status='UNEXPECTED_FAILURE', primary_notes=[], complete=False)
    late = dict(role=role, boundary_seen=False, writer_visits=0, complete=False)
    primary = primary_tb = None
    try:
        assert not any(n in sys.modules for n in ('numpy', 'scipy', 'skimage', 'skimagexx'))
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            import numpy as np
            features = provider.feature_map(np)
        assert not notices, 'Import/feature warning; zero public targets required'
        assert provider.cpu_record() == go['cpu']
        if role == 'default':
            assert features == go['base_features']
        else:
            provider.check_intervention(go['base_features'], features, go['disabled_targets'])
        record.update(features=features, cpu=provider.cpu_record(), environment_delta={
            'NPY_DISABLE_CPU_FEATURES': os.environ.get('NPY_DISABLE_CPU_FEATURES', '')})
        # One read-only pretarget save. Do not constrain or normalize sticky bits.
        common = module(Path(go['source_root']) / 'benchmarks/next_util_execution.py', '_vi_public_fp_gate')
        driver_context = json.loads((Path(go['source_root']) / c['driver_context']).read_bytes())
        save, _, _ = common.fp_controller(Path(go['source_root']), driver_context)
        raw = bytes(save())
        cw, sw, mxcsr = (int.from_bytes(raw[a:a+n], 'little') for a, n in ((0, 2), (4, 2), (28, 4)))
        record['pretarget_FP'] = dict(raw32=raw.hex(), control=cw, status=sw, mxcsr=mxcsr)
        assert cw == 0x037f and (mxcsr & ~0x3f) == 0x1f80, 'Unexpected control/mask/RNE; zero targets'
        wrapper = module(ROOT / c['wrapper'], '_vi_public_unchanged_wrapper')
        argv = [str(ROOT / c['wrapper']), '--binding', authority['binding'], '--root-go', authority['go'],
                '--diagnostic-go', authority['diagnostic_go'], '--diagnostic-go-sha256', authority['diagnostic_go_sha256'],
                '--output', authority['output']]
        previous = sys.argv
        try:
            sys.argv = argv
            try:
                run_with_late_gate(wrapper, go, c, late)
            except BaseException as error:
                primary, primary_tb = error, error.__traceback__
        finally:
            sys.argv = previous
        record['late_gate'] = late
        site = None
        if primary is not None:
            record.update(primary_type=type(primary).__name__, primary_error=str(primary),
                          primary_traceback=''.join(traceback.format_exception(primary)),
                          primary_notes=list(getattr(primary, '__notes__', ())))
            leaf = primary.__traceback__
            while leaf.tb_next is not None:
                leaf = leaf.tb_next
            if Path(leaf.tb_frame.f_code.co_filename).resolve() == Path(go['source_root']) / c['expected_failure_site']['path']:
                site = dict(c['expected_failure_site'], line=leaf.tb_lineno,
                            function=leaf.tb_frame.f_code.co_name, type=type(primary).__name__)
            assert not record['primary_notes'], 'Secondary restoration/writer failure'
        output = Path(authority['output'])
        assert not output.with_suffix('.failure.json').exists(), 'Fallback writer is not complete capture'
        value = json.loads(output.read_bytes())
        calls_path = folder / 'first-cohort/vi/calls.jsonl'
        calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
        record.update(status=classify(value, calls, site, c, late), complete=True,
                      value_path=str(output), value_sha256=sha(output), calls_path=str(calls_path),
                      calls_sha256=sha(calls_path), failure_site=site)
    except BaseException as error:
        record.update(status='UNEXPECTED_FAILURE', capture_error=dict(type=type(error).__name__, error=str(error)))
        if primary is None:
            primary, primary_tb = error, error.__traceback__
    try:
        write_json(folder / 'child.json', record)
    except BaseException as error:
        if primary is not None:
            primary.add_note('Public witness record writer failed: ' + type(error).__name__)
        else:
            primary, primary_tb = error, error.__traceback__
    if primary is not None:
        raise primary.with_traceback(primary_tb)
    return record


def prepare_role(go, c, role):
    folder = Path(go['output_root']) / role
    folder.mkdir(exist_ok=False)
    helper = Path(go['helper_binding_path']).read_bytes()
    with (folder / 'helper-binding.json').open('xb') as stream:
        stream.write(helper)
    probe = module(ROOT / c['probe'], '_vi_public_role_probe')
    _, manifest = probe.read_manifest(ROOT)
    authorities = probe.prepare_cohorts(ROOT, Path(go['source_root']), folder, manifest,
        go['interpreter'], Path(go['installed_site']), go['deadline_utc'])
    item = authorities['vi']
    output = folder / 'vi-value-diagnostic.json'
    diagnostic = folder / 'authority-vi/diagnostic-go.json'
    write_json(diagnostic, dict(status='ROOT_AUTHORIZED_ONE_VI_VALUE_DIAGNOSTIC',
        api='metrics.variation_of_information', wrapper_sha256=sha(ROOT / c['wrapper']),
        context_sha256=sha(ROOT / c['value_context']), binding_sha256=sha(item['binding']),
        cohort_GO_sha256=sha(item['go']), source_revision=go['source_revision'], output_path=str(output),
        deadline_utc=go['deadline_utc'], maximum_original_topcalls=1, maximum_native_topcalls=1,
        sampler_body_calls=0, timed_batches=0, controller_revision=go['controller_revision'],
        diagnostic_purpose=c['purpose'], role=role))
    authority = dict(binding=item['binding'], go=item['go'], diagnostic_go=str(diagnostic),
                     diagnostic_go_sha256=sha(diagnostic), output=str(output))
    if role == 'disabled':
        authority['baseline_terminal_sha256'] = sha(folder.parent / 'default/terminal.json')
    write_json(folder / 'authority.json', authority)
    return folder, authority, probe, manifest


def run_role(go, c, role, go_path, go_pin):
    folder, authority, probe, manifest = prepare_role(go, c, role)
    env = dict(os.environ)
    if role == 'disabled':
        env['NPY_DISABLE_CPU_FEATURES'] = ','.join(go['disabled_targets'])
    command = [go['interpreter'], str(Path(__file__).resolve()), '--go', str(go_path),
        '--go-sha256', go_pin, '--child-role', role, '--authority-sha256', sha(folder / 'authority.json')]
    write_json(folder / 'command.json', dict(argv=command, environment_delta={}
        if role == 'default' else {'NPY_DISABLE_CPU_FEATURES': env['NPY_DISABLE_CPU_FEATURES']}))
    with (folder / 'stdout.log').open('x') as out, (folder / 'stderr.log').open('x') as err:
        done = subprocess.run(command, env=env, stdout=out, stderr=err)
    record = dict(role=role, exit=done.returncode, source_helper_verified=False)
    write_json(folder / 'exit.json', record)
    try:
        binding = json.loads(Path(authority['binding']).read_bytes())
        cohort_go = json.loads(Path(authority['go']).read_bytes())
        probe.verify_source_state(binding, manifest)
        probe.verify_helper_state(binding, cohort_go)
        record['source_helper_verified'] = True
    except BaseException as error:
        record['postcheck_error'] = dict(type=type(error).__name__, error=str(error))
    verifier = [go['interpreter'], str(Path(go['source_root']) / 'tools/ci_measurement_artifacts.py'),
        'verify', '--wheel-dir', str(Path(go['runtime_path']).parent / 'wheels'), '--artifact', go['runtime_path']]
    write_json(folder / 'postidentity-command.json', verifier)
    with (folder / 'postidentity.log').open('x') as log:
        verified = subprocess.run(verifier, env=env, stdout=log, stderr=subprocess.STDOUT)
    record['postidentity_exit'] = verified.returncode
    write_json(folder / 'postidentity-exit.json', dict(exit=verified.returncode))
    path = folder / 'child.json'
    record['child'] = None
    if path.is_file():
        try:
            record['child'] = json.loads(path.read_bytes())
        except (ValueError, OSError) as error:
            record['capture_read_error'] = dict(type=type(error).__name__, error=str(error))
    if record['child'] and record['child'].get('complete'):
        value_path = Path(record['child']['value_path'])
        assert sha(value_path) == record['child']['value_sha256']
        record['value'] = json.loads(value_path.read_bytes())
    write_json(folder / 'terminal.json', record)
    return record


def usable(result, status):
    return (result.get('source_helper_verified') is True and result.get('postidentity_exit') == 0
            and (result.get('child') or {}).get('complete') is True
            and result['child']['status'] == status
            and (status != 'EXPECTED_WORD_MISMATCH' or
                 result['child'].get('late_gate', {}).get('independent_late_predicates_passed') is True))


def compare_control(default, disabled):
    left, right = default['value'], disabled['value']
    for key in ('reference_before', 'native_before', 'reference_FP', 'native_FP'):
        if left['strict_trace'][key] != right['strict_trace'][key]:
            return 'NOT_COMPARABLE_TARGET_FP'
    if not usable(disabled, 'PARITY_PASS') or disabled['exit'] != 0:
        return 'DISABLED_CONTROL_FAILED'
    if left['actual']['raw_hex'] != right['actual']['raw_hex']:
        return 'NATIVE_WORDS_CHANGED_NO_LOCALIZATION'
    return 'OBSERVED_CURRENT_PUBLIC_FEATURE_SENSITIVE_WORD_MISMATCH'


def campaign(go, c, go_path, go_pin):
    folder = Path(go['output_root'])
    folder.mkdir(exist_ok=False)
    first = run_role(go, c, 'default', go_path, go_pin)
    result = dict(default=first, disabled=None, old_9V45_causal_credit=False, timed_batches=0)
    code = first['exit'] or 1
    if usable(first, 'PARITY_PASS') and first['exit'] == 0:
        result['status'] = 'DEFAULT_PUBLIC_PARITY_PASS_STOPPED_AFTER_ONE_PAIR'
        code = 0
    elif (first['exit'] == 1 and usable(first, 'EXPECTED_WORD_MISMATCH')
          and go['allow_disabled_after_expected_word_mismatch']):
        second = run_role(go, c, 'disabled', go_path, go_pin)
        result['disabled'] = second
        result['status'] = compare_control(first, second) if second.get('value') else 'DISABLED_CONTROL_FAILED'
        # Always preserve the baseline's nonzero exit and retained primary.
    else:
        result['status'] = 'STOPPED_BASELINE_FAILURE_NO_CONTROL_AUTHORITY_OR_INCOMPLETE_CAPTURE'
    result['exit'] = code
    write_json(folder / 'summary.json', result)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-gate', type=Path)
    parser.add_argument('--go', type=Path)
    parser.add_argument('--go-sha256')
    parser.add_argument('--child-role', choices=('default', 'disabled'))
    parser.add_argument('--authority-sha256')
    args = parser.parse_args()
    if args.feature_gate:
        assert not args.go and not args.child_role
        feature_gate(args.feature_gate)
        return
    assert args.go and args.go_sha256, 'UNBOUND public witness'
    go, c, provider = check_go(args.go, args.go_sha256, args.child_role)
    if args.child_role:
        assert args.authority_sha256
        child(go, c, provider, args.child_role, args.authority_sha256)
    else:
        raise SystemExit(campaign(go, c, args.go, args.go_sha256))


if __name__ == '__main__':
    main()
