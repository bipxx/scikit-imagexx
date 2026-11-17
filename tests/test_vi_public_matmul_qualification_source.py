"""Inert current qualification/Windows metadata guards; never call image code."""
import ast
import collections
import copy
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
ADDED_SELECTORS = ('tests/test_vi_public_matmul_provider_source.py', 'tests/test_vi_public_matmul_sdk_source.py',
    'tests/test_vi_public_matmul_qualification_source.py', 'tests/test_vi4_public_matmul_capture.py',
    'tests/test_vi_controller.py', 'tests/test_vi_cohort_value_diagnostic.py',
    'tests/test_vi_value_diagnostic_workflow.py', 'tests/test_vi_log2_provider_diagnostic.py')
PATHS = dict(core='C:/synthetic/package/skimagexx/_core.pyd', frontend='C:/synthetic/package/skimagexx/_variation_of_information.py')


def load_sealers():
    tree = ast.parse((ROOT / 'tools/public_qualification.py').read_bytes())
    names = {'require', 'applicable_public_groups', 'validate_VI_current_large_rows',
             'validate_VI_current_public_rows', 'validate_VI_windows_workers'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    constants = {target.id: ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                 for target in node.targets if isinstance(target, ast.Name) and target.id.startswith('VI_')}
    scope = dict(Path=Path, json=json, hashlib=hashlib, re=re, collections=collections,
                 QualificationError=RuntimeError, PROFILES={name: {} for name in ('minimum', 'modern', 'debug', 'windows')}, **constants)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'current-sealers-only', 'exec'), scope)
    return scope


def put_rows(directory, rows):
    directory.mkdir()
    for index, row in enumerate(rows):
        (directory / (str(index) + '.json')).write_text(json.dumps(row) + '\n', encoding='utf-8')


def public_rows(windows=False):
    q = load_sealers(); rows = []; raw = '00' * 16; fp = '00' * 32
    array = dict(type=['numpy', 'ndarray'], dtype='<f8', shape=[2], strides=[8], nbytes=16,
                 flags=dict(owndata=True), raw_hex=raw, payload_sha256=hashlib.sha256(bytes.fromhex(raw)).hexdigest())
    natural = dict(policy=dict(divide='warn', over='warn', under='ignore', invalid='warn'), handler_type='NoneType',
                   handler_module=None, handler_name=None, handler_identity='1', handler_code=None, buffer=8192)
    code = dict(identity='20', first_line=68, source_path='synthetic/label_pair_metrics_observer.py',
                source_canonical_sha256=q['VI_OBSERVER_CANONICAL_SHA256'])
    for worker in (1, 4):
        for case in q['VI_CURRENT_PUBLIC_CASE_IDS']:
            error = case.endswith('/table-with-None-images'); table = '/table-' in case
            spec = dict(id=case)
            if table: spec['table'] = 'normalized-array'
            if error: spec['images_none'] = True
            events = [dict(provider=name, dtype='<f8') for name in q['VI_CURRENT_PROVIDER_ORDER']]
            markers = ({'get_num_threads': 1, 'label_pair_check_storage': 3, 'variation_of_information_csr': 1} if table else
                       {'get_num_threads': 2, 'label_pair_check_storage': 7, 'label_pair_counts_preweighted': 1, 'variation_of_information_csr': 1})
            observed = dict(id=case, before={'synthetic': True}, after={'synthetic': True}, result=None if error else {'synthetic': True},
                error=dict(type='AttributeError', module='builtins', message="'NoneType' object has no attribute 'shape'") if error else None,
                warnings=[], events=[], output_input_aliases={} if error else {'0': {'input': False}}, input_state_changed=False,
                checks=dict(policy_restored=True, input_buffer_addresses_stable=True))
            row = dict(case=spec, workers=worker, timing=False, primary=None, diagnostics=[], counts=dict(original=1, native=1),
                reference_observation=copy.deepcopy(observed), native_observation=copy.deepcopy(observed),
                provider_events=[[] if error else events], returned=[] if error else [key for key, count in markers.items() for _ in range(count)],
                accepted=[not error], suffix=[] if error or table else [True])
            if not error: row.update(expected=copy.deepcopy(array), actual=copy.deepcopy(array))
            if windows:
                left = dict(natural, handler_type='function', handler_module='label_pair_metrics_observer',
                            handler_name='observe.<locals>.callback', handler_identity='10', handler_code=code)
                right = dict(left, handler_identity='11')  # observe creates distinct callbacks; shared code, no normalization.
                row.update(FP_scope=q['VI_WINDOWS_FP_SCOPE'], native_core_path=PATHS['core'], native_frontend_path=PATHS['frontend'],
                    public_signature='(image0=None, image1=None, *, table=None, ignore_labels=())', provider_types=dict(log2='ufunc', matmul='ufunc'),
                    software_context_unchanged=True, namespace_identity_unchanged=True, software_restored=True,
                    natural_software=copy.deepcopy(natural), restored_software=dict(reference=copy.deepcopy(natural), native=copy.deepcopy(natural)),
                    target_software=dict(reference=[copy.deepcopy(left), copy.deepcopy(left)], native=[copy.deepcopy(right), copy.deepcopy(right)]))
            else:
                state = dict(semantic=dict(control=895, status=32, mxcsr=8098), raw32=fp)
                row.update(natural_caller_raw32=fp, caller32=dict(saved=fp, actual=fp, exact=True),
                           target_FP=dict(reference=[copy.deepcopy(state), copy.deepcopy(state)], native=[copy.deepcopy(state), copy.deepcopy(state)]))
            rows.append(row)
    return rows


def large_rows():
    small = public_rows(); rows = []
    for worker in (1, 4):
        for index in range(4):
            value = copy.deepcopy(next(v for v in small if v['workers'] == worker and v['case']['id'].endswith('/perfect')))
            value.update(index=index, requirements=dict(get_num_threads=2, label_pair_check_storage=7, label_pair_counts_preweighted=1, variation_of_information_csr=1),
                provider='current true numpy.log2 and numpy.matmul ufunc computation', provider_types=dict(log2='ufunc', matmul='ufunc'),
                provider_batches=value['provider_events'][0], output_pair_alias=False, output_input_aliases=False,
                input_before={'synthetic': True}, input_after={'synthetic': True}, pointers_before={'input': 1}, pointers_after={'input': 1},
                pointers_stable=True, software_context_unchanged=True, namespace_identity_unchanged=True)
            fp = value['natural_caller_raw32']; semantic = dict(control=895, status=32, mxcsr=8098)
            value.update(reference_FP=dict(before=semantic, after=semantic, before_raw32=fp, after_raw32=fp),
                         native_FP=dict(before=semantic, after=semantic, before_raw32=fp, after_raw32=fp))
            rows.append(value)
    return rows


def historical_workflow_source():
    from tools.mark_lbp_fp_context import VI_PUBLIC_MATMUL_CONTEXT_SHA256
    from tools.vi_public_matmul_context import speed_predecessor_bytes
    return speed_predecessor_bytes(ROOT, '.github/workflows/public-qualification.yml',
        (ROOT / '.github/workflows/public-qualification.yml').read_bytes(), VI_PUBLIC_MATMUL_CONTEXT_SHA256, True)


def workflow_source_selectors(raw):
    """Read pytest argv only; paths after the pipe are tee output filenames."""
    addition = (' ' + ' '.join(ADDED_SELECTORS)).encode()
    assert raw.count(addition) == 1
    assert hashlib.sha256(raw.replace(addition, b'', 1)).hexdigest() == '7252a926d63bd61c32a7e150782f08dd2e9aaeaa1da2e84c89cb1367b7f274b6'
    commands = [line for line in raw.decode().splitlines() if 'python -m pytest -q --capture=sys' in line]
    assert len(commands) == 1
    command = commands[0]
    assert command.count(' 2>&1 | tee ') == 1
    pytest_argv = shlex.split(command.split(' 2>&1 | tee ', 1)[0])
    selectors = [word for word in pytest_argv if word.startswith('tests/')]
    assert len(selectors) == len(set(selectors)) == 181 and selectors[-8:] == list(ADDED_SELECTORS)
    return selectors


def test_workflow_preserves_exact173_then_complete8_new_source_modules():
    raw = historical_workflow_source()
    assert len(workflow_source_selectors(raw)) == 181


def test_workflow_rejects_new_paths_after_tee_instead_of_pytest():
    raw = historical_workflow_source()
    addition = (' ' + ' '.join(ADDED_SELECTORS)).encode()
    before = raw.replace(addition, b'', 1)
    lines = before.splitlines(keepends=True)
    index = next(i for i, line in enumerate(lines) if b'python -m pytest -q --capture=sys' in line)
    lines[index] = lines[index].rstrip(b'\n') + addition + b'\n'
    bad = b''.join(lines)
    # Old whole-line token counting would falsely report181 for this mutation.
    line = lines[index].decode()
    assert len([word for word in shlex.split(line) if word.startswith('tests/')]) == 181
    assert len([word for word in shlex.split(line.split(' 2>&1 | tee ', 1)[0]) if word.startswith('tests/')]) == 173
    with pytest.raises(AssertionError): workflow_source_selectors(bad)


def test_V8_preserves_V7_and_distinguishes_all_current_gates():
    old_raw = (ROOT / 'coverage/contracts/current-public-qualification-v7.json').read_bytes()
    assert hashlib.sha256(old_raw).hexdigest() == 'b3b756eb2fb64a0334670f51e02f4a1768f566a5766e36463aaf3936e6a5ad72'
    old = json.loads(old_raw); new = json.loads((ROOT / 'coverage/contracts/current-public-qualification-v8.json').read_bytes()); q = load_sealers()
    for name, value in old.items():
        if name not in ('required_selectors', 'VI_current_runtime_large_reference'): assert new[name] == value
    assert new['required_selectors'] == old['required_selectors'] + ['tests/test_vi_public_matmul_provider_public.py', 'tests/test_vi_public_matmul_windows_public.py']
    for field, constant in [('VI_current_runtime_large_reference', 'VI_CURRENT_LARGE_REFERENCE_POLICY'),
                            ('VI_current_runtime_public_reference', 'VI_CURRENT_PUBLIC_REFERENCE_POLICY'),
                            ('VI_current_provider_controls', 'VI_CURRENT_PROVIDER_CONTROLS_POLICY')]: assert new[field] == q[constant]
    assert new['execution_authority'] is None


@pytest.mark.parametrize('profile', ('minimum', 'modern', 'debug', 'windows'))
def test_groups_preserve_old_and_select_exact_platform_contract(profile):
    q = load_sealers(); base = {'public-post203': ('test_variation_of_information_integration.py',), 'old': ('preserved',)}
    groups = q['applicable_public_groups'](base, profile)
    assert all(groups[name] == value for name, value in base.items())
    if profile == 'windows':
        assert groups['public-VI-current-windows'] == ('test_vi_public_matmul_windows_public.py',)
        assert 'public-VI-current-public-linux' not in groups and 'public-VI-large-saved-linux' not in groups
    else:
        assert groups['public-VI-current-public-linux'] == ('test_vi_public_matmul_provider_public.py',)
        assert groups['public-VI-large-saved-linux'] == ('test_variation_of_information_large_saved.py', 'test_vi_public_log2_provider_large.py')
        assert groups['public-VI-saved-linux'] == ('test_variation_of_information_public.py', 'test_variation_of_information_current_counts_public.py')
        assert 'public-VI-current-windows' not in groups


def test_collector_sdk_extractor_and_runtime_record_wiring():
    source = (ROOT / 'tools/public_qualification.py').read_text(); tree = ast.parse(source)
    additions = [node for node in tree.body if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == 'DRAW_SOURCE_PATHS']
    paths = {name for node in additions for name in ast.literal_eval(node.value)}
    for name in ('coverage/contracts/vi-public-matmul-source-history-v1.json', 'coverage/contracts/vi-public-matmul-source-history-v2.json',
                 'tools/vi_public_matmul_context.py', 'tests/vi_public_matmul_source_view.py',
                 'tests/test_vi_public_matmul_windows_public.py', 'coverage/contracts/current-public-qualification-v8.json',
                 'tests/variation_of_information_matmul_controls_v1.hpp', 'tests/cmake_consumer/variation_of_information_matmul_consumer_v1.cpp',
                 'tests/test_vi_public_matmul_sdk_source.py', 'tools/vi4_public_matmul_capture.py', 'coverage/contracts/vi4-public-matmul-capture-v1.json',
                 'coverage/contracts/vi-public-matmul-provider-benchmark-v2.json', 'benchmarks/vi_public_matmul_provider_execution_v2.py',
                 'benchmarks/vi_public_matmul_provider_consumer_v2.py', 'tests/test_vi4_public_matmul_capture.py'):
        assert name in paths
    sdk = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'sdk')
    assert 'tests/variation_of_information_matmul_controls_v1.hpp' in ast.unparse(sdk)
    for token in ('VI_PUBLIC_MATMUL_SMALL_RECORD_DIR', 'VI_PUBLIC_MATMUL_WINDOWS_RECORD_DIR', 'VI_PUBLIC_MATMUL_WINDOWS_WORKER_RECORD_DIR', "row['tests']==56", "len(callers)==56", "row['tests']==34", "row['tests']==16"):
        assert token in source
    assert "'test_variation_of_information_integration.py'" in source


@pytest.mark.parametrize('windows', (False, True))
def test_current_public_sealer_accepts_distinct_complete_metadata(tmp_path, windows):
    directory = tmp_path / 'rows'; put_rows(directory, public_rows(windows))
    load_sealers()['validate_VI_current_public_rows'](directory, windows=windows, installed_paths=PATHS if windows else None)


@pytest.mark.parametrize('fault', ('words', 'dtype', 'primary', 'calls', 'providers', 'FP', 'caller', 'error'))
def test_Linux_public_sealer_rejects_every_missing_gate(tmp_path, fault):
    rows = public_rows(); row = rows[0]
    if fault == 'words': row['actual']['raw_hex'] = '01' * 16
    if fault == 'dtype': row['expected']['dtype'] = row['actual']['dtype'] = '<f4'
    if fault == 'primary': row['primary'] = {'type': 'AssertionError'}
    if fault == 'calls': row['counts']['native'] = 2
    if fault == 'providers': row['provider_events'][0].pop()
    if fault == 'FP': row['target_FP']['native'][1]['semantic']['status'] = 1
    if fault == 'caller': row['caller32']['exact'] = False
    if fault == 'error': rows[7]['accepted'] = [True]
    directory = tmp_path / 'rows'; put_rows(directory, rows)
    with pytest.raises(RuntimeError): load_sealers()['validate_VI_current_public_rows'](directory)


@pytest.mark.parametrize('fault', ('hardware', 'owner', 'target_identity', 'callback_code', 'callback_source', 'natural_restore', 'policy'))
def test_Windows_sealer_rejects_false_or_missing_software_claims(tmp_path, fault):
    rows = public_rows(True); row = rows[0]
    if fault == 'hardware': row['caller32'] = {'exact': True}
    if fault == 'owner': row['native_core_path'] = 'other.pyd'
    if fault == 'target_identity': row['target_software']['native'][1]['handler_identity'] = 'changed'
    if fault == 'callback_code':
        for state in row['target_software']['native']: state['handler_code']['identity'] = 'different'
    if fault == 'callback_source':
        for role in row['target_software']:
            for state in row['target_software'][role]: state['handler_code']['source_canonical_sha256'] = '0' * 64
    if fault == 'natural_restore': row['restored_software']['native']['handler_identity'] = 'other'
    if fault == 'policy':
        for state in row['target_software']['native']: state['buffer'] = 4096
    directory = tmp_path / 'rows'; put_rows(directory, rows)
    with pytest.raises(RuntimeError): load_sealers()['validate_VI_current_public_rows'](directory, windows=True, installed_paths=PATHS)


def test_large_sealer_retains_strict_four_event_policy(tmp_path):
    directory = tmp_path / 'large'; rows = large_rows(); put_rows(directory, rows)
    q = load_sealers(); q['validate_VI_current_large_rows'](directory)
    rows[0]['provider_batches'] = rows[0]['provider_batches'][::2]
    (directory / '0.json').write_text(json.dumps(rows[0]), encoding='utf-8')
    with pytest.raises(RuntimeError): q['validate_VI_current_large_rows'](directory)


def test_Windows_source_uses_existing_calls_without_hardware_FP_or_identity_normalization():
    raw = (ROOT / 'tests/test_vi_public_matmul_windows_public.py').read_text(); tree = ast.parse(raw)
    assert not any(isinstance(node, ast.Name) and node.id in {'backend', 'FEnv', 'ctypes'} for node in ast.walk(tree))
    assert not any(isinstance(node, ast.Attribute) and node.attr in {'fegetenv', 'reference_fenv_restore', 'settrace'} for node in ast.walk(tree))
    assert 'reference_state[1].__code__ is native_state[1].__code__' in raw
    assert "restored['reference'][1] is policy[1]" in raw and "restored['native'][1] is policy[1]" in raw
    assert 'native_core_path=' in raw and 'native_frontend_path=' in raw and 'public_signature=' in raw
    assert 'call_once(call, args, options, snapshot, guard, outputs' in raw
    assert 'retain_record(path, record, outputs, encode)' in raw


def test_Windows34_actual_inert_collection_preserves_imported_control_names(tmp_path):
    """Collect source clones with pass bodies:0image imports/targets/fixtures run."""
    public = ast.parse((ROOT / 'tests/test_variation_of_information_public.py').read_bytes())
    windows = ast.parse((ROOT / 'tests/test_vi_public_matmul_windows_public.py').read_bytes())
    portable = {'test_owned_entropy_uses_only_explicit_log2_and_matmul', 'test_current_provider_primary_stops_later_callbacks', 'test_current_matmul_rejects_nonplain_scalar_transport'}
    funcs = [copy.deepcopy(node) for node in public.body if isinstance(node, ast.FunctionDef) and node.name in portable]
    for node in funcs: node.body = [ast.Pass()]
    specs = [{'id': name} for name in load_sealers()['VI_CURRENT_PUBLIC_CASE_IDS']]
    stub = "import pytest\nSPECS=" + repr(specs) + "\n@pytest.fixture(scope='module')\ndef target(): return None\n"
    (tmp_path / 'test_variation_of_information_public.py').write_text(stub + ast.unparse(ast.Module(body=funcs, type_ignores=[])), encoding='utf-8')
    nodes = [copy.deepcopy(node) for node in windows.body if isinstance(node, ast.ImportFrom) and node.module == 'test_variation_of_information_public']
    for node in windows.body:
        if isinstance(node, ast.FunctionDef) and node.name in ('workers', 'test_windows_current_public_original_pair'):
            clone = copy.deepcopy(node); clone.body = [ast.Pass()]; nodes.append(clone)
    file = tmp_path / 'test_windows_source_clone.py'
    file.write_text('import pytest\n' + ast.unparse(ast.Module(body=nodes, type_ignores=[])), encoding='utf-8')
    ids = tmp_path / 'IDs.json'
    code = "import sys,json,runpy\nsys.path[:0]=" + repr([str(tmp_path), *sys.path]) + "\ndef deny(event,args):\n if event=='import' and args[0].split('.')[0] in ('numpy','scipy','skimage','skimagexx'): raise RuntimeError('science forbidden')\nsys.addaudithook(deny)\nimport pytest\nclass Inventory:\n def pytest_collection_finish(self,session):\n  open(" + repr(str(ids)) + ",'w').write(json.dumps([item.nodeid for item in session.items]))\nraise SystemExit(pytest.main(" + repr(['--collect-only', '-q', '-c', str(ROOT / 'pyproject.toml'), str(file), '-o', 'cache_dir=' + str(tmp_path / 'cache')]) + ",plugins=[Inventory()]))\n"
    done = subprocess.run([sys.executable, '-I', '-S', '-c', code], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    names = json.loads(ids.read_bytes()); assert len(names) == len(set(names)) == 34
    counts = collections.Counter(name.split('::')[1].split('[')[0] for name in names)
    assert counts == dict(test_windows_current_public_original_pair=16, test_owned_entropy_uses_only_explicit_log2_and_matmul=2,
                         test_current_provider_primary_stops_later_callbacks=8, test_current_matmul_rejects_nonplain_scalar_transport=8)


def test_source_derived_full_saved56_and_portable18_counts():
    def multiplicity(function):
        count = 1
        for decorator in function.decorator_list:
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute) and decorator.func.attr == 'parametrize':
                value = decorator.args[1]
                if isinstance(value, ast.Name) and value.id == 'SPECS': size = 8
                elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'range': size = len(range(*[ast.literal_eval(v) for v in value.args]))
                else: size = len(ast.literal_eval(value))
                count *= size
        return count * 2  # Actual unchanged workers fixture params1/4.
    public = ast.parse((ROOT / 'tests/test_variation_of_information_public.py').read_bytes())
    counts = ast.parse((ROOT / 'tests/test_variation_of_information_current_counts_public.py').read_bytes())
    functions = {node.name: node for node in public.body if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')}
    assert sum(multiplicity(node) for node in functions.values()) == 44
    assert sum(multiplicity(node) for node in counts.body if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')) == 12
    names = ('test_owned_entropy_uses_only_explicit_log2_and_matmul', 'test_current_provider_primary_stops_later_callbacks', 'test_current_matmul_rejects_nonplain_scalar_transport')
    assert sum(multiplicity(functions[name]) for name in names) == 18


def test_Windows_worker_records_require_full34_and_default1(tmp_path):
    rows = []
    for worker in (1, 4):
        for name, count in dict(test_windows_current_public_original_pair=8, test_owned_entropy_uses_only_explicit_log2_and_matmul=1,
                                test_current_provider_primary_stops_later_callbacks=4, test_current_matmul_rejects_nonplain_scalar_transport=4).items():
            for index in range(count): rows.append(dict(node=name + '-' + str(worker) + '-' + str(index), test=name,
                selected=worker, before=1, after=1, exact_default1=True, FP_scope=load_sealers()['VI_WINDOWS_FP_SCOPE']))
    directory = tmp_path / 'workers'; put_rows(directory, rows); validate = load_sealers()['validate_VI_windows_workers']
    validate(directory)
    rows[0]['after'] = 4; (directory / '0.json').write_text(json.dumps(rows[0]), encoding='utf-8')
    with pytest.raises(RuntimeError): validate(directory)


def test_Windows_metadata_encoder_failure_still_retains_raw_outputs_and_first_failure():
    tree = ast.parse((ROOT / 'tests/test_vi_public_matmul_windows_public.py').read_bytes())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'retain_windows_record')
    failure = ValueError('software encoding'); writer_failure = OSError('raw writer'); retained = []
    def fail(*args): raise failure
    def writer(path, record, outputs, encode): retained.append(copy.deepcopy(outputs))
    scope = dict(retain_target_software=fail, retain_record=writer)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'metadata-helper-only', 'exec'), scope)
    record = dict(diagnostics=[]); outputs = {'expected': 'returned-before-metadata', 'actual': 'returned-before-metadata'}
    with pytest.raises(ValueError) as caught: scope['retain_windows_record'](None, record, outputs, None, {}, {})
    assert caught.value is failure and retained == [outputs] and record['diagnostics'][0]['stage'] == 'target software encoding'
    def broken_writer(*args): raise writer_failure
    scope['retain_record'] = broken_writer
    with pytest.raises(ValueError) as caught: scope['retain_windows_record'](None, dict(diagnostics=[]), outputs, None, {}, {})
    assert caught.value is failure and any('raw writer' in note for note in failure.__notes__)
