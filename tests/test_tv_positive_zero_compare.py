"""Synthetic metadata guards only; never import or execute scientific packages."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


t = load('_tv_zero_controller_metadata', 'benchmarks/tv_positive_zero_compare.py')
a = t.a
# Reuse immutable synthetic report construction, never its product admission.
old_tests = load('_tv_zero_legacy_metadata_fixtures', 'tests/test_tv_chambolle_admission_compare.py')


@pytest.fixture
def experiment():
    # Inert metadata only: these placeholder digests/IDs never admit a product.
    products = {'before': copy.deepcopy(a.PLAN['old_certificate']['products']), 'after': {}}
    qualification = {}
    for index, profile in enumerate((*a.PROFILES, 'debug')):
        qualification[profile] = dict(passed=True, qualification_sha256=str(index+1)*64)
        if profile == 'debug':
            continue
        pin = dict(revision=a.AFTER_PRODUCT, qualification_sha256=qualification[profile]['qualification_sha256'],
            product_sha256='2'*64, wheel_sha256='3'*64, source_sha256='4'*64,
            scope=a._scope('after'), artifact_run='40237475475', artifact_name='rank-statistics-product-'+profile)
        products['after'][profile] = dict(pin=pin, core_sha256='5'*64,
            artifact=dict(id=100+index, digest='6'*64, run='40237475475', name=pin['artifact_name']))
    return dict(schema=a.EXPERIMENT_SCHEMA, status='root-reviewed', controller_revision='c'*40,
        before_product_source=a.BEFORE_PRODUCT, after_product_source=a.AFTER_PRODUCT,
        original_version='0.26.0', rounds=7, workers=1, case_count=400, timed_count=70,
        factory_sha256=a.FROZEN_FACTORY_SHA256,
        manifest_sha256=a.sha(ROOT/'coverage/contracts/tv-chambolle-benchmark.json'), plan_sha256=a.PLAN_SHA256,
        root_before_admission=copy.deepcopy(a.PLAN['old_certificate']['root_admission']),
        root_after_admission=dict(source=a.AFTER_PRODUCT, passed=True, sha256='7'*64,
            run='40237475475', profiles=qualification), products=products)


def report(experiment, measure=False):
    value = old_tests.synthetic_report(experiment, measure)
    value.update(schema=a.REPORT_SCHEMA, before_product_source=a.BEFORE_PRODUCT,
        after_product_source=a.AFTER_PRODUCT,
        source_comparison=dict(schema='tv-positive-zero-source-comparison-v1',
            whole_core_equivalent=False, helper_causal_attribution=False,
            attribution=copy.deepcopy(a.PLAN['attribution'])), timing_fixture_checks={})
    value.pop('intended_before_source')
    for group in value['worker_identities'].values():
        for identity in group.values():
            identity['guard_environment'] = dict(optimize=0, custom_startup=False)
    if measure:
        value['timing_fixture_checks'] = {role: copy.deepcopy(
            value['default_checks']['normal'][role]['checks'][0]['records']) for role in a.ROLES}
    return value


def test_valid_closed_synthetic_certificate_and_historical_frames(experiment):
    assert a.validate_experiment(experiment, 'c'*40) is experiment
    assert experiment['products']['before'] == a.PLAN['old_certificate']['products']
    assert experiment['root_before_admission']['source'] == a.BEFORE_PRODUCT
    assert experiment['products']['before']['modern']['pin']['scope']['image_paths'] == 192
    assert experiment['products']['after']['modern']['pin']['scope']['image_paths'] == 203
    assert a._legacy.BEFORE_PRODUCT == '5e068a5a2267c3a917701d5348449972fad162a7'
    assert a._legacy.AFTER_PRODUCT == a.BEFORE_PRODUCT
    assert a.AFTER_PRODUCT == a.PLAN['after_source'] == 'ea35a2481880301b8ce7b4eb2d77665adb813ca2'
    assert a.PLAN['after_qualification_run'] == '40237475475'
    assert a.PLAN['after_whole_product_accepted'] is False
    assert a.PLAN['predecessor']['after_source'] == '7417676bbb628e75feac1be49aac900bc5af7b15'
    assert a.PLAN['failed_after_run'] == a.PLAN['predecessor']['failed_after_run'] == '40200919182'


@pytest.mark.parametrize('field,value', [
    ('schema', 'tv-chambolle-admission-experiment-v1'), ('schema', 'tv-positive-zero-experiment-v1'),
    ('status', 'proposal'),
    ('controller_revision', 'd'*40), ('before_product_source', a._legacy.BEFORE_PRODUCT),
    ('after_product_source', a.BEFORE_PRODUCT),
    ('after_product_source', '7417676bbb628e75feac1be49aac900bc5af7b15'), ('original_version', '0.25.2'),
    ('rounds', True), ('rounds', 6), ('workers', True), ('workers', 4),
    ('case_count', True), ('case_count', 399), ('timed_count', True), ('timed_count', 69),
    ('plan_sha256', '0'*64), ('factory_sha256', '0'*64), ('manifest_sha256', '0'*64),
    ('root_after_admission', None), ('root_before_admission', None)])
def test_closed_certificate_negatives(experiment, field, value):
    a.validate_experiment(experiment, 'c'*40)
    experiment[field] = value
    with pytest.raises(RuntimeError):
        a.validate_experiment(experiment, 'c'*40)


@pytest.mark.parametrize('mutation', ['failed741', 'unreviewedrun', 'run_type', 'failed_source', 'failed_product',
    'debug', 'faileddebug', 'boolpass',
    'oldframe', 'oldscope', 'scope', 'boolscope', 'family', 'wheelpin', 'artifactrun',
    'artifactname', 'artifactid', 'artifactdigest', 'duplicateartifact', 'missingprofile',
    'qualification', 'unknownfield', 'oldrootbool'])
def test_product_roles_scope_and_actual_admission_are_closed(experiment, mutation):
    a.validate_experiment(experiment, 'c'*40)
    root = experiment['root_after_admission']; frame = experiment['products']['after']['modern']
    if mutation == 'failed741': root['run'] = a.PLAN['failed_after_run']
    elif mutation == 'unreviewedrun':
        root['run'] = '999999'
        for product in experiment['products']['after'].values():
            product['pin']['artifact_run'] = product['artifact']['run'] = root['run']
    elif mutation == 'run_type': root['run'] = 40237475475
    elif mutation == 'failed_source': root['source'] = a.PLAN['predecessor']['after_source']
    elif mutation == 'failed_product': frame['pin']['revision'] = a.PLAN['predecessor']['after_source']
    elif mutation == 'debug': root['profiles'].pop('debug')
    elif mutation == 'faileddebug': root['profiles']['debug']['passed'] = False
    elif mutation == 'boolpass': root['passed'] = 1
    elif mutation == 'oldframe': experiment['products']['before']['modern']['core_sha256'] = '0'*64
    elif mutation == 'oldscope': experiment['products']['before']['modern']['pin']['scope']['image_paths'] = 203
    elif mutation == 'scope': frame['pin']['scope']['image_paths'] = 192
    elif mutation == 'boolscope': frame['pin']['scope']['metadata_paths'] = True
    elif mutation == 'family': frame['pin']['scope']['sdk_family_names'][-1] = 'tv_denoise'
    elif mutation == 'wheelpin': frame['pin'].pop('wheel_sha256')
    elif mutation == 'artifactrun': frame['artifact']['run'] = '888'
    elif mutation == 'artifactname': frame['artifact']['name'] = 'private-core'
    elif mutation == 'artifactid': frame['artifact']['id'] = True
    elif mutation == 'artifactdigest': frame['artifact']['digest'] = 'x'*64
    elif mutation == 'duplicateartifact': frame['artifact']['id'] = experiment['products']['after']['windows']['artifact']['id']
    elif mutation == 'missingprofile': experiment['products']['after'].pop('minimum')
    elif mutation == 'qualification': frame['pin']['qualification_sha256'] = '8'*64
    elif mutation == 'oldrootbool': experiment['root_before_admission']['passed'] = 1
    else: experiment['ignored_case_ids'] = [330]
    with pytest.raises(RuntimeError):
        a.validate_experiment(experiment, 'c'*40)


def test_certificate_raw_hash_and_size_precede_parsing(tmp_path, monkeypatch):
    path = tmp_path/'bad.json'; path.write_bytes(b'not JSON')
    monkeypatch.setattr(a, 'verify_controller_checkout', lambda _: pytest.fail('checkout reached before SHA'))
    with pytest.raises(RuntimeError, match='raw SHA'):
        a.load_experiment(path, '0'*64, 'c'*40)
    path.write_bytes(b'x'*(a.CERTIFICATE_LIMIT+1))
    with pytest.raises(RuntimeError, match='bounded'):
        a.load_experiment(path, a.sha(path), 'c'*40)


def test_failed_after_blocks_before_fixture_import(tmp_path, monkeypatch, experiment):
    experiment['root_after_admission'] = None
    path = tmp_path/'experiment.json'; path.write_text(json.dumps(experiment), encoding='utf-8')
    monkeypatch.setattr(a, 'verify_controller_checkout', lambda _: None)
    monkeypatch.setattr(t, 'factory', lambda *_: pytest.fail('fixture import before admission'))
    with pytest.raises(RuntimeError, match='AFTER'):
        a.load_experiment(path, a.sha(path), 'c'*40)


@pytest.mark.parametrize('measure', [False, True])
def test_complete_400_70_7_round_reports(experiment, measure):
    value = report(experiment, measure)
    assert t.validate_report(value, experiment) is value
    if measure:
        assert len(value['results']) == 70
        assert sum(len(v) for v in value['timing_fixture_checks'].values()) == 210
        assert sum(len(row['raw_samples']) for row in value['results']) == 1470
        assert value['summary']['overall']['slower_than_before']
        assert value['summary']['overall']['slower_than_original']


@pytest.mark.parametrize('mutation', ['legacy', 'mode', 'defaultrow', 'defaultpolicy',
    'finite', 'forward', 'actualmissing', 'actualorder', 'actualstate', 'actualdispatch',
    'actualdigest', 'actualrole', 'round', 'rawtime', 'summary', 'core', 'environment',
    'wholecore', 'causal', 'attribution'])
def test_reports_require_all_checks_and_preserve_attribution(experiment, mutation):
    value = report(experiment, True); t.validate_report(value, experiment)
    default = value['default_checks']['normal']['after']['checks'][0]['records'][0]
    actual = value['timing_fixture_checks']['after']
    if mutation == 'legacy': value['checks']['normal']['after']['checks'][0]['digests'].pop()
    elif mutation == 'mode': value['checks']['scalar'] = {}
    elif mutation == 'defaultrow': value['default_checks']['normal']['after']['checks'][0]['records'].pop()
    elif mutation == 'defaultpolicy': default['state_before']['policy']['under'] = 'warn'
    elif mutation == 'finite': default['dispatch']['finite']['c_return'] = 0
    elif mutation == 'forward': default['dispatch']['original']['denied'] = 1
    elif mutation == 'actualmissing': actual.pop()
    elif mutation == 'actualorder': actual.reverse()
    elif mutation == 'actualstate': actual[0]['state_before']['workers'] = 4
    elif mutation == 'actualdispatch': actual[0]['dispatch']['finite']['accepted'] = 0
    elif mutation == 'actualdigest': actual[0]['observation_digest'] = '0'*64
    elif mutation == 'actualrole': value['timing_fixture_checks'].pop('original')
    elif mutation == 'round': value['results'][0]['raw_samples'].pop()
    elif mutation == 'rawtime': value['results'][0]['raw_samples'][0]['elapsed_ns'] += 1
    elif mutation == 'summary': value['summary']['overall']['case_count'] = 69
    elif mutation == 'core': value['worker_identities']['normal']['after']['core_sha256'] = '0'*64
    elif mutation == 'environment': value['worker_identities']['normal']['after']['guard_environment']['optimize'] = 1
    elif mutation == 'wholecore': value['source_comparison']['whole_core_equivalent'] = True
    elif mutation == 'causal': value['source_comparison']['helper_causal_attribution'] = True
    else: value['source_comparison']['attribution']['allowed'] = 'isolated helper speedup'
    with pytest.raises(RuntimeError):
        t.validate_report(value, experiment)


def nodes(path):
    return {n.name: n for n in ast.parse(path.read_bytes()).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


@pytest.mark.parametrize('name', ['factory', 'experiment', 'generation', 'rotating_triads',
    'sample_admission', 'verify_environment', 'worker_permit', 'validate_authentication_index',
    'verify_authentication_artifacts', 'default_state', 'DefaultWitness', 'default_bases',
    'summarize', 'verify_barrier'])
def test_preserved_scientific_transport_and_arithmetic_nodes(name):
    old = nodes(ROOT/'benchmarks/tv_chambolle_admission_compare.py')
    new = nodes(ROOT/'benchmarks/tv_positive_zero_compare.py')
    assert ast.dump(old[name], include_attributes=False) == ast.dump(new[name], include_attributes=False)


def test_default_observer_inverse_changes_only_fixture_intake():
    old = nodes(ROOT/'benchmarks/tv_chambolle_admission_compare.py')['observe_default']
    new = nodes(ROOT/'benchmarks/tv_positive_zero_compare.py')['observe_default']
    new.args.kwonlyargs = []; new.args.kw_defaults = []
    for node in ast.walk(new):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.IfExp):
            assert ast.unparse(node.value.test) == 'prepared is None'
            node.value = node.value.body
    assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)


def test_exact_timing_closures_and_loops_are_preserved():
    frozen = nodes(ROOT/'benchmarks/tv_chambolle.py')['timing_callback']
    new = nodes(ROOT/'benchmarks/tv_positive_zero_compare.py')['prepare_timing_callback']
    for node_type in (ast.FunctionDef, ast.Lambda):
        left = [n for n in ast.walk(frozen) if isinstance(n, node_type) and (not isinstance(n, ast.FunctionDef) or n.name == 'call')]
        right = [n for n in ast.walk(new) if isinstance(n, node_type) and (not isinstance(n, ast.FunctionDef) or n.name == 'call')]
        assert [ast.dump(n, include_attributes=False) for n in left] == [ast.dump(n, include_attributes=False) for n in right]
    old = nodes(ROOT/'benchmarks/tv_chambolle_admission_compare.py')['worker_main']
    new_worker = nodes(ROOT/'benchmarks/tv_positive_zero_compare.py')['worker_main']
    timed = lambda node: [ast.dump(n, include_attributes=False) for n in ast.walk(node)
        if isinstance(n, ast.For) and ast.unparse(n.iter) == 'range(loops)']
    assert timed(old) == timed(new_worker) and len(timed(old)) == 1


def test_observations_retained_before_assertion_and_every_timer_order():
    current = nodes(ROOT/'benchmarks/tv_positive_zero_compare.py')
    for name in ('check_default_worker', 'prepare_timing_callback'):
        source = ast.unparse(current[name])
        assert source.index('retain_observation(') < source.index('a.validate_default_state(')
    worker = ast.unparse(current['worker_main'])
    assert worker.index('prepare_timing_callback(') < worker.index('time.perf_counter_ns()')
    assert "operation == 'warm'" in worker and 'partial_timing_fixture_checks' in worker
    controller = ast.unparse(current['controller'])
    assert controller.index("operation='validate-default'") < controller.index("operation='warm'")
    assert controller.index("'actual callback fixture differs before calibration'") < controller.index('calibration =')
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in ('seterr', 'seterrcall', 'setbufsize')
                   for name in ('observe_default', 'prepare_timing_callback', 'check_default_worker')
                   for n in ast.walk(current[name]))


def test_child_environment_inverse_preserves_transport_and_claims():
    old = t.transport_source
    assert t.transport_derived.replace(t.transport_environment_new, t.transport_environment_old) == old
    for key in ('PYTHONOPTIMIZE', 'PYTHONPATH', 'PYTHONHOME', 'PYTEST_ADDOPTS', 'PYTEST_PLUGINS'):
        assert key.encode() in t.transport_environment_new
    assert a._legacy.AFTER_PRODUCT == a.BEFORE_PRODUCT


def test_optimized_python_rejected_before_inherited_guards():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-B', '-O', '-c',
        "import sys;sys.path.insert(0,sys.argv[1]);import tools.tv_positive_zero_artifacts", str(ROOT)],
        capture_output=True, text=True, env=env)
    assert result.returncode != 0 and 'optimized Python cannot execute comparison guards' in result.stderr
    assert 'numpy' not in result.stdout


def test_new_controller_exact_byte_attributes_and_no_scientific_imports():
    names = a.PLAN['owned_new_paths']
    result = subprocess.check_output(['git', '-C', str(ROOT), 'check-attr', 'text', '--', *names], text=True)
    assert len(result.splitlines()) == len(names)
    assert all(line.endswith(': text: unset') for line in result.splitlines())
    assert not any(name in sys.modules for name in ('numpy', 'scipy', 'skimage', 'skimagexx'))


def test_local_source_slice_exact_inverse_and_unreviewed_change(tmp_path, monkeypatch):
    before = tmp_path/'before'; after = tmp_path/'after'
    for path in (before/'src', after/'src'): path.mkdir(parents=True)
    old = b'prefix\nbool zero_bytes() { return false; }\nsuffix\n'
    new = b'prefix\nbool zero_bytes() { return true; }\nsuffix\n'
    (before/'src/tv_denoise.cpp').write_bytes(old); (after/'src/tv_denoise.cpp').write_bytes(new)
    pins = {role: dict(canonical_lf_sha256=hashlib.sha256(raw).hexdigest())
            for role, raw in (('before', old), ('after', new))}
    plan = dict(full_source_delta={'src/tv_denoise.cpp': pins}, tv_source_slice={'src/tv_denoise.cpp': pins},
        helper_inverse=dict(before_body='bool zero_bytes() { return false; }', after_body='bool zero_bytes() { return true; }'))
    monkeypatch.setattr(a, 'PLAN', plan)
    assert a.verify_source_slice(before, after)['helper_only_inverse'] is True
    (after/'src/unreviewed.cpp').write_text('unexpected', encoding='utf-8')
    with pytest.raises(RuntimeError, match='complete qualified source delta'):
        a.verify_source_slice(before, after)


def test_runtime_comparison_retains_provider_closure():
    base = dict(runtime=dict(python='3.13.14', versions={'scikit-imagexx': '0.1', 'numpy': '2.5.4'},
        packages_sha256={'skimagexx': {'_core': 'a'}, 'numpy': {'a': 'b'}}, sdk_sha256={'lib': 'c'},
        dependency_libraries_sha256={'numpy': {'lib': 'd'}}), wheels={
            'xx.whl': {'name': 'scikit-imagexx', 'sha256': 'a'}, 'np.whl': {'name': 'numpy', 'sha256': 'b'}})
    changed = copy.deepcopy(base); changed['runtime']['packages_sha256']['skimagexx']['_core'] = 'new'
    changed['runtime']['sdk_sha256']['lib'] = 'new'; changed['wheels']['xx.whl']['sha256'] = 'new'
    assert a.reference_runtime(base) == a.reference_runtime(changed)
    changed['runtime']['dependency_libraries_sha256']['numpy']['lib'] = 'different'
    assert a.reference_runtime(base) != a.reference_runtime(changed)


def synthetic_identity(tmp_path):
    harness = load('_tv_zero_rank_metadata_inputs', 'tests/test_rank_statistics_compare.py')
    path = tmp_path/'product'; facts = harness.synthetic_build_facts(path)
    prototype = facts['translation_units'][0]
    for relative in a.PLAN['relevant_translation_units']:
        unit = copy.deepcopy(prototype); old = unit['source']
        unit['source'] = str(Path(facts['source_root'])/relative)
        Path(unit['source']).write_bytes(b'INERT METADATA SOURCE; NOT COMPILED')
        oldout = unit['outputs'][0]; newout = str(Path(oldout).with_name(Path(relative).name+'.obj'))
        Path(newout).write_bytes(b'INERT METADATA OBJECT; NOT LOADABLE'); unit['outputs'] = [newout]
        unit['argv'] = [v.replace(old, unit['source']).replace(oldout, newout) for v in unit['argv']]
        facts['translation_units'].append(unit)
        facts['link_commands'][0]['inputs'].append(newout); facts['link_commands'][0]['argv'].append(newout)
        facts['link_inputs'].append(dict(path=newout, generated=True))
    identity, _ = harness.capture_synthetic_build(path, facts)
    return identity


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'flag', 'header', 'compiler', 'configuration'])
def test_actual_tv_build_slice_keeps_all_declared_inputs(tmp_path, mutation):
    identity = synthetic_identity(tmp_path); baseline = a.build_slice(identity)
    changed = copy.deepcopy(identity)
    unit = next(u for u in changed['raw']['translation_units'] if u['source'].replace('\\', '/').endswith('/src/tv_denoise.cpp'))
    if mutation == 'missing': changed['raw']['translation_units'].remove(unit)
    elif mutation == 'duplicate': changed['raw']['translation_units'].append(copy.deepcopy(unit))
    elif mutation == 'flag': unit['argv'].append('-ffast-math')
    elif mutation == 'header':
        record = changed['file_records']['headers'][unit['dependencies'][0]]
        record['sha256'] = '0'*64; record['canonical_lf_sha256'] = '0'*64
    elif mutation == 'compiler': changed['common']['binaries']['compiler']['sha256'] = '0'*64
    else: changed['raw']['configuration']['lto'] = not changed['raw']['configuration']['lto']
    if mutation in ('missing', 'duplicate'):
        with pytest.raises(RuntimeError): a.build_slice(changed)
    else:
        assert a.build_slice(changed) != baseline


def test_generic_complete_product_verifier_is_first_boundary(tmp_path, monkeypatch, experiment):
    class Deny:
        def config(self): return {}
        def verify_product(self, *args): raise RuntimeError('generic full product rejected')
    monkeypatch.setattr(a, 'rank_controller', lambda: Deny())
    monkeypatch.setattr(a, 'read_json', lambda *_: pytest.fail('later package reads before complete product guard'))
    with pytest.raises(RuntimeError, match='generic full product rejected'):
        a.verify_product(tmp_path, 'after', 'modern', experiment)


def test_scope_builder_does_not_expose_mutable_plan_members():
    before = copy.deepcopy(a.PLAN['scopes']['after'])
    a._scope('after')['sdk_family_names'][-1] = 'synthetic mutation'
    assert a.PLAN['scopes']['after'] == before


def test_successor_receipts_keep_complete_historical_barrier_checks(tmp_path):
    paths = []
    for profile in a.PROFILES:
        value = dict(schema=a.RECEIPT_SCHEMA, passed=True, profile=profile,
            experiment_sha256='1'*64, generation='123', attempt='1', modes=['normal', 'scalar'],
            workers=[1, 4], case_count=400, timed_count=70, default_validation_schema=a.DEFAULT_SCHEMA,
            default_timed_count=70, default_reports={'normal': '2'*64, 'scalar': '3'*64},
            reports={'normal': '4'*64, 'scalar': '5'*64}, common_build_sha256='6'*64)
        path = tmp_path/(profile+'.json'); path.write_text(json.dumps(value)); paths.append(path)
    originals = [p.read_bytes() for p in paths]
    assert set(a.barrier(paths, '1'*64, '123', '1')) == set(a.PROFILES)
    assert [p.read_bytes() for p in paths] == originals
    with pytest.raises(RuntimeError, match='stale'):
        a.barrier(paths, '1'*64, '124', '1')
    with pytest.raises(RuntimeError, match='duplicate'):
        a.barrier([paths[0], paths[0], paths[2]], '1'*64, '123', '1')
    value['schema'] = 'tv-admission-preparation-v2'; paths[-1].write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match='successor'):
        a.barrier(paths, '1'*64, '123', '1')


def test_raw_failure_journal_is_lossless_and_flushed(capsys):
    payload = {'case_index': 330, 'error': {'type': 'RuntimeError', 'message': 'synthetic'}, 'raw': '00ff'}
    t.retain_observation('synthetic-negative-control', payload)
    value = json.loads(capsys.readouterr().err)
    assert value == dict(schema='tv-zero-observation-journal-v1', phase='synthetic-negative-control', value=payload)
