"""Fixed subset authority and inert host orchestration; no science imports."""
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('lbp_only_outer', ROOT / 'tools/lbp_only_comparison.py')
TOOL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TOOL)
CONTEXT = json.loads((ROOT / TOOL.CONTEXT).read_bytes())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


def root_go(tmp_path):
    admission = write(tmp_path / 'admission.json', dict(admitted_controller_sha256=TOOL.sha(TOOL.__file__)))
    now = datetime.now(timezone.utc)
    go = dict(status='ROOT_AUTHORIZED_LBP_ONLY_COMPARISON', context_sha256=TOOL.sha(ROOT / TOOL.CONTEXT),
        controller_sha256=TOOL.sha(TOOL.__file__), parent_sha256=CONTEXT['parent_sha256'],
        scope=CONTEXT['scope'], MARK_authorized=False,
        controller_admission=dict(path=str(admission), sha256=TOOL.sha(admission)),
        created_utc=(now-timedelta(minutes=1)).isoformat(), deadline_utc=(now+timedelta(minutes=10)).isoformat())
    return write(tmp_path / 'go.json', go)


def test_exact_historical_intake_and_fixed_scope():
    outer = TOOL.parent(ROOT)
    path, old = outer.read_manifest(ROOT)
    assert TOOL.sha(path) == CONTEXT['parent_sha256']
    assert old['execution_scope']['canonical_cases'] == 12
    assert old['execution_scope']['canonical_raw_samples'] == 168
    assert CONTEXT['case_ids'] == ['gray-u8-C', 'RGB-f32-F', 'RGB-f64-reverse-wrap', 'RGB-f64-C-large']
    assert CONTEXT['scope']['canonical_raw_samples'] == 4 * 2 * 7
    assert CONTEXT['scope']['warmup_groups'] == 4 * 2
    assert CONTEXT['root_GO'] is CONTEXT['controller_admission'] is CONTEXT['deadline_utc'] is None
    for name, pin in CONTEXT['dependencies_sha256'].items():
        assert TOOL.sha(ROOT / name) == pin


def test_authorize_explicit_subset_preserves_parent_and_driver(tmp_path):
    go = root_go(tmp_path)
    _, selected, _ = TOOL.authorize(ROOT, go, TOOL.sha(go))
    old = json.loads((ROOT / CONTEXT['parent_path']).read_bytes())
    assert set(selected['first_cohort']['routes']) == {'lbp'}
    assert selected['first_cohort']['routes']['lbp'] == old['first_cohort']['routes']['lbp']
    assert len(selected['families']) == 1
    assert selected['execution_scope']['canonical_cases'] == 4
    assert selected['execution_scope']['untimed_topcalls'] == dict(original=1, native=1)
    for name in ('prepared_source', 'producer', 'core_sha256', 'runtime_sha256', 'helper', 'benchmark_fingerprint'):
        assert selected[name] == old[name]
    assert TOOL.sha(ROOT / CONTEXT['parent_path']) == CONTEXT['parent_sha256']


@pytest.mark.parametrize('field,value', [
    ('status', 'UNBOUND'), ('parent_sha256', '0'*64), ('context_sha256', '0'*64),
    ('controller_sha256', '0'*64), ('MARK_authorized', True),
    ('scope', dict(canonical_cases=12)), ('deadline_utc', '2000-01-01T00:00:00+00:00'),
    ('created_utc', '2100-01-01T00:00:00+00:00'),
])
def test_changed_authority_refused(tmp_path, field, value):
    path = root_go(tmp_path); go = json.loads(path.read_bytes()); go[field] = value; write(path, go)
    with pytest.raises(AssertionError):
        TOOL.authorize(ROOT, path, TOOL.sha(path))


def test_unbound_or_wrong_external_hash_refuses_before_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(TOOL, 'parent', lambda *_: pytest.fail('No parent/helper/target before GO'))
    with pytest.raises(AssertionError): TOOL.authorize(ROOT, None, None)
    path = root_go(tmp_path)
    with pytest.raises(AssertionError): TOOL.authorize(ROOT, path, '0'*64)


def test_controller_admission_or_excess_clock_refused(tmp_path):
    path = root_go(tmp_path); value = json.loads(path.read_bytes())
    Path(value['controller_admission']['path']).write_text('{}')
    with pytest.raises(AssertionError): TOOL.authorize(ROOT, path, TOOL.sha(path))
    path = root_go(tmp_path); value = json.loads(path.read_bytes())
    value['deadline_utc'] = (datetime.now(timezone.utc)+timedelta(hours=3)).isoformat()
    write(path, value)
    with pytest.raises(AssertionError): TOOL.authorize(ROOT, path, TOOL.sha(path))


def test_prepare_emits_only_lbp_authority_inert(tmp_path, monkeypatch):
    path = root_go(tmp_path)
    outer, manifest, go = TOOL.authorize(ROOT, path, TOOL.sha(path))
    controller, snapshot, results, site = [tmp_path/name for name in ('controller','snapshot','results','site')]
    for folder in (controller, snapshot, results, site): folder.mkdir()
    first = manifest['first_cohort']
    for prefix in ('root_admission','root_stager'):
        name = first[prefix+'_path']; member = write(controller/name, {'synthetic': prefix})
        first[prefix+'_sha256'] = TOOL.sha(member)
    runtime = write(tmp_path/'stage/runtime.json', dict(runtime=dict(sdk_sha256={'a':'b'}, versions=dict(numpy='metadata-only'))))
    manifest['SDK_members'] = 1; manifest['runtime_sha256'] = TOOL.sha(runtime)
    core = site/first['core_member']; core.parent.mkdir(parents=True, exist_ok=True); core.write_bytes(b'inert-not-a-core')
    manifest['core_sha256'] = TOOL.sha(core)
    write(results/'helper-binding.json', dict(synthetic=True))
    route = first['routes']['lbp']
    write(snapshot/route['contract'], dict(route=route, route_id='lbp'))
    checked = []
    monkeypatch.setattr(outer, 'verify_source_state', lambda b, m: checked.append((b,m)))
    monkeypatch.setattr(TOOL, 'authorize', lambda *_: (outer, manifest, go))
    authority = TOOL.prepare_cohort(controller,snapshot,results,'python',site,path,'pin')
    assert len(checked) == 1
    binding = json.loads(Path(authority['binding']).read_bytes()); grant = json.loads(Path(authority['go']).read_bytes())
    assert binding['routes'] == {'lbp': route} and grant['routes'] == ['lbp']
    assert grant['maximum_original_topcalls'] == grant['maximum_native_topcalls'] == 1
    assert grant['timed_batches'] == 0 and grant['subset_authority']['MARK_authorized'] is False
    assert set(json.loads((results/'cohort-authorities.json').read_bytes())) == {'lbp'}
    assert not (results/'authority-mark').exists()
    with pytest.raises(FileExistsError):
        TOOL.prepare_cohort(controller,snapshot,results,'python',site,path,'pin')


def setup_host(tmp_path, monkeypatch):
    controller = tmp_path / 'controller'; snapshot = tmp_path / 'snapshot'; results = tmp_path / 'results'
    controller.mkdir(); snapshot.mkdir(); results.mkdir()
    path = root_go(tmp_path)
    outer, manifest, go = TOOL.authorize(ROOT, path, TOOL.sha(path))
    runtime = write(tmp_path / 'stage/runtime.json', {'synthetic': True})
    manifest['runtime_sha256'] = TOOL.sha(runtime)
    binding = write(results / 'authority-lbp/binding.json', dict(output_root=str((results/'first-cohort').resolve())))
    authority = dict(binding=str(binding), go=str(results/'authority-lbp/root-go.json'), measurement=str(results/'authority-lbp/measurement.json'))
    terminal = dict(status='PASS_REAL_CLI_STOP_BEFORE_BODY', route='lbp', api='feature.draw_multiblock_lbp',
        source=manifest['first_cohort']['source_revision'], root_binding_sha256=TOOL.sha(binding),
        counts=dict(original=1, native=1), sampler_body_calls=0, timed_batches=0,
        outer_state_restored=True, poststop_source_authority_unchanged=True)
    monkeypatch.setattr(TOOL, 'authorize', lambda *_: (outer, copy.deepcopy(manifest), go))
    return controller, snapshot, results, authority, terminal, manifest, path


@pytest.mark.parametrize('exit_code,verify_exit', [(0, 0), (2, 0), (0, 3)])
def test_one_real_cli_argv_and_once_verifier_inert(tmp_path, monkeypatch, exit_code, verify_exit):
    controller, _, results, authority, terminal, _, go = setup_host(tmp_path, monkeypatch)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if '--route' in command:
            assert command[-2:] == ['--route', 'lbp']
            assert 'mark' not in command
            write(results / 'first-cohort/lbp/terminal.json', terminal)
            return SimpleNamespace(returncode=exit_code)
        assert command[2] == 'verify'
        return SimpleNamespace(returncode=verify_exit)
    monkeypatch.setattr(TOOL.subprocess, 'run', run)
    if exit_code or verify_exit:
        with pytest.raises(AssertionError): TOOL.run_cohort(controller, results, 'python', authority, go, 'pin')
        assert not (results/'first-cohort-summary.json').exists()
    else:
        summary = TOOL.run_cohort(controller, results, 'python', authority, go, 'pin')
        assert len(summary['routes']) == 1 and summary['original_public_topcalls'] == summary['native_public_topcalls'] == 1
    assert len(calls) == (1 if exit_code else 2)


@pytest.mark.parametrize('field,value', [('outer_state_restored', False), ('poststop_source_authority_unchanged', False),
    ('root_binding_sha256', 'wrong'), ('sampler_body_calls', 1), ('counts', dict(original=2,native=1))])
def test_c1_or_budget_refusal_before_verifier(tmp_path, monkeypatch, field, value):
    controller, _, results, authority, terminal, _, go = setup_host(tmp_path, monkeypatch)
    terminal[field] = value
    def run(command, **kwargs):
        assert '--route' in command
        write(results/'first-cohort/lbp/terminal.json', terminal)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(TOOL.subprocess, 'run', run)
    with pytest.raises(AssertionError): TOOL.run_cohort(controller, results, 'python', authority, go, 'pin')
    assert not (results/'first-cohort-summary.json').exists()


@pytest.mark.parametrize('verified', [True, False])
def test_measurement_output_containment_and_actual_loader(tmp_path, monkeypatch, verified):
    controller, snapshot, results, authority, terminal, manifest, go = setup_host(tmp_path, monkeypatch)
    write(results/'first-cohort/lbp/terminal.json', terminal)
    write(results/'first-cohort-summary.json', dict(status='PASS', postbatch_offline_identity_verified=verified,
        original_public_topcalls=1, native_public_topcalls=1, timed_batches=0, sampler_body_calls=0, routes=[terminal]))
    calls = []
    outer = TOOL.parent(ROOT)
    def validate(root, output, selected, authorities):
        assert output == results/'first-cohort'
        assert set(selected['first_cohort']['routes']) == {'lbp'}
        return 'synthetic report control only'
    # Same module object returned by authorize; replace only report I/O in this inert control.
    original_authorize = TOOL.authorize
    monkeypatch.setattr(TOOL, 'authorize', lambda *args: (outer, original_authorize(*args)[1], original_authorize(*args)[2]))
    monkeypatch.setattr(outer, 'validate_reports', validate)
    def run(command, **kwargs):
        calls.append(command)
        if '--output' in command:
            assert Path(command[command.index('--output')+1]).resolve().is_relative_to(Path(json.loads(Path(authority['binding']).read_bytes())['output_root']))
            context_path = 'coverage/contracts/draw-multiblock-lbp-benchmark-execution-v5.json'
            context = json.loads((ROOT/context_path).read_bytes())
            # The unchanged DDA measurement loader consumes the actual generated envelope.
            binding = {k: manifest[k] for k in ('prepared_source','producer','core_sha256','runtime_sha256')}
            binding['source_revision'] = manifest['first_cohort']['source_revision']
            outer.driver_adapter(ROOT).validate_measurement(authority['measurement'], binding, context,
                TOOL.sha(authority['binding']), context_path=context_path)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(TOOL.subprocess, 'run', run)
    if not verified:
        with pytest.raises(AssertionError): TOOL.run_measurement(controller, snapshot, results, 'python', authority, go, 'pin')
        assert not calls and not Path(authority['measurement']).exists()
    else:
        assert TOOL.run_measurement(controller, snapshot, results, 'python', authority, go, 'pin') == 'synthetic report control only'
        assert len(calls) == 3 and sum('--output' in call for call in calls) == 1
