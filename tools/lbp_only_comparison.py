"""Fixed LBP-only authority around the unchanged qualified public-call route.

No target runs at import. Root must supply a separately admitted, externally
hashed subset GO; the old MARK/LBP manifest is authenticated, never rewritten.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

CONTEXT = 'coverage/contracts/lbp-only-comparison-v1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parent(controller):
    path = Path(controller) / 'tools/mark_lbp_first_cohort_probe.py'
    spec = importlib.util.spec_from_file_location('_lbp_parent_outer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authorize(controller, go_path, go_sha256):
    """Authenticate root subset authority before helper preparation or science."""
    controller = Path(controller)
    context_path = controller / CONTEXT
    context = json.loads(context_path.read_bytes())
    assert go_path is not None and go_sha256 is not None, 'UNBOUND LBP-only GO'
    assert sha(go_path) == go_sha256
    go = json.loads(Path(go_path).read_bytes())
    for name, pin in context['dependencies_sha256'].items():
        assert sha(controller / name) == pin, name
    assert go['status'] == 'ROOT_AUTHORIZED_LBP_ONLY_COMPARISON'
    assert go['context_sha256'] == sha(context_path)
    assert go['controller_sha256'] == sha(__file__)
    assert go['parent_sha256'] == context['parent_sha256']
    assert go['scope'] == context['scope']
    assert go['MARK_authorized'] is False
    admission = go['controller_admission']
    assert sha(admission['path']) == admission['sha256']
    admitted = json.loads(Path(admission['path']).read_bytes())
    assert admitted['admitted_controller_sha256'] == go['controller_sha256']
    created = datetime.fromisoformat(go['created_utc'])
    deadline = datetime.fromisoformat(go['deadline_utc'])
    assert created.tzinfo is not None and deadline.tzinfo is not None
    assert created <= datetime.now(timezone.utc) < deadline
    assert 0 < (deadline - created).total_seconds() <= 7200
    outer = parent(controller)
    path, manifest = outer.read_manifest(controller)
    assert sha(path) == context['parent_sha256']
    route = manifest['first_cohort']['routes']['lbp']
    assert route['api'] == context['api']
    assert [case['id'] for case in route['cases']] == context['case_ids']
    selected = json.loads(json.dumps(manifest))
    selected['first_cohort']['routes'] = {'lbp': route}
    selected['families'] = {Path(route['driver']).stem: manifest['families'][Path(route['driver']).stem]}
    selected['execution_scope'].update(canonical_cases=4, canonical_raw_samples=56,
                                       warmup_groups=8, untimed_topcalls=dict(original=1, native=1))
    selected['subset_authority'] = dict(path=str(Path(go_path).resolve()), sha256=go_sha256,
                                      parent_sha256=context['parent_sha256'], MARK_authorized=False)
    selected['deadline_utc'] = go['deadline_utc']
    return outer, selected, go


def prepare_cohort(controller, snapshot, results, python, site, go_path, go_sha256):
    outer, manifest, go = authorize(controller, go_path, go_sha256)
    controller, snapshot, results = map(Path, (controller, snapshot, results))
    first = manifest['first_cohort']; route = first['routes']['lbp']
    admission = controller / first['root_admission_path']
    stager = controller / first['root_stager_path']
    runtime = controller.parent / 'stage/runtime.json'
    assert sha(admission) == first['root_admission_sha256']
    assert sha(stager) == first['root_stager_sha256']
    assert sha(runtime) == manifest['runtime_sha256']
    actual = json.loads(runtime.read_bytes()); sdk = actual['runtime']['sdk_sha256']
    assert len(sdk) == manifest['SDK_members']
    core = Path(site) / first['core_member']
    assert core.resolve().is_relative_to(Path(site).resolve()) and sha(core) == manifest['core_sha256']
    helper = results / 'helper-binding.json'; assert helper.is_file()
    folder = results / 'authority-lbp'; folder.mkdir(exist_ok=False)
    binding_path, root_go = folder / 'binding.json', folder / 'root-go.json'
    value = dict(status='ROOT_AUTHORIZED_GENUINE_FIRST_COHORTS', product_kind='GENUINE_INSTALLED_WHOLE_PRODUCT',
        source_root=str(snapshot.resolve()), source_revision=first['source_revision'],
        git_argv=['git', '-C', str(snapshot.resolve())], prepared_source=manifest['prepared_source'],
        producer=manifest['producer'], interpreter=str(python), installed_site=str(site),
        numpy=actual['runtime']['versions']['numpy'], core_path=str(core.resolve()),
        core_sha256=manifest['core_sha256'], SDK_members=sdk, runtime_path=str(runtime.resolve()),
        runtime_sha256=manifest['runtime_sha256'], genuine_root_admission_path=str(admission.resolve()),
        genuine_root_admission_sha256=first['root_admission_sha256'],
        genuine_root_stager_path=str(stager.resolve()), genuine_root_stager_sha256=first['root_stager_sha256'],
        helper_binding_path=str(helper.resolve()), helper_binding_sha256=sha(helper),
        output_root=str((results / 'first-cohort').resolve()), routes={'lbp': route},
        base_contract_sha256=sha(snapshot / route['contract']), root_go_path=str(root_go.resolve()))
    context = json.loads((snapshot / route['contract']).read_bytes())
    assert context['route'] == route and context['route_id'] == 'lbp'
    outer.verify_source_state(value, manifest)
    outer.write_json(binding_path, value)
    outer.write_json(root_go, dict(binding_sha256=sha(binding_path), probe_sha256=first['probe_sha256'],
        routes=['lbp'], timed_batches=0, maximum_original_topcalls=1, maximum_native_topcalls=1,
        deadline_utc=go['deadline_utc'], helper_binding_sha256=sha(helper),
        subset_authority=manifest['subset_authority']))
    authority = dict(binding=str(binding_path.resolve()), go=str(root_go.resolve()),
                     measurement=str((folder / 'measurement.json').resolve()))
    outer.write_json(results / 'cohort-authorities.json', {'lbp': authority})
    return authority


def run_cohort(controller, results, python, authority, go_path, go_sha256):
    outer, manifest, _ = authorize(controller, go_path, go_sha256)
    results = Path(results)
    command = [str(python), str(Path(controller) / manifest['first_cohort']['probe_path']),
               '--binding', authority['binding'], '--root-go', authority['go'], '--route', 'lbp']
    outer.write_json(results / 'first-cohort-lbp-command.json', command)
    with (results / 'first-cohort-lbp.log').open('w') as log:
        done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    outer.write_json(results / 'first-cohort-lbp-exit.json', dict(exit=done.returncode))
    assert done.returncode == 0, 'Stop; no timing or automatic retry'
    terminal = json.loads((results / 'first-cohort/lbp/terminal.json').read_bytes())
    check_terminal(terminal, manifest, authority)
    # Same existing offline verifier, once after this one-cohort batch.
    runtime = Path(controller).parent / 'stage/runtime.json'
    assert sha(runtime) == manifest['runtime_sha256']
    verifier = [str(python), str(Path(controller).parent / 'snapshot/tools/ci_measurement_artifacts.py'),
                'verify', '--wheel-dir', str(Path(controller).parent / 'stage/wheels'), '--artifact', str(runtime)]
    with (results / 'first-cohort-identity-after.log').open('w') as log:
        checked = subprocess.run(verifier, stdout=log, stderr=subprocess.STDOUT)
    outer.write_json(results / 'first-cohort-identity-after-exit.json', dict(exit=checked.returncode))
    assert checked.returncode == 0, 'Postbatch verifier failed; no timing'
    assert sha(runtime) == manifest['runtime_sha256']
    authorize(controller, go_path, go_sha256)
    summary = dict(status='PASS', original_public_topcalls=1, native_public_topcalls=1,
        timed_batches=0, sampler_body_calls=0, postbatch_offline_identity_verified=True, routes=[terminal])
    outer.write_json(results / 'first-cohort-summary.json', summary)
    return summary


def check_terminal(done, manifest, authority):
    assert done['status'] == 'PASS_REAL_CLI_STOP_BEFORE_BODY'
    assert done['route'] == 'lbp' and done['api'] == 'feature.draw_multiblock_lbp'
    assert done['source'] == manifest['first_cohort']['source_revision']
    assert done['root_binding_sha256'] == sha(authority['binding'])
    assert done['counts'] == dict(original=1, native=1)
    assert done['timed_batches'] == done['sampler_body_calls'] == 0
    assert done['outer_state_restored'] is done['poststop_source_authority_unchanged'] is True


def run_measurement(controller, snapshot, results, python, authority, go_path, go_sha256):
    outer, manifest, go = authorize(controller, go_path, go_sha256)
    results = Path(results)
    summary = json.loads((results / 'first-cohort-summary.json').read_bytes())
    assert summary['status'] == 'PASS' and summary['postbatch_offline_identity_verified'] is True
    assert summary['original_public_topcalls'] == summary['native_public_topcalls'] == 1
    assert summary['timed_batches'] == summary['sampler_body_calls'] == 0
    terminal_path = results / 'first-cohort/lbp/terminal.json'
    done = json.loads(terminal_path.read_bytes())
    assert summary['routes'] == [done]
    check_terminal(done, manifest, authority)
    target = Path(authority['measurement']); assert not target.exists()
    manifest['completed_first_cohort'] = dict(path=str(terminal_path.resolve()), sha256=sha(terminal_path),
                                             binding_sha256=sha(authority['binding']))
    manifest['scoped_parent_sha256'] = manifest['subset_authority']['parent_sha256']
    manifest['scoped_api'] = 'feature.draw_multiblock_lbp'
    outer.write_json(target, manifest)
    route = manifest['first_cohort']['routes']['lbp']
    output = results / 'first-cohort/lbp.json'
    command = [str(python), str(Path(snapshot) / route['driver']), '--threads', '1', '--repeat', '7',
               '--api', route['api'], '--output', str(output), '--root-binding', authority['binding'],
               '--measurement-go', str(target)]
    outer.write_json(results / 'lbp-command.json', command)
    runtime = Path(controller).parent / 'stage/runtime.json'
    verifier = [str(python), str(Path(snapshot) / 'tools/ci_measurement_artifacts.py'), 'verify',
                '--wheel-dir', str(Path(snapshot).parent / 'stage/wheels'), '--artifact', str(runtime)]
    with (results / 'lbp-identity-before.log').open('w') as log:
        subprocess.run(verifier, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (results / 'lbp.log').open('w') as log:
        measured = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    outer.write_json(results / 'lbp-exit.json', dict(exit=measured.returncode))
    assert measured.returncode == 0, 'Measurement failed; no automatic retry'
    with (results / 'lbp-identity-after.log').open('w') as log:
        subprocess.run(verifier, stdout=log, stderr=subprocess.STDOUT, check=True)
    authorize(controller, go_path, go_sha256)
    # Reuse the unchanged full report/journal validator at the actual bound output root.
    return outer.validate_reports(snapshot, results / 'first-cohort', manifest, {'lbp': authority})
