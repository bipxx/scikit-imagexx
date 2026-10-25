"""Hosted H diagnostic of four unchanged failed products; no build or timing."""
import sys
if sys.flags.optimize:
    raise RuntimeError('H diagnostic requires unoptimized Python')

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / 'coverage/evidence/h-failed-product-diagnostic-inputs.json'
INPUTS_SHA256 = 'dd2e46313ef7e28856950169a8b9524e956b031d69ebf92e561eb014ceab0579'
FAILED_SOURCE = '7417676bbb628e75feac1be49aac900bc5af7b15'
FAILED_RUN = 40200919182
GATE_SOURCE = 'ea35a2481880301b8ce7b4eb2d77665adb813ca2'
OUTER_CAP = 1536 * 1024**2


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = ROOT / 'tools/li_installed_diagnostic.py'
if hashlib.sha256(helper.read_bytes()).hexdigest() != '3a86441c81f5cc7950353a9048f80d7301741cde32e585198ee8fbbcd5301a68':
    raise RuntimeError('Reviewed Li byte/transport helpers changed')
LI = load_module('h_diagnostic_retained_li_helpers', helper)
require, sha, save, selected_member = LI.require, LI.sha, LI.save, LI.selected_member


def validate_inputs(data, profile):
    require(data['schema'] == 'h-same-failed-product-fixed-inputs-v1'
        and data['failed_source'] == FAILED_SOURCE and data['failed_run'] == FAILED_RUN
        and data['corrected_gate_source'] == GATE_SOURCE and data['whole_product_acceptance'] is False
        and set(data['products']) == {'windows', 'modern', 'minimum', 'debug'}, 'Fixed H inputs differ')
    product = data['products'][profile]
    require(product['failed_source'] == FAILED_SOURCE and product['failed_run'] == FAILED_RUN,
        'Must preserve original failed product identity')
    require(product['artifact']['size_in_bytes'] <= OUTER_CAP, 'Artifact exceeds fixed cap')
    require((product['product_member_count'], product['SDK_member_count']) == (120, 68), 'Wrong product scope')
    return product


def project_gate(before, gate):
    require(len(before) == gate['before_bytes'] and hashlib.sha256(before).hexdigest() == gate['before_sha256'], 'Archived gate changed')
    old, new = gate['old_projection'].encode(), gate['new_projection'].encode()
    require(before.count(old) == 1 and new not in before, 'Projection boundary differs')
    after = before.replace(old, new)
    require(len(after) == gate['after_bytes'] and hashlib.sha256(after).hexdigest() == gate['after_sha256']
        and after.count(new) == 1 and after.replace(new, old) == before, 'Exact ea35 projection inverse failed')
    return after


def environment(mode=None):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith('SKIMAGEXX_') or key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONOPTIMIZE',
            'PYTEST_ADDOPTS', 'PYTEST_PLUGINS', 'LD_PRELOAD'):
            env.pop(key)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
        OMP_NUM_THREADS='1', OMP_DYNAMIC='FALSE', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1')
    if mode == 'scalar':
        env['SKIMAGEXX_DISABLE_SIMD'] = '1'
    return env


def command(out, label, args, env, timeout):
    code, expired = None, False
    with (out / (label + '.stdout')).open('xb') as stdout, (out / (label + '.stderr')).open('xb') as stderr:
        try:
            code = subprocess.run(args, stdout=stdout, stderr=stderr, env=env, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            expired = True
        finally:
            save(out / (label + '-exit.json'), dict(command=args, exit_code=code, timed_out=expired,
                timeout_seconds=timeout, operation_timings=0))
    return code


def materialize(source_zip, state, source):
    require(state['source']['revision'] == FAILED_SOURCE, 'State source differs')
    pins = state['source']['files_sha256']
    with zipfile.ZipFile(source_zip) as archive:
        require(archive.comment.decode() == FAILED_SOURCE, 'Source archive revision differs')
        entries = archive.infolist()
        names = [v.filename for v in entries if not v.is_dir()]
        require(len(names) == len(set(names)) == len({v.casefold() for v in names})
            and set(names) == set(pins), 'Source member closure differs')
        require(sum(v.file_size for v in entries) <= 1024**3, 'Source decoded size exceeds cap')
        for info in entries:
            name = info.filename
            path = PurePosixPath(name)
            require(not path.is_absolute() and '..' not in path.parts and '\\' not in name and ':' not in name,
                'Unsafe source member')
            if info.is_dir():
                continue
            require((info.external_attr >> 16) & 0o170000 in (0, 0o100000), 'Source member is not regular')
            raw = archive.read(info)  # CRC validates the unchanged raw bytes.
            require(hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest() == pins[name], 'Source state hash differs: ' + name)
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(raw)


def installed_product(package):
    dist = importlib.metadata.distribution('scikit-imagexx')
    expected = package['members_sha256']
    record = next(n for n in expected if n.endswith('.dist-info/RECORD'))
    rows = list(csv.reader(io.StringIO(Path(dist.locate_file(record)).read_text())))
    require(all(len(row) == 3 for row in rows), 'Malformed installed RECORD row')
    names = [r[0] for r in rows]
    extras = {record.rsplit('/', 1)[0] + '/' + n for n in ('INSTALLER', 'REQUESTED', 'direct_url.json')}
    require(len(names) == len(set(names)) and set(expected) <= set(names) <= set(expected) | extras, 'Installed product member set differs')
    for name, digest, size in rows:
        raw = Path(dist.locate_file(name)).read_bytes()
        if name != record:
            require(digest == 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b'=').decode()
                and int(size) == len(raw), 'Installed RECORD differs: ' + name)
            if name in expected:
                require(hashlib.sha256(raw).hexdigest() == expected[name], 'Installed archived member changed: ' + name)
        else:
            require(not digest and not size, 'Invalid installed RECORD self entry')
    core = Path(dist.locate_file(package['sole_core'])).resolve()
    return dict(path=str(core), sha256=sha(core), installed_record_sha256=sha(Path(dist.locate_file(record))),
        pip_generated_members=sorted(set(names) - set(expected)))


def validate_reports(output, mode, product, core, ids, unsafe):
    summary = json.loads(output.read_bytes())
    policies = {label:dict(rows=1772, safe=1744, blocked=28, guards=14, fallback_enabled=fallback)
        for label, fallback in (('strict', False), ('default', True))}
    require(summary == dict(logical_ids=886, policies=policies, rows=3544, private_adapter=False,
        full_API_complete=False, operation_timings=0), 'Incomplete or relabelled H summary')
    expected_order = [(name, worker) for worker in (1, 4) for name in ids]
    for label, fallback in (('strict', False), ('default', True)):
        report_path = output.with_name(output.stem + '-' + label + '.json')
        report = json.loads(report_path.read_bytes())
        rows = report['observations']
        journal = [json.loads(line) for line in report_path.with_suffix('.jsonl').read_text().splitlines()]
        require(rows == journal and [(r['id'], r['workers']) for r in rows] == expected_order, 'Missing, duplicate or relabelled H rows')
        require(report['schema'] == 'h-extrema-installed-public886-v1' and report['failure'] is None
            and report['fallback_enabled'] is fallback and report['profile'] == ('windows' if product['interpreter']['os_name'] == 'nt' else 'minimum' if product['runtime_pins']['numpy'] == '1.24.4' else 'modern')
            and report['core'] == {k:core[k] for k in ('path', 'sha256')}
            and report['default_workers'] == report['final_workers'] == 1
            and report['disable_simd'] == ('1' if mode == 'scalar' else None)
            and report['private_adapter'] is report['full_API_complete'] is False
            and report['operation_timings'] == 0, 'H report identity/state differs')
        require(mode != 'scalar' or report['has_avx2'] is False, 'Scalar override ignored')
        require(len(report['guards']) == len({g['id'] for g in report['guards']}) == 14
            and all(g['passed'] for g in report['guards']), 'Incomplete H guards')
        for row in rows:
            require(not row.get('forwarding_attempts'), 'Upstream forwarding recorded')
            if row['id'] in unsafe:
                require(row['status'] == 'source-unsafe-blocked' and row['actual_calls'] == 0, 'Unsafe input executed')
            else:
                require(row['public_observation_exact'] and row['intermediate_stages_exact']
                    and row['actual']['original_numeric_completed'] == 0, 'Actual H parity failed')


def diagnostic(profile, output, revision):
    require(os.environ.get('GITHUB_ACTIONS') == 'true', 'Hosted diagnostic only')
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        == revision == os.environ['GITHUB_SHA'], 'Controller source differs')
    require(sha(INPUTS) == INPUTS_SHA256, 'Fixed diagnostic inputs changed')
    data = json.loads(INPUTS.read_bytes()); product = validate_inputs(data, profile)
    require(sha(sys.executable) == product['interpreter']['sha256']
        and sys.version == product['interpreter']['version']
        and sys.implementation.cache_tag == product['interpreter']['cache_tag'], 'Interpreter differs from failed product')
    require(not output.exists() and output.is_relative_to((ROOT / 'build').resolve()), 'Fresh workspace output required')
    require(shutil.disk_usage(ROOT).free >= 16 * 1024**3, 'Insufficient free disk')
    output.mkdir(parents=True); payload = output / 'payload'; payload.mkdir()
    artifact = product['artifact']; endpoint = f'repos/bipxx/scikit-imagexx/actions/artifacts/{artifact["id"]}'
    api_raw = subprocess.check_output(['gh', 'api', endpoint]); (output / 'artifact-api.json').write_bytes(api_raw)
    api = json.loads(api_raw)
    require(all(api[k] == artifact[k] for k in ('id', 'name', 'digest', 'size_in_bytes'))
        and api['workflow_run']['id'] == FAILED_RUN and api['workflow_run']['head_sha'] == FAILED_SOURCE
        and not api['expired'], 'Archived artifact identity differs')
    archive_path = payload / 'original-artifact.zip'
    with archive_path.open('xb') as stream, (output / 'download.stderr').open('xb') as stderr:
        code = subprocess.run(['gh', 'api', endpoint + '/zip'], stdout=stream, stderr=stderr).returncode
    save(output / 'download-exit.json', dict(exit_code=code))
    require(code == 0 and archive_path.stat().st_size == artifact['size_in_bytes']
        and sha(archive_path) == artifact['digest'].removeprefix('sha256:'), 'Artifact transport failed')
    wheels = payload / 'wheels'; wheels.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        def select(pin):
            return selected_member(archive, pin['member'], pin['bytes'], pin['sha256'])
        source_zip = payload / 'source.zip'; source_zip.write_bytes(select(product['source_zip']))
        wheel = wheels / PurePosixPath(product['product_wheel']['member']).name
        wheel.write_bytes(select(product['product_wheel']))
        for name, pin in product['runtime_wheels'].items():
            require(PurePosixPath(name).name == name, 'Unsafe wheel filename')
            (wheels / name).write_bytes(select(pin))
        for name, pin in product['source_metadata'].items():
            (output / name).write_bytes(select(pin))
    state = json.loads((output / 'state.json').read_bytes()); source = payload / 'source'
    materialize(source_zip, state, source)
    q = load_module('h_archived_product_authentication', source / 'tools/public_qualification.py')
    package = q.inspect_wheel(wheel, source_zip, FAILED_SOURCE, scope='h_extrema23')
    require(package == json.loads((output / 'package.json').read_bytes()), 'Archived product receipt differs')
    target = source / data['gate']['path']; target.write_bytes(project_gate(target.read_bytes(), data['gate']))
    (output / 'executed-gate.py').write_bytes(target.read_bytes())
    policy = json.loads((source / 'coverage/contracts/h-extrema-integer.json').read_bytes())
    floating = json.loads((source / 'coverage/contracts/h-extrema-floating-plan.json').read_bytes())
    ids = policy['all_case_ids'] + [s['id'] for s in floating['cases']]
    require(len(ids) == len(set(ids)) == 886 and hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
        == data['cohort']['ids_sha256'] and policy['source_unsafe_blocked_ids'] == data['cohort']['unsafe_ids'], 'Frozen cohort differs')
    lines = [f'{pin["name"]}=={pin["version"]} --hash=sha256:{pin["sha256"]}' for pin in product['runtime_wheels'].values()]
    lines.append('scikit-imagexx==0.1.0 --hash=sha256:' + product['product_wheel']['sha256'])
    requirements = output / 'requirements.txt'; requirements.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    require(command(output, 'offline-install', [sys.executable, '-I', '-m', 'pip', 'install', '--no-index', '--no-deps',
        '--no-compile', '--require-hashes', '--only-binary=:all:', '--force-reinstall', '--find-links', str(wheels),
        '-r', str(requirements)], environment(), 600) == 0, 'Offline installation failed')
    for name, version in product['runtime_pins'].items():
        require(importlib.metadata.version(name) == version, 'Runtime version differs: ' + name)
    core = installed_product(package)
    require(core['sha256'] == product['core']['sha256'], 'Installed core differs')
    save(output / 'admission.json', dict(failed_source=FAILED_SOURCE, failed_run=FAILED_RUN, controller=revision,
        artifact_id=artifact['id'], core=core, interpreter=product['interpreter'], runtime_pins=product['runtime_pins'],
        source_archive=product['source_zip'], gate_source=GATE_SOURCE, gate_sha256=sha(target),
        product_accepted=False, sys_flags_optimize=sys.flags.optimize))
    outcomes = {}
    for mode in ('normal', 'scalar'):
        report = output / (mode + '-observations.json')
        code = command(output, mode, [sys.executable, '-I', '-X', 'utf8', '-B', str(target), '--audit-installed', str(report)], environment(mode), 1200)
        failure = None
        try:
            require(code == 0, 'Child failed; preserved exit/stdout/stderr and any partial journals')
            validate_reports(report, mode, product, core, ids, set(data['cohort']['unsafe_ids']))
        except (RuntimeError, ValueError, KeyError, TypeError, OSError) as error:
            failure = dict(type=type(error).__name__, message=str(error))
        outcomes[mode] = dict(exit_code=code, validation_error=failure)
    save(output / 'diagnosis.json', dict(profile=profile, outcomes=outcomes, failed_source=FAILED_SOURCE,
        gate_source=GATE_SOURCE, product_accepted=False, operation_timings=0,
        scope='Installed H diagnostic only; original whole qualification remains failed'))
    require(sum(p.stat().st_size for p in output.rglob('*') if p.is_file() and not p.is_relative_to(payload))
        <= 256 * 1024**2, 'Diagnostic evidence exceeds planned cap; preserve all output')
    require(all(v['exit_code'] == 0 and v['validation_error'] is None for v in outcomes.values()), 'H diagnostic failed; all evidence retained')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('windows', 'modern', 'minimum', 'debug'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--controller-revision', required=True)
    args = parser.parse_args()
    diagnostic(args.profile, args.output.resolve(), args.controller_revision)
