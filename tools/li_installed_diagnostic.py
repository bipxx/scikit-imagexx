"""Fixed hosted diagnostic of two archived, failed public200 products. No builds."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / 'coverage/evidence/li-failed-product-diagnostic-inputs.json'
INPUTS_SHA256 = '57eba925bba327b77b29f9f1893996004bff14bdc091c81984a59b4b19a362d5'
FAILED_SOURCE = '28350015823b36333085ce8ce04e3dcc70065572'
FAILED_RUN = 40174473080
SOURCE_FILES = frozenset(('tests/test_li_u16_bounded.py',
    'tests/test_iterative_threshold_contracts.py', 'coverage/contracts/li-u16-bounded.json',
    'coverage/contracts/iterative-thresholds.json'))
OUTER_CAP = 1288490188
OUTPUT_CAP = 256 * 1024**2


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def selected_member(archive, name, size, digest):
    """Read only a single exact, bounded regular member; zipfile verifies CRC."""
    path = PurePosixPath(name)
    require(not path.is_absolute() and '..' not in path.parts and '\\' not in name and ':' not in name,
            'unsafe selected member')
    require(archive.namelist().count(name) == 1, 'missing/duplicate selected member')
    info = archive.getinfo(name)
    require(not info.is_dir() and (info.external_attr >> 16) & 0o170000 in (0, 0o100000),
            'selected member is not regular')
    require(info.file_size == size and 0 < size <= 256 * 1024**2,
            'selected member size differs')
    raw = archive.read(name)
    require(len(raw) == size and hashlib.sha256(raw).hexdigest() == digest,
            'selected member hash differs')
    return raw


def command(out, label, args, env=None):
    with (out / (label + '.stdout')).open('xb') as stdout, \
            (out / (label + '.stderr')).open('xb') as stderr:
        try:
            result = subprocess.run(args, stdout=stdout, stderr=stderr, env=env,
                timeout=180 if label in ('normal', 'scalar') else 600)
            code, timed_out = result.returncode, False
        except subprocess.TimeoutExpired:
            code, timed_out = None, True
    save(out / (label + '-exit.json'), dict(command=args, exit_code=code, timed_out=timed_out))
    return code


def validate_inputs(data, profile):
    require(data['schema'] == 'li-same-failed-product-fixed-inputs-v1'
            and set(data['products']) == {'modern', 'windows'}
            and data['no_product_acceptance'] is True, 'fixed diagnostic inputs differ')
    product = data['products'][profile]
    require(product['source'] == FAILED_SOURCE and product['run'] == FAILED_RUN,
            'diagnostic must use original failed product')
    require(set(product['source_files']) == SOURCE_FILES, 'original source set differs')
    require(product['artifact']['size_in_bytes'] <= OUTER_CAP, 'outer artifact exceeds cap')
    return product


def diagnostic(profile, output, revision):
    require(os.environ.get('GITHUB_ACTIONS') == 'true', 'hosted diagnostic only')
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
            text=True).strip() == revision == os.environ['GITHUB_SHA'], 'controller revision differs')
    require(sha(INPUTS) == INPUTS_SHA256, 'input bytes changed')
    data = json.loads(INPUTS.read_bytes())
    product = validate_inputs(data, profile)
    require(sha(sys.executable) == product['interpreter']['sha256'], 'interpreter bytes differ')
    require(sys.implementation.cache_tag == product['interpreter']['cache_tag'], 'Python ABI differs')
    require(not output.exists() and output.resolve().is_relative_to((ROOT / 'build').resolve()),
            'output must be new and inside workspace build')
    require(shutil.disk_usage(ROOT).free >= 16 * 1024**3, 'insufficient free disk')
    output.mkdir(parents=True)
    payload = output / 'payload'
    payload.mkdir()
    artifact = product['artifact']
    endpoint = f'repos/bipxx/scikit-imagexx/actions/artifacts/{artifact["id"]}'
    api_raw = subprocess.check_output(['gh', 'api', endpoint])
    (output / 'artifact-api.json').write_bytes(api_raw)
    api = json.loads(api_raw)
    require(all(api[k] == artifact[k] for k in ('id', 'name', 'digest', 'size_in_bytes'))
            and api['workflow_run']['id'] == FAILED_RUN
            and api['workflow_run']['head_sha'] == FAILED_SOURCE and not api['expired'],
            'archived artifact identity differs')
    archive_path = payload / 'original-artifact.zip'
    with archive_path.open('xb') as stream, (output / 'download.stderr').open('xb') as stderr:
        result = subprocess.run(['gh', 'api', endpoint + '/zip'], stdout=stream, stderr=stderr)
    save(output / 'download-exit.json', dict(exit_code=result.returncode))
    require(result.returncode == 0 and archive_path.stat().st_size == artifact['size_in_bytes']
            and sha(archive_path) == artifact['digest'].removeprefix('sha256:'),
            'archived artifact transport failed')
    wheels = payload / 'wheels'
    wheels.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        source_pin = product['source_zip']
        source_zip = payload / 'original-source.zip'
        source_zip.write_bytes(selected_member(archive, source_pin['artifact_member'],
            source_pin['raw_bytes'], source_pin['raw_sha256']))
        pin = product['product_wheel']
        wheel = wheels / PurePosixPath(pin['artifact_member']).name
        wheel.write_bytes(selected_member(archive, pin['artifact_member'], pin['raw_bytes'], pin['raw_sha256']))
        for name, pin in product['runtime_wheels'].items():
            require(PurePosixPath(name).name == name, 'unsafe wheel filename')
            (wheels / name).write_bytes(selected_member(archive, pin['artifact_member'], pin['bytes'], pin['sha256']))
    source = output / 'source'
    with zipfile.ZipFile(source_zip) as archive:
        for name, pin in product['source_files'].items():
            require(pin['bytes'] <= 32 * 1024**2, 'source member exceeds cap')
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(selected_member(archive, name, pin['bytes'], pin['sha256']))
    gate = ROOT / 'tests/test_li_u16_pipeline.py'
    require(sha(gate) == data['gate_sha256'], 'diagnostic gate differs from reviewed87b')
    target_gate = source / 'tests/test_li_u16_pipeline.py'
    target_gate.write_bytes(gate.read_bytes())
    requirements = output / 'requirements.txt'
    lines = [f'{pin["name"]}=={pin["version"]} --hash=sha256:{pin["sha256"]}'
             for pin in product['runtime_wheels'].values()]
    lines.append('scikit-imagexx==0.1.0 --hash=sha256:' + product['product_wheel']['raw_sha256'])
    requirements.write_bytes(('\n'.join(lines) + '\n').encode())
    require(command(output, 'offline-install', [sys.executable, '-I', '-m', 'pip', 'install',
        '--no-index', '--no-deps', '--require-hashes', '--only-binary=:all:', '--force-reinstall',
        '--find-links', str(wheels), '-r', str(requirements)]) == 0, 'offline installation failed')
    for name, version in product['runtime_pins'].items():
        require(importlib.metadata.version(name) == version, 'installed runtime version differs: ' + name)
    distribution = importlib.metadata.distribution('scikit-imagexx')
    core = Path(distribution.locate_file(product['core']['wheel_member']))
    require(core.stat().st_size == product['core']['bytes'] and sha(core) == product['core']['sha256'],
            'installed archived core differs')
    save(output / 'admission.json', dict(failed_source=FAILED_SOURCE, failed_run=FAILED_RUN,
        artifact_id=artifact['id'], installed_core_sha256=sha(core), gate_sha256=sha(target_gate),
        original_source_files=product['source_files'], runtime_versions=product['runtime_pins'],
        interpreter_sha256=sha(sys.executable), controller=revision, product_accepted=False))
    exits, journals = {}, {}
    for mode in ('normal', 'scalar'):
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        env.pop('SKIMAGEXX_DISABLE_SIMD', None)
        if mode == 'scalar':
            env['SKIMAGEXX_DISABLE_SIMD'] = '1'
        journal = output / (mode + '-observations.jsonl')
        exits[mode] = command(output, mode, [sys.executable, '-I', '-X', 'utf8', '-B',
            str(target_gate), '--audit-installed-product', str(journal)], env)
        journals[mode] = journal.is_file() and journal.stat().st_size > 0
    save(output / 'diagnosis.json', dict(schema='li-same-failed-product-diagnostic-v1',
        profile=profile, child_exit_codes=exits, journals_present=journals,
        product_accepted=False, operation_timings=0,
        status='diagnostic-complete; child parity success/failure is recorded separately'))
    require(sum(p.stat().st_size for p in output.rglob('*')
        if p.is_file() and not p.is_relative_to(payload)) <= OUTPUT_CAP, 'diagnostic output exceeds cap')
    require(all(journals.values()),
            'child failed before retaining decisive observation; inspect preserved stderr/status')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('modern', 'windows'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--controller-revision', required=True)
    args = parser.parse_args()
    diagnostic(args.profile, args.output.resolve(), args.controller_revision)
