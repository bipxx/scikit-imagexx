"""Freeze and verify offline CI measurement inputs; never measure or build."""
import argparse
from email.parser import BytesParser
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def source_state(root):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    assert not git('status', '--porcelain'), 'preparation/measurement source must be clean'
    paths = git('ls-files').splitlines()
    return {'revision': git('rev-parse', 'HEAD'), 'files_sha256': {
        name: hashlib.sha256((root / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        for name in paths if (root / name).is_file()}}


def wheel_records(directory):
    records = {}
    names = set()
    for path in sorted(directory.glob('*.whl')):
        with zipfile.ZipFile(path) as archive:
            entries = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
            assert len(entries) == 1, 'ambiguous wheel metadata'
            info = BytesParser().parsebytes(archive.read(entries[0]))
        name = canonical(info['Name'])
        assert name not in names, 'multiple wheels for one distribution'
        names.add(name)
        records[path.name] = {'name': name, 'version': info['Version'], 'sha256': sha(path)}
    assert 'scikit-imagexx' in names and {'numpy', 'scipy', 'scikit-image'} <= names
    return records


def package_files(directory):
    return {p.relative_to(directory).as_posix(): sha(p) for p in directory.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def sdk_member(name):
    return name.startswith(('include/skimagexx/', 'lib/cmake/skimagexx/',
                            'share/licenses/scikit-imagexx/')) or name in (
                                'lib/skimagexx.lib', 'lib/libskimagexx.a')


def installed_state(wheels, records):
    versions = {item['name']: metadata.version(item['name']) for item in records.values()}
    assert all(versions[item['name']] == item['version'] for item in records.values()), 'installed dependency differs from frozen wheel'
    packages = {}
    for name in ('numpy', 'scipy', 'skimage', 'skimagexx'):
        spec = importlib.util.find_spec(name)
        assert spec and spec.origin
        packages[name] = package_files(Path(spec.origin).parent)
    libraries = {}
    for name in ('numpy', 'scipy', 'skimage'):
        directory = Path(importlib.util.find_spec(name).origin).parent
        libraries[name] = package_files(directory.parent / (directory.name + '.libs'))
    native = next(filename for filename, item in records.items() if item['name'] == 'scikit-imagexx')
    with zipfile.ZipFile(wheels / native) as archive:
        members = {n: hashlib.sha256(archive.read(n)).hexdigest()
                   for n in archive.namelist() if not n.endswith('/')}
    expected = {n.removeprefix('skimagexx/'): value for n, value in members.items()
                if n.startswith('skimagexx/')}
    assert packages['skimagexx'] == expected, 'complete installed package differs from exact wheel'
    sdk = {n: value for n, value in members.items() if sdk_member(n)}
    assert 'include/skimagexx/image.hpp' in sdk and any(n.endswith(('.a', '.lib')) for n in sdk), 'wheel lacks complete SDK'
    site = Path(importlib.util.find_spec('skimagexx').origin).parent.parent
    for name, value in sdk.items():
        assert sha(site / name) == value, 'installed SDK differs from wheel: ' + name
    for prefix in ('include/skimagexx', 'lib/cmake/skimagexx', 'share/licenses/scikit-imagexx'):
        assert {p.relative_to(site).as_posix() for p in (site / prefix).rglob('*') if p.is_file()} == {n for n in sdk if n.startswith(prefix + '/')}, 'installed SDK file set differs'
    return {'python': sys.version, 'implementation': platform.python_implementation(),
            'cache_tag': sys.implementation.cache_tag, 'machine': platform.machine(),
            'system': platform.system(), 'versions': versions, 'packages_sha256': packages,
            'sdk_sha256': sdk, 'dependency_libraries_sha256': libraries}


def inventory_snapshot(root):
    """Retain generated inventory without changing the checked source inputs."""
    paths = [root / 'coverage/api.json', root / 'coverage/README.md']
    original = {p: p.read_bytes() for p in paths}
    try:
        subprocess.run([sys.executable, str(root / 'tools/inventory.py')], check=True)
        output = root / 'build/ci-artifacts/inventory'
        output.mkdir(parents=True, exist_ok=True)
        for path in paths:
            (output / path.name).write_bytes(path.read_bytes())
    finally:
        for path, value in original.items():
            path.write_bytes(value)


def freeze(root, wheels, output):
    assert not output.exists(), 'choose a fresh artifact manifest'
    records = wheel_records(wheels)
    state = installed_state(wheels, records)
    lock = '\n'.join(f"{item['name']}=={item['version']} --hash=sha256:{item['sha256']}"
                     for item in sorted(records.values(), key=lambda x: x['name'])) + '\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name('runtime-lock.txt')
    lock_path.write_text(lock, encoding='utf-8')
    result = {'schema': 'ci-offline-measurement-v1', 'source': source_state(root),
              'wheels': records, 'runtime': state, 'lock_sha256': sha(lock_path)}
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def verify(root, wheels, artifact):
    expected = json.loads(artifact.read_text(encoding='utf-8'))
    assert expected['schema'] == 'ci-offline-measurement-v1'
    assert source_state(root) == expected['source'], 'source differs from preparation'
    assert wheel_records(wheels) == expected['wheels'], 'wheel file set or bytes changed'
    assert sha(artifact.with_name('runtime-lock.txt')) == expected['lock_sha256'], 'runtime lock changed'
    assert installed_state(wheels, expected['wheels']) == expected['runtime'], 'runtime/package/SDK differs from preparation'
    return expected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'verify', 'debug-assertions', 'inventory'])
    parser.add_argument('--wheel-dir', type=Path)
    parser.add_argument('--artifact', type=Path)
    args = parser.parse_args(argv)
    if args.mode == 'inventory':
        inventory_snapshot(ROOT)
        return
    if args.mode == 'debug-assertions':
        from skimagexx import _core
        assert getattr(_core, '_consumer_gil_assertions', False) is True, 'pybind GIL assertions must be active'
        print('pybind GIL assertions active in installed Debug extension')
        return
    if args.wheel_dir is None or args.artifact is None:
        parser.error('--wheel-dir and --artifact are required')
    result = (freeze if args.mode == 'freeze' else verify)(ROOT, args.wheel_dir, args.artifact)
    print(json.dumps({'status': args.mode, 'revision': result['source']['revision'],
                      'wheel_count': len(result['wheels']), 'artifact_sha256': sha(args.artifact)}))


if __name__ == '__main__':
    main()
