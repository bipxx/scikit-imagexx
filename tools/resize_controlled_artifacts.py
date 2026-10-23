"""Authenticate the initial resize backend and seal controlled experiment inputs."""
import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.provenance import source_fingerprint
from tools import ci_measurement_artifacts as artifacts

INITIAL_RUN = '39957875987'
INITIAL_REVISION = '66e5f78e4f5736597373bd5ccca0378f52d67980'
INITIAL_SOURCE = '570221bce8e8a5e6321bddeca57f7a4c24cee5086800df3c6c94d7c09a60ac59'
PROFILES = {
    'windows-modern': ('Windows', (3, 13), '2.5.4', '1.18.1'),
    'linux-modern': ('Linux', (3, 13), '2.5.4', '1.18.1'),
    'linux-minimum': ('Linux', (3, 11), '1.24.4', '1.15.3'),
}
PINS = {
    'windows-modern': {'measurement': '846bc812e63dfc9b8983de6905436f18bb4e86aad81f8830a86ca18e76ae93d4', 'manifest': 'd54effdec9e957978e3d8d4e9f39137921ec02d8c6ce7d159889be86020ea4cf', 'wheel': 'ba9bf17034b335a4f8713ab893697db4d79831851a9fc151c37a8797ac16c82a'},
    'linux-modern': {'measurement': '0624990b19e213f82d29364948c8309e05803599be174f3983079d33d4796f8c', 'manifest': '7d97e01717bcba17c0f3f1d59b1b4fd91c9610302d98d7f8460adf414432b11a', 'wheel': 'ffddc7b6f575f5237c2963f96b1956aa82619029b4cf1d585dcb973c7c7f6f4f'},
    'linux-minimum': {'measurement': '3ead45c70dfc77d0e117483630b909c3b292835b0f1f06e6cc95436460a92e8c', 'manifest': 'b19ad23599713a8268708577303ff489cc4a57611e52de69d0f681e88ec0f82c', 'wheel': 'ed165d500deb4a3415d18ddf9bf0bff9816978818448abb525f2c11c321ce778'},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree(directory):
    return {p.relative_to(directory).as_posix(): sha(p) for p in directory.rglob('*')
            if p.is_file() and '__pycache__' not in p.relative_to(directory).parts
            and p.suffix != '.pyc' and p != directory / 'controlled-capsule.json'}


def corpus_ast(path):
    names = {'image', 'corpus', 'serial', 'spec', 'observe', 'warning_key', 'assert_result'}
    constants = {'SEED', 'CORPUS_VERSION', 'TYPES', 'MODES', 'APIS'}
    nodes = [node for node in ast.parse(path.read_text()).body if
             (isinstance(node, ast.FunctionDef) and node.name in names) or
             (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets))]
    assert {node.name for node in nodes if isinstance(node, ast.FunctionDef)} == names
    return hashlib.sha256(ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False).encode()).hexdigest()


def baseline_inputs(initial, source, profile, check_runtime=True):
    """Only the three authenticated artifacts from the fixed original run qualify."""
    assert profile in PROFILES, 'unknown baseline runtime profile'
    manifest_path = initial / 'prepared/exact-package/baseline-source.json'
    measurement_path = initial / 'results/resize-performance.json'
    wheels = list((initial / 'prepared/wheels').glob('*.whl'))
    assert len(wheels) == 1, 'one exact initial wheel required'
    paths = {'manifest': manifest_path, 'measurement': measurement_path, 'wheel': wheels[0]}
    assert {name: sha(path) for name, path in paths.items()} == PINS[profile], 'initial artifact identity differs from pinned run'
    manifest = json.loads(manifest_path.read_text())
    measurement = json.loads(measurement_path.read_text())
    data = measurement['metadata']
    assert measurement['status'] == 'complete-measurement' and not measurement['failures']
    assert data['github_run_id'] == INITIAL_RUN and data['source_revision'] == manifest['revision'] == INITIAL_REVISION
    assert len(measurement['checks']) == data['case_count'] == 373 and len(measurement['results']) == data['timing_eligible_case_count'] == 372
    assert all(row['status'] == 'passed' for row in measurement['checks'])
    assert len({row['case_index'] for row in measurement['results']}) == 372
    assert data['requested_workers'] == [1] and all(row['threads'] == 1 for row in measurement['results'])
    fingerprint = source_fingerprint(source)
    assert fingerprint['source_sha256'] == INITIAL_SOURCE, 'old source fingerprint differs from initial source'
    for key, value in fingerprint.items():
        assert manifest[key] == data[key] == value, 'old source/manifest/measurement mismatch: ' + key
    assert corpus_ast(source / 'benchmarks/resize.py') == corpus_ast(ROOT / 'benchmarks/resize.py'), 'initial corpus/data/tolerance/reference functions changed'
    assert data['prepared_source']['manifest_sha256'] == sha(manifest_path)
    with zipfile.ZipFile(wheels[0]) as archive:
        members = {n.removeprefix('skimagexx/'): archive.read(n) for n in archive.namelist()
                   if n.startswith('skimagexx/') and not n.endswith('/')}
    package = initial / 'prepared/exact-package/_baseline'
    assert artifacts.package_files(package) == {n: hashlib.sha256(v).hexdigest() for n, v in members.items()}, 'complete preserved old package differs from initial wheel'
    python = {n: hashlib.sha256(v.replace(b'\r\n', b'\n')).hexdigest() for n, v in members.items() if n.endswith('.py')}
    expected_python = {p.relative_to(source / 'python/skimagexx').as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                       for p in (source / 'python/skimagexx').rglob('*.py')}
    assert python == expected_python == data['current_python_sha256'], 'all old Python must match source and measured package'
    binary = {n: hashlib.sha256(v).hexdigest() for n, v in members.items() if Path(n).suffix in ('.pyd', '.so')}
    assert binary and binary == manifest['binary_sha256'] == data['current_binary_sha256'], 'old extension differs from measured binary'
    system, abi, numpy_version, scipy_version = PROFILES[profile]
    assert data['platform'].startswith(system) and tuple(map(int, re.match(r'(\d+)\.(\d+)', data['python']).groups())) == abi
    assert (data['numpy'], data['scipy'], data['scikit_image']) == (numpy_version, scipy_version, '0.26.0')
    if check_runtime:
        import numpy, scipy, skimage
        assert platform.system() == system and platform.python_implementation() == 'CPython' and sys.version_info[:2] == abi, 'baseline OS/CPython ABI does not match current runtime'
        assert (numpy.__version__, scipy.__version__, skimage.__version__) == (numpy_version, scipy_version, '0.26.0'), 'baseline numerical runtime does not match'
    return {'profile': profile, 'package': str(package.resolve()), 'binary_sha256': binary,
            'python_sha256': python, 'manifest': manifest, 'initial_measurement': data,
            'authenticated_sha256': PINS[profile]}


def load_baseline(info):
    package = Path(info['package'])
    name = '_resize_initial_' + info['authenticated_sha256']['wheel'][:16]
    assert name not in sys.modules, 'baseline already loaded; use a fresh process'
    spec = importlib.util.spec_from_file_location(name, package / '__init__.py', submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
        getattr(module, 'transform')
        importlib.import_module(name + '.coverage')
    finally:
        sys.dont_write_bytecode = bytecode
    assert Path(module._core.__file__).resolve().parent == package.resolve()
    assert {Path(module._core.__file__).name: sha(module._core.__file__)} == info['binary_sha256']
    return module


def export_source(archive, output):
    assert not output.exists(), 'choose a fresh old-source directory'
    with zipfile.ZipFile(archive) as data:
        for name in data.namelist():
            assert (output / name).resolve().is_relative_to(output.resolve()), 'source member escapes output'
        data.extractall(output)


def current_source(capsule, root):
    wheels = []
    for path in (capsule / 'wheels').glob('*.whl'):
        with zipfile.ZipFile(path) as archive:
            if 'skimagexx/__init__.py' in archive.namelist():
                wheels.append(path)
    assert len(wheels) == 1, 'one current product wheel required'
    with zipfile.ZipFile(wheels[0]) as archive:
        python = {n.removeprefix('skimagexx/'): archive.read(n).replace(b'\r\n', b'\n')
                  for n in archive.namelist() if n.startswith('skimagexx/') and n.endswith('.py')}
        assert python == {p.relative_to(root / 'python/skimagexx').as_posix(): p.read_bytes().replace(b'\r\n', b'\n')
                          for p in (root / 'python/skimagexx').rglob('*.py')}, 'current wheel Python differs from complete source'
        for name in archive.namelist():
            if name.startswith('include/skimagexx/') and not name.endswith('/'):
                assert archive.read(name).replace(b'\r\n', b'\n') == (root / name).read_bytes().replace(b'\r\n', b'\n'), 'current SDK header differs from source'
    return source_fingerprint(root)


def freeze(capsule, profile, root=ROOT):
    manifest = capsule / 'controlled-capsule.json'
    assert not manifest.exists(), 'choose a fresh controlled capsule'
    initial = baseline_inputs(capsule / 'initial', capsule / 'old-source', profile)
    current = current_source(capsule, root)
    artifacts.freeze(root, capsule / 'wheels', capsule / 'current-runtime.json')
    payload = {'schema': 'resize-controlled-capsule-v1', 'profile': profile,
               'initial_sha256': initial['authenticated_sha256'], 'current_source': current, 'files_sha256': tree(capsule)}
    manifest.write_text(json.dumps(payload, indent=2) + '\n')
    return verify(capsule, root)


def verify(capsule, root=ROOT):
    manifest = capsule / 'controlled-capsule.json'
    data = json.loads(manifest.read_text())
    assert data['schema'] == 'resize-controlled-capsule-v1'
    assert data['files_sha256'] == tree(capsule), 'controlled capsule file set or bytes changed'
    assert data['initial_sha256'] == PINS[data['profile']]
    artifacts.verify(root, capsule / 'wheels', capsule / 'current-runtime.json')
    assert current_source(capsule, root) == data['current_source']
    info = baseline_inputs(capsule / 'initial', capsule / 'old-source', data['profile'])
    return {'manifest_sha256': sha(manifest), 'profile': data['profile'],
            'initial': info, 'current_runtime_manifest_sha256': sha(capsule / 'current-runtime.json')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'verify', 'export-source'])
    parser.add_argument('--capsule', type=Path)
    parser.add_argument('--profile', choices=PROFILES)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.mode == 'export-source':
        assert args.archive and args.output
        export_source(args.archive, args.output)
    elif args.mode == 'freeze':
        assert args.capsule and args.profile
        print(json.dumps(freeze(args.capsule, args.profile), indent=2))
    else:
        assert args.capsule
        print(json.dumps(verify(args.capsule), indent=2))


if __name__ == '__main__':
    main()
