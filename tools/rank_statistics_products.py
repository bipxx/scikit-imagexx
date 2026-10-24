"""Stage a bounded, still-unaccepted controlled product on its qualification host.

Only original sealed bytes are retained. No image imports, builds, measurements,
or accepted experiment pins are created. Run after the independent public
qualification verifier has completed successfully; keep the full raw artifact.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ('windows', 'modern', 'minimum')
CONTROLLERS = ('tools/rank_statistics_products.py', 'tools/public_qualification.py',
               'tools/build_identity_facts.py', 'benchmarks/rank_statistics_compare.py',
               'benchmarks/rank_statistics.py', 'tools/ci_measurement_artifacts.py',
               'coverage/contracts/rank-statistics-comparison.json',
               '.github/workflows/public-qualification.yml')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def safe(name):
    p = PurePosixPath(name)
    require(name and not p.is_absolute() and '..' not in p.parts
            and '\\' not in name and ':' not in name and p.as_posix() == name,
            'unsafe product member: ' + name)
    return p


def plain(path):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), 'regular sealed file required: ' + str(path))
    for parent in path.parents:
        require(not parent.is_symlink(), 'symlink parent forbidden: ' + str(parent))
    return path


def tree(path):
    result = {}
    for item in Path(path).rglob('*'):
        require(not item.is_symlink(), 'product symlink forbidden')
        if item.is_file():
            result[item.relative_to(path).as_posix()] = sha(plain(item))
    return result


def checked(out, name, qualification):
    safe(name)
    p = plain(Path(out) / name)
    record = qualification['records'].get(name)
    require(isinstance(record, dict) and record.get('bytes') == p.stat().st_size
            and record.get('sha256') == sha(p), 'original qualification record differs: ' + name)
    return p


def retain(source, target):
    """Link immutable sealed inputs where possible; never link a worktree file."""
    source = plain(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not target.exists(), 'staged output already exists')
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)
    require(sha(source) == sha(target), 'retained bytes differ')


def controller():
    path = ROOT / 'benchmarks/rank_statistics_compare.py'
    spec = importlib.util.spec_from_file_location('_stager_comparison_controller', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(callable(getattr(module, 'verify_build_identity', None)),
            'prospective build-identity verifier missing; product staging blocked')
    return module


def extract_source(archive_path, target, expected):
    with zipfile.ZipFile(archive_path) as archive:
        files = [i for i in archive.infolist() if not i.is_dir()]
        names = [i.filename for i in files]
        require(len(names) == len(set(names)) and set(names) == set(expected),
                'complete source archive file set differs')
        for info in archive.infolist():
            safe(info.filename.rstrip('/'))
            mode = info.external_attr >> 16
            require(not stat.S_ISLNK(mode), 'source ZIP symlink forbidden')
        target.mkdir()
        for info in files:
            name = info.filename
            data = archive.read(info)
            require(hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest() == expected[name],
                    'qualified source archive bytes differ: ' + name)
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(data)
    require(tree(target) == {i.filename: hashlib.sha256(archive_data).hexdigest()
                            for i, archive_data in source_members(archive_path)},
            'extracted full source differs')


def source_members(path):
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if not info.is_dir():
                yield info, archive.read(info)


def validate_payload(path, qualification, profile, compare):
    """Validate a staged tree without manufacturing accepted product pins."""
    source = path / 'source'
    cfg = load(source / 'coverage/contracts/rank-statistics-comparison.json')
    for name in CONTROLLERS:
        safe(name)
        require((source / name).is_file() and (ROOT / name).is_file(),
                'same-source controller missing: ' + name)
        require((source / name).read_bytes().replace(b'\r\n', b'\n')
                == (ROOT / name).read_bytes().replace(b'\r\n', b'\n'),
                'same-source controller differs: ' + name)
    for name, expected in cfg['source_text_sha256'].items():
        safe(name)
        require(hashlib.sha256((source / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == expected,
                'frozen R4 controller pin differs: ' + name)
    corpus = load(source / 'coverage/contracts/rank-statistics-benchmark.json')
    require(corpus['case_contract_sha256'] == cfg['case_contract_sha256']
            and len(corpus['case_inventory']) == 1379, 'fixed1379 corpus differs')
    runtime = load(path / 'runtime.json')
    require(runtime['schema'] == 'ci-offline-measurement-v1'
            and runtime['source'] == qualification['source'], 'runtime/source differs')
    require(sha(path / 'runtime-lock.txt') == runtime['lock_sha256'], 'runtime lock differs')
    wheels = {}
    for wheel in sorted((path / 'wheels').glob('*.whl')):
        row = compare.verify_wheel(wheel)
        require(row['name'] not in wheels, 'duplicate runtime distribution')
        wheels[row['name']] = row
    require({'scikit-imagexx', 'scikit-image', 'numpy', 'scipy'} <= wheels.keys(), 'wheel closure missing')
    require({k: {n: v[n] for n in ('name', 'version', 'sha256')} for k, v in wheels.items()}
            == {v['name']: v for v in runtime['wheels'].values()}, 'complete runtime wheel closure differs')
    require(wheels['scikit-image']['version'] == '0.26.0'
            and all(wheels[n]['version'] == cfg['profiles'][profile][n] for n in ('numpy', 'scipy')),
            'reference runtime pins differ')
    product = wheels['scikit-imagexx']
    sdk = {n: h for n, h in product['members'].items()
           if n.startswith(('include/skimagexx/', 'lib/cmake/skimagexx/', 'share/licenses/scikit-imagexx/'))
           or n in ('lib/skimagexx.lib', 'lib/libskimagexx.a')}
    receipt = load(path / 'sdk.json')
    require(len(sdk) == 59 and sdk == receipt['SDK_sha256'] == runtime['runtime']['sdk_sha256'],
            'complete59 SDK differs')
    module = ast.parse((source / 'tools/public_qualification.py').read_text(encoding='utf-8'))
    families = [ast.literal_eval(n.value) for n in module.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'FAMILIES' for t in n.targets)]
    require(len(families) == 1 and list(families[0]) == receipt['families'], 'source SDK family identities differ')
    require(len(receipt['families']) == len(set(receipt['families'])) == 14
            and receipt['checks'] == qualification['SDK_checks'] == 28, 'SDK14/28 incomplete')
    for prefix, directory, suffix in [('include/skimagexx/', source / 'include/skimagexx', None),
                                      ('skimagexx/', source / 'python/skimagexx', '.py')]:
        actual = {n: h for n, h in product['members'].items()
                  if n.startswith(prefix) and (suffix is None or n.endswith(suffix))}
        expected = {prefix + p.relative_to(directory).as_posix(): p
                    for p in directory.rglob('*') if p.is_file() and (suffix is None or p.suffix == suffix)}
        require(set(actual) == set(expected), 'complete wheel/source header or Python set differs')
        wheel_path = next(w for w in (path / 'wheels').glob('*.whl') if sha(w) == product['sha256'])
        with zipfile.ZipFile(wheel_path) as archive:
            for name, p in expected.items():
                require(archive.read(name).replace(b'\r\n', b'\n') == p.read_bytes().replace(b'\r\n', b'\n'),
                        'wheel/source header or Python bytes differ: ' + name)
    return compare.verify_build_identity(path, profile, qualification,
                                         qualification['source']['revision'], wheel=product)


def stage(qualification_dir, verification_receipt, output, profile, artifact_run):
    require(profile in PROFILES, 'only three Release comparison profiles; Debug stays in full qualification artifact')
    require(str(artifact_run).isdigit(), 'original qualification artifact run required')
    qout = Path(qualification_dir).absolute()
    plain(qout / 'manifest.json')  # Reject aliases through mutable symlink parents.
    qout = qout.resolve()
    output = Path(output).absolute()
    for parent in output.parents:
        require(not parent.is_symlink(), 'product output symlink parent forbidden')
    output = output.resolve()
    require(not output.exists() and not output.is_relative_to(qout)
            and not qout.is_relative_to(output), 'fresh disjoint product output required')
    qpath = plain(qout / 'manifest.json')
    q = load(qpath)
    require(q.get('schema') == 'public-qualification-v2' and q.get('passed') is True
            and not q.get('failures') and q.get('no_timing') is True and q.get('profile') == profile,
            'original terminal passed qualification required')
    verified = load(plain(verification_receipt))
    require(verified == dict(passed=True, manifest_sha256=sha(qpath), records_rehashed=len(q['records'])),
            'independent successful verification receipt differs')
    expected = q['expected_registry']
    require(expected['image_count'] == 190 and expected['metadata_count'] == 3
            and expected['registered_count'] == len(set(expected['functions'])) == len(expected['functions']) == 193,
            'genuine190+3 registry qualification required')
    state = load(checked(qout, 'state.json', q))
    require(state['source'] == q['source'] and state['profile'] == profile, 'qualified state differs')
    binding = q.get('build_identity')
    require(isinstance(binding, dict) and binding == state.get('build_identity')
            and binding.get('schema') == 'rank-statistics-build-identity-v2', 'prospective build identity pins missing')
    for name, key in [('build-identity.json', 'identity_sha256'), ('build-identity-facts.json', 'facts_sha256')]:
        require(sha(checked(qout, name, q)) == binding.get(key), 'build identity state pin differs')
    identity = load(qout / 'build-identity.json')
    require(identity.get('schema') == binding['schema'], 'build identity schema differs')
    capsule = 'capsule-statistics/'
    artifact = load(checked(qout, capsule + 'rank-statistics-artifacts.json', q))
    require(artifact['schema'] == 'rank-statistics-exact-offline-v1', 'statistics capsule schema differs')
    for name, digest in artifact['files_sha256'].items():
        require(sha(checked(qout, capsule + name, q)) == digest, 'statistics capsule seal differs')
    require(sha(checked(qout, capsule + 'rank-statistics-runtime.json', q))
            == artifact['runtime_manifest_sha256'], 'capsule runtime receipt differs')
    # Load only the same frozen standard-library comparison controller, never fixtures.
    compare = controller()
    output.mkdir(parents=True)
    selected = {'qualification.json': qpath, 'sdk.json': checked(qout, 'sdk.json', q),
                'source.zip': checked(qout, 'source.zip', q),
                'runtime.json': checked(qout, capsule + 'rank-statistics-runtime.json', q),
                'runtime-lock.txt': checked(qout, capsule + 'runtime-lock.txt', q),
                'qualification-evidence/state.json': checked(qout, 'state.json', q),
                'qualification-evidence/package.json': checked(qout, 'package.json', q),
                'qualification-evidence/statistics-capsule.json': checked(qout, capsule + 'rank-statistics-artifacts.json', q),
                'qualification-evidence/independent-verification.json': plain(verification_receipt)}
    for name in ('build-identity.json', 'build-identity-facts.json', *identity['evidence_records']):
        safe(name)
        if name not in ('build-identity.json', 'build-identity-facts.json'):
            require(name.startswith('build-identity-evidence/'), 'identity evidence outside namespace')
        selected[name] = checked(qout, name, q)
    runtime = load(selected['runtime.json'])
    require(runtime['source'] == q['source'], 'capsule source differs')
    for filename in runtime['wheels']:
        safe(filename)
        require(Path(filename).name == filename and filename.endswith('.whl'), 'invalid wheel filename')
        selected['wheels/' + filename] = checked(qout, capsule + 'wheels/' + filename, q)
    product_name = state['product_wheel']
    require(Path(product_name).name == product_name, 'invalid product wheel name')
    native = checked(qout, 'product-wheels/' + product_name, q)
    require('wheels/' + product_name in selected and sha(native) == sha(selected['wheels/' + product_name]),
            'exact qualified native wheel absent from runtime closure')
    package = load(selected['qualification-evidence/package.json'])
    require(package['wheel_sha256'] == sha(native), 'qualified package wheel differs')
    for name, source in selected.items():
        retain(source, output / name)
    extract_source(output / 'source.zip', output / 'source', q['source']['files_sha256'])
    require(sha(output / 'source/coverage/contracts/rank-statistics-benchmark.json')
            == artifact['corpus_manifest_sha256'], 'original capsule corpus binding differs')
    build = validate_payload(output, q, profile, compare)
    provenance = dict(status='staged-unaccepted', original_run=str(artifact_run),
                      original_artifact='public-qualification-' + profile,
                      original_manifest_sha256=sha(qpath), original_source_archive_sha256=sha(output / 'source.zip'),
                      independent_verification_sha256=sha(verification_receipt), build_identity_sha256=build['sha256'],
                      selected_original_records={name: dict(sha256=sha(p), bytes=p.stat().st_size)
                                                 for name, p in selected.items()},
                      boundary='No accepted product pin. Full original qualification artifact retained separately; outer artifact ID/digest assigned after upload and audited by root.')
    manifest = dict(schema='rank-statistics-comparison-product-v1', revision=q['source']['revision'],
                    profile=profile, status='staged-unaccepted', provenance=provenance, files_sha256=tree(output))
    (output / 'product.json').write_bytes((json.dumps(manifest, indent=2) + '\n').encode())
    require(tree(output) == dict(manifest['files_sha256'], **{'product.json': sha(output / 'product.json')}),
            'product tree changed during seal')
    return dict(status='staged-unaccepted', profile=profile, product_sha256=sha(output / 'product.json'),
                files=len(manifest['files_sha256']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--verification-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', choices=PROFILES, required=True)
    parser.add_argument('--artifact-run', required=True)
    args = parser.parse_args(argv)
    require(args.output.resolve().is_relative_to(ROOT / 'build'), 'CLI output must stay in ignored source build directory')
    print(json.dumps(stage(args.qualification, args.verification_receipt, args.output,
                           args.profile, args.artifact_run)))


if __name__ == '__main__':
    main()
