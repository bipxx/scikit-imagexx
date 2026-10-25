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
MAX_TREE_RUNTIME_SOURCE_PINS = {'src/max_tree.cpp': '1808c8581b465a96a5d5ee9cccd73ee269d17893877784933230173221f1e135',
 'include/skimagexx/max_tree.hpp': '60b33407a721e57b3629e671f0f822ab1dd344ccbaddfcf526a9e3f93ca9bd1c',
 'src/simd_max_tree_avx2.cpp': '2bdb72ade563581161038b8cf32260c6330e028e938590c7a382dc7c5fa92de4',
 'src/max_tree_stages.hpp': '8afefbbdfe06282b5524607c4a5a831ceaeddb59cf75047e33fbbaabee125184',
 'src/max_tree_stages.cpp': 'd1961c7d0b4dd0940a765e76e7fb6f64aee8edfcfb3255c69b4e030f3b42d9e6',
 'src/bind_max_tree.cpp': 'f4764b61180293ab98e474d68b40b214774ff367fd07e0ca4034c3705bbee2c5',
 'python/skimagexx/_max_tree.py': '3519902c42187e3f79ebe1bf5e17bdf05b562518e773fee05d743f79160db3ae'}
MAX_TREE_ORACLE_PATH = 'tests/cmake_consumer/max_tree_source_oracle.bin'
MAX_TREE_ORACLE_SHA256 = 'e7162c6aa8a43434e0c6db0f835912dfc48b64eedca7e79192cab61bc39d93b5'
RECONSTRUCTION_RUNTIME_SOURCE_PINS = {'include/skimagexx/reconstruction.hpp': '334aae965c0bff534146e7a7cc18a18c180beb3d0d47cba5dc83c9c30de234a8', 'src/reconstruction.cpp': 'a81b1e7acb903e6ecc1fa900c736521bc8b917282467e9b9e3a087ddf4317e2b', 'src/simd_reconstruction_avx2.cpp': '133c63473a284d163c6ba689b030266861a2b653ff5628b28f926bb82366d5fb', 'src/bind_reconstruction.cpp': '8d609e6ed59969ee6f48e5f32f2726cef2f21fed3ab940d91941b3a56ff32c94', 'python/skimagexx/_reconstruction.py': 'c190d7e8c879da82facdd4bbfd9d751535eed4a3f08870b7088d14b4372d4fd0', 'tests/test_reconstruction_native.cpp': 'eecb84170c45f15c563d380a58a447e35a82038b5d1bfad0337d8fbd89692b16'}
RECONSTRUCTION_ORACLE_PATH = 'tests/cmake_consumer/reconstruction_source_oracle.bin'
RECONSTRUCTION_ORACLE_SHA256 = 'b3c8f265e92a431ba282ad715856d9c7f4d24834b878876c63a90dcf703e860e'
RECONSTRUCTION_TRACE_POLICY_PINS = {'tests/reconstruction_trace_policy.py': '946ad095e76630438440209eb065c7482543ff1e57d2dce7a3c499f768e364d6', 'tests/test_reconstruction_contracts.py': '38b12d918c120e86d09fa1b0be246bb82e25b483184da650fbbae695e2d11bcf', 'tests/test_reconstruction_trace_policy.py': '722a000c7d0087a5883362d030eb4a48ddef6527a6d96f99be0ca2aa8fe4d592', 'coverage/evidence/reconstruction-trace-policy-migration.json': 'ebc9b988148d0d87e49d5a93eb69a1f74a71688cd5cc6c6dea85b3137ccef978', 'coverage/evidence/reconstruction-trace-policy-observations.json': '39b8a1b0e07018198550e624abd3955cf49928e87521b91a0936ca859312ed2e'}
NMI_PATH = 'metrics.normalized_mutual_information'
NMI_IMPORTED_SOURCE_PINS = {'include/skimagexx/normalized_mutual_information.hpp': '76e3cf0e895650f2aa79c0c61c596b8ce29bdc5f26b4bccd130e9d34108167f4',
 'src/normalized_mutual_information.cpp': 'cc9150a323e2bee04cb18be0f32008b306c9ee312d8fa39569174fe5ad426ac3',
 'src/bind_normalized_mutual_information.cpp': '423ee0bf149c3280b36a9229c4f3f783749b459a579359644c3623101222620d',
 'python/skimagexx/_normalized_mutual_information.py': 'b968a0b15852178254be7f28d7384651bc0bd071b312714ac60ad720cf53c49d',
 'tests/test_normalized_mutual_information_native.cpp': '0a5369154f1011067b17435fa851d6cf066ba0b9482d8b0ab1bdd469796476ba',
 'tests/test_normalized_mutual_information.py': '98ec869f16cac1f0a9898218eeafa4a9a478ae79270fd61d5df694ebbc21bd0c',
 'tests/test_normalized_mutual_information_cases.py': '02209513c6b724db9b7cb9e6b1d7a0a8b5021e3a6e2b1837cb6d1b34bb327b8b',
 'coverage/contracts/normalized-mutual-information.json': 'b043549b814f9d2fc7aa4d31f6134eee2162782e8aec78c1b035e7f6b90845c1'}
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


def qualification_scope(source):
    """Read authenticated source literals; never execute a source checkout."""
    return qualification_scope_text((source / 'tools/public_qualification.py').read_text(encoding='utf-8'))


def qualification_scope_text(text):
    module = ast.parse(text)
    values = {}
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('FAMILIES', 'PRODUCT_SCOPE', 'PRODUCT_MEMBERS', 'SDK_MEMBERS'):
                    require(target.id not in values, 'duplicate source qualification literal')
                    values[target.id] = ast.literal_eval(node.value)
    families = values.get('FAMILIES')
    require(type(families) is tuple and len(families) == len(set(families)), 'source SDK family identities differ')
    if set(values) == {'FAMILIES'}:
        # Original authenticated rank14 controllers predate the explicit census.
        require(len(families) == 14, 'original source SDK14 incomplete')
        return dict(name='rank14', families=list(families), members=104, sdk=59)
    require(set(values) == {'FAMILIES', 'PRODUCT_SCOPE', 'PRODUCT_MEMBERS', 'SDK_MEMBERS'},
            'source explicit product scope incomplete')
    scopes = {'tv17': (108, 62, ('tv_chambolle', 'tv_reductions', 'tv_denoise')),
              'bregman18': (110, 63, ('tv_chambolle', 'tv_reductions', 'tv_denoise', 'tv_bregman')),
              'max_tree19': (112, 64, ('tv_chambolle', 'tv_reductions', 'tv_denoise', 'tv_bregman', 'max_tree')),
              'matrix20': (114, 65, ('tv_chambolle', 'tv_reductions', 'tv_denoise', 'tv_bregman', 'max_tree', 'matrix_transform')),
              'reconstruction21': (116,66,('tv_chambolle','tv_reductions','tv_denoise','tv_bregman','max_tree','matrix_transform','reconstruction')),
              'nmi22': (118,67,('tv_chambolle','tv_reductions','tv_denoise','tv_bregman','max_tree','matrix_transform','reconstruction','normalized_mutual_information'))}
    require(values['PRODUCT_SCOPE'] in scopes, 'source TV/Bregman scope incomplete')
    members, sdk, suffix = scopes[values['PRODUCT_SCOPE']]
    require(type(values['PRODUCT_MEMBERS']) is int and type(values['SDK_MEMBERS']) is int
            and values['PRODUCT_MEMBERS'] == members and values['SDK_MEMBERS'] == sdk
            and len(families) == 14 + len(suffix) and families[-len(suffix):] == suffix,
            'source TV/Bregman product/SDK scope incomplete')
    return dict(name=values['PRODUCT_SCOPE'], families=list(families), members=members, sdk=sdk)


def validate_reconstruction_source(archive,snapshot,scope):
    if scope['name'] not in ('reconstruction21','nmi22'):return
    tree=ast.parse(archive.read('tools/public_qualification.py'));declared={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('RECONSTRUCTION_RUNTIME_SOURCE_PINS','RECONSTRUCTION_TRACE_POLICY_PINS','RECONSTRUCTION_ORACLE_PATH','RECONSTRUCTION_ORACLE_SHA256'):
                    require(target.id not in declared,'duplicate reconstruction source pin declaration')
                    declared[target.id]=ast.literal_eval(node.value)
    require(declared==dict(RECONSTRUCTION_RUNTIME_SOURCE_PINS=RECONSTRUCTION_RUNTIME_SOURCE_PINS,RECONSTRUCTION_TRACE_POLICY_PINS=RECONSTRUCTION_TRACE_POLICY_PINS,RECONSTRUCTION_ORACLE_PATH=RECONSTRUCTION_ORACLE_PATH,RECONSTRUCTION_ORACLE_SHA256=RECONSTRUCTION_ORACLE_SHA256),'reconstruction source pins absent or changed')
    for name,want in {**RECONSTRUCTION_RUNTIME_SOURCE_PINS,**RECONSTRUCTION_TRACE_POLICY_PINS,RECONSTRUCTION_ORACLE_PATH:RECONSTRUCTION_ORACLE_SHA256}.items():
        require(name in archive.namelist(),'reconstruction source absent: '+name)
        raw=archive.read(name);canonical=raw if name==RECONSTRUCTION_ORACLE_PATH else raw.replace(b'\r\n',b'\n')
        require(hashlib.sha256(canonical).hexdigest()==want and snapshot['files_sha256'].get(name)==want,'reconstruction source bytes differ: '+name)


def validate_max_tree_source(archive, snapshot, scope):
    """Data-only source admission; old rank/TV/Bregman envelopes keep their scope."""
    if scope['name'] not in ('max_tree19', 'matrix20', 'reconstruction21', 'nmi22'):
        return
    tree = ast.parse(archive.read('tools/public_qualification.py'))
    declared = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('MAX_TREE_RUNTIME_SOURCE_PINS', 'MAX_TREE_ORACLE_PATH', 'MAX_TREE_ORACLE_SHA256'):
                    require(target.id not in declared, 'duplicate corrected max-tree source pin declaration')
                    declared[target.id] = ast.literal_eval(node.value)
    require(declared == dict(MAX_TREE_RUNTIME_SOURCE_PINS=MAX_TREE_RUNTIME_SOURCE_PINS,
                MAX_TREE_ORACLE_PATH=MAX_TREE_ORACLE_PATH, MAX_TREE_ORACLE_SHA256=MAX_TREE_ORACLE_SHA256),
            'corrected max-tree source pins absent or changed')
    for name, expected in {**MAX_TREE_RUNTIME_SOURCE_PINS, MAX_TREE_ORACLE_PATH: MAX_TREE_ORACLE_SHA256}.items():
        require(name in archive.namelist(), 'corrected max-tree source absent: ' + name)
        raw = archive.read(name)
        canonical = raw if name == MAX_TREE_ORACLE_PATH else raw.replace(b'\r\n', b'\n')
        require(hashlib.sha256(canonical).hexdigest() == expected
                and snapshot['files_sha256'].get(name) == expected,
                'corrected max-tree source bytes differ: ' + name)


def validate_nmi_source(archive, snapshot, scope):
    """Authenticate the frozen NMI imports without executing scientific code."""
    if scope['name'] != 'nmi22':
        return
    tree = ast.parse(archive.read('tools/public_qualification.py'))
    declared = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'NMI_IMPORTED_SOURCE_PINS':
                    require(declared is None, 'duplicate NMI source pin declaration')
                    require(isinstance(node.value, ast.Dict), 'NMI source pins must be a literal dictionary')
                    keys = [ast.literal_eval(key) for key in node.value.keys]
                    require(len(keys) == len(set(keys)), 'duplicate NMI source pin key')
                    declared = ast.literal_eval(node.value)
    require(declared == NMI_IMPORTED_SOURCE_PINS, 'NMI source pins absent or changed')
    for name, expected in NMI_IMPORTED_SOURCE_PINS.items():
        require(archive.namelist().count(name) == 1, 'NMI source absent or duplicate: ' + name)
        canonical = archive.read(name).replace(b'\r\n', b'\n')
        require(hashlib.sha256(canonical).hexdigest() == expected
                and snapshot['files_sha256'].get(name) == expected,
                'NMI source bytes differ: ' + name)


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
    scope = qualification_scope(source)
    require(len(product['members']) == scope['members'], 'complete product member census differs')
    require(len([n for n in product['members'] if n.startswith('skimagexx/_core.')
                 and n.endswith(('.so', '.pyd'))]) == 1, 'sole production core differs')
    sdk = {n: h for n, h in product['members'].items()
           if n.startswith(('include/skimagexx/', 'lib/cmake/skimagexx/', 'share/licenses/scikit-imagexx/'))
           or n in ('lib/skimagexx.lib', 'lib/libskimagexx.a')}
    receipt = load(path / 'sdk.json')
    require(len(sdk) == scope['sdk'] and sdk == receipt['SDK_sha256'] == runtime['runtime']['sdk_sha256'],
            'complete SDK differs: ' + scope['name'])
    require(scope['families'] == receipt['families'], 'source SDK family identities differ')
    require(len(receipt['families']) == len(set(receipt['families'])) == len(scope['families'])
            and receipt['checks'] == qualification['SDK_checks'] == 2*len(scope['families']), 'SDK family/check count incomplete')
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
    with zipfile.ZipFile(checked(qout, 'source.zip', q)) as archive:
        text = archive.read('tools/public_qualification.py')
        require(hashlib.sha256(text.replace(b'\r\n', b'\n')).hexdigest()
                == q['source']['files_sha256']['tools/public_qualification.py'],
                'qualified source controller differs before registry admission')
        scope = qualification_scope_text(text.decode('utf-8'))
        validate_max_tree_source(archive, q['source'], scope)
        validate_reconstruction_source(archive,q['source'],scope)
        validate_nmi_source(archive,q['source'],scope)
    image_count = {'rank14': 190, 'tv17': 191, 'bregman18': 192, 'max_tree19': 198, 'matrix20': 199,'reconstruction21':200,'nmi22':201}[scope['name']]
    require(expected['image_count'] == image_count and expected['metadata_count'] == 3
            and expected['registered_count'] == len(set(expected['functions'])) == len(expected['functions']) == image_count+3,
            f'genuine{image_count}+3 registry qualification required')
    require(('restoration.denoise_tv_chambolle' in expected['functions']) == (scope['name'] != 'rank14'),
            'qualified TV registry/source scope differs')
    require(('restoration.denoise_tv_bregman' in expected['functions']) == (scope['name'] in ('bregman18', 'max_tree19', 'matrix20', 'reconstruction21', 'nmi22')),
            'qualified Bregman registry/source scope differs')
    max_tree = tuple('morphology.' + name for name in ('max_tree', 'max_tree_local_maxima',
                     'area_opening', 'area_closing', 'diameter_opening', 'diameter_closing'))
    require(all(expected['functions'].count(name) == 1 for name in max_tree) if scope['name'] in ('max_tree19', 'matrix20', 'reconstruction21', 'nmi22')
            else not any(name in expected['functions'] for name in max_tree),
            'qualified max-tree registry/source scope differs')
    require(('morphology.reconstruction' in expected['functions'])==(scope['name'] in ('reconstruction21','nmi22')),
            'source reconstruction scope/registry differs')
    require(('transform.matrix_transform' in expected['functions']) == (scope['name'] in ('matrix20','reconstruction21','nmi22')),
            'qualified matrix registry/source scope differs')
    require(expected['functions'].count(NMI_PATH) == 1 if scope['name'] == 'nmi22'
            else NMI_PATH not in expected['functions'], 'qualified NMI registry/source scope differs')
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
