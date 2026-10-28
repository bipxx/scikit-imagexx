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
H_EXTREMA_PATHS = ('morphology.h_maxima', 'morphology.h_minima')
H_EXTREMA_RUNTIME_SOURCE_PINS = {'include/skimagexx/h_extrema.hpp': 'e56ee6af110857ed50657aaf61d0ec1a8afc608c57c96a56a63eefeaab6dc9fd', 'src/h_extrema.cpp': '59cd1d8022926d1be06c2ecd2a68a817f9dbd8623b2d1f5f7871212a4ab2b817', 'src/bind_h_extrema.cpp': '732f4da1e6cd0d06c2c5126cc6d62df11cc4893ae59ea650ca41adc10cbdffe1', 'python/skimagexx/_h_extrema.py': '966f35f953e1bf630da84ddf0cd9de25662f0586c74d7cd1f93bbeb8b432790e'}
H_EXTREMA_FROZEN_RAW_PINS = {'tests/test_h_extrema_native.cpp': '228c3dc2e0e8041d9d49454b7b95cf3ab5947ebcf13a27145c94cdf70930c438', 'tests/test_h_extrema_native_pipeline.py': 'c04bfaea2b69744bb99336546b6e01ced556c9bebb87cf6d5b626c3a3bf106b5', 'tests/test_h_extrema_floating_native.cpp': '41b347cd259b67af14bb162394d5ef63edf62825a8632c3a93cf9a052362daca', 'tests/test_h_extrema_floating_pipeline.py': '86d4f0491ea8fc89230a300b4b45d85f8a8ce92af1d381e7abe6e52b2f59f148', 'coverage/contracts/h-extrema.json': 'ed02daef587eb7a408c578725a85e2fa9031a8beb5e63879dc32ee3667dcda0d', 'coverage/contracts/h-extrema-plan.json': '652ac143392a9005712fb8ab42a7bb53163da4be15ed848a2e88e3b004d6029b', 'coverage/contracts/h-extrema-integer.json': 'e91c0947845184fded73472c66128efe889ae795c6a323d800f923560699af51', 'coverage/contracts/h-extrema-floating.json': '566ed99dfcfe70ea9d24deb1e8237389de9ed838a32d3909571c29a8d9a2a8df', 'coverage/contracts/h-extrema-floating-plan.json': '67d6b11cbe3d78215aa637f167dd1cbf3a7543f568d44e9cda93dcea1cf901a0', 'coverage/contracts/h-extrema-object-preentry.json': '10383844bf54762cf09293f5270cd6d49713371bba5d558b172f0293e512dff8', 'tests/test_h_extrema_contracts.py': '82fd81ecc8a268878e4356231f2d046814619de1aabc2b0d925da14ed4c8c4d4', 'tests/test_h_extrema_boundaries.py': 'a12bbbc4ae1eb2a4fc6e24c924f363de5d7b7d2279eb2a7253b3757b75ea4a71', 'tests/h_extrema_reference.py': 'ac314e885d91bc91be5099cf5619d3ad2a6debc9de8cd237de50cd99160f42ce', 'tests/h_extrema_floating_reference.py': 'a6220e0d3bfdd4624fd15f896f51d73e2a3bc4e370500fefd14340f8c532bf04', 'tests/test_h_extrema_floating_cases.py': '171a6823112ff980b71fadae6788dc55daa26ff21c629a5fa2f554ff8f26fd4a', 'tests/h_extrema_source_provenance.py': '962d99e39beeadc6a534525e1b97aee22ad5ea40623b8cbaa7844e15f548929e', 'tests/test_h_extrema_source_provenance.py': 'e222069d656c99cd89e0bb4230825373331464e95c3fb86672817909d76da423', 'coverage/evidence/h-extrema-reference-assertion-migration.json': '923b435a24165befd53044dbd71df8b4a36fed2562c4f187b31f6300d9bdfb81', 'coverage/evidence/h-extrema-reference-assertion-migration-9b.json': 'bdf3c49437f9b2a33486efc41f3a237c4a39a0b2c66884636a880311ae261754', 'tests/cmake_consumer/h_extrema_source_oracle.bin': '091de3e03a728ada6e181a9ed22f7ceca071d66a5c8f1df968c09c39905716a5', 'tests/cmake_consumer/h_extrema_source_oracle.cmake': '4f09f0ff38d88e4d983e9e0e1c0d80233584928ad13c577e8b6ad57026dda2cd', 'coverage/evidence/h-extrema-floating-private-admission.json': 'f6e0fdf58c9171d2313dc38a854ed4d7948e73baeed4eff6ebffdd40caa06d20', 'tests/h_extrema_public_authentication.py': 'd3850e055d900099e229b4356a8b40bc8524150b6a0679178838c33d174c7680', 'coverage/evidence/h-extrema-reconstruction-reference-integration.json': 'aaa12eeee8d07052427cc71ec2906308f6b27e5f9728085bd136b245ed48e0dd', 'tests/test_reconstruction_contracts.py': '920150a40c65ae5eb57bd3b17f96a269b040de3d26b2c83d6fdf69ed0d83d61a', 'coverage/evidence/reconstruction-public-assertion-migration.json': '21eb60b8665449d4ad7adf194e5f00c48e117d790b328652f69561337d6d18fc', 'coverage/contracts/reconstruction.json': '621981325f5e6939d3fb4d1c94e52910b369ef3ca798c326bdb93322a12b504c', 'coverage/evidence/reconstruction-trace-policy-migration.json': 'ebc9b988148d0d87e49d5a93eb69a1f74a71688cd5cc6c6dea85b3137ccef978', 'tests/reconstruction_trace_policy.py': '946ad095e76630438440209eb065c7482543ff1e57d2dce7a3c499f768e364d6'}
H_EXTREMA_HISTORICAL_RAW_PINS = {'tests/test_h_extrema_contracts.py': '00ad418879bc5b4ffbeebca0f78497f03d06da9cc1b31f60cd8313aa80327151', 'tests/h_extrema_reference.py': 'bc9efc45e338d6c447284df244185d8101d4615e6f72abfe500ced05f71bdfbf', 'tests/h_extrema_floating_reference.py': '389e5cffe1b399c0deb3c8dce8424a414b9b2329e16907e6afa86415fab0dfb0', 'tests/test_h_extrema_floating_cases.py': 'f127526cd66c5043edd66d82518db3abb2a8956d3b29f1931057566ec090de1e'}
H_EXTREMA_LEGACY_RUNTIME_SOURCE_PINS = {'include/skimagexx/h_extrema.hpp': 'fc934e0223908b2728d3b3f5493825d5221a8a83a037658ec9702a69952b0ad1', 'src/h_extrema.cpp': 'd67fce22ee797815671c94ccd8489795c1708a49e09019541e686009860e395f', 'src/bind_h_extrema.cpp': '0932da3b09ae5ec50879f7611f4f2842be159f1ae3bb17b5854e31d096f06b86', 'python/skimagexx/_h_extrema.py': '60b19c3130123043ae86294a2fc20a269e331275670ceb5d9bc5cef2d569869a'}
H_EXTREMA_LEGACY_FROZEN_RAW_PINS = {'coverage/contracts/h-extrema-integer.json': 'e91c0947845184fded73472c66128efe889ae795c6a323d800f923560699af51', 'coverage/contracts/h-extrema-plan.json': '652ac143392a9005712fb8ab42a7bb53163da4be15ed848a2e88e3b004d6029b', 'coverage/contracts/h-extrema.json': 'ed02daef587eb7a408c578725a85e2fa9031a8beb5e63879dc32ee3667dcda0d', 'tests/h_extrema_reference.py': 'bc9efc45e338d6c447284df244185d8101d4615e6f72abfe500ced05f71bdfbf', 'tests/test_h_extrema_boundaries.py': 'a12bbbc4ae1eb2a4fc6e24c924f363de5d7b7d2279eb2a7253b3757b75ea4a71', 'tests/test_h_extrema_contracts.py': '00ad418879bc5b4ffbeebca0f78497f03d06da9cc1b31f60cd8313aa80327151', 'tests/test_h_extrema_native.cpp': '228c3dc2e0e8041d9d49454b7b95cf3ab5947ebcf13a27145c94cdf70930c438', 'tests/test_h_extrema_native_pipeline.py': 'c04bfaea2b69744bb99336546b6e01ced556c9bebb87cf6d5b626c3a3bf106b5'}
H_EXTREMA_ORACLE_PATH = 'tests/cmake_consumer/h_extrema_source_oracle.bin'
H_EXTREMA_ORACLE_SHA256 = '091de3e03a728ada6e181a9ed22f7ceca071d66a5c8f1df968c09c39905716a5'

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


def qualification_scope(source, profile='modern'):
    """Read authenticated source literals; never execute a source checkout."""
    names = [p.relative_to(source).as_posix() for prefix in ('python/skimagexx', 'include')
             for p in (source / prefix).rglob('*') if p.is_file()]
    return qualification_scope_text((source / 'tools/public_qualification.py').read_text(encoding='utf-8'),
        read_bytes=lambda name: (source / name).read_bytes(), names=names, profile=profile)


def qualification_scope_text(text, *, read_bytes=None, names=None, profile='modern'):
    module = ast.parse(text)
    values = {}
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('FAMILIES', 'PRODUCT_SCOPE', 'PRODUCT_MEMBERS', 'SDK_MEMBERS', 'CURRENT_PRODUCT_SCOPE', 'CURRENT_FAMILIES'):
                    require(target.id not in values, 'duplicate source qualification literal')
                    if target.id == 'CURRENT_FAMILIES':
                        require(isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add)
                                and isinstance(node.value.left, ast.Name) and node.value.left.id == 'FAMILIES',
                                'current client declaration must extend closed historical families')
                        extension = ast.literal_eval(node.value.right)
                        require(type(extension) is tuple, 'current client extension must be a tuple')
                        values[target.id] = values['FAMILIES'] + extension
                    else:
                        values[target.id] = ast.literal_eval(node.value)
    current = values.pop('CURRENT_PRODUCT_SCOPE', None)
    current_families = values.pop('CURRENT_FAMILIES', None)
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
              'nmi22': (118,67,('tv_chambolle','tv_reductions','tv_denoise','tv_bregman','max_tree','matrix_transform','reconstruction','normalized_mutual_information')),
              'h_extrema22': (118,67,('tv_chambolle', 'tv_reductions', 'tv_denoise', 'tv_bregman', 'max_tree', 'matrix_transform', 'reconstruction', 'h_extrema')),
              'h_extrema23': (120,68,('tv_chambolle', 'tv_reductions', 'tv_denoise', 'tv_bregman', 'max_tree', 'matrix_transform', 'reconstruction', 'normalized_mutual_information', 'h_extrema'))}
    require(values['PRODUCT_SCOPE'] in scopes, 'source TV/Bregman scope incomplete')
    members, sdk, suffix = scopes[values['PRODUCT_SCOPE']]
    require(type(values['PRODUCT_MEMBERS']) is int and type(values['SDK_MEMBERS']) is int
            and values['PRODUCT_MEMBERS'] == members and values['SDK_MEMBERS'] == sdk
            and len(families) == 14 + len(suffix) and families[-len(suffix):] == suffix,
            'source TV/Bregman product/SDK scope incomplete')
    if current is not None:
        require(current == 'current236' and read_bytes is not None and names is not None,
                'current source scope requires authenticated install closure')
        # Reuse our trusted stdlib-only qualifier's project-specific install rules;
        # never execute the authenticated source archive or an extracted checkout.
        spec = importlib.util.spec_from_file_location('_stager_current_qualification', Path(__file__).with_name('public_qualification.py'))
        tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
        contract = tool.current_contract(read_bytes)
        layout = tool.current_install_layout(read_bytes, names, profile)
        require(tuple(families) == tool.FAMILIES and current_families == tool.CURRENT_FAMILIES, 'current installed SDK client scope differs')
        return dict(name=current, families=list(current_families), members=layout['product_count'], sdk=layout['SDK_count'],
                    image_count=len(contract['image_paths']), metadata_count=len(contract['metadata_paths']),
                    functions=sorted(contract['image_paths'] + contract['metadata_paths']),
                    install_members=layout['members_without_core'], SDK_members=layout['SDK_members'],
                    SDK_client_scope=contract['SDK_scope'])
    return dict(name=values['PRODUCT_SCOPE'], families=list(families), members=members, sdk=sdk)


def validate_reconstruction_source(archive,snapshot,scope):
    if scope['name'] not in ('reconstruction21','nmi22','h_extrema22','h_extrema23','current236'):return
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
    if scope['name'] not in ('max_tree19', 'matrix20', 'reconstruction21', 'nmi22','h_extrema22','h_extrema23','current236'):
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


def validate_h_extrema_source(archive,snapshot,scope):
    """Closed source-only H lineage; no algorithm or candidate imports."""
    if scope['name'] not in ('h_extrema22','h_extrema23','current236'):return
    legacy=scope['name']=='h_extrema22'
    runtime=H_EXTREMA_LEGACY_RUNTIME_SOURCE_PINS if legacy else H_EXTREMA_RUNTIME_SOURCE_PINS
    frozen=H_EXTREMA_LEGACY_FROZEN_RAW_PINS if legacy else H_EXTREMA_FROZEN_RAW_PINS
    expected=dict(H_EXTREMA_RUNTIME_SOURCE_PINS=runtime,H_EXTREMA_FROZEN_RAW_PINS=frozen,
                  H_EXTREMA_ORACLE_PATH=H_EXTREMA_ORACLE_PATH,H_EXTREMA_ORACLE_SHA256=H_EXTREMA_ORACLE_SHA256)
    if not legacy:expected['H_EXTREMA_HISTORICAL_RAW_PINS']=H_EXTREMA_HISTORICAL_RAW_PINS
    declared={}
    for node in ast.parse(archive.read('tools/public_qualification.py')).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in expected:
                    require(target.id not in declared,'duplicate H source declaration')
                    declared[target.id]=ast.literal_eval(node.value)
    require(declared==expected,'H source pin declarations absent or changed')
    for name,want in {**runtime,**frozen,H_EXTREMA_ORACLE_PATH:H_EXTREMA_ORACLE_SHA256}.items():
        require(archive.namelist().count(name)==1,'H source absent or duplicate: '+name)
        raw=archive.read(name);canonical=raw.replace(b'\r\n',b'\n')
        checked=canonical if name in runtime else raw
        require(hashlib.sha256(checked).hexdigest()==want and snapshot['files_sha256'].get(name)==hashlib.sha256(canonical).hexdigest(),
                'H source bytes differ: '+name)
    if not legacy:
        name='tests/h_extrema_source_provenance.py';raw=archive.read(name)
        namespace={'__name__':'_h_stager_closed_metadata','__file__':name}
        exec(compile(raw,name,'exec'),namespace)  # Exact fixed stdlib reader authenticated above.
        record=json.loads(archive.read('coverage/evidence/h-extrema-reference-assertion-migration.json'))
        current={name:archive.read(name) for name in H_EXTREMA_HISTORICAL_RAW_PINS}
        require(namespace['verify_record'](record,current,raw),'H closed metadata inverse failed')
        require({name:entry['original_raw_sha256'] for name,entry in record['files'].items()}==H_EXTREMA_HISTORICAL_RAW_PINS,
                'H historical source declaration differs')
        name='tests/h_extrema_public_authentication.py'
        adapter={'__name__':'_h_stager_cross_family_metadata','__file__':name}
        exec(compile(archive.read(name),name,'exec'),adapter)
        historical=adapter['validate_reconstruction'](archive.read)
        old=json.loads(archive.read('coverage/contracts/h-extrema-plan.json'))
        floating=json.loads(archive.read('coverage/contracts/h-extrema-floating-plan.json'))
        for name,expected in {**old['frozen_files'],**floating['preserved_files'],**floating['reference_files']}.items():
            if hashlib.sha256(archive.read(name)).hexdigest()==expected:continue
            if name in H_EXTREMA_HISTORICAL_RAW_PINS:
                require(expected==H_EXTREMA_HISTORICAL_RAW_PINS[name],'H historical pin mismatch')
            else:
                require(name==adapter['PATH'] and hashlib.sha256(historical).hexdigest()==expected,
                        'H cross-family reference mismatch: '+name)


def validate_nmi_source(archive, snapshot, scope):
    """Authenticate the frozen NMI imports without executing scientific code."""
    if scope['name'] not in ('nmi22','h_extrema23','current236'):
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
    scope = qualification_scope(source, profile)
    require(len(product['members']) == scope['members'], 'complete product member census differs')
    require(len([n for n in product['members'] if n.startswith('skimagexx/_core.')
                 and n.endswith(('.so', '.pyd'))]) == 1, 'sole production core differs')
    if scope['name'] == 'current236':
        cores = {n for n in product['members'] if n.startswith('skimagexx/_core.') and n.endswith(('.so', '.pyd'))}
        require(set(product['members']) == set(scope['install_members']) | cores, 'current staged install closure differs')
    sdk = {n: h for n, h in product['members'].items()
           if n.startswith(('include/skimagexx/', 'lib/cmake/skimagexx/', 'share/licenses/scikit-imagexx/'))
           or n in ('lib/skimagexx.lib', 'lib/libskimagexx.a')}
    if scope['name'] == 'current236':
        require(set(sdk) == set(scope['SDK_members']), 'current staged SDK file closure differs')
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
        scope = qualification_scope_text(text.decode('utf-8'), read_bytes=archive.read, names=archive.namelist(), profile=profile)
        validate_max_tree_source(archive, q['source'], scope)
        validate_reconstruction_source(archive,q['source'],scope)
        validate_nmi_source(archive,q['source'],scope)
        validate_h_extrema_source(archive,q['source'],scope)
    image_count = scope['image_count'] if scope['name'] == 'current236' else {'rank14': 190, 'tv17': 191, 'bregman18': 192, 'max_tree19': 198, 'matrix20': 199,'reconstruction21':200,'nmi22':201,'h_extrema22':202,'h_extrema23':203}[scope['name']]
    if scope['name'] == 'current236':
        require(expected['functions'] == scope['functions'] and expected['metadata_count'] == scope['metadata_count'],
                'current exact registry contract differs')
    require(expected['image_count'] == image_count and expected['metadata_count'] == 3
            and expected['registered_count'] == len(set(expected['functions'])) == len(expected['functions']) == image_count+3,
            f'genuine{image_count}+3 registry qualification required')
    require(('restoration.denoise_tv_chambolle' in expected['functions']) == (scope['name'] != 'rank14'),
            'qualified TV registry/source scope differs')
    require(('restoration.denoise_tv_bregman' in expected['functions']) == (scope['name'] in ('bregman18', 'max_tree19', 'matrix20', 'reconstruction21', 'nmi22','h_extrema22','h_extrema23','current236')),
            'qualified Bregman registry/source scope differs')
    max_tree = tuple('morphology.' + name for name in ('max_tree', 'max_tree_local_maxima',
                     'area_opening', 'area_closing', 'diameter_opening', 'diameter_closing'))
    require(all(expected['functions'].count(name) == 1 for name in max_tree) if scope['name'] in ('max_tree19', 'matrix20', 'reconstruction21', 'nmi22','h_extrema22','h_extrema23','current236')
            else not any(name in expected['functions'] for name in max_tree),
            'qualified max-tree registry/source scope differs')
    require(('morphology.reconstruction' in expected['functions'])==(scope['name'] in ('reconstruction21','nmi22','h_extrema22','h_extrema23','current236')),
            'source reconstruction scope/registry differs')
    require(('transform.matrix_transform' in expected['functions']) == (scope['name'] in ('matrix20','reconstruction21','nmi22','h_extrema22','h_extrema23','current236')),
            'qualified matrix registry/source scope differs')
    require(all(expected['functions'].count(n)==1 for n in H_EXTREMA_PATHS) if scope['name'] in ('h_extrema22','h_extrema23','current236') else not any(n in expected['functions'] for n in H_EXTREMA_PATHS), 'H-extrema registry missing/partial/downgraded')
    require(expected['functions'].count(NMI_PATH) == 1 if scope['name'] in ('nmi22','h_extrema23','current236')
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
