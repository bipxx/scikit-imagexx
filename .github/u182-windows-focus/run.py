"""Optional current306 hosted Windows preparation and focused admission."""

import argparse
from collections import Counter
from hashlib import sha256
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

SOURCE = '5b94b20d30e572164a6473c8bb029e18d4a7e706'
TREE = '2ab7c8bd13965df3d48e565b1c6b779739d0974c'
MANIFEST_SHA256 = '59102d76caa69c2931fd539b755837af940b88b72c14b9977fa0ca5cbf649abe'
COMPLETE_SOURCE_RECEIPT_SHA256 = 'dddc2f67d7a758f86af6131aa73f95b37da4a6c3bb59f0f608a02c6fb1bd714e'  # Root binds the actual immutable passing receipt.
WORKFLOW_SHA256 = '772089faf0c8abf79c4ed7d2c2542598e20b3dc3271898bb31d0b68ba6ea779b'
SOURCE_RUN = 45088211262
SOURCE_RAW_PATHS = 508
CAPACITY = 44 * 2**30
FAMILIES = {
    'match_descriptors': ('feature', 'match_descriptors_bool'),
    'thin': ('morphology', 'thin_binary'),
    'skeletonize': ('morphology', 'skeletonize_zhang'),
    'hough_circle': ('transform', 'hough_circle_accumulate'),
    'hough_line_peaks': ('transform', 'hough_line_peak_indices'),
}
APIS = {namespace + '.' + name for name, (namespace, _) in FAMILIES.items()}
MODULES = tuple('tests/test_' + name + '_public.py' for name in FAMILIES)
GENERIC = {
    'tests/test_api.py::test_registry_matches_per_function_tests': 'registry',
    'tests/test_api.py::test_every_native_function': 'public',
    'tests/test_dtype_aliases.py::test_every_native_default_accepts_equivalent_descriptor': 'alias',
    'tests/test_dtype_metadata.py::test_every_native_entry_dtype_metadata': 'metadata',
}


def require(condition, why):
    if not condition:
        raise SystemExit('REFUSED: ' + why)


def digest(path):
    with path.open('rb') as stream:
        checksum = sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            checksum.update(chunk)
        return checksum.hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def gate(receipt, manifest):
    require(isinstance(COMPLETE_SOURCE_RECEIPT_SHA256, str), 'root complete-source receipt is unbound')
    require(receipt.is_file() and digest(receipt) == COMPLETE_SOURCE_RECEIPT_SHA256,
            'immutable root complete-source receipt differs')
    require(manifest.is_file() and digest(manifest) == MANIFEST_SHA256, 'source manifest differs')
    proof, binding = read(receipt), read(manifest)
    require(proof['status'] == 'passed_complete_exact_source_both'
            and (proof['source'], proof['tree'], proof['workflow_sha256']) == (SOURCE, TREE, WORKFLOW_SHA256)
            and proof['source_manifest_sha256'] == MANIFEST_SHA256 and proof['source_raw_paths'] == SOURCE_RAW_PATHS
            and proof['selector_count'] == 186 and set(proof['profiles']) == {'windows311', 'windows313'},
            'root complete-source receipt does not attest this scope')
    for profile, row in proof['profiles'].items():
        xml = row['xml']
        require(row['run'] == SOURCE_RUN and row['exit'] == 0 and row['job'] > 0
                and xml['tests'] > 0 and xml['failures'] == xml['errors'] == xml['skipped'] == 0
                and any(name.endswith('/command.json') for name in row['records_sha256'])
                and any(name.endswith('/results.xml') for name in row['records_sha256']),
                'incomplete or failed source receipt: ' + profile)
    require((binding['source'], binding['tree'], binding['workflow_sha256']) == (SOURCE, TREE, WORKFLOW_SHA256)
            and len(binding['current_scope']) == SOURCE_RAW_PATHS and len(set(binding['selectors'])) == len(binding['selectors']) == 186,
            'wrong source manifest provenance')
    return binding


def clear_git_environment(environment):
    # Match the current qualifier's Git environment policy for parent and children.
    for key in tuple(environment):
        if key.upper().startswith('GIT_'):
            del environment[key]


def source_entry(root, manifest):
    require(os.name == 'nt' and sys.version_info[:3] == (3, 13, 16) and struct.calcsize('P') == 8
            and sys.implementation.name == 'cpython' and sys.dont_write_bytecode and not sys.flags.optimize,
            'expected Windows x64 CPython 3.13.16, assertions on and -B')
    clear_git_environment(os.environ)
    git = shutil.which('git.exe')
    require(git is not None, 'native Windows Git missing')

    def read_git(*args):
        return subprocess.check_output([git, '-C', str(root), *args])

    require(read_git('rev-parse', 'HEAD', 'HEAD^{tree}').decode().splitlines() == [SOURCE, TREE]
            and not read_git('status', '--porcelain=v1', '--untracked-files=all'), 'candidate source is not pristine')
    for name, pin in manifest['current_scope'].items():
        path = root / name
        require(path.is_file() and path.resolve().is_relative_to(root) and digest(path) == pin, 'source bytes: ' + name)
    return dict(git=git, git_version=read_git('--version').decode().strip(), python=sys.version,
                executable=str(Path(sys.executable).resolve()), executable_sha256=digest(Path(sys.executable)))


def qualifier(root):
    path = root / 'tools/public_qualification.py'
    spec = importlib.util.spec_from_file_location('_current306_qualifier', path)
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
    return q


def product_identity(q, out, state):
    package = out / 'package'
    record = read(out / 'package.json')
    members = record['members_sha256']
    sdk = record['SDK_sha256']
    core = record['sole_core']
    record_names = [n for n in members if n.endswith('.dist-info/RECORD')]
    require(state['profile'] == 'windows' and state['source']['revision'] == SOURCE
            and state['profile_config']['config'] == 'Release'
            and record['source_revision'] == SOURCE and record['product_scope'] == 'current306'
            and record['record_verified'] is True and len(members) == 282 and len(sdk) == 125
            and sdk == {n: v for n, v in members.items() if n.startswith(('include/', 'lib/', 'share/'))}
            and len(record_names) == 1 and core.endswith('.cp313-win_amd64.pyd')
            and digest(out / 'product-wheels' / state['product_wheel']) == record['wheel_sha256'],
            'unexpected producer/product layout or wheel identity')
    q.verify_tree(package, members)
    return record, dict(wheel=record['wheel_sha256'], core=members[core], record=members[record_names[0]])


def stock_readback(out, stock_entries, q):
    wheels = q.RUNTIME_WHEELS['windows']
    required = {'numpy', 'scipy', 'scikit-image', 'pytest'}
    require({p.name for p in (out / 'runtime-wheels').glob('*.whl')} == {v['filename'] for v in wheels.values()},
            'exact runtime wheel names differ')
    packages = {}
    for name, prefix in (('numpy', 'numpy'), ('scipy', 'scipy'), ('scikit-image', 'skimage'), ('pytest', 'pytest')):
        wanted = wheels[name]
        wheel = out / 'runtime-wheels' / wanted['filename']
        metadata = q.wheel_metadata(wheel)
        module = sys.modules[prefix]
        site = Path(module.__file__).resolve().parent.parent
        require(site.is_relative_to(out / 'runtime-env') and metadata['sha256'] == wanted['sha256']
                and module.__version__ == metadata['version'], 'wrong stock version/wheel/site: ' + name)
        packages[name] = {'version': metadata['version'], 'wheel_sha256': wanted['sha256'], 'members': {}}
        with zipfile.ZipFile(wheel) as archive:
            selected = {Path(module.__file__).resolve()}
            if name == 'scikit-image':
                selected.update(Path(inspect.getsourcefile(inspect.unwrap(f))).resolve() for f in stock_entries.values())
            bundled = site / (prefix + '.libs')
            binary_members = {n for n in archive.namelist()
                              if n.startswith(prefix + '.libs/') and n.lower().endswith(('.pyd', '.dll'))}
            installed_binaries = {p.relative_to(site).as_posix() for p in bundled.rglob('*')
                                  if p.is_file() and p.suffix.lower() in ('.pyd', '.dll')}
            require(installed_binaries == binary_members, 'original provider library census: ' + name)
            selected.update(site / n for n in installed_binaries)
            for key, module in tuple(sys.modules.items()):
                filename = getattr(module, '__file__', None)
                if key.startswith(prefix + '.') and filename and filename.lower().endswith(('.pyd', '.dll')):
                    selected.add(Path(filename).resolve())
            for path in sorted(selected):
                member = path.relative_to(site).as_posix()
                raw = digest(path)
                require(raw == sha256(archive.read(member)).hexdigest(), 'loaded original/provider member: ' + member)
                packages[name]['members'][member] = raw
    require(set(packages) == required, 'incomplete runtime proof')
    return packages


def check_suite(xml, expected):
    root = ET.parse(xml).getroot()
    suites = (root,) if root.tag == 'testsuite' else root.findall('testsuite')
    cases = root.findall('.//testcase')
    require(bool(suites) and len(cases) == expected
            and sum(int(s.attrib['tests']) for s in suites) == expected
            and not any(int(s.get(k, '0')) for s in suites for k in ('failures', 'errors', 'skipped'))
            and not any(case.findall(k) for case in cases for k in ('failure', 'error', 'skipped')),
            'incomplete or failed JUnit: ' + xml.name)


def public_child(root, out, proof, mode, q):
    state = read(out / 'state.json')
    record, pins = product_identity(q, out, state)
    require((os.environ.get('SKIMAGEXX_DISABLE_SIMD') == '1') == (mode == 'scalar')
            and (mode == 'scalar' or 'SKIMAGEXX_DISABLE_SIMD' not in os.environ)
            and all(os.environ.get(key) == '1' for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')),
            'runtime pools or dispatch mode differ')
    require(Path(sys.executable).resolve() == q.python_in(out / 'runtime-env').resolve(), 'wrong child interpreter')
    package = out / 'package'
    sys.path.insert(0, str(package))
    os.chdir(root)
    sys.path.append(str(root))
    import numpy
    import pytest
    import scipy
    import skimage
    import skimagexx as xx
    from skimagexx.coverage import native_dtypes

    require(Path(xx.__file__).resolve() == package / 'skimagexx/__init__.py'
            and Path(xx._core.__file__).resolve() == package / record['sole_core']
            and xx.get_num_threads() == xx._core.get_num_threads() == 1
            and xx.has_avx2() == (mode == 'normal'), 'different genuine package/core, native default worker or mode')
    stock = {}
    for name, (namespace, symbol) in FAMILIES.items():
        original = getattr(getattr(skimage, namespace), name)
        public = getattr(getattr(xx, namespace), name)
        owner = sys.modules['skimagexx._' + name]
        entry = getattr(xx._core, symbol)
        require(getattr(owner, name) is public and Path(owner.__file__).resolve() == package / f'skimagexx/_{name}.py'
                and inspect.signature(original) == inspect.signature(public)
                and inspect.isbuiltin(entry) and entry.__module__ == xx._core.__name__, 'incorrect route: ' + name)
        stock[name] = original
    stock_before = stock_readback(out, stock, q)
    terminal_nodes = []
    family_counts = {}
    generic_counts = {}

    class Selection:
        def pytest_collection_modifyitems(self, config, items):
            selected, deselected = [], []
            cover = Counter()
            for item in items:
                key = item.nodeid.split('[', 1)[0]
                module = item.nodeid.split('::', 1)[0]
                if module in MODULES:
                    family_counts[module] = family_counts.get(module, 0) + 1
                    selected.append(item)
                    continue
                group = GENERIC.get(key)
                params = getattr(getattr(item, 'callspec', None), 'params', {})
                api = params.get('api')
                if group == 'public':
                    api = params['native'].__name__.split('.', 1)[1] + '.' + params['name']
                if group == 'registry' or api in APIS:
                    require(group is not None, 'unexpected generic case')
                    dtype = params.get('dtype')
                    if dtype is not None:
                        dtype = numpy.dtype(dtype).name
                    workers = params.get('configuration') if group != 'alias' else params.get('native')
                    cover[(group, api, dtype, workers)] += 1
                    generic_counts[group] = generic_counts.get(group, 0) + 1
                    selected.append(item)
                else:
                    deselected.append(item)
            require(set(family_counts) == set(MODULES) and generic_counts.get('registry') == 2, 'family or registry collection incomplete')
            for api in APIS:
                for group in ('public', 'metadata', 'alias'):
                    for dtype in native_dtypes(api):
                        by_workers = {workers for (kind, name, dt, workers) in cover if (kind, name, dt) == (group, api, dtype)}
                        require(by_workers == {1, 4}, f'generic scope incomplete: {group}/{api}/{dtype}')
            items[:] = selected
            config.hook.pytest_deselected(items=deselected)

        def pytest_collection_finish(self, session):
            terminal_nodes.extend(item.nodeid for item in session.items)
            require(len(terminal_nodes) == len(set(terminal_nodes)) == sum(family_counts.values()) + sum(generic_counts.values())
                    and any(Path(filename).resolve() == root / 'tests/conftest.py'
                            for plugin in session.config.pluginmanager.get_plugins()
                            if isinstance(filename := getattr(plugin, '__file__', None), str)),
                    'duplicate nodes, unexpected selection or missing real conftest')

    xml = proof / f'public-{mode}.xml'
    argv = ['-q', '--capture=sys', '--tb=short', *MODULES, *GENERIC, '--junitxml=' + str(xml),
            '--basetemp=' + str(out / ('pytest-' + mode)), '-o', 'cache_dir=' + str(out / ('cache-' + mode))]
    result = int(pytest.main(argv, plugins=[Selection()]))
    stock_after = stock_readback(out, stock, q)
    require(all(stock_before[name]['members'].items() <= stock_after[name]['members'].items() for name in stock_before),
            'loaded original or dependency readback changed')
    write(proof / f'public-{mode}.json', dict(source=SOURCE, tree=TREE, mode=mode, core=record['sole_core'],
         product=pins, original=stock_after, node_ids=terminal_nodes, family_counts=family_counts,
         generic_counts=generic_counts, argv=argv, pytest_exit=result))
    require(result == 0 and xx.get_num_threads() == xx._core.get_num_threads() == 1, 'focused public execution failed')
    check_suite(xml, len(terminal_nodes))
    require(product_identity(q, out, state)[1] == pins, 'product changed during focus')

def prepare_focused(root, out, proof, entry, q, args):
    require(not out.exists() and not proof.exists() and root not in proof.parents, 'fresh external evidence and output required')
    free = min(shutil.disk_usage(root).free, shutil.disk_usage(proof.parent).free)
    require(free >= CAPACITY, 'fresh producer capacity below 44 GiB')
    proof.mkdir()
    write(proof / 'entry.json', dict(source=SOURCE, tree=TREE, identity=entry,
         source_receipt_sha256=COMPLETE_SOURCE_RECEIPT_SHA256, manifest_sha256=MANIFEST_SHA256,
         available_free_bytes=free, required_free_bytes=CAPACITY))
    journal = q.Journal(proof)
    paths = ('plan.json', 'state.json', 'package.json', 'build-headers.json', 'build-identity.json',
             'interpreter-provenance.json', 'native-build-properties.json', 'sdk.json',
             'windows-sdk-selection.json', 'windows-sdk-toolchain-verification.json')
    try:
        q.must_run(journal, 'prepare', [sys.executable, '-B', root / 'tools/public_qualification.py',
                   'prepare', '--profile', 'windows', '--expected-native-images', '306',
                   '--source-revision', SOURCE, '--output', out])
        state = read(out / 'state.json')
        record, pins = product_identity(q, out, state)
        require(state['source'] == q.source_state(root)
                and state['interpreter_provenance']['base'] == q.base_interpreter(), 'exact producer source or bootstrap differs')
        wheel_toolchain = q.read_windows_toolchain(out / 'native-build', 'Release')
        write(proof / 'producer-toolchain.json', wheel_toolchain)
        runtime = q.python_in(out / 'runtime-env')
        for mode in ('normal', 'scalar'):
            q.must_run(journal, 'public-' + mode, [runtime, '-I', '-B', Path(__file__).resolve(),
                '--source', root, '--manifest', args.manifest, '--complete-source-receipt', args.complete_source_receipt,
                '--evidence', proof, '--public-child', mode], q.environment(scalar=mode == 'scalar'))
        a, b = (read(proof / ('public-' + mode + '.json')) for mode in ('normal', 'scalar'))
        require(a['node_ids'] == b['node_ids'] and a['product'] == b['product'] == pins
                and a['family_counts'] == b['family_counts'] and a['generic_counts'] == b['generic_counts'],
                'unexpected difference between dispatch selectors or product')

        sdk_journal = q.Journal(out)

        class SDKJournal:
            def run(self, label, command, env=None):
                if label in ('sdk-normal', 'sdk-scalar'):
                    command = [*command, '--parallel', '1', '--no-tests=error', '--output-junit', proof / (label + '.xml')]
                return sdk_journal.run(label, command, env)

        # Reuse the exact Windows relocation, source extraction, toolchain
        # verification and listing code in this separate diagnostic module.
        previous = q.CURRENT_FAMILIES
        require(tuple(previous[-5:]) == tuple(FAMILIES), 'consumer order differs')
        try:
            q.CURRENT_FAMILIES = tuple(FAMILIES)
            q.sdk(out, state, SDKJournal())
        finally:
            q.CURRENT_FAMILIES = previous
        sdk = read(out / 'sdk.json')
        require(sdk['families'] == list(FAMILIES) and sdk['checks'] == 10 and sdk['SDK_sha256'] == record['SDK_sha256'],
                'incorrect relocated SDK focus')
        for mode in ('normal', 'scalar'):
            check_suite(proof / ('sdk-' + mode + '.xml'), len(FAMILIES))

        # Only then prepare the bounded offline wheel transport for a later,
        # separately authorized and quiescent timing runner.
        transport = out / 'transport-wheels'
        transport.mkdir()
        for path in sorted((out / 'runtime-wheels').glob('*.whl')):
            os.link(path, transport / path.name)
        original_product = out / 'product-wheels' / state['product_wheel']
        os.link(original_product, transport / original_product.name)
        q.must_run(journal, 'install-candidate-for-offline-export', [runtime, '-m', 'pip', 'install', '--no-index',
            '--no-deps', '--no-compile', transport / original_product.name])
        for stage in ('freeze', 'verify'):
            q.must_run(journal, 'offline-' + stage, [runtime, '-B', root / 'tools/ci_measurement_artifacts.py',
                stage, '--wheel-dir', transport, '--artifact', proof / 'measurement.json'])
        lock = read(proof / 'measurement.json')
        require(lock['source'] == state['source']
                and lock['runtime']['sdk_sha256'] == record['SDK_sha256']
                and {name: item['sha256'] for name, item in lock['wheels'].items()} ==
                    {name: item['sha256'] for name, item in state['runtime_wheels'].items()} |
                    {original_product.name: pins['wheel']},
                'unapproved offline transport closure')
        require(q.source_state(root) == state['source'] and product_identity(q, out, state)[1] == pins,
                'source or product changed during focus')
        write(proof / 'terminal.json', dict(status='passed_focused_windows', source=SOURCE, tree=TREE, profile='windows',
              product=pins, modes=['normal', 'scalar'], public_checks={mode: len(r['node_ids'])
              for mode, r in zip(('normal', 'scalar'), (a, b), strict=True)}, sdk_checks=10,
              complete_source_receipt_sha256=COMPLETE_SOURCE_RECEIPT_SHA256,
              transport_sha256={path.name: digest(path) for path in sorted(transport.glob('*.whl'))},
              timing_authorized=False, full_qualification=False))
    finally:
        # The remote runner has a disposable work area. Only these exact files
        # and logs are exported; the source ZIP and bootstrap archives stay local.
        for name in (*paths, 'pinned-runtime-inputs.txt', 'commands.json', 'initial-public-admission.log',
                     'build-headers.log', 'build-whole-product.log', 'gather-build-identity-facts.log',
                     'toolchain-cmake-capabilities.log', 'toolchain-visual-studio-instances.log',
                     'sdk-configure.log', 'sdk-build.log', 'sdk-list.log', 'sdk-normal.log', 'sdk-scalar.log'):
            path = out / name
            if path.is_file():
                shutil.copyfile(path, proof / ('producer-' + name))
        for path in sorted(out.glob('failure-*.json')):
            shutil.copyfile(path, proof / path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--complete-source-receipt', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--public-child', choices=('normal', 'scalar'))
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.complete_source_receipt = args.complete_source_receipt.resolve()
    manifest = gate(args.complete_source_receipt, args.manifest)
    if args.preflight:
        print('Exact complete-source receipt accepted for optional Windows focus')
        return
    require(args.source is not None and args.evidence is not None, 'exact source and evidence paths required')
    root = args.source.resolve(strict=True)
    proof = args.evidence.resolve()
    entry = source_entry(root, manifest)
    q = qualifier(root)
    out = root / 'build/qp306'
    if args.public_child:
        require(proof.is_dir(), 'existing evidence missing')
        public_child(root, out, proof, args.public_child, q)
    else:
        prepare_focused(root, out, proof, entry, q, args)


if __name__ == '__main__':
    main()
