"""Prepare one source-bound weighted-hue experiment; never execute its body."""
import argparse
import ast
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import struct
import subprocess
import sys


AUTHORITY = 'ROOT_AUTHORIZED_WEIGHTED_CONTRACTION_PREPARATION_V1'
EPOCH = 'a5f3aa9748b77540690189067fb1178385bb7500'
DRIVER_SHA = '9ec911d0c04e88c9178e02a3d0b7fb4fa0fb9438777f86bf78c9e92fe6df8e94'
CPP_SHA = 'dd8dd59f31f0d9a51bcf6f95f735a4f3965010a0e6486320c9c5989781720fef'
ENTRY_SHA = '83f31987b65ee379b55e05bb4b9347d9f089f32019daa074bd6f5a8b67f0867f'
GO_SHA = 'd1096586c2c091e416ea7142229e65296bc05ccede893193808f69d7705d4a56'
ORIGINAL_SHA = 'da85c0746c1f49ccb60dbfd76556b1cc79d855026382cf1e2e0488b7f362b7b6'
CAPS = dict(original=1, baseline=1, strict=1, backend=1, fegetenv=10,
            restore=3, public_color=0, timing=0)
ABI = [32, 0, 4, 8, 12, 20, 28, 8]
PINS = {
    'src/distance_metrics.hpp': '63ae09db1a66f8493440cb2e11d6d6078462c000673b99ec3e6afa5f6a199293',
    'tests/test_weighted_distance.py': '81433215024975b06e4dd2253bf18039d2ceb70a444ff70ce3d45b43a3eca118',
    'CMakeLists.txt': '06ead19d2848e3d83fd3041a03cdb7ea6600abaf82faa8112497487ec3535fcc',
    'tests/util_views_fenv.py': '275cfab25633b20489d1f173ac8fba16563ae8ffeed2e7f649d6e48338cef894',
    'tests/util_views_fenv_restore.c': '97ff5740e8eb30c75b6e38729a6ada89ec47f8ac20fd0ceb172e06aeca9f2ae6',
    'tests/util_views_fenv_build.py': '2876f223a689bf020b6869d6d6a752536685a80140ae695ea130721d52510b2d',
    'tests/util_views_fenv_probe.py': '98421b0d0229fb2e52e38e4a7a233e6405c7810d63758af18037601b0dc1ee8b',
    'coverage/contracts/util-views-fenv-v1.json': '7b8e83d46f981be96090dacadf9a068739197748073ce083b0653673ad08dace',
}
INJECTIONS = (
    'CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS', 'CMAKE_ARGS',
    'CMAKE_TOOLCHAIN_FILE', 'LD_PRELOAD', 'LD_AUDIT', 'LD_HWCAP_MASK',
    'GLIBC_TUNABLES', 'NPY_DISABLE_CPU_FEATURES', 'NPY_ENABLE_CPU_FEATURES',
    'SKIMAGEXX_DISABLE_SIMD', 'OPENBLAS_CORETYPE', 'MKL_DEBUG_CPU_TYPE',
    'PYTHONPATH', 'PYTHONHOME',
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_record(path):
    return dict(path=str(path), bytes=Path(path).stat().st_size, sha256=sha(path))


def dump(path, record):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')


def check_environment(environ, base_prefix):
    require(not any(environ.get(key) for key in INJECTIONS), 'CPU/flag/environment injection')
    require(environ.get('PYTHONOPTIMIZE', '0') == '0', 'Python assertions must remain enabled')
    loader = environ.get('LD_LIBRARY_PATH', '')
    allowed = str(Path(base_prefix).resolve() / 'lib')
    require(not loader or (loader == allowed and Path(allowed).is_dir()),
            'Only the exact interpreter base_prefix/lib loader path is allowed')
    require(all(environ.get(key, '1') == '1' for key in
                ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS')),
            'One-thread environment required')
    return dict(LD_LIBRARY_PATH=loader, allowed_loader_path=allowed,
                absent_injections=list(INJECTIONS))


def check_host(host):
    require(host['system'] == 'Linux' and host['machine'] == 'x86_64'
            and host['os'].get('ID') == 'ubuntu' and host['os'].get('VERSION_ID') == '26.04'
            and host['glibc'] == 'glibc 2.43' and host['python'] == '3.11.17'
            and host['byteorder'] == 'little', 'Required Ubuntu26/glibc2.43/Python3.11.17 host differs')


def check_capacity(free):
    peak, reserve, floor = 4 * 1024**3, 20 * 1024**3, 30 * 1024**3
    required = max(peak + reserve, floor)
    require(free and min(free.values()) >= required, 'Insufficient scratch capacity before allocation')
    return dict(estimated_peak_bytes=peak, concurrent_bytes=0, reserve_bytes=reserve,
                minimum_free_bytes=floor, required_free_bytes=required, observed_free_bytes=free)


def literal_assignment(source, name):
    matches = [node.value for node in ast.parse(source).body
               if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                                      for t in node.targets)]
    require(len(matches) == 1, 'Expected one literal assignment: ' + name)
    return ast.literal_eval(matches[0])


def runtime_lock(source):
    wheels = literal_assignment(source, 'RUNTIME_WHEELS')['minimum']
    expected = {'imageio', 'iniconfig', 'lazy-loader', 'networkx', 'numpy', 'packaging',
                'pillow', 'pluggy', 'pygments', 'pytest', 'scikit-image', 'scipy', 'tifffile'}
    require(set(wheels) == expected, 'Minimum runtime wheel tuple differs')
    versions, lines = {}, []
    for name, row in sorted(wheels.items()):
        fields = row['filename'].split('-')
        require(fields[0].replace('_', '-').lower() == name and row['filename'].endswith('.whl')
                and re.fullmatch('[0-9a-f]{64}', row['sha256']), 'Malformed pinned runtime wheel')
        versions[name] = fields[1]
        lines.append(name + '==' + fields[1] + ' --hash=sha256:' + row['sha256'])
    require({key: versions[key] for key in ('numpy', 'scipy', 'scikit-image')} ==
            {'numpy': '1.24.4', 'scipy': '1.15.3', 'scikit-image': '0.26.0'}, 'Runtime version scope differs')
    return wheels, versions, '\n'.join(lines) + '\n'


def variant_commands(compdb, root, cpp, output):
    matches = [row for row in compdb if
               (Path(row['directory']) / row['file']).resolve() == (root / 'src/distance.cpp').resolve()]
    require(len(matches) == 1, 'Expected exactly one native distance.cpp command')
    row = matches[0]
    original = row.get('arguments') or shlex.split(row['command'])
    require(original and all(isinstance(x, str) and not x.startswith('@') for x in original),
            'Unexpanded compiler response command')
    require('-ffp-contract=off' not in original, 'Ordinary baseline already disables contraction; zero body')
    require(original.count('-c') == original.count('-o') == 1
            and original.index('-o') + 1 < len(original), 'Unexpected compile command topology')
    source_index = original.index('-c') + 1
    require(source_index < len(original) and
            (Path(row['directory']) / original[source_index]).resolve() == (root / 'src/distance.cpp').resolve(),
            'Compile command source differs')
    commands = {}
    for role in ('baseline', 'strict'):
        command = list(original)
        command[source_index] = str(cpp)
        command[command.index('-o') + 1] = str(output / (role + '.so'))
        for flag in ('-MF', '-MT', '-MQ'):
            for index in [i for i, token in enumerate(command) if token == flag]:
                require(index + 1 < len(command), 'Truncated dependency output')
                command[index + 1] = str(output / (role + ('.d' if flag == '-MF' else '.so')))
        command.remove('-c')
        # The original TU finds this header through its implicit source directory.
        command += ['-iquote', str(root / 'src'), '-shared']
        if role == 'strict':
            command.append('-ffp-contract=off')
        commands[role] = command
    normalized = [x.replace(str(output / 'strict'), str(output / 'baseline'))
                  for x in commands['strict'][:-1]]
    require(normalized == commands['baseline'], 'Compiler variants have more than the sole contraction delta')
    return row, original, commands


def contraction(codegen):
    require('<weighted_hue_graph_v1>:' in codegen, 'Exported primitive disassembly absent')
    return bool(re.search(r'\bv?f(?:n?madd|n?msub)[0-9a-z]*\b', codegen, re.I))


def header_command(command):
    result = []
    index = 0
    while index < len(command):
        token = command[index]
        if token in ('-o', '-MF', '-MT', '-MQ'):
            require(index + 1 < len(command), 'Truncated dependency command')
            index += 2
        elif token in ('-shared', '-MD', '-MMD', '-MP'):
            index += 1
        else:
            result.append(token)
            index += 1
    return result + ['-M', '-MT', 'weighted_hue_graph_v1']


def source_pins(root):
    for name, digest in PINS.items():
        require(hashlib.sha256((root / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == digest,
                'Source epoch differs: ' + name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='.')
    parser.add_argument('--output', default='build/weighted-contraction-v1')
    parser.add_argument('--expected-revision')
    parser.add_argument('--authority', default='UNBOUND')
    args = parser.parse_args(argv)
    require(__debug__ and args.authority == AUTHORITY, 'UNBOUND preparation')
    require(args.expected_revision and re.fullmatch('[0-9a-f]{40}', args.expected_revision), 'Exact revision required')
    root, out = Path(args.source).resolve(), Path(args.output).resolve()
    require(not out.exists() and out != root / 'build' and out.is_relative_to(root / 'build'),
            'Fresh dedicated ignored output required')
    environment = check_environment(os.environ, sys.base_prefix)
    host = dict(system=platform.system(), machine=platform.machine(), os=platform.freedesktop_os_release(),
                glibc=os.confstr('CS_GNU_LIBC_VERSION'), python=platform.python_version(), byteorder=sys.byteorder,
                executable=sys.executable, base_prefix=sys.base_prefix, kernel=platform.release(),
                ImageOS=os.environ.get('ImageOS'), ImageVersion=os.environ.get('ImageVersion'))
    check_host(host)
    capacity_paths = [root, Path(os.environ['RUNNER_TEMP'])]
    capacity = check_capacity({str(path): shutil.disk_usage(path).free for path in capacity_paths})
    source_pins(root)
    family = root / 'tools/diagnostics/weighted_contraction'
    cpp, driver = family / 'retained_hue_graph_v2.cpp', family / 'run_retained_hue_witness_v1.py'
    require(sha(cpp) == CPP_SHA and sha(driver) == DRIVER_SHA, 'Frozen witness source differs')
    require(sha(family / 'entry-UNBOUND-v2.json') == ENTRY_SHA
            and sha(family / 'input-UNBOUND-v2.json') == GO_SHA, 'Frozen binding template differs')
    entry_template = json.loads((family / 'entry-UNBOUND-v2.json').read_text(encoding='utf-8'))
    go_template = json.loads((family / 'input-UNBOUND-v2.json').read_text(encoding='utf-8'))
    require(go_template['canonical_source_pins'] == PINS and go_template['caps'] == CAPS,
            'Frozen binding source/counter scope differs')
    wheels, versions, lock = runtime_lock((root / 'tools/public_qualification.py').read_text(encoding='utf-8'))
    out.mkdir(parents=True)
    journal = []

    def run(label, command, cwd=root):
        command = [str(x) for x in command]
        record = dict(argv=command, cwd=str(cwd), started_utc=datetime.now(timezone.utc).isoformat())
        dump(out / (label + '-command.json'), record)
        result = subprocess.run(command, cwd=cwd, capture_output=True)
        (out / (label + '.stdout')).write_bytes(result.stdout)
        (out / (label + '.stderr')).write_bytes(result.stderr)
        record.update(exit_code=result.returncode, ended_utc=datetime.now(timezone.utc).isoformat())
        dump(out / (label + '-terminal.json'), record)
        journal.append(record)
        require(result.returncode == 0, label + ' failed; no dependent action')
        return result.stdout.decode('utf-8', errors='strict')

    try:
        dump(out / 'host-capacity-environment.json', dict(host=host, capacity=capacity, environment=environment))
        revision = run('source-HEAD', ['git', '--no-optional-locks', 'rev-parse', 'HEAD']).strip()
        require(revision == args.expected_revision, 'Checkout revision differs')
        require(not run('source-clean', ['git', '--no-optional-locks', 'status', '--porcelain', '--untracked-files=no']).strip(),
                'Tracked source is dirty')
        run('CPU', ['lscpu'])
        (out / 'runtime-lock.txt').write_text(lock, encoding='utf-8', newline='\n')
        dump(out / 'runtime-wheel-inputs.json', wheels)
        run('runtime-install', [sys.executable, '-m', 'pip', 'install', '--require-hashes',
            '--only-binary=:all:', '--no-deps', '--no-cache-dir', '--force-reinstall', '-r', out / 'runtime-lock.txt'])
        observed = {name: importlib.metadata.version(name) for name in versions}
        require(observed == versions, 'Installed runtime tuple differs')
        original_dist = importlib.metadata.distribution('scikit-image')
        original_files = [original_dist.locate_file(p).resolve() for p in original_dist.files
                          if p.as_posix() == 'skimage/color/delta_e.py']
        require(len(original_files) == 1 and sha(original_files[0]) == ORIGINAL_SHA, 'Pristine original source differs')
        dump(out / 'runtime-metadata.json', dict(versions=observed, python=host['python'],
             original_source=dict(path=str(original_files[0]), sha256=ORIGINAL_SHA), scientific_imports=0))
        numpy_dist = importlib.metadata.distribution('numpy')
        numpy_binaries = [numpy_dist.locate_file(p).resolve() for p in numpy_dist.files
                          if p.name.startswith('_multiarray_umath.') and p.suffix == '.so']
        require(len(numpy_binaries) == 1, 'Unique installed NumPy core binary required')
        dump(out / 'numpy-binary.json', dict(path=str(numpy_binaries[0]), sha256=sha(numpy_binaries[0])))
        run('numpy-ELF', ['readelf', '-h', '-Ws', '-V', numpy_binaries[0]])
        tools = {}
        for tool in ('cmake', 'ninja'):
            path = shutil.which(tool)
            require(path, 'Required ordinary build tool absent: ' + tool)
            tools[tool] = file_record(Path(path).resolve())
            tools[tool]['version'] = run(tool + '-version', [path, '--version'])
        dump(out / 'ordinary-build-tools.json', tools)
        native = out / 'native-plan'
        run('configure', ['cmake', '-S', root, '-B', native, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
            '-DSKIMAGEXX_BUILD_PYTHON=OFF', '-DBUILD_TESTING=OFF', '-DSKIMAGEXX_AVX2=ON',
            '-DSKIMAGEXX_OPENMP=ON', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'])
        compdb = json.loads((native / 'compile_commands.json').read_text(encoding='utf-8'))
        row, original, commands = variant_commands(compdb, root, cpp, out)
        compiler = shutil.which(original[0])
        require(compiler, 'Ordinary compiler is absent')
        compiler = Path(compiler).absolute()
        compiler_record = dict(invoked=str(compiler), realpath=str(compiler.resolve()), sha256=sha(compiler))
        compiler_record['version'] = run('compiler-version', [compiler, '--version'])
        compiler_record['target'] = run('compiler-target', [compiler, '-dumpmachine']).strip()
        backend = Path(run('compiler-backend', [compiler, '-print-prog-name=cc1plus']).strip()).resolve()
        require(backend.is_file() and compiler_record['target'].startswith('x86_64'), 'GNU C++ backend/target differs')
        compiler_record.update(backend=str(backend), backend_sha256=sha(backend))
        run('compiler-default-target', [compiler, '-Q', '--help=target'])
        run('compiler-default-optimizers', [compiler, '-Q', '--help=optimizers'])
        dump(out / 'compiler-and-command-adaptation.json', dict(compiler=compiler_record,
             ordinary_row=row, ordinary_argv=original, variants=commands,
             adaptation='Replace input/output/dependency topology; remove -c; common -iquote original-src and -shared; strict only appends -ffp-contract=off'))
        variants, fused, dso_hashes = {}, {}, {}
        for role in ('baseline', 'strict'):
            dependencies = run(role + '-headers', header_command(commands[role]), cwd=Path(row['directory']))
            headers = shlex.split(dependencies.split(':', 1)[1].replace('\\\n', ' '))
            dump(out / (role + '-headers.json'), [dict(path=str((Path(row['directory']) / p).resolve()),
                 sha256=sha(Path(row['directory']) / p)) for p in headers])
            run(role + '-compile', commands[role], cwd=Path(row['directory']))
            library = out / (role + '.so')
            raw = library.read_bytes()
            require(raw[:6] == b'\x7fELF\x02\x01' and struct.unpack_from('<H', raw, 18)[0] == 62,
                    'Primitive is not little-endian ELF64 x86_64')
            run(role + '-ELF', ['readelf', '-h', '-Ws', '-V', library])
            exports = run(role + '-exports', ['nm', '-D', '--defined-only', library])
            require(re.search(r'\bT weighted_hue_graph_v1$', exports, re.M), 'Primitive dynamic export absent')
            codegen = run(role + '-codegen', ['objdump', '-d', '--disassemble=weighted_hue_graph_v1', library])
            fused[role] = contraction(codegen)
            dependencies = run(role + '-DSOs', ['ldd', library])
            libraries = sorted(set(re.findall(r'(?:=>\s*)?(/[^\s()]+)', dependencies)))
            for path in libraries:
                if path not in dso_hashes:
                    dso_hashes[path] = sha(path)
            dump(out / (role + '-DSOs.json'), [dict(path=p, sha256=dso_hashes[p]) for p in libraries])
            variants[role] = dict(path=str(library), sha256=sha(library))
        fenv_directory = out / 'F1'
        run('F1-build-controls', [sys.executable, root / 'tests/util_views_fenv_build.py',
            '--output', fenv_directory, '--control-python', sys.executable])
        receipt = fenv_directory / 'receipt.json'
        spec = importlib.util.spec_from_file_location('preparation_fenv_validation', root / 'tests/util_views_fenv.py')
        fenv = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fenv)
        helper = fenv.validate_receipt(receipt, sha(receipt))
        require(helper['status'] == 'PASS_SOURCE_BUILT_FENV32' and helper['ABI'] == ABI
                and helper['glibc'] == '2.43', 'Fresh F1 source/control/glibc receipt differs')
        source_pins(root)
        require(sha(cpp) == CPP_SHA and sha(driver) == DRIVER_SHA, 'Witness source changed during preparation')
        require(run('source-postHEAD', ['git', '--no-optional-locks', 'rev-parse', 'HEAD']).strip() == revision
                and not run('source-postclean', ['git', '--no-optional-locks', 'status', '--porcelain', '--untracked-files=no']).strip(),
                'Source changed during preparation')
        entry = dict(entry_template)
        entry.update(status='PASS_WEIGHTED_HUE_ENTRY_AND_CODEGEN_V1', source_revision=revision,
            production_epoch=EPOCH, runtime_versions=dict(python=host['python'], numpy=observed['numpy'],
                scipy=observed['scipy'], skimage=observed['scikit-image']), os_version='26.04', glibc='2.43',
            cpp_sha256=CPP_SHA, sole_change_is_contract_off=True, baseline_already_contract_off=False,
            baseline_has_contraction=fused['baseline'], strict_has_contraction=fused['strict'], variants=variants)
        entry['evidence'] = {name: file_record(out / name) for name in
            ('host-capacity-environment.json', 'runtime-wheel-inputs.json', 'runtime-metadata.json',
             'numpy-binary.json', 'ordinary-build-tools.json', 'compiler-and-command-adaptation.json',
             'baseline-codegen.stdout', 'strict-codegen.stdout', 'baseline-exports.stdout', 'strict-exports.stdout')}
        entry['native_metadata'] = {name: file_record(native / name)
                                    for name in ('CMakeCache.txt', 'compile_commands.json', 'build.ninja')}
        entry['fenv_receipt'] = file_record(receipt)
        dump(out / 'entry.json', entry)
        issued = datetime.now(timezone.utc)
        go = dict(go_template)
        go.update(status='ROOT_GO_WEIGHTED_HUE_CONTRACTION_V1', production_epoch=EPOCH,
            source_revision=revision, driver_sha256=DRIVER_SHA, cpp_source=dict(path=str(cpp), sha256=CPP_SHA),
            maximum_invocations=1, retries=0, issued_utc=issued.isoformat(),
            deadline_utc=(issued + timedelta(seconds=1800)).isoformat(), caps=CAPS,
            entry=dict(path=str(out / 'entry.json'), sha256=sha(out / 'entry.json')), source_root=str(root),
            canonical_source_pins=PINS, variants=variants,
            original_source=dict(path=str(original_files[0]), sha256=ORIGINAL_SHA),
            fenv_receipt=dict(path=str(receipt), sha256=sha(receipt)), output=str(out / 'body'),
            authority=dict(build=0, helper=1, scientific_import_or_setup=1, original_get_dH2=1,
                           native_primitives=2, public_color=0, dispatch=0, timing=0),
            note='Conditional root-authorized fixed three-observation body only; preparation did not execute it.')
        dump(out / 'ROOT_GO.json', go)
        digest = sha(out / 'ROOT_GO.json')
        (out / 'ROOT_GO.sha256').write_text(digest + '\n', encoding='utf-8', newline='\n')
        body_env = dict(UTIL_VIEWS_FENV_MODE='source-built-v1', UTIL_VIEWS_FENV_RECEIPT=str(receipt),
            UTIL_VIEWS_FENV_RECEIPT_SHA256=sha(receipt), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        dump(out / 'body-command.json', dict(argv=[sys.executable, '-B', '-X', 'utf8', '-I', str(driver),
             '--binding', str(out / 'ROOT_GO.json'), '--binding-sha256', digest], env=body_env))
        dump(out / 'preparation-terminal.json', dict(status='PREPARED_BODY_NOT_EXECUTED',
             source_revision=revision, commands=len(journal), original_calls=0, primitive_calls=0,
             helper_controls='32positive/6negative actual fresh F1', entry_sha256=sha(out / 'entry.json'),
             body_GO_sha256=digest, cleanup='No archives retained; pip uses transient uncached transport. No cleanup attempted.'))
        print(json.dumps(dict(status='PREPARED_BODY_NOT_EXECUTED', binding=str(out / 'ROOT_GO.json'),
                              binding_sha256=digest, command=str(out / 'body-command.json'))))
    except BaseException as error:
        dump(out / 'preparation-failure.json', dict(status='FAILED_PREPARATION_ZERO_BODY',
             type=type(error).__name__, message=str(error), completed_commands=len(journal),
             original_calls=0, primitive_calls=0))
        raise


if __name__ == '__main__':
    main()
