"""Validate the scoped compiler correction on the existing Ubuntu CI runtime."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile

root = Path.cwd()
source = root / 'source'
output = root / 'proof'
output.mkdir()
temp_root = Path(os.environ['RUNNER_TEMP'])
assert all(shutil.disk_usage(p).free >= 30 * 1024**3 for p in (root, temp_root))
assert sys.version_info[:3] == (3, 13, 16)
assert os.confstr('CS_GNU_LIBC_VERSION') == 'glibc 2.43'
assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == os.environ['PRODUCT_SOURCE']
base = json.loads((root / 'diagnostic-cases.json').read_bytes())
prior = json.loads((root / 'diagnostic-stock-wheels.json').read_bytes())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
prior_nodes = [node for node in base['groups']['native']
               if node.split('::', 1)[0] == 'tests/test_perceptual.py']
assert len(prior_nodes) == 6

archive = temp_root / 'scikit-imagexx-failed-wheel/wheels'
all_wheels = sorted(archive.glob('*.whl'))
deps = [path for path in all_wheels if not path.name.startswith(('scikit_imagexx-', 'cmake-'))]
assert len(deps) == 13
assert all(sha(path) == prior[path.name] for path in deps)
modules = ['tests/test_perceptual.py', 'tests/test_color_transforms.py', 'tests/test_color_distance.py']
units = {'simd_color_avx2.cpp'}

with tempfile.TemporaryDirectory(dir=temp_root, prefix='strict-ubuntu-wheel-') as scratch:
    scratch = Path(scratch)
    wheels = scratch / 'wheels'
    build = scratch / 'build'
    package = scratch / 'site'
    env = dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL='3')
    with (output / 'build.log').open('xb') as log:
        subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps',
                        '--wheel-dir', str(wheels), '--config-settings=build-dir=' + str(build),
                        '--config-settings=cmake.define.CMAKE_EXPORT_COMPILE_COMMANDS=ON', '.'],
                       cwd=source, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    wheel, = wheels.glob('scikit_imagexx-*.whl')
    wheel_sha = sha(wheel)
    proof_package = output / 'package'
    with zipfile.ZipFile(wheel) as wheel_archive:
        for entry in wheel_archive.infolist():
            if entry.is_dir() or not entry.filename.startswith('skimagexx/'):
                continue
            target = proof_package / entry.filename
            assert target.resolve().is_relative_to(proof_package.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(wheel_archive.read(entry))
    proof_core, = (proof_package / 'skimagexx').glob('_core*.so')
    (output / 'wheel-evidence.json').write_text(json.dumps(
        {'source': os.environ['PRODUCT_SOURCE'], 'wheel_sha256': wheel_sha, 'core_sha256': sha(proof_core)},
        indent=2
    ) + '\n', encoding='utf-8')
    compile_db, = build.glob('compile_commands.json')
    command_rows = json.loads(compile_db.read_bytes())
    checked = {}
    for row in command_rows:
        name = Path(row['file']).name
        if name not in units:
            continue
        assert name not in checked
        args = shlex.split(row['command'])
        assert args.count('-ffp-contract=off') == 1
        obj = Path(row['directory']) / args[args.index('-o') + 1]
        asm = subprocess.check_output(['objdump', '-d', str(obj)], text=True)
        fused = re.findall(r'\bv(?:fmadd|fnmadd|fmsub|fnmsub)\w*', asm)
        # The same SIMD object also contains explicitly fused SVML distance code.
        checked[name] = {'argv': args, 'object_sha256': sha(obj), 'fma_in_object': len(fused)}
    assert checked.keys() == units
    (output / 'compile_commands.json').write_bytes(compile_db.read_bytes())
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                    '--no-cache-dir', '--no-compile', '--target', str(package), str(wheel), *map(str, deps)], check=True)
    for path in all_wheels + [wheel]:
        path.unlink()
    core, = (package / 'skimagexx').glob('_core*.so')
    assert sha(core) == sha(proof_core)
    verification = (
        "import ctypes, json, os, sys, numpy, scipy, skimage, skimagexx, pytest;"
        "assert (numpy.__version__, scipy.__version__, skimage.__version__, pytest.__version__)=="
        "('2.5.4','1.18.1','0.26.0','9.2.0');"
        "assert all(m.__file__.startswith(sys.argv[1]+'/') for m in (numpy, scipy, skimage, skimagexx, pytest));"
        "assert skimagexx._core.__file__==sys.argv[2];"
        "assert ctypes.CDLL(None).fegetround()==0;"
        "print(json.dumps(dict(python=sys.version, glibc=os.confstr('CS_GNU_LIBC_VERSION'), "
        "core=skimagexx._core.__file__, fenv_rounding=0, "
        "cpu_features=numpy._core._multiarray_umath.__cpu_features__)))"
    )
    env = dict(os.environ, PYTHONPATH=str(package), PYTHONDONTWRITEBYTECODE='1')
    assert not any(name in env for name in ('NPY_DISABLE_CPU_FEATURES', 'NPY_ENABLE_CPU_FEATURES', 'SKIMAGEXX_DISABLE_SIMD'))
    with (output / 'runtime.json').open('xb') as log:
        subprocess.run([sys.executable, '-B', '-c', verification, str(package), str(core)], cwd=source, env=env, stdout=log, check=True)
    outcomes = {}
    for mode in ('normal', 'disabled'):
        xml = output / f'{mode}.xml'
        testenv = dict(env)
        if mode == 'disabled':
            testenv['SKIMAGEXX_DISABLE_SIMD'] = '1'
        command = [sys.executable, '-B', '-m', 'pytest', '-q', f'--junitxml={xml}', *modules]
        with (output / f'{mode}.log').open('xb') as log:
            run = subprocess.run(command, cwd=source, env=testenv, stdout=log, stderr=subprocess.STDOUT)
        suite = ET.parse(xml).getroot().find('testsuite')
        nodes = {
            'tests/' + row.get('classname').split('.')[-1] + '.py::' + row.get('name'): row
            for row in suite.findall('testcase')
        }
        wanted = [nodes[node] for node in prior_nodes]
        checked_live = sum(all(row.find(key) is None for key in ('failure', 'error', 'skipped')) for row in wanted)
        outcomes[mode] = {'actual_exit': run.returncode, 'counts': suite.attrib,
                          'formerly_failed_passed': checked_live, 'command': command}
        print(mode, json.dumps(outcomes[mode]), flush=True)
    record = {'product_source': os.environ['PRODUCT_SOURCE'], 'wheel_sha256': wheel_sha,
              'core_sha256': sha(core), 'changed_units': checked, 'comparison': outcomes,
              'stock_dependencies_sha256': {path.name: prior[path.name] for path in deps},
              'timing_batches': 0, 'transport_wheels_left': len(list(archive.glob('*.whl')))}
    (output / 'receipt.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    assert all(row['actual_exit'] == 0 and row['formerly_failed_passed'] == len(prior_nodes) for row in outcomes.values())
