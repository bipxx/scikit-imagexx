"""Private, read-only diagnostic of three observed CI failure groups."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

root = Path.cwd()
data = json.loads((root / 'diagnostic-cases.json').read_bytes())
source = root / 'source'
output = root / 'diagnostic-evidence'
output.mkdir()
assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == data['source']
assert shutil.disk_usage(output).free >= 30 * 1024**3
assert shutil.disk_usage(os.environ['RUNNER_TEMP']).free >= 30 * 1024**3
assert sys.version_info[:3] == (3, 13, 16) and os.confstr('CS_GNU_LIBC_VERSION') == 'glibc 2.43'
assert 'VERSION_ID="26.04"' in Path('/etc/os-release').read_text(encoding='utf-8')
bundle = Path(os.environ['RUNNER_TEMP']) / 'scikit-imagexx-failed-wheel'
archive = bundle / 'wheels'
wheels = sorted(path for path in archive.glob('*.whl') if not path.name.startswith('cmake-'))
assert len(wheels) == 14
native, = [path for path in wheels if path.name.startswith('scikit_imagexx-')]
assert hashlib.sha256(native.read_bytes()).hexdigest() == data['native_wheel_sha256']
hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in wheels}
with tempfile.TemporaryDirectory(dir=os.environ['RUNNER_TEMP'], prefix='installed-ci26-') as temp:
    prefix = Path(temp)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                    '--no-cache-dir', '--target', temp, *map(str, wheels)], check=True)
    for path in archive.glob('*.whl'):
        path.unlink()
    core, = (prefix / 'skimagexx').glob('_core*.so')
    assert hashlib.sha256(core.read_bytes()).hexdigest() == data['core_sha256']
    env = dict(os.environ, PYTHONPATH=temp, PYTHONDONTWRITEBYTECODE='1',
               SKIMAGEXX_RECONSTRUCTION_PRIMARY_SOURCE=str(source / 'tests/data/reconstruction-primary-0.26'))
    env.pop('SKIMAGEXX_DISABLE_SIMD', None)
    env['NPY_DISABLE_CPU_FEATURES'] = ','.join(data['numpy_disabled_features'])
    probe = (
        "import ctypes, json, os, platform, sys, numpy, scipy, skimage, skimagexx, pytest;"
        "assert (numpy.__version__, scipy.__version__, skimage.__version__, pytest.__version__)=="
        "('2.5.4','1.18.1','0.26.0','9.2.0');"
        "assert all(m.__file__.startswith(sys.argv[1]+'/') for m in (numpy, scipy, skimage, skimagexx, pytest));"
        "assert skimagexx._core.__file__==sys.argv[2];"
        "print(json.dumps(dict(python=sys.version, platform=platform.platform(), glibc=os.confstr('CS_GNU_LIBC_VERSION'),"
        "rounding=ctypes.CDLL(None).fegetround(), core=skimagexx._core.__file__, numpy=numpy.__version__,"
        "scipy=scipy.__version__, skimage=skimage.__version__, pytest=pytest.__version__, "
        "cpu_features=numpy._core._multiarray_umath.__cpu_features__),indent=2))"
    )
    with (output / 'runtime.json').open('xb') as out:
        subprocess.run([sys.executable, '-B', '-c', probe, temp, str(core)], cwd=source, env=env, stdout=out, check=True)
    feature_observation = json.loads((output / 'runtime.json').read_bytes())['cpu_features']
    assert feature_observation['AVX2'] and feature_observation['FMA3']
    assert not any(feature_observation[name] for name in data['numpy_disabled_features']), feature_observation
    executions = {}
    for label, nodes in data['groups'].items():
        for mode in (('normal', 'disabled') if label == 'native' else ('normal',)):
            case = f'{label}-{mode}'
            xml = output / f'{case}.xml'
            testenv = dict(env)
            if mode == 'disabled':
                testenv['SKIMAGEXX_DISABLE_SIMD'] = '1'
            argv = [sys.executable, '-B', '-m', 'pytest', '-q', f'--junitxml={xml}', *nodes]
            with (output / f'{case}.log').open('xb') as out:
                run = subprocess.run(argv, cwd=source, env=testenv, stdout=out, stderr=subprocess.STDOUT)
            assert xml.is_file(), (case, run.returncode)
            suite = ET.parse(xml).getroot().find('testsuite').attrib
            assert int(suite['tests']) == len(nodes) and int(suite['skipped']) == 0, (case, suite)
            executions[case] = {'exit': run.returncode, 'argv': argv, 'counts': suite}
            print(case, run.returncode, json.dumps(suite), flush=True)
    receipt = {'run_base': 44207792316, 'source': data['source'], 'installed_wheels': hashes,
               'core_sha256': data['core_sha256'], 'executions': executions,
               'numpy_disabled_features': data['numpy_disabled_features'], 'builds': 0, 'timing_batches': 0, 'transport_wheels_left': len(list(archive.glob('*.whl')))}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
