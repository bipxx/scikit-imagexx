"""Measure the admitted Ubuntu wheel unpacked from the strict parity job."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

source = Path.cwd() / 'source'
output = Path.cwd() / 'speed-proof'
output.mkdir()
tmp = Path(os.environ['RUNNER_TEMP'])
assert all(shutil.disk_usage(path).free >= 30 * 1024**3 for path in (output, tmp))
assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == os.environ['PRODUCT_SOURCE']
assert sys.version_info[:3] == (3, 13, 16) and os.confstr('CS_GNU_LIBC_VERSION') == 'glibc 2.43'
strict = tmp / 'scikit-imagexx-strict-proof'
wheels = tmp / 'scikit-imagexx-failed-wheel/wheels'
proof = json.loads((Path.cwd() / 'diagnostic-farid-color-parity-lock.json').read_bytes())
assert int(os.environ['PARITY_RUN']) == proof['run']
raw = (strict / 'wheel-evidence.json').read_bytes()
assert hashlib.sha256(raw).hexdigest() == proof['wheel_evidence_sha256'] == os.environ['EXPECTED_STRICT_RECEIPT_SHA256']
receipt = json.loads(raw)
assert receipt['source'] == proof['source'] == os.environ['PRODUCT_SOURCE']
assert receipt['core_sha256'] == proof['core_sha256']
for mode, expected in proof['tests'].items():
    xml = (strict / f'{mode}.xml').read_bytes()
    assert len(xml) == expected['bytes'] and hashlib.sha256(xml).hexdigest() == expected['sha256']
    suite = ET.fromstring(xml).find('testsuite')
    assert len(suite.findall('testcase')) == int(suite.get('tests')) == expected['count'] == 6966
    assert not any(int(suite.get(key)) for key in ('failures','errors','skipped'))
product = strict / 'package'
core, = (product / 'skimagexx').glob('_core*.so')
assert hashlib.sha256(core.read_bytes()).hexdigest() == receipt['core_sha256']
deps = {name: digest for name, digest in proof['dependency_wheels'].items() if not name.startswith('scikit_imagexx-')}
assert len(deps) == 13
assert all(hashlib.sha256((wheels / name).read_bytes()).hexdigest() == digest for name, digest in deps.items())
patterns = ['filters.farid*', 'color.*luv*', 'color.rgb2hsv', 'color.hsv2rgb',
            'color.xyz2lab', 'color.lab2xyz', 'color.rgb2lab', 'color.lab2rgb', 'color.lab2lch', 'color.lch2lab']
scoped = sorted(os.sched_getaffinity(0))
assert scoped
cpu = scoped[0]
os.sched_setaffinity(0, {cpu})
quiet = {'cpu': cpu, 'load_before': os.getloadavg(),
         'active_processes': subprocess.check_output(['ps', '-eo', 'stat,comm'], text=True)}
with tempfile.TemporaryDirectory(dir=tmp, prefix='runtime-ubuntu-speed-') as target:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                    '--no-cache-dir', '--target', target, *[str(wheels / name) for name in sorted(deps)]], check=True)
    for path in wheels.glob('*.whl'):
        path.unlink()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(product), target)), PYTHONDONTWRITEBYTECODE='1')
    assert not any(name in env for name in ('NPY_DISABLE_CPU_FEATURES', 'NPY_ENABLE_CPU_FEATURES', 'SKIMAGEXX_DISABLE_SIMD'))
    probe = (
        "import ctypes,json,os,platform,sys,numpy,scipy,skimage,skimagexx;"
        "assert (numpy.__version__,scipy.__version__,skimage.__version__)==('2.5.4','1.18.1','0.26.0');"
        "assert skimagexx._core.__file__==sys.argv[1] and ctypes.CDLL(None).fegetround()==0;"
        "assert all(m.__file__.startswith(sys.argv[2]+'/') for m in (numpy,scipy,skimage));"
        "print(json.dumps(dict(python=sys.version,platform=platform.platform(),"
        "cpu_features=numpy._core._multiarray_umath.__cpu_features__,cpu_affinity=sorted(os.sched_getaffinity(0)))))"
    )
    with (output / 'runtime.json').open('xb') as log:
        subprocess.run([sys.executable, '-B', '-c', probe, str(core), target], cwd=source, env=env, stdout=log, check=True)
    reports = {}
    for mode in ('normal', 'disabled'):
        options = [arg for pattern in patterns for arg in ('--api', pattern)]
        result = output / f'{mode}.json'
        cmd = [sys.executable, '-B', 'benchmarks/coverage.py', '--sizes', '64', '--threads', '1',
               '--repeat', '7', *options, '--output', str(result)]
        mode_env = dict(env)
        if mode == 'disabled':
            mode_env['SKIMAGEXX_DISABLE_SIMD'] = '1'
        with (output / f'{mode}.log').open('xb') as log:
            subprocess.run(cmd, cwd=source, env=mode_env, stdout=log, stderr=subprocess.STDOUT, check=True)
        data = json.loads(result.read_bytes())
        assert len(data['results']) == 45 and data['metadata']['fallback_enabled'] is False
        cases = {(row['api'], row['size'], row['dtype']) for row in data['results']}
        assert len(cases) == 45
        for row in data['results']:
            assert row['threads'] == 1 and len(row['samples_ms']) == 2
            original, current = row['samples_ms']
            assert len(original) == len(current) == 7 and min(original + current) > 0
            assert math.isclose(row['speedup'], statistics.median(original) / statistics.median(current))
        losses = [
            {key: row[key] for key in ('api', 'dtype', 'size', 'speedup', 'reference_ms', 'cpp_ms')}
            for row in data['results'] if row['speedup'] < 1
        ]
        reports[mode] = {'argv': cmd, 'status': 'pass' if not losses else 'speed gaps',
                         'count': len(cases), 'losses': sorted(losses, key=lambda x: x['speedup']),
                         'raw_report_sha256': hashlib.sha256(result.read_bytes()).hexdigest()}
        print(mode, 'cases', len(cases), 'slower', len(losses), flush=True)
    (output / 'receipt.json').write_text(json.dumps(dict(
        source=os.environ['PRODUCT_SOURCE'], core_sha256=receipt['core_sha256'], cpu=quiet, reports=reports,
        isolated_job=True, no_builds_or_pytest=True, transport_wheels_left=len(list(wheels.glob('*.whl')))
    ), indent=2)+'\n', encoding='utf-8')

    if any(row['api'] in {'color.xyz2lab', 'color.rgb2lab', 'color.rgb2luv'} and
           row['dtype'] in {'float32', 'float64'} for row in reports['normal']['losses']):
        probe = [sys.executable, '-B', str(Path.cwd() / 'diagnostic-float-cie-occupancy.py'),
                 str(output / 'normal.json'), str(output / 'float-branch-probe.json')]
        with (output / 'float-branch-probe.log').open('xb') as log:
            trial = subprocess.run(probe, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
        (output / 'float-branch-probe-terminal.json').write_text(json.dumps(
            {'argv': probe, 'actual_exit': trial.returncode, 'new_inputs': True}, indent=2
        ) + '\n', encoding='utf-8')
