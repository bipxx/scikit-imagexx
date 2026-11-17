"""Validate the scalar contraction correction on the matching CI runtime."""
import argparse
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

from prepare_weighted_contraction import (
    check_capacity, check_environment, check_host, dump, require, runtime_lock, sha,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'build/weighted-validation'
CMAKE_SHA = 'b90c19854f85101ea47560c5e5f8d3b232dc793adf0f4e3a02153b2c204ae849'
MODULES = ('test_ciede2000_promotion.py', 'test_color_distance.py', 'test_weighted_distance.py',
           'test_distance_math_provider.py', 'test_distance_scalar_promotion.py')
METRICS = {'color.deltaE_cie76': 19, 'color.deltaE_ciede94': 29,
           'color.deltaE_ciede2000': 27, 'color.deltaE_cmc': 25}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def source_checks():
    require(sha(ROOT / 'CMakeLists.txt') == CMAKE_SHA, 'Scalar build source differs')
    requires = 'set_source_files_properties(src/distance.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")'
    cmake = (ROOT / 'CMakeLists.txt').read_text(encoding='utf-8')
    require(cmake.count(requires) == 1 and cmake.index(requires) > cmake.index('target_compile_options(skimagexx PRIVATE /W4 /fp:precise)'),
            'Correction must touch only the non-MSVC scalar TU')
    require(all((ROOT / 'tests' / module).is_file() for module in MODULES), 'Missing affected public selectors')
    wheels, versions, lock = runtime_lock((ROOT / 'tools/public_qualification.py').read_text(encoding='utf-8'))
    recipe = (ROOT / '.github/workflows/distance-overlap-diagnostics.yml').read_text(encoding='utf-8')
    require(recipe.count("python-version: '3.11.17'") == 2 and 'needs: parity' in recipe
            and recipe.count('runs-on: ubuntu-latest') == 2 and 'matrix:' not in recipe,
            'Expected independent parity and timing jobs')
    require('actions/download-artifact@v4' in recipe and recipe.count('validate_weighted_correction.py ') == 2,
            'Expected measured reuse of the sealed installed product')
    return wheels, versions, lock


def entry(phase):
    _, versions, lock = source_checks()
    env = check_environment(os.environ, sys.base_prefix)
    host = dict(system=platform.system(), machine=platform.machine(), os=platform.freedesktop_os_release(),
                glibc=os.confstr('CS_GNU_LIBC_VERSION'), python=platform.python_version(), byteorder=sys.byteorder)
    check_host(host)
    temp = Path(os.environ['RUNNER_TEMP'])
    capacity = check_capacity({str(p): shutil.disk_usage(p).free for p in (ROOT, temp)})
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    require(revision == os.environ['GITHUB_SHA'], 'Checkout revision differs')
    require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT),
            'Tracked source is dirty')
    if phase == 'parity':
        OUT.mkdir(parents=True)
    else:
        require((OUT / 'parity-pass.json').is_file(), 'No admitted producer')
    dump(OUT / (phase + '-entry.json'), dict(host=host, capacity=capacity, environment=env, revision=revision))
    lock_path = OUT / (phase + '-runtime-lock.txt')
    lock_path.write_text(lock, encoding='utf-8', newline='\n')
    run(phase + '-install-runtime', sys.executable, '-m', 'pip', 'install', '--require-hashes',
        '--only-binary=:all:', '--no-deps', '--no-cache-dir', '--force-reinstall', '-r', lock_path)
    require({name: importlib.metadata.version(name) for name in versions} == versions, 'Runtime lock did not resolve')
    return temp, revision


def run(label, *argv, env=None, check=True):
    command = [str(x) for x in argv]
    print(shlex.join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, env=env)
    dump(OUT / (label + '-terminal.json'), dict(argv=command, exit_code=result.returncode))
    require(not check or result.returncode == 0, label + ' failed')
    return result.returncode


def installed_env():
    return dict(os.environ, PYTHONPATH=str(OUT / 'site'), PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')


def observe_fp():
    import ctypes

    import numpy as np
    import skimagexx as xx
    from skimage import color

    spec = importlib.util.spec_from_file_location('actual_fenv', ROOT / 'tests/util_views_fenv.py')
    fenv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fenv)
    libc, helper = fenv.backend()
    require((OUT / 'site').resolve() in Path(xx._core.__file__).resolve().parents and xx.get_num_threads() == 1,
            'Expected the admitted native product at default worker count')
    xx.set_fallback_enabled(False)
    lab = np.frombuffer(bytes.fromhex('0b28efc67f1242c0a5e9cdbb755142c020bfe2a6ed7f4c40'), dtype='<f8')

    def save():
        state = fenv.FEnv()
        require(libc.fegetenv(ctypes.byref(state)) == 0, 'FP snapshot failed')
        return state

    records = []
    for name in ('deltaE_ciede94', 'deltaE_cmc'):
        for role, module in (('original', color), ('installed', xx.color)):
            call = getattr(module, name)
            before = save()
            try:
                output = call(lab, lab)
                after = save()
            finally:
                require(helper.reference_fenv_restore(ctypes.byref(before), 32) == 0, 'Full32 restore failed')
            restored = save()
            require(bytes(restored) == bytes(before) and after.control == before.control
                    and (after.mxcsr & ~63) == (before.mxcsr & ~63), 'Caller FP scope changed')
            word = np.asarray(output).tobytes().hex()
            require(word == '0000000000000000', 'Witnessed scalar public self pair differs')
            records.append(dict(metric=name, role=role, result=word, before=bytes(before).hex(),
                                after=bytes(after).hex(), restored=bytes(restored).hex()))
    mode = 'disabled' if os.environ.get('SKIMAGEXX_DISABLE_SIMD') else 'normal'
    dump(OUT / (mode + '-public-fp.json'), dict(records=records, calls=4, restores=4))


def parity(temp, revision):
    env = installed_env()
    with tempfile.TemporaryDirectory(prefix='weighted-release-', dir=temp) as directory:
        scratch = Path(directory)
        build = scratch / 'native'
        run('wheel', sys.executable, '-m', 'pip', 'wheel', '.', '--verbose', '--no-deps', '--no-cache-dir',
            '--wheel-dir', scratch, '--config-settings=build-dir=' + str(build),
            '--config-settings=cmake.define.CMAKE_EXPORT_COMPILE_COMMANDS=ON')
        rows = [r for r in read_json(build / 'compile_commands.json') if Path(r['file']).name == 'distance.cpp']
        require(len(rows) == 1 and shlex.split(rows[0]['command']).count('-ffp-contract=off') == 1,
                'Emitted scalar compile command misses the approved flag')
        dump(OUT / 'distance-compile.json', rows[0])
        wheels = list(scratch.glob('scikit_imagexx-*.whl'))
        require(len(wheels) == 1, 'Expected one corrected wheel')
        run('install-product', sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
            '--target', OUT / 'site', wheels[0])
        cores = list((OUT / 'site/skimagexx').glob('_core*.so'))
        require(len(cores) == 1, 'Expected the installed native core')
        prefix = scratch / 'sdk'
        run('install-sdk', 'cmake', '--install', build, '--prefix', prefix)
        consumer = scratch / 'consumer'
        run('sdk-configure', 'cmake', '-S', ROOT / 'tests/cmake_consumer/weighted_distance_diagnostic',
            '-B', consumer, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_PREFIX_PATH=' + str(prefix),
            '-DSKIMAGEXX_EXPECT_PREFIX=' + str(prefix), '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON')
        run('sdk-build', 'cmake', '--build', consumer, '--target', 'weighted_distance_consumer', '--parallel', '2', '--verbose')
        shutil.copyfile(consumer / 'weighted-distance.map', OUT / 'weighted-distance.map')
        run('sdk', 'ctest', '--test-dir', consumer, '-R', '^installed_weighted_distance$', '--output-on-failure',
            '--output-junit', OUT / 'sdk.xml')
    runtime_code = "import skimagexx as xx; print(xx._core.__file__); assert xx.get_num_threads()==1 and xx.has_avx2()"
    run('installed-ownership', sys.executable, '-c', runtime_code, env=env)
    helper_dir = OUT / 'fenv'
    run('fenv-controls', sys.executable, ROOT / 'tests/util_views_fenv_build.py',
        '--output', helper_dir, '--control-python', sys.executable)
    receipt = helper_dir / 'receipt.json'
    helper_receipt = read_json(receipt)
    require(helper_receipt['status'] == 'PASS_SOURCE_BUILT_FENV32' and helper_receipt['glibc'] == '2.43',
            'Expected fresh full32 helper on the actual host')
    env = dict(env, UTIL_VIEWS_FENV_MODE='source-built-v1', UTIL_VIEWS_FENV_RECEIPT=str(receipt),
               UTIL_VIEWS_FENV_RECEIPT_SHA256=sha(receipt))
    results = {}
    for mode in ('normal', 'disabled'):
        mode_env = dict(env, SKIMAGEXX_DISABLE_SIMD='1') if mode == 'disabled' else env
        xml = OUT / (mode + '.xml')
        code = run(mode, sys.executable, '-m', 'pytest', '-q', '--maxfail=0',
                   *(ROOT / 'tests' / x for x in MODULES), '--junitxml', xml, env=mode_env, check=False)
        results[mode] = dict(exit_code=code, xml=xml.name)
    require(all(r['exit_code'] == 0 for r in results.values()), 'Installed public parity failed')
    for row in results.values():
        nodes = list(ET.parse(OUT / row['xml']).iter('testcase'))
        require(len(nodes) == 3360 and not any(t.tag in {'failure','error','skipped'} for n in nodes for t in n),
                'Installed suite did not complete all required nodes')
        row.update(cases=len(nodes), xml_sha256=sha(OUT / row['xml']))
    for mode in results:
        mode_env = dict(env, SKIMAGEXX_DISABLE_SIMD='1') if mode == 'disabled' else env
        run(mode + '-public-fp', sys.executable, '-B', __file__, 'fp', env=mode_env)
        results[mode]['fp_sha256'] = sha(OUT / (mode + '-public-fp.json'))
    files = {p.relative_to(OUT).as_posix(): sha(p) for p in sorted((OUT / 'site').rglob('*')) if p.is_file()}
    dump(OUT / 'parity-pass.json', dict(source=revision, cmake_sha256=CMAKE_SHA, files=files,
         core=cores[0].relative_to(OUT).as_posix(), results=results, sdk_xml_sha256=sha(OUT / 'sdk.xml')))


def measure(revision):
    product = read_json(OUT / 'parity-pass.json')
    require(product['source'] == revision and product['cmake_sha256'] == CMAKE_SHA, 'Producer source differs')
    require({p.relative_to(OUT).as_posix(): sha(p) for p in sorted((OUT / 'site').rglob('*')) if p.is_file()}
            == product['files'], 'Installed product payload differs')
    report = OUT / 'distance.json'
    run('benchmark', sys.executable, ROOT / 'benchmarks/distance.py', '--size', '512', '--threads', '1',
        '--repeat', '7', '--output', report, env=installed_env())
    rows = read_json(report)['results']
    require(len(rows) == 123 and {api: sum(row['api'] == api for row in rows) for api in METRICS} == METRICS,
            'Benchmark workload coverage differs')
    affected = [r for r in rows if r['api'] in METRICS]
    for row in affected:
        a, b = row['samples_ms']
        require(row['threads'] == 1 and len(a) == len(b) == 7 and all(math.isfinite(x) and x > 0 for x in a+b)
                and statistics.median(a) / statistics.median(b) == row['speedup'], 'Invalid raw timing row')
    losses = [r for r in affected if r['speedup'] < 1]
    dump(OUT / 'speed-result.json', dict(source=revision, cases=len(affected), slower=len(losses),
         minimum_speedup=min(r['speedup'] for r in affected), losses=losses, report_sha256=sha(report)))
    require(not losses, 'At least one affected public-call case is slower than the original')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('source', 'parity', 'measure', 'fp'))
    phase = parser.parse_args().phase
    if phase == 'source':
        source_checks()
        print(json.dumps(dict(status='PASS_WEIGHTED_VALIDATION_SOURCE', python=platform.python_version(),
                              selectors=MODULES, scientific_calls=0)))
        return
    if phase == 'fp':
        observe_fp()
        return
    temp, revision = entry(phase)
    if phase == 'parity':
        parity(temp, revision)
    else:
        measure(revision)


if __name__ == '__main__':
    main()
