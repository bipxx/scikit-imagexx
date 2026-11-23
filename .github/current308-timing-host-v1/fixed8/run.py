"""Fixed current308 Windows installed public timing. All science requires root GO."""

import argparse
import base64
import csv
import ctypes
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from importlib import import_module, metadata
import inspect
import json
import os
from pathlib import Path
import runpy
import shutil
import struct
import subprocess
import sys
import warnings


HEAD = 'e44a720dd2c84eedbd7e75641829c6c4f76d9a6c'
TREE = 'c501bff41f16a4eecbde630a7d16bfa1884ceac8'
PARENT = 'fc9f029391f6943695f625fa64aa6673838ba8d9'
SCOPE = 'current308-windows-installed-fixed8-v1'
HERE = Path(__file__).resolve().parent
VERSIONS = {'python': '3.13.16', 'numpy': '2.5.4', 'scipy': '1.18.1', 'scikit-image': '0.26.0'}
LIMITS = ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS',
          'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_THREAD_LIMIT')
FAMILIES = {
    'convex_hull_image': ('morphology', 'convex_hull_image_binary', 'skimage.morphology.convex_hull'),
    'convex_hull_object': ('morphology', 'convex_hull_object_binary', 'skimage.morphology.convex_hull'),
}
STOCK_MODULES = {
    'scikit-image': {
        'skimage', 'skimage.morphology.convex_hull', 'skimage.morphology._convex_hull',
        'skimage.measure._label', 'skimage.measure._pnpoly', 'skimage.measure.pnpoly',
    },
    'scipy': {'scipy', 'scipy.ndimage._ni_label', 'scipy.spatial._qhull'},
    'numpy': {'numpy', 'numpy._core._multiarray_umath'},
}



def digest(path):
    with Path(path).open('rb') as stream:
        from hashlib import file_digest
        return file_digest(stream, 'sha256').hexdigest()


def pinned(info):
    path = Path(info['path'])
    assert path.is_absolute() and digest(path) == info['sha256'], path
    return path.resolve()


def loaded_python_dll():
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.GetModuleHandleW.argtypes = (ctypes.c_wchar_p,)
    api.GetModuleHandleW.restype = ctypes.c_void_p
    api.GetModuleFileNameW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint)
    handle = api.GetModuleHandleW('python313.dll')
    assert handle
    target = ctypes.create_unicode_buffer(32768)
    assert 0 < api.GetModuleFileNameW(handle, target, len(target)) < len(target)
    return Path(target.value).resolve(strict=True)


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, row):
    with path.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(row, stream, indent=2, allow_nan=False)
        stream.write('\n')


def append(path, row):
    with path.open('a', encoding='utf8', newline='\n') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()


def git(command, *args):
    return subprocess.check_output([*command, *args])


def source_entry(config, plan):
    source = Path(config['source_root']).resolve(strict=True)
    git_exe = pinned(config['git'])
    assert git_exe.name.lower() == 'git.exe'
    command = [str(git_exe), '--no-optional-locks', '-C', str(source)]
    assert git(command, 'rev-parse', '--show-toplevel').decode().strip().replace('\\', '/').lower() == source.as_posix().lower()
    assert git(command, 'rev-parse', 'HEAD', 'HEAD^{tree}', 'HEAD^').decode().splitlines() == [HEAD, TREE, PARENT]
    assert not git(command, 'status', '--porcelain=v1', '--untracked-files=all')
    # The plan pins raw Git blobs. Normalize only the materialized comparison.
    for name, pin in plan['sources'].items():
        raw = git(command, 'show', 'HEAD:' + name)
        assert sha256(raw).hexdigest() == pin, name
        checkout = (source / name).read_bytes()
        assert checkout.replace(b'\r\n', b'\n') == raw.replace(b'\r\n', b'\n'), name
    return source


def wheel_members(info, *, full=False):
    site = Path(info['site']).resolve(strict=True)
    record = pinned(info['record'])
    assert record.parent.is_relative_to(site) and record.name == 'RECORD'
    assert record.parent.parent == site and record.parent.name.endswith('.dist-info')
    with record.open(encoding='utf8', newline='') as stream:
        records = list(csv.reader(stream))
    index = {name: (value, size) for name, value, size in records}
    assert len(index) == len(records)
    distribution = metadata.PathDistribution(record.parent)
    assert distribution.metadata['Name'].lower().replace('_', '-') == info['distribution']
    assert distribution.version == info['version']
    def check(name, *, allow_record=False):
        path = (site / name).resolve(strict=True)
        assert path.is_relative_to(site) and not Path(name).is_absolute() and '..' not in Path(name).parts
        value, size = index[name]
        if allow_record and path == record:
            assert not value and not size
            return path
        encoded = base64.urlsafe_b64encode(bytes.fromhex(digest(path))).decode().rstrip('=')
        assert value == 'sha256=' + encoded and size and path.stat().st_size == int(size), name
        return path
    if full:
        assert info['wheel_tag'] == 'cp313-cp313-win_amd64'
        wheel = check(record.parent.name + '/WHEEL')
        assert 'Tag: ' + info['wheel_tag'] in wheel.read_text(encoding='utf8').splitlines()
        assert len(info['wheel_members']) == 286 and len(set(info['wheel_members'])) == 286
        package_code = {path.relative_to(site).as_posix() for path in (site / 'skimagexx').rglob('*')
                        if path.is_file() and path.suffix.lower() in {'.py', '.pyd', '.dll'}}
        assert package_code <= set(info['wheel_members'])
        assert record.relative_to(site).as_posix() in info['wheel_members']
        for name in info['wheel_members']:
            check(name, allow_record=True)
    return site, check


def windows_pe(path, init):
    raw = path.read_bytes()
    offset = struct.unpack_from('<I', raw, 60)[0]
    assert raw[:2] == b'MZ' and raw[offset:offset+6] == b'PE\0\0\x64\x86'
    assert init.encode() + b'\0' in raw


def affinity():
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.GetCurrentProcess.restype = ctypes.c_void_p
    api.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    api.GetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
    process, system = ctypes.c_size_t(), ctypes.c_size_t()
    handle = api.GetCurrentProcess()
    return api, handle, process, system


def cpu0(pin=False):
    api, handle, process, system = affinity()
    if pin:
        assert api.SetProcessAffinityMask(handle, 1), ctypes.get_last_error()
    assert api.GetProcessAffinityMask(handle, ctypes.byref(process), ctypes.byref(system)), ctypes.get_last_error()
    assert process.value == 1 and system.value & 1


def authenticate(binding, plan):
    assert os.name == 'nt' and sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
    assert not sys.flags.optimize and sys.maxsize == 2**63-1
    assert all(os.environ.get(key) == '1' for key in LIMITS)
    assert not any(key.startswith('SKIMAGEXX_') or key in {
        'PYTHONPATH', 'PYTHONHOME', 'NPY_DISABLE_CPU_FEATURES', 'OPENBLAS_CORETYPE', 'MKL_DEBUG_CPU_TYPE',
        'LD_PRELOAD', 'GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE'} for key in os.environ)
    assert Path(sys.executable).samefile(pinned(binding['python']))
    assert sys.version.split()[0] == VERSIONS['python'] and sys.implementation.cache_tag == 'cpython-313'
    assert pinned(binding['python_dll']) == loaded_python_dll()
    source = source_entry(binding, plan)
    for name in ('producer', 'focused_public', 'relocated_sdk'):
        # Root binds independently admitted terminal receipts for this product.
        pinned(binding[name])
    product = binding['candidate']
    assert binding['source'] == HEAD and binding['tree'] == TREE and binding['scope'] == SCOPE
    assert product['distribution'] == 'scikit-imagexx' and product['version'] == '0.1.0'
    assert len(bytes.fromhex(product['wheel_sha256'])) == 32
    site, check = wheel_members(product, full=True)
    assert list((site/'skimagexx').glob('_core*.pyd')) == [
        site/'skimagexx/_core.cp313-win_amd64.pyd']
    core = pinned(product['core'])
    assert core == check('skimagexx/_core.cp313-win_amd64.pyd')
    windows_pe(core, 'PyInit__core')
    for family, (namespace, _, _) in FAMILIES.items():
        for member in (f'skimagexx/_{family}.py', f'skimagexx/{namespace}.py'):
            assert check(member).read_bytes().replace(b'\r\n', b'\n') == (
                source/'python'/member).read_bytes().replace(b'\r\n', b'\n')
    assert set(binding['stock']) == set(STOCK_MODULES)
    stock_sites = set()
    stock_paths = {}
    stock_blas = set()
    for name, modules in STOCK_MODULES.items():
        stock = binding['stock'][name]
        assert stock['distribution'] == name and stock['version'] == VERSIONS[name]
        assert stock['input_wheel_sha256'] == plan['stock_wheels'][name]
        stock_site, stock_check = wheel_members(stock)
        stock_sites.add(stock_site)
        if name in ('numpy', 'scipy'):
            blas = stock['openblas_member']
            assert blas.startswith(name + '.libs/libscipy_openblas') and blas.endswith('.dll')
            stock_blas.add(stock_check(blas))
        assert set(stock['modules']) == modules
        for module in modules:
            member = stock['modules'][module]
            path = stock_paths[module] = stock_check(member)
            stem = Path(member).name.split('.', 1)[0]
            assert stem in (module.rsplit('.', 1)[-1], '__init__'), module
            if path.suffix == '.pyd':
                windows_pe(path, 'PyInit_' + stem)
    assert len(stock_sites) == 1
    stock_site, = stock_sites
    assert site != stock_site and not (stock_site/'skimagexx').exists()
    pool_paths = {pinned(info) for info in binding['loaded_pools']}
    assert len(stock_blas) == 2 and stock_blas < pool_paths
    return source, site, stock_site, stock_paths, core, pool_paths


def array_snapshot(array, np):
    assert type(array) is np.ndarray
    return dict(type='numpy.ndarray', dtype=array.dtype.str, metadata=array.dtype.metadata,
                shape=list(array.shape), strides=list(array.strides),
                owns=array.flags.owndata, c=array.flags.c_contiguous, f=array.flags.f_contiguous,
                aligned=array.flags.aligned, writeable=array.flags.writeable,
                base_type=type(array.base).__name__, sha256=sha256(array.tobytes()).hexdigest())


def benchmark_cases(source, np, family, candidate, stock, entry, snapshots):
    adapter = runpy.run_path(str(source/'benchmarks'/f'{family}.py'))
    pairs = adapter['public_call_pairs']
    if family == 'convex_hull_image':
        cases = pairs(stock, candidate, entry)
    else:
        cases = pairs(stock, candidate)
    module = import_module('skimage.morphology.convex_hull')
    qhull = import_module('scipy.spatial._qhull').ConvexHull
    assert module.ConvexHull is qhull
    for name, reference, native in cases:
        expected = snapshots[name]
        image, = reference.args
        assert native.args[0] is image
        assert array_snapshot(image, np) == expected['input']
        assert reference.keywords == native.keywords == expected['kwargs']
        guard = input_guard(reference.args, reference.keywords, np)
        qhull_calls = 0

        def record_qhull(*args, **kwargs):
            nonlocal qhull_calls
            qhull_calls += 1
            return qhull(*args, **kwargs)

        def stock_witness():
            result = reference()
            assert qhull_calls == expected['stock_qhull_calls'] > 0
            return result

        module.ConvexHull = record_qhull
        try:
            old, new = witness(stock_witness, native, stock, entry)
        finally:
            module.ConvexHull = qhull
        assert qhull_calls == expected['stock_qhull_calls']
        assert array_snapshot(old, np) == array_snapshot(new, np) == expected['output']
        assert not np.shares_memory(old, image) and not np.shares_memory(new, image)
        guard()
        yield name, reference, native, guard


def input_guard(args, kwargs, np):
    def state():
        def one(a):
            if type(a) is np.ndarray:
                return (a.shape, a.strides, a.dtype.str, a.dtype.metadata, a.flags.num,
                        type(a.base), sha256(a.tobytes()).hexdigest())
            return repr(a)
        return tuple(map(one, args)), {key: one(value) for key, value in kwargs.items()}
    before = state()
    def verify():
        assert state() == before, 'public input changed'
    return verify


def witness(reference, native, stock, entry):
    original = inspect.unwrap(stock).__code__
    calls = []
    def observe(frame, event, value):
        assert event != 'call' or frame.f_code is not original, 'candidate entered stock public code'
        if event == 'c_return' and value is entry:
            calls.append(value)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        old = reference()
        prior = sys.getprofile()
        try:
            sys.setprofile(observe)
            new = native()
        finally:
            sys.setprofile(prior)
    assert calls == [entry]
    return old, new


def execute(binding, plan, paths, output, deadline):
    source, candidate_site, stock_site, stock_paths, expected_core, pool_paths = paths
    sys.path[:0] = [str(candidate_site), str(stock_site)]
    import numpy as np
    import scipy
    import skimage
    import skimagexx as xx
    from skimagexx import _compat, _core
    assert (_core.__file__ and Path(_core.__file__).resolve() == expected_core
            and Path(xx.__file__).resolve() == candidate_site/'skimagexx/__init__.py')
    assert (np.__version__, scipy.__version__, skimage.__version__) == tuple(VERSIONS[key] for key in ('numpy', 'scipy', 'scikit-image'))
    pool_file = pinned(binding['threadpoolctl'])
    assert pool_file.parent == stock_site
    import threadpoolctl
    assert Path(threadpoolctl.__file__).resolve() == pool_file
    for name, expected in stock_paths.items():
        assert Path(import_module(name).__file__).resolve() == expected, name
    xx.set_fallback_enabled(False)
    assert not _compat._fallback_enabled and xx.get_num_threads() == 1
    measure = runpy.run_path(str(source/'tools/montage_sustained.py'))['measure']
    builtin_entries = []
    cases = []
    def readback():
        cpu0()
        assert xx.get_num_threads() == 1 and not _compat._fallback_enabled
        pools = threadpoolctl.threadpool_info()
        assert pools and all(p['num_threads'] == 1 for p in pools)
        assert {Path(p['filepath']).resolve() for p in pools} == pool_paths
        assert sum(p['internal_api'] == 'openblas' for p in pools) >= 2
        assert any(p['internal_api'] == 'openmp' for p in pools)
        for name, expected in stock_paths.items():
            assert Path(import_module(name).__file__).resolve() == expected, name
        return pools

    correct = output/'correctness.jsonl'
    results = output/'results.jsonl'
    batches = output/'raw-batches.jsonl'
    with threadpoolctl.threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error')
        for family, (namespace, symbol, original_module) in FAMILIES.items():
            candidate = getattr(getattr(xx, namespace), family)
            original = getattr(getattr(skimage, namespace), family)
            owned = import_module(f'skimagexx._{family}')
            assert Path(owned.__file__).resolve() == candidate_site/f'skimagexx/_{family}.py'
            assert candidate is getattr(owned, family)
            assert original is getattr(import_module(original_module), family)
            assert inspect.signature(candidate) == inspect.signature(original)
            entry = getattr(_core, symbol)
            assert inspect.isbuiltin(entry) and entry.__module__ == 'skimagexx._core'
            builtin_entries.append(symbol)
            found = []
            append(correct, dict(family=family, phase='checking'))
            verified_cases = iter(benchmark_cases(source, np, family, candidate, original, entry, plan['snapshots'][family]))
            for expected_name in plan['cases'][family]:
                append(correct, dict(family=family, case=expected_name, phase='checking'))
                name, reference, native, guard = next(verified_cases)
                assert name == expected_name
                append(correct, dict(family=family, case=name, phase='passed', native=symbol))
                found.append(name)
                cases.append((family, name, reference, native, guard))
            assert next(verified_cases, None) is None
            assert found == plan['cases'][family], (family, found)
        assert len(cases) == 8 and len(set(builtin_entries)) == 2
        pools = readback()
        write(output/'providers.json', dict(pools=pools, stock={n:str(p) for n,p in stock_paths.items()},
                                           builtin_entries=builtin_entries))
        ratios = {}
        for index, (family, name, reference, native, guard) in enumerate(cases):
            case = f'{family}/{name}'
            append(results, dict(case=case, phase='sampling'))
            guard()
            timing = measure(reference, native, 7, 100, journal=batches, case_index=index,
                             case_id=case, pass_index=1, deadline=deadline)
            append(results, dict(case=case, phase='measured', speedup=timing['speedup'],
                                 samples_ms=timing['samples_ms'], warmup_elapsed_ns=timing['warmup_elapsed_ns']))
            guard()
            assert all(value >= 250_000_000 for value in timing['warmup_elapsed_ns'])
            for side in (0, 1):
                valid = [row for row in timing['batch_records'] if row['function'] == side and row['accepted']]
                assert len(valid) == 7 and all(row['complete'] and row['elapsed_ns'] >= 100_000_000 for row in valid)
            assert readback() == pools
            ratios[case] = timing['speedup']
            append(results, dict(case=case, phase='verified', parity=ratios[case] >= 1.0))
    assert Path(_core.__file__).resolve() == expected_core
    return ratios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--go', type=Path)
    mode.add_argument('--check-binding', type=Path)
    parser.add_argument('--go-sha256')
    parser.add_argument('--binding-sha256')
    parser.add_argument('--plan-sha256')
    args = parser.parse_args()
    if args.check_binding:
        if not args.binding_sha256 or not args.plan_sha256:
            parser.error('binding and plan digests are mandatory for entry check')
        plan_path = pinned(dict(path=str(HERE/'plan-v1.json'), sha256=args.plan_sha256))
        plan = read(plan_path)
        assert (plan['source'],plan['tree'],plan['scope'],plan['versions']) == (HEAD,TREE,SCOPE,VERSIONS)
        binding = read(pinned(dict(path=str(args.check_binding), sha256=args.binding_sha256)))
        paths = authenticate(binding, plan)
        print(json.dumps(dict(source=HEAD, tree=TREE, runtime=sys.version.split()[0],
                              core=str(paths[4]), core_sha256=digest(paths[4]), scientific_imports=0)))
        return 0
    if not args.go or not args.go_sha256:
        print('REFUSED_NO_EXACT_ROOT_TIMING_AUTHORITY', file=sys.stderr)
        return 2
    go = read(pinned(dict(path=str(args.go), sha256=args.go_sha256)))
    assert (go['authorized_by'], go['scope'], go['source'], go['tree'], go['exclusive'], go['cpu'], go['workers']) == (
        '/root', SCOPE, HEAD, TREE, True, 0, 1)
    assert pinned(go['launcher']) == Path(__file__).resolve()
    plan_path = pinned(go['plan'])
    assert plan_path == HERE/'plan-v1.json'
    plan = read(plan_path)
    assert (plan['source'],plan['tree'],plan['scope'],plan['versions']) == (HEAD,TREE,SCOPE,VERSIONS)
    assert list(plan['cases']) == list(FAMILIES) and all(len(value) == 4 for value in plan['cases'].values())
    binding = read(pinned(go['binding']))
    quiet = read(pinned(go['quiet']))
    assert quiet['windows_competitors'] == quiet['wsl_competitors'] == 0 and quiet['host'] == os.environ['COMPUTERNAME']
    now = datetime.now(timezone.utc)
    deadline = datetime.fromisoformat(go['deadline_utc'])
    assert deadline.utcoffset().total_seconds() == 0 and now < deadline
    assert 0 <= (now - datetime.fromisoformat(quiet['utc'])).total_seconds() <= 120
    output = Path(go['output'])
    assert output.is_absolute() and not output.exists() and output.parent.is_dir()
    assert shutil.disk_usage(output.parent).free >= 150 * 2**20
    paths = authenticate(binding, plan)
    assert datetime.now(timezone.utc) < deadline
    cpu0(pin=True)
    output.mkdir()
    try:
        ratios = execute(binding, plan, paths, output, go['deadline_utc'])
        assert authenticate(binding, plan) == paths and datetime.now(timezone.utc) < deadline
        assert sum(p.stat().st_size for p in output.iterdir() if p.is_file()) < 8 * 2**20
    except Exception as error:
        write(output/'terminal.json', dict(scope=SCOPE, source=HEAD, status='failed',
                                           error_type=type(error).__name__, error=str(error)))
        raise
    passed = all(value >= 1.0 for value in ratios.values())
    assert len(ratios) == 8
    write(output/'terminal.json', dict(scope=SCOPE, source=HEAD, status='passed' if passed else 'below_parity',
               authority_sha256=args.go_sha256, product_sha256=binding['candidate']['core']['sha256'],
               per_case=ratios, accepted_batches=112,
               files={p.name:digest(p) for p in output.iterdir() if p.is_file()}))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
