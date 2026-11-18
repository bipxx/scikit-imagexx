"""Fixed64 fc7d/eb1f installed comparison, with no original algorithm calls."""
import argparse
from contextlib import ExitStack
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import select
import shutil
import statistics
import subprocess
import sys

ARMS = {
    'A': dict(source='fc7d10f17382bc56c8b33c2fe5c418fa58f69921',
              receipt='284a5c5982fe39cab9cefc2a4eb5ef1c1f198c2d832a7202e575ab4924fd5829',
              core='d5df5b90ecb3603cdfce9a721663e23d76e06cf0896f9315ea2386fdf3964250'),
    'B': dict(source='eb1f37e441b7d1262cb43f2205f39d3fec6c598c',
              receipt='7169208dc2dd6c9718b83aa4f5e9e82b95e448f079d67598a38b1a1a93994c2b',
              core='17fc01c7a8d1753feebcf30699c26d85dfb15daf6bfea53d341cc495a785b2e3'),
}
FIXTURE_SHA = '998ebe9172df05ffe0499ac3e4b3c582761244b28195b7f9a77ab7b30b78fa84'
TIMER_SHA = 'c28ce23805ab39ad55d806f42a227b6269b30927b36491f7f9a062e2ed4dd513'
CASES = {'xyz2lab': 128, 'rgb2lab': 64}
ORDERS = ('AB', 'BA') * 4
POOLS = dict(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1',
             BLIS_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE')
DISALLOWED_ENV = ('SKIMAGEXX_DISABLE_SIMD', 'NPY_DISABLE_CPU_FEATURES', 'NPY_ENABLE_CPU_FEATURES')
CALLER_FILES = {
    'color.py': 'b0eeb0cb49f5dccf0ed668a0a6e61ed5501c2a300604ebfacb9d2c605bbc8e2f',
    '_compat.py': 'e3d8893372d6f8fe511d5d3a9890b63bc01ab711ee1fa81bff48946ba847b357',
    '__init__.py': '4987e9d1d2e348a8c1d5ac2bb8479f645da977a4c4fee9739aea2b90c9ed4c27',
    '_whitepoints.py': '13865b87745a2ddb7a030a78b9c5f7650b0b58482f6d683d140ed6ad2b72a7c5',
}


def need(condition, label):
    if not condition:
        raise RuntimeError(label)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def emit(handle, row):
    handle.write(json.dumps(row, allow_nan=False) + '\n')
    handle.flush()


def environment():
    need(sys.platform == 'linux' and platform.machine() == 'x86_64' and sys.byteorder == 'little', 'unsupported OS/ISA')
    need(sys.implementation.name == 'cpython' and sys.version_info[:3] == (3, 13, 16) and
         'GCC 15.2.0' in sys.version, 'wrong CPython build')
    libc = os.confstr('CS_GNU_LIBC_VERSION')
    need(libc == 'glibc 2.43', 'wrong glibc')
    need(not set(DISALLOWED_ENV) & os.environ.keys(), 'unexpected CPU or SIMD override')
    return dict(python=sys.version, executable=str(Path(sys.executable).resolve()), glibc=libc)


def fenv():
    # glibc Linux x86-64 fenv_t has seven 32-bit x87 words and final MXCSR.
    # environment() establishes the pinned ABI before every use of this helper.
    libm = ctypes.CDLL('libm.so.6')
    state = (ctypes.c_uint32 * 8)()
    need(libm.fegetenv(ctypes.byref(state)) == 0 and libm.fegetround() == 0, 'unexpected rounding mode')
    mxcsr = state[7]
    need(mxcsr & 0xffc0 == 0x1f80, 'unexpected SIMD exception masks, rounding or flush modes')
    return f'0x{mxcsr:x}'


def worker(args, arm):
    identity = environment()
    package = (args.baseline_package if arm == 'A' else args.wide_package).resolve(strict=True)
    stock = args.stock.resolve(strict=True)
    need(all(os.environ.get(key) == value for key, value in POOLS.items()), 'pool overrides changed')
    need(os.sched_getaffinity(0) == {args.cpu}, 'worker affinity mismatch')
    need(digest(args.timer) == TIMER_SHA, 'timer bytes changed')
    need(all(digest(package / 'skimagexx' / name) == sha for name, sha in CALLER_FILES.items()),
         'installed color callers differ')
    sys.path[:0] = [str(package), str(stock)]

    # run.py imports original scikit-image and SciPy during fixture setup.
    # Only its existing batch() is called. It is passed owned callables only.
    spec = importlib.util.spec_from_file_location('coverage_fixed_timer', args.timer)
    timer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(timer)
    import numpy as np
    import scipy
    import skimage
    import skimagexx as xx
    from skimagexx import _core, color
    from numpy._core import _multiarray_umath

    for module, version in ((np, '2.5.4'), (scipy, '1.18.1'), (skimage, '0.26.0')):
        need(module.__version__ == version and Path(module.__file__).resolve().is_relative_to(stock),
             f'dependency mismatch: {module.__name__}')
    numpy_binary = Path(_multiarray_umath.__file__).resolve()
    need(numpy_binary.is_relative_to(stock), 'unexpected NumPy binary owner')
    core = Path(_core.__file__).resolve()
    need(core.parent == package / 'skimagexx' and digest(core) == ARMS[arm]['core'], 'wrong core loaded')
    need(Path(color.__file__).resolve().parent == core.parent, 'wrong public color caller loaded')
    need(tuple(core.parent.glob('_core*.so')) == (core,), 'ambiguous installed binary')
    features = dict(_multiarray_umath.__cpu_features__)
    need(features.get('X86_V4') is True and features.get('AVX512_SKX') is True and xx.has_avx2(),
         'no eligible OS-effective x86-64-v4')
    xx.set_fallback_enabled(False)
    xx.set_num_threads(1)
    pkgid = Path(f'/sys/devices/system/cpu/cpu{args.cpu}/topology/physical_package_id')
    sections = [dict(line.split(':', 1) for line in block.splitlines() if ':' in line)
                for block in Path('/proc/cpuinfo').read_text().split('\n\n')]
    fields = [{key.strip(): value.strip() for key, value in section.items()} for section in sections]
    model, = (field['model name'] for field in fields if field.get('processor') == str(args.cpu))
    scope = dict(affinity=sorted(os.sched_getaffinity(0)), physical_package_id=int(pkgid.read_text()),
                 cpu_model=model, numpy_binary=str(numpy_binary), numpy_sha256=digest(numpy_binary), cpu_features=features,
                 pools={key: os.environ[key] for key in POOLS})
    fixture = args.fixture.read_bytes()
    need(len(fixture) == 98304 and hashlib.sha256(fixture).hexdigest() == FIXTURE_SHA, 'wrong fixture bytes')
    image = np.frombuffer(fixture, dtype='<f8').reshape((64, 64, 3)).copy(order='C')
    need(image.flags.owndata and image.flags.c_contiguous and image.strides == (1536, 24, 8), 'not the packed fixed64 input')
    calls = {case: getattr(color, case) for case in CASES}
    need(all(call.__module__ == 'skimagexx.color' for call in calls.values()), 'unexpected public function owner')

    def state():
        need(xx.get_num_threads() == 1 and xx.set_fallback_enabled(False) is False, 'native worker/fallback mismatch')
        need(os.sched_getaffinity(0) == {args.cpu} and len(list(Path('/proc/self/task').iterdir())) == 1,
             'unexpected native process threads or affinity')
        return fenv()

    emit(sys.stdout, dict(event='ready', arm=arm, core=str(core), core_sha256=ARMS[arm]['core'], callers=CALLER_FILES,
                          scope=scope, identity=identity, mxcsr=state()))
    warmed = False
    for line in sys.stdin:
        order = json.loads(line)
        if order == 'warm':
            need(not warmed, 'unexpected repeated warmup')
            for case, call in calls.items():
                output = call(image)
                need(output.shape == image.shape and output.dtype == image.dtype, f'wrong {case} output layout')
                del output
            warmed = True
            emit(sys.stdout, dict(event='warmed', arm=arm, mxcsr=state()))
            continue
        need(warmed and order['case'] in calls, 'unexpected measurement instruction')
        case = order['case']
        call = calls[case]
        state()
        sample = timer.batch(lambda: call(image), CASES[case])
        need(math.isfinite(sample) and sample > 0, 'invalid latency')
        emit(sys.stdout, dict(event='sample', arm=arm, case=case, pair=order['pair'],
                              loops=CASES[case], ms=sample, mxcsr=state()))


def main(args):
    args.out.mkdir()
    receipts = {arm: getattr(args, path) for arm, path in (('A', 'baseline_receipt'), ('B', 'wide_receipt'))}
    started = completed = 0
    exits = {}
    processes = {}
    events = []
    failure = None
    with (args.out / 'journal.jsonl').open('x', encoding='utf-8') as journal, ExitStack() as stack:
        def record(row):
            emit(journal, row)
            os.fsync(journal.fileno())

        try:
            identity = environment()
            need(all(shutil.disk_usage(p).free >= 30 * 1024**3 for p in (args.out, args.stock, args.baseline_package, args.wide_package)),
                 'scratch reserve not satisfied')
            need(args.cpu in os.sched_getaffinity(0), 'requested CPU not available')
            os.sched_setaffinity(0, {args.cpu})
            need(digest(args.timer) == TIMER_SHA and digest(args.fixture) == FIXTURE_SHA, 'benchmark/fixture bytes mismatch')
            deps = None
            for arm, path in receipts.items():
                arm_receipt = json.loads(path.read_bytes())
                pin = ARMS[arm]
                need(digest(path) == pin['receipt'] and arm_receipt['product_source'] == pin['source'] and
                     arm_receipt['core_sha256'] == pin['core'], f'installed receipt mismatch: {arm}')
                if deps is None:
                    deps = arm_receipt['stock_dependencies_sha256']
                need(len(deps) == 13 and deps == arm_receipt['stock_dependencies_sha256'], 'stock pins differ between installed products')
            stock_record = json.loads(args.stock_receipt.read_bytes())
            need(Path(stock_record['stock_root']).resolve(strict=True) == args.stock.resolve(strict=True) and
                 stock_record['wheel_sha256'] == deps, 'stock dependency materialization receipt mismatch')
            active = subprocess.check_output(['/usr/bin/ps', '-eo', 'stat=,comm='], text=True).splitlines()
            blocked = {'cc1', 'cc1plus', 'gcc', 'g++', 'clang', 'clang++', 'cmake', 'ninja', 'make', 'ctest', 'pytest', 'rustc'}
            need(not any(row.split()[-1] in blocked for row in active if row.split()), 'concurrent build or test process')
            env = dict(os.environ, **POOLS)
            for arm in ARMS:
                error = stack.enter_context((args.out / f'{arm}.stderr').open('x', encoding='utf-8'))
                argv = [sys.executable, '-B', '-I', __file__, *sys.argv[1:], '--worker', arm]
                processes[arm] = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                  stderr=error, env=env, text=True, encoding='utf-8')

            def exchange(arm, request=None):
                process = processes[arm]
                if request is not None:
                    emit(process.stdin, request)
                readable, _, _ = select.select([process.stdout], [], [], 30)
                need(readable, f'{arm} reply deadline exceeded')
                raw = process.stdout.readline()
                need(raw, f'{arm} worker closed its proof pipe')
                response = json.loads(raw)
                need(response['arm'] == arm, 'wrong response arm')
                return response

            ready = {arm: exchange(arm) for arm in ARMS}
            for row in ready.values():
                record(row)
            need(all(row['event'] == 'ready' and row['identity'] == identity for row in ready.values()) and
                 ready['A']['scope'] == ready['B']['scope'] and ready['A']['scope']['affinity'] == [args.cpu],
                 'workers do not share one runtime and CPU')
            record(dict(event='eligible', stock_receipt_sha256=digest(args.stock_receipt), input_sha256=FIXTURE_SHA,
                        timer_sha256=TIMER_SHA, load=list(os.getloadavg()), process_snapshot=active,
                        stock_wheel_sha256=deps, original_imported_during_setup=True, original_reference_calls=0,
                        timing_order=ORDERS))
            for arm in ARMS:
                row = exchange(arm, 'warm')
                need(row['event'] == 'warmed', 'warmup incomplete')
                record(row)
            for pair, order in enumerate(ORDERS, 1):
                for case, loops in CASES.items():
                    for arm in order:
                        record(dict(event='batch-request', pair=pair, arm=arm, case=case, loops=loops))
                        started += 1
                        row = exchange(arm, dict(pair=pair, case=case))
                        need((row['event'], row['pair'], row['case'], row['loops']) == ('sample', pair, case, loops), 'wrong sample identity')
                        events.append(row)
                        completed += 1
                        record(row)
        except (Exception, KeyboardInterrupt) as exc:
            failure = f'{type(exc).__name__}: {exc}'
        finally:
            for process in processes.values():
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            for arm, process in processes.items():
                try:
                    exits[arm] = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    exits[arm] = process.wait(timeout=10)
                process.stdout.close()
    passed = failure is None and started == completed == 32 and exits == {'A': 0, 'B': 0}
    results = {}
    if passed:
        for case in CASES:
            rows = {arm: [row['ms'] for row in events if row['case'] == case and row['arm'] == arm] for arm in ARMS}
            ratios = [a / b for a, b in zip(rows['A'], rows['B'], strict=True)]
            results[case] = dict(median_ms={arm: statistics.median(values) for arm, values in rows.items()},
                                 median_paired_A_over_B=statistics.median(ratios), per_pair_A_over_B=ratios,
                                 paired_A_over_B_by_order={order: statistics.median(r for r, o in zip(ratios, ORDERS, strict=True) if o == order) for order in ('AB', 'BA')})
    with (args.out / 'terminal.json').open('x', encoding='utf-8') as terminal:
        emit(terminal, dict(status='passed' if passed else 'refused_or_failed', failure=failure, arms=ARMS,
                            workers=exits, batches_requested=started, batches_completed=completed,
                            original_reference_calls=0, installed_comparison_only=True, samples=events, results=results))
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-package', 'wide-package', 'baseline-receipt', 'wide-receipt',
                 'stock', 'stock-receipt', 'timer', 'fixture', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cpu', type=int, required=True)
    parser.add_argument('--worker', choices=ARMS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args, args.worker)
    else:
        raise SystemExit(main(args))
