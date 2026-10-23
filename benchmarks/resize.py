"""Fixed, allocation-inclusive resize/rescale comparison with scikit-image 0.26.0.

Local use is untimed: --check-only, optionally --reference-only while the native
API is pending. Measurements require an artifact-fed GitHub Actions job and an
exact preserved-package manifest. No workload is dropped because it is slow.
"""
try:
    from .run import measure
    from .provenance import source_fingerprint
except ImportError:
    from run import measure
    from provenance import source_fingerprint

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import traceback
import warnings

import numpy as np
import scipy
import skimage
from skimage import transform as reference
import skimagexx as xx
from skimagexx.coverage import native_functions

SEED = 5781
CORPUS_VERSION = 'resize-rescale-v1'
TYPES = (np.bool_, np.uint8, np.int8, np.uint16, np.int16, np.uint32,
         np.int32, np.uint64, np.int64, np.float16, np.float32, np.float64)
MODES = ('constant', 'edge', 'symmetric', 'reflect', 'wrap')
APIS = ('transform.resize', 'transform.rescale')


@dataclass(frozen=True)
class Case:
    api: str
    label: str
    image: np.ndarray
    target: object
    options: dict
    expect_error: bool = False


def image(dtype, shape, seed):
    rng = np.random.default_rng(seed)
    if np.dtype(dtype).kind == 'f':
        return rng.normal(size=shape).astype(dtype)
    raw = rng.bit_generator.random_raw(int(np.prod(shape))).reshape(shape)
    if dtype is np.bool_:
        return (raw & 1).astype(dtype)
    result = raw.astype(dtype)
    result.flat[0] = np.iinfo(dtype).min
    result.flat[-1] = np.iinfo(dtype).max
    return result


def corpus():
    """Fixed shapes and options; changes require a new corpus version/hash."""
    for index, dtype in enumerate(TYPES):
        a = image(dtype, (512, 513), SEED+index)
        ordinary_order = 0 if dtype is np.bool_ else 1
        unaligned = np.ndarray(a.shape, a.dtype, buffer=bytearray(a.nbytes+1), offset=1)
        unaligned[...] = a
        readonly = a.view()
        readonly.flags.writeable = False
        variants = [
            ('down-default', a, (257, 259), {}),
            ('up-default', a, (641, 643), {}),
            ('nearest-down', a, (257, 259), dict(order=0, anti_aliasing=False)),
            ('nearest-aa', a, (257, 259), dict(order=0, anti_aliasing=dtype is not np.bool_)),
            ('preserve-range', a, (257, 259), dict(order=ordinary_order, preserve_range=True)),
            ('identity', a, a.shape, {}),
            ('tiny', a[:3, :5], (5, 7), {}),
            ('fortran', np.asfortranarray(a), (257, 259), {}),
            ('reversed', a[::-1, ::-1], (257, 259), {}),
            ('stepped', a[::2, ::2], (173, 175), {}),
            ('endian', a.astype(a.dtype.newbyteorder('>')), (257, 259), {}),
            ('unaligned', unaligned, (257, 259), {}),
            ('broadcast', np.broadcast_to(a[:1], a.shape), (257, 259), {}),
            ('readonly', readonly, (257, 259), {}),
            ('zero', np.zeros_like(a), (257, 259), {}),
        ]
        for label, values, target, options in variants:
            # Upstream's endian-half default path is unsupported. Keep its
            # error parity in the frozen corpus, plus a valid preserved-range
            # endian-half operation below; never call an error a speedup.
            yield Case('resize', label, values, target, options,
                       expect_error=dtype is np.float16 and label == 'endian')
        if dtype is np.float16:
            yield Case('resize', 'endian-preserve-range', a.astype('>f2'), (257, 259),
                       dict(preserve_range=True))
        yield Case('resize', 'volume', image(dtype, (9, 11, 13), SEED+100+index), (5, 17, 9), {})
        for label, scale in [('scalar-scale', .7), ('anisotropic-scale', (.7, 1.3)), ('unit-scale', 1.)]:
            yield Case('rescale', label, a, scale, {})
        channels = image(dtype, (65, 67, 3), SEED+200+index)
        for label, values, axis in [('channels-last', channels, -1),
                                    ('channels-first', np.moveaxis(channels, -1, 0), 0),
                                    ('channels-middle', np.moveaxis(channels, -1, 1), 1)]:
            yield Case('rescale', label, values, (.7, 1.3), dict(channel_axis=axis))
    for index, dtype in enumerate((np.float32, np.float64)):
        a = image(dtype, (128, 129), SEED+300+index)
        for mode in MODES:
            for order in range(6):
                yield Case('resize', f'{mode}-order{order}', a, (97, 173),
                           dict(order=order, mode=mode, cval=-.3, anti_aliasing=False))
        yield Case('resize', 'unclipped-cubic', a, (97, 173),
                   dict(order=5, mode='constant', cval=3., clip=False, anti_aliasing=False))
        yield Case('resize', 'mixed-zero-identity', np.resize(np.array([0., -0.], dtype), (512, 513)),
                   (512, 513), dict(order=0, anti_aliasing=False))
        yield Case('resize', 'nan-identity', np.full((512, 513), np.nan, dtype),
                   (512, 513), dict(order=0, anti_aliasing=False))
    for index, dtype in enumerate((np.uint8, np.int64, np.uint64)):
        a = image(dtype, (128, 129), SEED+400+index)
        for mode in MODES:
            for order in (3, 5):
                yield Case('resize', f'{mode}-integer-order{order}', a, (97, 173),
                           dict(order=order, mode=mode, cval=.3, anti_aliasing=False))
    for index, dtype in enumerate((np.float32, np.float64, np.uint16, np.uint64)):
        a = image(dtype, (512, 513), SEED+500+index)
        for label, sigma in [('zero', 0.), ('scalar', .7), ('anisotropic', (.3, 1.3))]:
            yield Case('resize', 'aa-'+label, a, (257, 259),
                       dict(order=0 if np.dtype(dtype).kind in 'iu' else 3,
                            anti_aliasing=True, anti_aliasing_sigma=sigma))


def serial(value):
    if isinstance(value, np.generic):
        return {'dtype': value.dtype.str, 'value': value.item()}
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(item) for item in value]
    return value


def spec(case):
    a = case.image
    return dict(api='transform.'+case.api, case=case.label, dtype=a.dtype.str,
                shape=list(a.shape), strides=list(a.strides), readonly=not a.flags.writeable,
                aligned=bool(a.flags.aligned), target=serial(case.target), options=serial(case.options),
                expected_error=case.expect_error,
                input_sha256=hashlib.sha256(a.tobytes()).hexdigest())


def warning_key(notices):
    return [(type(w.message).__name__, str(w.message)) for w in notices]


def observe(fn):
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            output, error = fn(), None
        except Exception as caught:
            output, error = None, caught
    return output, warning_key(notices), error


def assert_result(actual, expected, source, options):
    assert type(actual) is type(expected) is np.ndarray
    assert actual.shape == expected.shape and actual.strides == expected.strides
    assert actual.dtype == expected.dtype and actual.dtype.metadata == expected.dtype.metadata
    assert not np.shares_memory(actual, source)
    if actual.dtype.kind in 'biu' or options.get('order') == 0:
        np.testing.assert_array_equal(actual, expected)
    else:
        tolerance = 3e-6 if actual.dtype == np.float32 else 3e-13
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True)
    zeros = (actual == 0) & (expected == 0)
    np.testing.assert_array_equal(np.signbit(actual[zeros]), np.signbit(expected[zeros]))
    if actual.dtype.kind in 'biu':
        return 0.
    finite = np.isfinite(actual) & np.isfinite(expected)
    return float(np.max(np.abs(actual[finite].astype(np.float64)-expected[finite].astype(np.float64)), initial=0.))


def binary_hashes():
    package = Path(xx.__file__).resolve().parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(package.iterdir()) if p.suffix in ('.pyd', '.so')}


def python_hashes():
    package = Path(xx.__file__).resolve().parent
    return {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for p in sorted(package.rglob('*.py'))}


def source_revision(root):
    try:
        return subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cpu_metadata():
    model = os.environ.get('PROCESSOR_IDENTIFIER', '') or platform.processor()
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding='utf-8').splitlines():
            if line.lower().startswith('model name'):
                model = line.partition(':')[2].strip()
                break
    core = getattr(np, '_core', None)
    if core is None:
        core = np.core
    features = dict(getattr(core._multiarray_umath, '__cpu_features__', {}))
    return dict(cpu_model=model, numpy_cpu_features=features)


def verify_prepared(root, path):
    prepared = json.loads(path.read_text(encoding='utf-8'))
    for key, value in source_fingerprint(root).items():
        assert prepared[key] == value, f'prepared {key} does not match checkout'
    assert prepared['revision'] == source_revision(root), 'prepared revision does not match checkout'
    assert prepared['binary_sha256'] == binary_hashes(), 'installed extension differs from prepared wheel'
    installed = Path(xx.__file__).resolve().parent
    sources = list((root/'python/skimagexx').rglob('*.py'))
    assert set(python_hashes()) == {p.relative_to(root/'python/skimagexx').as_posix() for p in sources}, 'installed Python file set differs from checkout'
    for source in sources:
        member = installed/source.relative_to(root/'python/skimagexx')
        assert member.read_bytes().replace(b'\r\n', b'\n') == source.read_bytes().replace(b'\r\n', b'\n'), str(member)
    return dict(manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **prepared)


def save(report, output):
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def summarize(results, workers, eligible_count):
    summaries = {}
    for count in workers:
        rows = [row for row in results if row['threads'] == count]
        assert len(rows) == eligible_count, 'cannot summarize an incomplete measurement corpus'
        speedups = [row['speedup'] for row in rows]
        summaries[str(count)] = dict(case_count=len(rows), primary=count == 1,
            geometric_mean_speedup=math.exp(math.fsum(math.log(value) for value in speedups)/len(speedups)),
            minimum_speedup=min(speedups), maximum_speedup=max(speedups),
            faster_cases=sum(value > 1 for value in speedups),
            slower_cases=[dict(case_index=row['case_index'], speedup=row['speedup'])
                          for row in rows if row['speedup'] < 1])
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, nargs='+', default=[1])
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--target-ms', type=float, default=15.)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--reference-only', action='store_true', help='Untimed corpus validation only; no native parity claim')
    parser.add_argument('--prepared-source', type=Path, help='Artifact baseline-source.json, required for timings')
    args = parser.parse_args(argv)
    if not args.threads or min(args.threads) <= 0 or 1 not in args.threads or args.repeat <= 0 or not math.isfinite(args.target_ms) or args.target_ms <= 0:
        parser.error('positive workers/repeats/target-ms required; include one worker')
    if len(set(args.threads)) != len(args.threads):
        parser.error('worker counts must be unique')
    if args.reference_only and not args.check_only:
        parser.error('--reference-only requires --check-only')
    if not args.check_only and (args.output is None or args.prepared_source is None):
        parser.error('measurements require --output and --prepared-source')
    if not args.check_only and os.environ.get('GITHUB_ACTIONS') != 'true':
        parser.error('measurements require an isolated Actions job; use --check-only locally')
    root = Path(__file__).resolve().parents[1]
    fingerprint = source_fingerprint(root)
    harness = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (Path(__file__).resolve(), root/'benchmarks/run.py', root/'benchmarks/provenance.py')}
    report = {'metadata': dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(), cpu=platform.processor(), cpu_identifier=os.environ.get('PROCESSOR_IDENTIFIER', ''),
        logical_cpus=os.cpu_count(), **cpu_metadata(), python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
        scikit_image=skimage.__version__, skimagexx=xx.__version__, avx2=xx.has_avx2(),
        source_revision=source_revision(root), **fingerprint, harness_sha256=harness,
        current_binary_sha256=binary_hashes(), current_python_sha256=python_hashes(),
        command=sys.argv if argv is None else list(argv),
        corpus_version=CORPUS_VERSION, seed=SEED, requested_workers=args.threads,
        check_only=args.check_only, reference_only=args.reference_only,
        validation_scope='reference-only; native not exercised' if args.reference_only else 'strict native public API versus original scikit-image',
        fallback_enabled=False, reference_thread_pools=1, repeat=args.repeat, target_ms=args.target_ms,
        cooldown_ms=20, sample_order_seed=81,
        scope='Warmed allocation-inclusive public Python calls, native conversion/normalization and output destruction included; input construction excluded',
        warning_timing_policy='Warnings compared during preflight; warning display suppressed symmetrically during timing',
        range_provider='C++ typed scan; runtime NumPy compiled reductions may resolve observable zero/NaN ordering',
        execution_environment='github-actions' if os.environ.get('GITHUB_ACTIONS') == 'true' else 'local-untimed',
        github_run_id=os.environ.get('GITHUB_RUN_ID'), github_run_attempt=os.environ.get('GITHUB_RUN_ATTEMPT'),
        github_job=os.environ.get('GITHUB_JOB'), github_sha=os.environ.get('GITHUB_SHA')),
        'status': 'preflight', 'checks': [], 'results': [], 'untimed_cases': [], 'failures': []}
    save(report, args.output)
    old_workers = xx.get_num_threads()
    old_fallback = xx.set_fallback_enabled(False)
    try:
        if skimage.__version__ != '0.26.0':
            raise RuntimeError('reference must be exactly scikit-image 0.26.0')
        if not args.reference_only:
            missing = sorted(set(APIS)-set(native_functions()))
            if missing:
                raise RuntimeError('public native APIs are not registered: '+', '.join(missing))
        if args.prepared_source is not None:
            report['metadata']['prepared_source'] = verify_prepared(root, args.prepared_source)
        cases = list(corpus())
        specs = [spec(case) for case in cases]
        contracts = [{key: value for key, value in item.items() if key != 'input_sha256'} for item in specs]
        report['metadata'].update(case_count=len(cases),
            timing_eligible_case_count=sum(not case.expect_error for case in cases),
            # NumPy/libm versions can change seeded float values by a bit.
            # Preserve both the stable workload contract and actual data hash.
            case_contract_sha256=hashlib.sha256(json.dumps(contracts, sort_keys=True).encode()).hexdigest(),
            case_spec_sha256=hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest(), case_specs=specs)
        report['status'] = 'checking'
        save(report, args.output)
        for index, case in enumerate(cases):
            row = dict(case_index=index, **specs[index])
            snapshot = case.image.tobytes()
            try:
                ref = getattr(reference, case.api)
                expected, notices, expected_error = observe(lambda: ref(case.image, case.target, **case.options))
                assert case.image.tobytes() == snapshot, 'reference mutated input'
                assert bool(expected_error is not None) == case.expect_error, 'unexpected reference success/error contract'
                row['warnings'] = notices
                if expected_error is not None:
                    row['expected_exception'] = dict(type=type(expected_error).__name__, message=str(expected_error))
                    report['untimed_cases'].append(dict(case_index=index, reason='reference-contract expected exception; not an image-processing timing'))
                else:
                    assert type(expected) is np.ndarray and not np.shares_memory(expected, case.image)
                    row.update(output_dtype=expected.dtype.str, output_shape=list(expected.shape),
                               output_strides=list(expected.strides))
                if not args.reference_only:
                    native = getattr(xx.transform, case.api)
                    first = None
                    errors = {}
                    error_workers = []
                    for workers in args.threads:
                        xx.set_num_threads(workers)
                        actual, actual_notices, actual_error = observe(lambda: native(case.image, case.target, **case.options))
                        assert actual_notices == notices, (actual_notices, notices)
                        assert case.image.tobytes() == snapshot, 'native mutated input'
                        if expected_error is not None:
                            assert type(actual_error) is type(expected_error) and str(actual_error) == str(expected_error)
                            error_workers.append(workers)
                            continue
                        if actual_error is not None:
                            raise actual_error
                        errors[str(workers)] = assert_result(actual, expected, case.image, case.options)
                        assert case.image.tobytes() == snapshot, 'native mutated input'
                        if first is None:
                            first = actual.tobytes()
                        else:
                            assert actual.tobytes() == first, 'native worker outputs differ in raw bytes'
                    row['max_abs_error_by_workers'] = errors
                    if error_workers:
                        row['error_parity_workers'] = error_workers
                row['status'] = 'passed'
            except Exception as error:
                row.update(status='failed', exception_type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
                report['failures'].append(row)
            report['checks'].append(row)
            save(report, args.output)
            print(f'CHECK {index+1}/{len(cases)} {case.api} {case.image.dtype} {case.label}: {row["status"]}', flush=True)
        if report['failures']:
            report['status'] = 'failed-parity'
            return 1
        if args.check_only:
            report['status'] = 'complete-reference-check' if args.reference_only else 'complete-parity-check'
            return 0
        report['status'] = 'measuring'
        order_rng = random.Random(81)
        sample_order = []
        for _ in range(args.repeat):
            order = ['scikit-image', 'current'];order_rng.shuffle(order);sample_order.append(order)
        report['metadata']['paired_sample_order'] = sample_order
        for index, case in enumerate(cases):
            if case.expect_error:
                continue
            ref, native = getattr(reference, case.api), getattr(xx.transform, case.api)
            for workers in args.threads:
                xx.set_num_threads(workers)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    timing = measure(lambda: ref(case.image, case.target, **case.options),
                                     lambda: native(case.image, case.target, **case.options), args.repeat, args.target_ms)
                row = dict(case_index=index, api='transform.'+case.api, case=case.label,
                           dtype=case.image.dtype.str, threads=workers,
                           slower_than_scikit_image=timing['speedup'] < 1, **timing)
                report['results'].append(row)
                save(report, args.output)
                print(f'MEASURE {case.api} {case.image.dtype} {case.label} workers={workers}: {row["speedup"]:.3f}x', flush=True)
        assert source_fingerprint(root) == fingerprint, 'product source changed during benchmark'
        for name, digest in harness.items():
            assert hashlib.sha256((root/'benchmarks'/name).read_bytes()).hexdigest() == digest, 'benchmark source changed'
        assert binary_hashes() == report['metadata']['current_binary_sha256'], 'extension changed during benchmark'
        assert python_hashes() == report['metadata']['current_python_sha256'], 'installed Python package changed during benchmark'
        report['summary_by_workers'] = summarize(report['results'], args.threads, report['metadata']['timing_eligible_case_count'])
        report['status'] = 'complete-measurement'
        return 0
    except Exception as error:
        report['failures'].append(dict(phase=report['status'], exception_type=type(error).__name__,
                                       message=str(error), traceback=traceback.format_exc()))
        report['status'] = 'failed'
        print(f'FAILED {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 1
    finally:
        xx.set_num_threads(old_workers)
        xx.set_fallback_enabled(old_fallback)
        save(report, args.output)


if __name__ == '__main__':
    raise SystemExit(main())
