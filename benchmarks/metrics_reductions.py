"""Fixed public MSE/NRMSE/PSNR parity and isolated end-to-end measurements.

Local execution is untimed (--check-only, optionally --reference-only). Timings
require an artifact-fed Actions job, a prepared exact-package manifest, and a
successful strict-native parity pass over the entire fixed corpus.
"""
try:
    from .run import measure
    from .provenance import source_fingerprint
    from .resize import (binary_hashes, python_hashes, source_revision,
                         cpu_metadata, verify_prepared, observe, save, summarize)
except ImportError:
    from run import measure
    from provenance import source_fingerprint
    from resize import (binary_hashes, python_hashes, source_revision,
                        cpu_metadata, verify_prepared, observe, save, summarize)

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
import sys
import traceback
import warnings

import numpy as np
import scipy
import skimage
from skimage import metrics as reference
import skimagexx as xx
from skimagexx.coverage import native_functions

SEED = 8723
CORPUS_VERSION = 'metrics-reductions-v1'
TYPES = (np.bool_, np.uint8, np.int8, np.uint16, np.int16, np.uint32,
         np.int32, np.uint64, np.int64, np.float16, np.float32, np.float64)
APIS = ('metrics.mean_squared_error', 'metrics.normalized_root_mse', 'metrics.peak_signal_noise_ratio')
MSE, NRMSE, PSNR = (api.split('.')[-1] for api in APIS)


@dataclass(frozen=True)
class Case:
    api: str
    label: str
    first: np.ndarray
    second: np.ndarray
    options: dict
    expect_error: bool = False


def image(dtype, shape, seed, signed=False):
    rng = np.random.default_rng(seed)
    if np.dtype(dtype).kind == 'f':
        return rng.uniform(-.75 if signed else .1, .9, size=shape).astype(dtype)
    raw = rng.bit_generator.random_raw(int(np.prod(shape))).reshape(shape)
    if dtype is np.bool_:
        return (raw & np.uint64(1)).astype(dtype)
    result = raw.astype(dtype)
    if result.size:
        result.flat[0] = np.iinfo(dtype).min
        result.flat[-1] = np.iinfo(dtype).max
    return result


def basic_cases(label, first, second):
    yield Case(MSE, label, first, second, {})
    yield Case(NRMSE, label, first, second, {})
    yield Case(PSNR, label, first, second, {})


def unaligned(values):
    result = np.ndarray(values.shape, values.dtype, buffer=bytearray(values.nbytes+1), offset=1)
    result[...] = values
    return result


def corpus():
    """All real promotions plus fixed scalar/large/layout/option contracts."""
    small = [image(dtype, (17, 19), SEED+i) for i, dtype in enumerate(TYPES)]
    other = [image(dtype, (17, 19), SEED+100+i) for i, dtype in enumerate(TYPES)]
    for i, first in enumerate(small):
        for j, second in enumerate(other):
            yield from basic_cases('promotion', first, second)
    for index, dtype in enumerate(TYPES):
        first, second = image(dtype, (512, 513), SEED+200+index), image(dtype, (512, 513), SEED+300+index)
        readonly = second.view(); readonly.flags.writeable = False
        variants = [
            ('large-c', first, second),
            ('fortran', np.asfortranarray(first), np.asfortranarray(second)),
            ('mixed-c-f', first, np.asfortranarray(second)),
            ('reversed', first[::-1, ::-1], second[::-1, ::-1]),
            ('stepped', first[::2, ::2], second[::2, ::2]),
            ('endian', first.astype(first.dtype.newbyteorder('>')), second.astype(second.dtype.newbyteorder('>'))),
            ('unaligned', unaligned(first), unaligned(second)),
            ('broadcast', np.broadcast_to(first[:1], first.shape), second),
            ('readonly', first, readonly),
            ('tagged', first.view(np.dtype(first.dtype,metadata={'unit':'raw'})),
                       second.view(np.dtype(second.dtype,metadata={'unit':'raw'}))),
        ]
        for label, a, b in variants:
            yield from basic_cases(label, a, b)
        if np.dtype(dtype).kind=='f':
            defect=first.copy()
            defect.flat[0]=np.nextafter(first.flat[0],np.asarray(np.inf,dtype=dtype))
            yield from basic_cases('one-ulp-defect',first,defect)
        for normalization in ('euclidean', 'min-max', 'mean', 'EuClIdEaN'):
            yield Case(NRMSE, 'normalization-'+normalization, first, second, dict(normalization=normalization))
        for label, value in [('python-float', 1.), ('typed-float', np.float32(.7)), ('typed-integer', np.uint16(255))]:
            yield Case(PSNR, 'range-'+label, first, second, dict(data_range=value))
        yield from basic_cases('scalar', image(dtype, (), SEED+400+index), image(dtype, (), SEED+500+index))
        yield from basic_cases('volume', image(dtype, (13, 17, 19), SEED+600+index), image(dtype, (13, 17, 19), SEED+700+index))
        yield from basic_cases('tiny', first[:3, :5], second[:3, :5])
        zeros = np.zeros((17, 19), dtype)
        yield from basic_cases('identical', first, first)
        yield from basic_cases('zero-reference', zeros, second[:17, :19])
        yield Case(NRMSE, 'zero-reference-min-max', zeros, second[:17, :19], dict(normalization='min-max'))
        yield Case(NRMSE, 'zero-reference-mean', zeros, second[:17, :19], dict(normalization='mean'))
    for index, dtype in enumerate((np.uint16, np.float32, np.float64)):
        yield from basic_cases('megapixel', image(dtype, (1024, 1025), SEED+800+index),
                               image(dtype, (1024, 1025), SEED+900+index))
    for index, dtype in enumerate((np.float16, np.float32, np.float64)):
        a = image(dtype, (257, 259), SEED+1000+index, signed=True)
        b = image(dtype, a.shape, SEED+1100+index, signed=True)
        for norm in ('euclidean', 'min-max', 'mean'):
            yield Case(NRMSE, 'signed-'+norm, a, b, dict(normalization=norm))
        yield Case(PSNR, 'signed-default-range', a, b, {})
        for label, values in [('nan', [np.nan, .5, .7]), ('infinity', [np.inf, .5, .7]),
                              ('all-nan', [np.nan]), ('signed-zero', [-0., 0.])]:
            a = np.resize(np.array(values, dtype), (17, 19)); b = np.zeros_like(a)
            # NaNs do not trip the default PSNR range check; infinity does.
            yield Case(MSE, label, a, b, {})
            for norm in ('euclidean', 'min-max', 'mean'):
                yield Case(NRMSE, label+'-'+norm, a, b, dict(normalization=norm))
            yield Case(PSNR, label+'-explicit-range', a, b, dict(data_range=1.))
            yield Case(PSNR, label+'-default-range', a, b, {}, expect_error=label == 'infinity')
        empty = np.empty((0, 7), dtype)
        yield Case(MSE, 'empty', empty, empty, {})
        yield Case(NRMSE, 'empty-euclidean', empty, empty, {})
        yield Case(NRMSE, 'empty-mean', empty, empty, dict(normalization='mean'))
        yield Case(NRMSE, 'empty-min-max', empty, empty, dict(normalization='min-max'), True)
        yield Case(PSNR, 'empty-explicit-range', empty, empty, dict(data_range=1.))
        yield Case(PSNR, 'empty-default-range', empty, empty, {}, True)
        # Finite extremes distinguish subtract/multiply/reduce overflow from
        # pre-existing infinities, including the denominator-before-MSE order.
        extreme=np.full((17,19),np.finfo(dtype).max,dtype)
        opposite=-extreme
        yield Case(MSE,'finite-extreme',extreme,opposite,{})
        for norm in ('euclidean','min-max','mean'):
            yield Case(NRMSE,'finite-extreme-'+norm,extreme,opposite,dict(normalization=norm))
        yield Case(PSNR,'finite-extreme-explicit-range',extreme,opposite,dict(data_range=1.))
        yield Case(PSNR,'finite-extreme-default-range',extreme,opposite,{},True)
    a, b = np.full((17, 19), .5), np.full((17, 19), .7)
    for label, value in [('zero', 0.), ('negative', -1.), ('nan', np.nan), ('infinity', np.inf),
                         ('zero-dimensional', np.asarray(1.)), ('numeric-string', '2.5')]:
        yield Case(PSNR, 'explicit-range-'+label, a, b, dict(data_range=value))
    yield Case(PSNR, 'outside-float-default-range', a+2, b, {}, True)
    for norm in ('unsupported', 3):
        yield Case(NRMSE, 'invalid-normalization-'+str(norm), a, b, dict(normalization=norm), True)
    yield Case(PSNR, 'invalid-data-range', a, b, dict(data_range='invalid'), True)
    for name in (MSE, NRMSE, PSNR):
        yield Case(name, 'shape-error', np.ones((3, 5)), np.ones((5, 3)), {}, True)


def serial(value):
    if isinstance(value, np.generic):
        return dict(dtype=value.dtype.str, value=serial(value.item()))
    if isinstance(value, np.ndarray):
        return dict(dtype=value.dtype.str, shape=list(value.shape), value=serial(value.tolist()))
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return 'nan' if math.isnan(value) else '-inf' if value < 0 else 'inf'
    return value


def input_spec(values):
    return dict(dtype=values.dtype.str, shape=list(values.shape), strides=list(values.strides),
                readonly=not values.flags.writeable, aligned=bool(values.flags.aligned),
                dtype_metadata=serial(dict(values.dtype.metadata)) if values.dtype.metadata is not None else None,
                input_sha256=hashlib.sha256(values.tobytes()).hexdigest())


def spec(case):
    return dict(api='metrics.'+case.api, case=case.label, first=input_spec(case.first),
                second=input_spec(case.second), options=serial(case.options), expected_error=case.expect_error)


def scalar_spec(value):
    return dict(type=type(value).__module__+'.'+type(value).__qualname__,
                dtype=np.asarray(value).dtype.str, value=serial(float(value)), signbit=bool(np.signbit(value)),
                bits_hex=np.asarray(value).tobytes().hex())


def assert_result(actual, expected, case):
    assert type(actual) is type(expected), (type(actual), type(expected))
    assert np.ndim(actual) == np.ndim(expected) == 0
    assert np.asarray(actual).dtype == np.asarray(expected).dtype
    work32 = all(a.dtype.type in (np.float16, np.float32) for a in (case.first, case.second))
    tolerance = 3e-6 if work32 else 3e-13
    # A zero MSE must not pass for a small positive error. Logarithmic PSNR
    # retains an absolute budget near 0 dB; MSE/NRMSE use relative precision.
    atol=tolerance if case.api==PSNR else 0.
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=atol, equal_nan=True)
    if actual == expected == 0:
        assert bool(np.signbit(actual)) == bool(np.signbit(expected)), 'scalar zero sign differs'
    return dict(rtol=tolerance, atol=atol,
                absolute_error=float(abs(actual-expected)) if np.isfinite(actual) and np.isfinite(expected) else None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, nargs='+', default=[1])
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--target-ms', type=float, default=15.)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--reference-only', action='store_true', help='Untimed reference contract only; no native parity claim')
    parser.add_argument('--prepared-source', type=Path)
    args = parser.parse_args(argv)
    if min(args.threads) <= 0 or 1 not in args.threads or len(set(args.threads)) != len(args.threads) or args.repeat <= 0 or not math.isfinite(args.target_ms) or args.target_ms <= 0:
        parser.error('positive distinct workers/repeats/target-ms required; include one worker')
    if args.reference_only and not args.check_only:
        parser.error('--reference-only requires --check-only')
    if not args.check_only and (args.output is None or args.prepared_source is None):
        parser.error('measurements require --output and --prepared-source')
    if not args.check_only and os.environ.get('GITHUB_ACTIONS') != 'true':
        parser.error('measurements require an isolated Actions job; use --check-only locally')
    root = Path(__file__).resolve().parents[1]
    fingerprint = source_fingerprint(root)
    harness = {name: hashlib.sha256((root/'benchmarks'/name).read_bytes()).hexdigest()
               for name in ('metrics_reductions.py', 'resize.py', 'run.py', 'provenance.py')}
    report = {'metadata': dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
        cpu=platform.processor(), logical_cpus=os.cpu_count(), **cpu_metadata(), python=sys.version,
        numpy=np.__version__, scipy=scipy.__version__, scikit_image=skimage.__version__, skimagexx=xx.__version__,
        avx2=xx.has_avx2(), source_revision=source_revision(root), **fingerprint, harness_sha256=harness,
        current_binary_sha256=binary_hashes(), current_python_sha256=python_hashes(), command=sys.argv if argv is None else list(argv),
        corpus_version=CORPUS_VERSION, seed=SEED, requested_workers=args.threads, fallback_enabled=False,
        check_only=args.check_only, reference_only=args.reference_only, reference_thread_pools=1,
        validation_scope='reference-only; native not exercised' if args.reference_only else 'strict native public API versus original scikit-image',
        scope='Warmed end-to-end public scalar metric calls, input conversion/normalization, temporary allocations and result destruction included; fixture construction excluded',
        warning_timing_policy='Warnings compared during preflight; warning display suppressed symmetrically during timing',
        repeat=args.repeat, target_ms=args.target_ms, cooldown_ms=20, sample_order_seed=81,
        execution_environment='github-actions' if os.environ.get('GITHUB_ACTIONS') == 'true' else 'local-untimed',
        github_run_id=os.environ.get('GITHUB_RUN_ID'), github_run_attempt=os.environ.get('GITHUB_RUN_ATTEMPT'),
        github_job=os.environ.get('GITHUB_JOB'), github_sha=os.environ.get('GITHUB_SHA')),
        'status': 'preflight', 'checks': [], 'results': [], 'untimed_cases': [], 'failures': []}
    save(report, args.output)
    old_workers = xx.get_num_threads(); old_fallback = xx.set_fallback_enabled(False)
    try:
        if skimage.__version__ != '0.26.0':
            raise RuntimeError('reference must be exactly scikit-image 0.26.0')
        if not args.reference_only:
            missing = sorted(set(APIS)-set(native_functions()))
            if missing:
                raise RuntimeError('public native APIs are not registered: '+', '.join(missing))
        if args.prepared_source is not None:
            report['metadata']['prepared_source'] = verify_prepared(root, args.prepared_source)
        cases = list(corpus()); specs = [spec(case) for case in cases]
        contracts = [dict(item, first={k:v for k,v in item['first'].items() if k!='input_sha256'},
                          second={k:v for k,v in item['second'].items() if k!='input_sha256'}) for item in specs]
        report['metadata'].update(case_count=len(cases), timing_eligible_case_count=sum(not case.expect_error for case in cases),
            case_contract_sha256=hashlib.sha256(json.dumps(contracts, sort_keys=True).encode()).hexdigest(),
            case_spec_sha256=hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest(), case_specs=specs)
        report['status']='checking'; save(report, args.output)
        for index, case in enumerate(cases):
            row = dict(case_index=index, **specs[index]); snapshots=(case.first.tobytes(),case.second.tobytes())
            try:
                ref=getattr(reference,case.api)
                expected, notices, expected_error=observe(lambda:ref(case.first,case.second,**case.options))
                assert (case.first.tobytes(),case.second.tobytes())==snapshots, 'reference mutated input'
                assert bool(expected_error is not None)==case.expect_error, 'unexpected reference success/error contract'
                row['warnings']=notices
                if expected_error is not None:
                    row['expected_exception']=dict(type=type(expected_error).__name__,message=str(expected_error))
                    report['untimed_cases'].append(dict(case_index=index,reason='reference-contract expected exception; not an image-processing timing'))
                else:
                    row['expected_scalar']=scalar_spec(expected)
                if not args.reference_only:
                    native=getattr(xx.metrics,case.api); first=None; errors={}; scalars={}; error_workers=[]
                    for workers in args.threads:
                        xx.set_num_threads(workers)
                        actual,actual_notices,actual_error=observe(lambda:native(case.first,case.second,**case.options))
                        assert actual_notices==notices,(actual_notices,notices)
                        assert (case.first.tobytes(),case.second.tobytes())==snapshots,'native mutated input'
                        if expected_error is not None:
                            assert type(actual_error) is type(expected_error) and str(actual_error)==str(expected_error)
                            error_workers.append(workers);continue
                        if actual_error is not None:raise actual_error
                        errors[str(workers)]=assert_result(actual,expected,case)
                        scalars[str(workers)]=scalar_spec(actual)
                        bits=np.asarray(actual).tobytes()
                        if first is None:first=bits
                        else:assert bits==first,'native worker scalar bits differ'
                    row['error_by_workers']=errors
                    row['actual_scalar_by_workers']=scalars
                    if error_workers:row['error_parity_workers']=error_workers
                row['status']='passed'
            except Exception as error:
                row.update(status='failed',exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
                report['failures'].append(row)
            report['checks'].append(row);save(report,args.output)
            print(f'CHECK {index+1}/{len(cases)} {case.api} {case.first.dtype}/{case.second.dtype} {case.label}: {row["status"]}',flush=True)
        if report['failures']:
            report['status']='failed-parity';return 1
        if args.check_only:
            report['status']='complete-reference-check' if args.reference_only else 'complete-parity-check';return 0
        report['status']='measuring';rng=random.Random(81);sample_order=[]
        for _ in range(args.repeat):
            order=['scikit-image','current'];rng.shuffle(order);sample_order.append(order)
        report['metadata']['paired_sample_order']=sample_order
        for index,case in enumerate(cases):
            if case.expect_error:continue
            ref,native=getattr(reference,case.api),getattr(xx.metrics,case.api)
            for workers in args.threads:
                xx.set_num_threads(workers)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    timing=measure(lambda:ref(case.first,case.second,**case.options),lambda:native(case.first,case.second,**case.options),args.repeat,args.target_ms)
                row=dict(case_index=index,api='metrics.'+case.api,case=case.label,first_dtype=case.first.dtype.str,
                         second_dtype=case.second.dtype.str,threads=workers,slower_than_scikit_image=timing['speedup']<1,**timing)
                report['results'].append(row);save(report,args.output)
                print(f'MEASURE {case.api} {case.first.dtype}/{case.second.dtype} {case.label} workers={workers}: {row["speedup"]:.3f}x',flush=True)
        assert source_fingerprint(root)==fingerprint,'product source changed during benchmark'
        for name,digest in harness.items():assert hashlib.sha256((root/'benchmarks'/name).read_bytes()).hexdigest()==digest,'benchmark source changed'
        assert binary_hashes()==report['metadata']['current_binary_sha256'],'extension changed during benchmark'
        assert python_hashes()==report['metadata']['current_python_sha256'],'installed Python package changed during benchmark'
        report['summary_by_workers']=summarize(report['results'],args.threads,report['metadata']['timing_eligible_case_count'])
        report['status']='complete-measurement';return 0
    except Exception as error:
        report['failures'].append(dict(phase=report['status'],exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        report['status']='failed';print(f'FAILED {type(error).__name__}: {error}',file=sys.stderr,flush=True);return 1
    finally:
        xx.set_num_threads(old_workers);xx.set_fallback_enabled(old_fallback);save(report,args.output)


if __name__=='__main__':raise SystemExit(main())
