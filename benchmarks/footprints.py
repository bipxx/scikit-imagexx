"""Fixed public footprint parity and isolated end-to-end measurements.

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
from pathlib import Path, PurePosixPath
import platform
import random
import sys
import traceback
import warnings
import zipfile

import numpy as np
import scipy
import skimage
from skimage import morphology as reference
import skimagexx as xx
from skimagexx.coverage import native_functions


SEED = 0
CORPUS_VERSION = 'footprints-v1'
NAMES = ('footprint_rectangle', 'diamond', 'disk', 'ball', 'mirror_footprint', 'pad_footprint')
APIS = tuple('morphology.' + name for name in NAMES)
TYPES = ('bool', 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32',
         'uint64', 'int64', 'float16', 'float32', 'float64')


@dataclass(frozen=True)
class Case:
    api: str
    label: str
    args: tuple
    options: dict
    work_kind: str = 'raster'
    expect_error: bool = False


def dtype_for(name):
    if name == 'structured': return np.dtype([('value', 'i2'), ('flag', '?')])
    if name == 'tagged': return np.dtype('uint8', metadata={'contract': 'footprint'})
    return np.dtype(name)


def corpus():
    """Fixed complete workloads; boundary tables never allocate radius250 balls."""
    for dtype_name in TYPES + ('complex64', 'object', 'U3', 'S3', '>i2', 'structured', 'tagged'):
        dtype = dtype_for(dtype_name)
        for name in NAMES[:4]:
            decompositions = ([None, 'sequence', 'separable'] if name == 'footprint_rectangle'
                              else [None, 'sequence', 'crosses'] if name == 'disk'
                              else [None, 'sequence'])
            for decomposition in decompositions:
                arg = (3, 5) if name == 'footprint_rectangle' else 3
                error = name == 'disk' and decomposition == 'crosses' and dtype_name in ('U3', 'S3', 'structured')
                yield Case(name, f'dtype={dtype_name}/decomposition={decomposition}', (arg,),
                           dict(dtype=dtype, decomposition=decomposition),
                           'raster' if decomposition is None else 'decomposition_plan', error)
        for name in NAMES[4:]:
            values = np.ones((17, 19), dtype=dtype)
            for label, array in [('c', values), ('fortran', np.asfortranarray(values)),
                                 ('reverse', values[::-1, ::-1]), ('strided', values[::2, ::2]),
                                 ('broadcast', np.broadcast_to(values[:1], values.shape))]:
                yield Case(name, f'dtype={dtype_name}/layout={label}', (array,), {},
                           'view_metadata' if name == 'mirror_footprint' else 'raster')
            readonly = values.copy(); readonly.flags.writeable = False
            yield Case(name, f'dtype={dtype_name}/layout=readonly', (readonly,), {},
                       'view_metadata' if name == 'mirror_footprint' else 'raster')
    for name in NAMES[:4]:
        for size, label in [(0, 'zero'), (1, 'unit'), (2, 'small'), (8, 'medium')]:
            arg = (size, size+2) if name == 'footprint_rectangle' else size
            yield Case(name, label, (arg,), {})
        for dtype in ['uint8', 'float32', 'float64']:
            arg = (1024, 1025) if name == 'footprint_rectangle' else 48 if name == 'ball' else 513
            yield Case(name, f'large/{dtype}', (arg,), dict(dtype=dtype))
        if name == 'footprint_rectangle':
            for shape in [(3,), (3, 5, 7), (1, 5), (2, 4), (0, 3)]:
                for decomposition in ['sequence', 'separable']:
                    yield Case(name, f'shape={shape}/{decomposition}', (shape,),
                               dict(decomposition=decomposition), 'decomposition_plan')
        else:
            for radius in [-2, -.5, .5, 1.5, np.int16(3), np.float32(2.5)]:
                yield Case(name, f'radius={type(radius).__name__}:{str(radius)}', (radius,), {})
            if name in ('disk', 'ball'):
                for strict in [False, True]:
                    yield Case(name, f'strict={strict}', (3,), dict(strict_radius=strict))
                maximum = 251 if name == 'disk' else 101
                for radius in [-maximum, -2, 0, 1, 5, maximum-1, maximum, maximum+1, 2.5, False]:
                    yield Case(name, f'table={radius!r}', (radius,), dict(decomposition='sequence'),
                               'decomposition_plan', radius is False or radius in (maximum, maximum+1, 2.5))
    for name in ['disk', 'ball']:
        yield Case(name, 'dtype=None', (2,), dict(dtype=None))
    for strict in [False, True]:
        for radius in [.1, 1.3, 20, 80]:
            yield Case('disk', f'crosses={radius}/strict={strict}', (radius,),
                       dict(decomposition='crosses', strict_radius=strict), 'decomposition_plan')
    for name in NAMES[4:]:
        for shape in [(0,), (1,), (2,), (2, 3, 4), (1, 3, 5), (1024, 1025)]:
            values = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)
            for pad_end in [False, True]:
                yield Case(name, f'shape={shape}/pad_end={pad_end}', (values,),
                           dict(pad_end=pad_end) if name == 'pad_footprint' else {},
                           'view_metadata' if name == 'mirror_footprint' else 'raster')
        a = np.arange(6, dtype=np.int16).reshape(2, 3)
        b = np.ones((3, 1), dtype=bool)
        for label, value in [('empty-sequence', ()), ('single-sequence', ((a, 2),)),
                             ('repeated-sequence', ((a, np.int64(0)), (a, True), (b, -2)))]:
            yield Case(name, label, (value,), {}, 'view_metadata' if name == 'mirror_footprint' else 'decomposition_plan')
        for label, value in [('nested-sequence', ((((a, 1),), 2),)),
                             ('nonintegral-repeat', ((a, 1.5),)), ('array-list', [[1, 0], [0, 1]]),
                             ('scalar-footprint', 3)]:
            yield Case(name, label, (value,), {}, 'view_metadata', True)
    for name in NAMES[:4]:
        yield Case(name, 'invalid-dtype', ((3, 5) if name == 'footprint_rectangle' else 2,),
                   dict(dtype='not-a-dtype'), expect_error=True)
        yield Case(name, 'invalid-decomposition', ((3, 5) if name == 'footprint_rectangle' else 2,),
                   dict(decomposition='invalid'), 'decomposition_plan', True)
    yield Case('footprint_rectangle', 'negative-shape', ((-1, 3),), {}, expect_error=True)
    yield Case('footprint_rectangle', 'fractional-shape', ((2.5, 3),), {}, expect_error=True)


def leaves(value):
    if isinstance(value, np.ndarray): return [value]
    if isinstance(value, (tuple, list)): return [a for v in value for a in leaves(v)]
    return []


def serial(value):
    if isinstance(value, np.dtype):
        return dict(dtype=value.str, descr=repr(value.descr), metadata=repr(value.metadata))
    if isinstance(value, np.generic):
        return dict(type=type(value).__name__, value=repr(value.item()))
    if isinstance(value, np.ndarray):
        return dict(array=True, dtype=serial(value.dtype), shape=list(value.shape), strides=list(value.strides),
                    readonly=not value.flags.writeable, aligned=bool(value.flags.aligned),
                    input_sha256=hashlib.sha256(repr(value.tolist()).encode()).hexdigest())
    if isinstance(value, dict): return {k:serial(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)): return dict(type=type(value).__name__, items=[serial(v) for v in value])
    return value


def spec(case):
    return dict(api='morphology.'+case.api, case=case.label, args=serial(case.args),
                options=serial(case.options), work_kind=case.work_kind, expected_error=case.expect_error)


def result_spec(value, inputs):
    if isinstance(value, np.ndarray):
        return dict(type=type(value).__name__, dtype=serial(value.dtype), shape=list(value.shape),
                    strides=list(value.strides), owned=bool(value.flags.owndata),
                    writeable=bool(value.flags.writeable), c=bool(value.flags.c_contiguous),
                    f=bool(value.flags.f_contiguous), values_sha256=hashlib.sha256(repr(value.tolist()).encode()).hexdigest(),
                    input_alias=[bool(np.shares_memory(value,a)) for a in inputs],
                    input_identity=[value is a for a in inputs])
    if isinstance(value, (tuple, list)):
        return dict(type=type(value).__name__, items=[result_spec(v,inputs) for v in value])
    return dict(type=type(value).__name__, value=serial(value))


def output_spec(value, inputs):
    outputs=leaves(value)
    return dict(structure=result_spec(value,inputs),
                output_alias=[[bool(np.shares_memory(a,b)) for b in outputs] for a in outputs])


def check_case(case, workers, reference_only, row):
    inputs=leaves(case.args)
    before=[result_spec(a,[]) for a in inputs]
    expected, notices, error=observe(lambda:getattr(reference,case.api)(*case.args,**case.options))
    assert [result_spec(a,[]) for a in inputs]==before, 'reference mutated input'
    assert (error is not None)==case.expect_error, 'unexpected original success/error contract'
    row['warnings']=notices
    if error is None:
        row['expected_output']=output_spec(expected,inputs)
    else:
        row['expected_exception']=dict(type=type(error).__name__,message=str(error))
    if reference_only:return
    actual_by_workers={};error_workers=[]
    for count in workers:
        xx.set_num_threads(count)
        assert xx.get_num_threads()==count, 'global worker setting was not applied'
        actual, actual_notices, actual_error=observe(lambda:getattr(xx.morphology,case.api)(*case.args,**case.options))
        assert [result_spec(a,[]) for a in inputs]==before, 'native mutated input'
        assert actual_notices==notices, (actual_notices,notices)
        if error is not None:
            assert type(actual_error) is type(error) and str(actual_error)==str(error)
            error_workers.append(count)
        else:
            if actual_error is not None:raise actual_error
            description=output_spec(actual,inputs)
            assert description==row['expected_output'], (description,row['expected_output'])
            actual_by_workers[str(count)]=description
    row['actual_output_by_workers']=actual_by_workers
    if error_workers:row['error_parity_workers']=error_workers


def package_hashes():
    package=Path(xx.__file__).resolve().parent
    return {p.relative_to(package).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(package.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}


def sdk_member(name):
    return (name.startswith(('include/skimagexx/','lib/cmake/skimagexx/','share/licenses/scikit-imagexx/'))
            or name in ('lib/skimagexx.lib','lib/libskimagexx.a'))


def verify_sdk(records):
    installed=Path(xx.__file__).resolve().parent.parent
    for name,digest in records.items():
        assert not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts and '\\' not in name and ':' not in name, 'invalid SDK member path'
        path=installed/name
        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==digest, 'installed SDK differs from wheel: '+name
    for prefix in ('include/skimagexx','lib/cmake/skimagexx','share/licenses/scikit-imagexx'):
        actual={p.relative_to(installed).as_posix() for p in (installed/prefix).rglob('*') if p.is_file()}
        assert actual=={n for n in records if n.startswith(prefix+'/')}, 'installed SDK file set differs: '+prefix


def seal_artifacts(prepared, wheel, source_root=None):
    """Preparation only: bind the exact wheel, complete package and installed SDK."""
    with zipfile.ZipFile(wheel) as archive:
        members={n:hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist() if not n.endswith('/')}
    package={n.removeprefix('skimagexx/'):h for n,h in members.items() if n.startswith('skimagexx/')}
    sdk={n:h for n,h in members.items() if sdk_member(n)}
    assert sdk and any(n.endswith('image.hpp') for n in sdk), 'wheel lacks C++ SDK'
    assert any(n.endswith(('.a','.lib')) for n in sdk), 'wheel lacks SDK static library'
    assert package==package_hashes(), 'installed complete package differs from wheel'
    verify_sdk(sdk)
    root=Path(__file__).resolve().parents[1] if source_root is None else source_root
    installed=Path(xx.__file__).resolve().parent.parent
    for name in sdk:
        if name.startswith('include/skimagexx/'):
            assert (root/name).read_bytes().replace(b'\r\n',b'\n')==(installed/name).read_bytes().replace(b'\r\n',b'\n'), 'SDK header differs from source: '+name
    seal=dict(schema='footprints-exact-wheel-sdk-v1',wheel_filename=wheel.name,
              wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
              package_sha256=package,sdk_sha256=sdk,
              prepared_source_sha256=hashlib.sha256(prepared.read_bytes()).hexdigest())
    output=prepared.with_name('footprints-artifacts.json')
    assert not output.exists(), 'choose a fresh prepared artifact seal'
    output.write_text(json.dumps(seal,indent=2)+'\n',encoding='utf-8')
    return verify_sealed(prepared)


def verify_sealed(prepared):
    path=prepared.with_name('footprints-artifacts.json')
    seal=json.loads(path.read_text(encoding='utf-8'))
    assert seal['schema']=='footprints-exact-wheel-sdk-v1'
    assert seal['prepared_source_sha256']==hashlib.sha256(prepared.read_bytes()).hexdigest(), 'prepared source manifest changed'
    # Artifact directory layout is fixed by the workflow on both hosts.
    filename=seal['wheel_filename']
    assert Path(filename).name==filename and '/' not in filename and '\\' not in filename and Path(filename).suffix=='.whl', 'invalid wheel filename'
    wheel=prepared.parent.parent/'wheels'/filename
    assert hashlib.sha256(wheel.read_bytes()).hexdigest()==seal['wheel_sha256'], 'prepared wheel changed'
    assert package_hashes()==seal['package_sha256'], 'installed complete package changed'
    verify_sdk(seal['sdk_sha256'])
    return dict(seal_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**seal)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, nargs='+', default=[1])
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--target-ms', type=float, default=15.)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--reference-only', action='store_true', help='Untimed reference contract only; no native parity claim')
    parser.add_argument('--prepared-source', type=Path)
    parser.add_argument('--seal-wheel', type=Path, help='Preparation-only exact wheel/SDK seal; requires --check-only and --prepared-source')
    args = parser.parse_args(argv)
    if min(args.threads) <= 0 or 1 not in args.threads or len(set(args.threads)) != len(args.threads) or args.repeat <= 0 or not math.isfinite(args.target_ms) or args.target_ms <= 0:
        parser.error('positive distinct workers/repeats/target-ms required; include one worker')
    if args.reference_only and not args.check_only:
        parser.error('--reference-only requires --check-only')
    if args.seal_wheel and (not args.check_only or args.prepared_source is None or args.reference_only):
        parser.error('--seal-wheel requires strict --check-only and --prepared-source')
    if not args.check_only and (args.output is None or args.prepared_source is None):
        parser.error('measurements require --output and --prepared-source')
    if not args.check_only and os.environ.get('GITHUB_ACTIONS') != 'true':
        parser.error('measurements require an isolated Actions job; use --check-only locally')
    root = Path(__file__).resolve().parents[1]
    fingerprint = source_fingerprint(root)
    harness = {name: hashlib.sha256((root/'benchmarks'/name).read_bytes()).hexdigest()
               for name in ('footprints.py', 'resize.py', 'run.py', 'provenance.py')}
    report = {'metadata': dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
        cpu=platform.processor(), logical_cpus=os.cpu_count(), **cpu_metadata(), python=sys.version,
        numpy=np.__version__, scipy=scipy.__version__, scikit_image=skimage.__version__, skimagexx=xx.__version__,
        avx2=xx.has_avx2(), source_revision=source_revision(root), **fingerprint, harness_sha256=harness,
        current_binary_sha256=binary_hashes(), current_python_sha256=python_hashes(), command=sys.argv if argv is None else list(argv),
        corpus_version=CORPUS_VERSION, seed=SEED, requested_workers=args.threads, fallback_enabled=False,
        check_only=args.check_only, reference_only=args.reference_only, reference_thread_pools=1,
        validation_scope='reference-only; native not exercised' if args.reference_only else 'strict native public API versus original scikit-image',
        scope='Warmed public calls including allocation, dtype providers, normalization, tuple assembly and destruction; fixture construction excluded. View/metadata and plan rows are separate from raster work; their ratios are not image-kernel speedups.',
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
            if args.seal_wheel:
                report['metadata']['exact_artifacts']=seal_artifacts(args.prepared_source,args.seal_wheel)
            elif not args.check_only:
                report['metadata']['exact_artifacts']=verify_sealed(args.prepared_source)
        cases = list(corpus()); specs = [spec(case) for case in cases]
        report['metadata'].update(case_count=len(cases), timing_eligible_case_count=sum(not c.expect_error for c in cases),
            case_contract_sha256=hashlib.sha256(json.dumps(specs,sort_keys=True).encode()).hexdigest(),
            case_spec_sha256=hashlib.sha256(json.dumps(specs,sort_keys=True).encode()).hexdigest(),case_specs=specs)
        report['status']='checking';save(report,args.output)
        for index,case in enumerate(cases):
            row=dict(case_index=index,**specs[index])
            try:
                check_case(case,args.threads,args.reference_only,row)
                if case.expect_error:
                    report['untimed_cases'].append(dict(case_index=index,reason='expected original exception; no timing'))
                row['status']='passed'
            except Exception as error:
                row.update(status='failed',exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
                report['failures'].append(row)
            report['checks'].append(row);save(report,args.output)
            print(f'CHECK {index+1}/{len(cases)} {case.api} {case.label}: {row["status"]}',flush=True)
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
            ref,native=getattr(reference,case.api),getattr(xx.morphology,case.api)
            for workers in args.threads:
                xx.set_num_threads(workers)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    timing=measure(lambda:ref(*case.args,**case.options),lambda:native(*case.args,**case.options),args.repeat,args.target_ms)
                row=dict(case_index=index,api='morphology.'+case.api,case=case.label,work_kind=case.work_kind,
                         image_kernel_speedup_claim=False,ratio_scope='public '+case.work_kind+' call',
                         threads=workers,slower_than_scikit_image=timing['speedup']<1,**timing)
                report['results'].append(row);save(report,args.output)
                print(f'MEASURE {case.api} {case.work_kind} {case.label} workers={workers}: {row["speedup"]:.3f}x',flush=True)
        assert source_fingerprint(root)==fingerprint,'product source changed during benchmark'
        for name,digest in harness.items():assert hashlib.sha256((root/'benchmarks'/name).read_bytes()).hexdigest()==digest,'benchmark source changed'
        assert binary_hashes()==report['metadata']['current_binary_sha256'],'extension changed during benchmark'
        assert python_hashes()==report['metadata']['current_python_sha256'],'installed Python package changed during benchmark'
        assert verify_sealed(args.prepared_source)==report['metadata']['exact_artifacts'], 'wheel/package/SDK artifacts changed during benchmark'
        report['summary_by_work_kind']={kind:summarize([r for r in report['results'] if r['work_kind']==kind],args.threads,sum(not c.expect_error and c.work_kind==kind for c in cases)) for kind in sorted({c.work_kind for c in cases if not c.expect_error})}
        report['status']='complete-measurement';return 0
    except Exception as error:
        report['failures'].append(dict(phase=report['status'],exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        report['status']='failed';print(f'FAILED {type(error).__name__}: {error}',file=sys.stderr,flush=True);return 1
    finally:
        xx.set_num_threads(old_workers);xx.set_fallback_enabled(old_fallback);save(report,args.output)


if __name__=='__main__':raise SystemExit(main())
