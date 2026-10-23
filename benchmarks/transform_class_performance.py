"""Artifact-fed public class timings; local runs are strictly untimed.

The unchanged 296-case contract runs in full before any measurement. Coordinate
calls, NumPy/SciPy providers and state/serialization are separate workloads.
"""
try:
    from . import transform_classes as contract
    from .run import measure
    from .resize import verify_prepared,summarize
    from .footprints import package_hashes,verify_sdk,sdk_member
except ImportError:
    import transform_classes as contract
    from run import measure
    from resize import verify_prepared,summarize
    from footprints import package_hashes,verify_sdk,sdk_member
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import traceback
import warnings
import zipfile
import numpy as np
import scipy
import skimage
import skimagexx as xx
from skimagexx.coverage import native_classes,class_contract

VERSION='public-transform-class-performance-v1'
CORPUS_SHA256='223f2bbd38d4215d716b38fe0086d6cc7e469f307ff5c1c481b4bc0d0bf9e428'
HARNESS=('transform_class_performance.py','transform_classes.py','footprints.py','resize.py','run.py','provenance.py')


def work_kind(case):
    if case.operation in ('apply','residuals'):return 'public-coordinate-call'
    if case.operation in ('estimate','from_estimate','inverse','compose'):return 'runtime-fitting-or-matrix-provider'
    return 'state-metadata-or-serialization'


def timed_scope(case):
    return {'estimate':'constructor-plus-estimate: fresh model each call, input fixtures excluded; includes reset/construction and state replacement',
            'from_estimate':'public from_estimate including returned model/failure construction; fixed input fixtures excluded',
            'construct':'public constructor plus fresh input matrix copy; immutable matrix fixture excluded',
            'compose':'public composition with both operand models prepared outside timer',
            'pickle':'pickle.dumps plus pickle.loads including bytes and reconstructed model',
            'properties':'public property dictionary assembly and descriptor access; fixed model excluded',
            'array':'public np.asarray conversion; may return a view, not an image kernel'}.get(case.operation,
            'public '+case.operation+' call; immutable inputs/model fixtures excluded, returned allocation and destruction included')


def prepared_action(namespace,case):
    """The original contract action with read-only fixtures and explicit resets.

    No callbacks exclude per-iteration estimator reset: frozen estimate action
    constructs a fresh target inside every call. Composition's right operand is
    prepared once instead of measuring unrelated identity construction.
    """
    call=contract.action(namespace,case)
    captures=inspect.getclosurevars(call).nonlocals
    if case.operation=='compose':
        model=captures['model'];right=getattr(namespace,case.model)()
        right.params.flags.writeable=False
        call=lambda:model+right
    for value in captures.values():
        if isinstance(value,np.ndarray):value.flags.writeable=False
        elif hasattr(type(value),'__array__') and hasattr(value,'params'):
            value.params.flags.writeable=False
    return call


def assert_observation(expected,actual,case):
    value,error,notices=expected;got,caught,actual_notices=actual
    assert notices==actual_notices
    if error is not None:
        if case.nominal_helper_error:assert type(caught).__name__==type(error).__name__ and isinstance(caught,AttributeError)
        else:assert type(caught) is type(error)
        assert contract.normalize(str(caught))==contract.normalize(str(error))
    else:
        if caught is not None:raise caught
        contract.compare(got,value)


def check_case(case,workers,reference_only):
    expected=contract.observe(contract.action(contract.original,case))
    assert (expected[1] is not None)==case.expected_error
    # Also compare prepared output/endstate against the pinned public action's
    # values; read-only fixture flags are an explicit timing-only scope change.
    prepared_expected=contract.observe(prepared_action(contract.original,case))
    if not case.expected_error:
        assert prepared_expected[1] is None
        value=expected[0];prepared=prepared_expected[0]
        if isinstance(value,np.ndarray):np.testing.assert_allclose(prepared,value,rtol=5e-11,atol=5e-11,equal_nan=True)
        elif hasattr(value,'params'):np.testing.assert_allclose(prepared.params,value.params,rtol=5e-11,atol=5e-11,equal_nan=True)
    assert prepared_expected[2]==expected[2]
    if reference_only:return
    for count in workers:
        xx.set_num_threads(count);assert xx.get_num_threads()==count
        assert_observation(expected,contract.observe(contract.action(xx.transform,case)),case)
        assert_observation(prepared_expected,contract.observe(prepared_action(xx.transform,case)),case)


def assert_public_identity():
    from skimagexx import _geometric
    assert _geometric._core is xx._core
    assert set(contract.APIS)<=set(native_classes()),'public native transform classes are not registered'
    for name in contract.NAMES:
        target=getattr(xx.transform,name)
        assert target is getattr(_geometric,name) and target is not getattr(contract.original,name)
        assert not target.__module__.startswith('skimage.')
    assert not any(k.startswith('SKIMAGEXX_') and any(v in k for v in ('CANDIDATE','BACKEND','MODULE_DIR')) for k in os.environ)


def prove_no_upstream_forwarding():
    """Preparation/check-only proof, never inside a measured callable."""
    originals={name:getattr(contract.original,name) for name in contract.NAMES}
    def reject(*args,**kwargs):raise AssertionError('upstream transform class forwarding')
    try:
        for name in originals:setattr(contract.original,name,reject)
        points=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[2.,-.5]])
        for name in contract.NAMES:
            cls=getattr(xx.transform,name);model=cls()
            model(points);model.inverse(points);model.residuals(points,points)
            assert isinstance(model+cls(),cls)
            assert isinstance(cls.from_estimate(points,points+[2.,-1.]),cls)
    finally:
        for name,value in originals.items():setattr(contract.original,name,value)


def seal_artifacts(prepared,wheel,root,corpus_hash):
    with zipfile.ZipFile(wheel) as z:members={n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if not n.endswith('/')}
    package={n[len('skimagexx/'):]:h for n,h in members.items() if n.startswith('skimagexx/')}
    sdk={n:h for n,h in members.items() if sdk_member(n)}
    assert package==package_hashes() and sdk and any(n.endswith(('.lib','.a')) for n in sdk)
    verify_sdk(sdk)
    installed=Path(xx.__file__).resolve().parent.parent
    for n in sdk:
        if n.startswith('include/skimagexx/'):assert (root/n).read_bytes().replace(b'\r\n',b'\n')==(installed/n).read_bytes().replace(b'\r\n',b'\n')
    seal=dict(schema='transform-classes-exact-wheel-sdk-v1',wheel_filename=wheel.name,wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),package_sha256=package,sdk_sha256=sdk,prepared_source_sha256=hashlib.sha256(prepared.read_bytes()).hexdigest(),corpus_sha256=corpus_hash,harness_sha256=harness_hashes(root))
    path=prepared.with_name('transform-classes-artifacts.json');assert not path.exists(),'choose fresh preparation seal'
    path.write_text(json.dumps(seal,indent=2)+'\n');return verify_sealed(prepared,root,corpus_hash)


def harness_hashes(root):return {name:hashlib.sha256((root/'benchmarks'/name).read_bytes()).hexdigest() for name in HARNESS}


def verify_sealed(prepared,root,corpus_hash):
    path=prepared.with_name('transform-classes-artifacts.json');seal=json.loads(path.read_text())
    assert seal['schema']=='transform-classes-exact-wheel-sdk-v1'
    assert seal['prepared_source_sha256']==hashlib.sha256(prepared.read_bytes()).hexdigest()
    name=seal['wheel_filename'];assert Path(name).name==name and '/' not in name and '\\' not in name and name.endswith('.whl')
    assert hashlib.sha256((prepared.parent.parent/'wheels'/name).read_bytes()).hexdigest()==seal['wheel_sha256']
    assert seal['package_sha256']==package_hashes() and seal['corpus_sha256']==corpus_hash
    assert seal['harness_sha256']==harness_hashes(root)
    verify_sdk(seal['sdk_sha256'])
    return dict(seal_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**seal)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only',action='store_true');p.add_argument('--reference-only',action='store_true')
    p.add_argument('--threads',nargs='+',type=int,default=[1]);p.add_argument('--repeat',type=int,default=7)
    p.add_argument('--target-ms',type=float,default=15.);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepared-source',type=Path);p.add_argument('--seal-wheel',type=Path);args=p.parse_args(argv)
    if not args.threads or min(args.threads)<1 or 1 not in args.threads or len(set(args.threads))!=len(args.threads) or args.repeat<1 or not math.isfinite(args.target_ms) or args.target_ms<=0:p.error('positive distinct workers including one, repeats and target-ms required')
    if args.reference_only and not args.check_only:p.error('reference-only requires check-only')
    if args.seal_wheel and (not args.check_only or args.reference_only or args.prepared_source is None):p.error('seal-wheel requires strict check-only and prepared-source')
    if not args.check_only and (args.prepared_source is None or os.environ.get('GITHUB_ACTIONS')!='true'):p.error('timings require isolated artifact-fed Actions; use check-only locally')
    root=Path(__file__).resolve().parents[1];cases=list(contract.corpus());specs=[asdict(c) for c in cases]
    corpus_hash=hashlib.sha256(json.dumps(specs,sort_keys=True).encode()).hexdigest()
    assert corpus_hash==CORPUS_SHA256,'pinned296 corpus changed; explicit contract version required'
    report=dict(status='preflight',metadata=dict(version=VERSION,timestamp_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,skimage=skimage.__version__,**contract.cpu_metadata(),**contract.source_fingerprint(root),source_revision=contract.source_revision(root),binary_sha256=contract.binary_hashes(),python_sha256=contract.python_hashes(),harness_sha256=harness_hashes(root),corpus_sha256=corpus_hash,case_count=len(cases),case_specs=specs,requested_workers=args.threads,primary_workers=1,check_only=args.check_only,reference_only=args.reference_only,provider_scope='Own C++ scalar/runtime AVX2 coordinates; NumPy fitting/SVD/normalization/inversion/composition and policy replay; SciPy3D Euler provider. No pure-C++ fitting claim.',provider_thread_environment={n:os.environ.get(n) for n in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS','OMP_WAIT_POLICY')},timing_scope='Warmed allocation/destruction inclusive public calls. Immutable fixtures/models excluded; constructor-plus-estimate includes fresh state reset each iteration. Warnings checked then suppressed symmetrically.',github_run_id=os.environ.get('GITHUB_RUN_ID'),github_run_attempt=os.environ.get('GITHUB_RUN_ATTEMPT'),github_sha=os.environ.get('GITHUB_SHA'),github_job=os.environ.get('GITHUB_JOB')),checks=[],untimed_cases=[],results=[],failures=[])
    workers=xx.get_num_threads();previous=xx.set_fallback_enabled(False)
    try:
        assert skimage.__version__=='0.26.0'
        if not args.reference_only:assert_public_identity();prove_no_upstream_forwarding()
        if args.prepared_source:
            report['metadata']['prepared_source']=verify_prepared(root,args.prepared_source)
            if args.seal_wheel:report['metadata']['exact_artifacts']=seal_artifacts(args.prepared_source,args.seal_wheel,root,corpus_hash)
            elif not args.check_only:report['metadata']['exact_artifacts']=verify_sealed(args.prepared_source,root,corpus_hash)
        for index,case in enumerate(cases):
            row=dict(case_index=index,**asdict(case),work_kind=work_kind(case),timed_scope=timed_scope(case))
            try:check_case(case,args.threads,args.reference_only);row['status']='passed'
            except Exception as caught:row.update(status='failed',message=str(caught),traceback=traceback.format_exc());report['failures'].append(row)
            report['checks'].append(row)
            if case.expected_error:report['untimed_cases'].append(dict(case_index=index,reason='expected error; retained untimed'))
            contract.save(report,args.output)
        if report['failures']:report['status']='failed-parity';return 1
        if args.check_only:report['status']='complete-reference-check' if args.reference_only else 'complete-parity-check';return 0
        assert all(os.environ.get(n)=='1' for n in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')),'external provider pools must be constrained independently'
        report['status']='measuring'
        rng=random.Random(81);orders=[]
        for _ in range(args.repeat):
            order=['scikit-image','current'];rng.shuffle(order);orders.append(order)
        report['metadata']['paired_sample_order']=orders
        for index,case in enumerate(cases):
            if case.expected_error:continue
            ref=prepared_action(contract.original,case);native=prepared_action(xx.transform,case)
            for count in args.threads:
                xx.set_num_threads(count)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore');timing=measure(ref,native,args.repeat,args.target_ms)
                report['results'].append(dict(case_index=index,model=case.model,operation=case.operation,label=case.label,work_kind=work_kind(case),timed_scope=timed_scope(case),threads=count,primary=count==1,image_kernel_speedup_claim=False,slower_than_scikit_image=timing['speedup']<1,**timing));contract.save(report,args.output)
        assert contract.source_fingerprint(root)=={key:report['metadata'][key] for key in contract.source_fingerprint(root)}
        assert harness_hashes(root)==report['metadata']['harness_sha256']
        assert contract.binary_hashes()==report['metadata']['binary_sha256'] and contract.python_hashes()==report['metadata']['python_sha256']
        assert verify_sealed(args.prepared_source,root,corpus_hash)==report['metadata']['exact_artifacts']
        report['summary_by_work_kind']={kind:summarize([r for r in report['results'] if r['work_kind']==kind],args.threads,sum(not c.expected_error and work_kind(c)==kind for c in cases)) for kind in sorted({work_kind(c) for c in cases})}
        report['status']='complete-measurement';return 0
    except Exception as caught:
        report['failures'].append(dict(phase=report['status'],message=str(caught),traceback=traceback.format_exc()));report['status']='failed';return 1
    finally:xx.set_num_threads(workers);xx.set_fallback_enabled(previous);contract.save(report,args.output)


if __name__=='__main__':raise SystemExit(main())
