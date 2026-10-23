"""Fixed untimed public class corpus; this tool never measures performance.

Coordinate, runtime fitting/matrix-provider and state/metadata operations are
classified separately. A future isolated measurement packet can reuse these
cases; neither a passing corpus nor this file's existence establishes speedup.
"""
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import re
import sys
import traceback
import warnings
import numpy as np
import scipy
import skimage
from skimage import transform as original
import skimagexx as xx
from skimagexx.coverage import native_classes, class_contract
try:
    from .provenance import source_fingerprint
    from .resize import binary_hashes,python_hashes,cpu_metadata,source_revision,save
except ImportError:
    from provenance import source_fingerprint
    from resize import binary_hashes,python_hashes,cpu_metadata,source_revision,save

NAMES=('ProjectiveTransform','AffineTransform','EuclideanTransform','SimilarityTransform')
APIS=tuple('transform.'+name for name in NAMES)
CORPUS_VERSION='transform-classes-contract-v1'


@dataclass(frozen=True)
class Case:
    model: str
    operation: str
    label: str
    dimensions: int=2
    count: int=7
    dtype: str='float64'
    layout: str='c'
    expected_error: bool=False
    nominal_helper_error: bool=False


def corpus():
    for name in NAMES:
        for dimensions in [2,3,4]:
            for dtype in ['float32','float64']:
                for layout in ['c','f','reverse','broadcast']:
                    yield Case(name,'apply','small',dimensions,19,dtype,layout)
        for dimensions in [2,3]:
            for dtype in ['float32','float64']:
                for count in [257,131073]:
                    for layout in ['c','f','reverse']:
                        yield Case(name,'apply','large',dimensions,count,dtype,layout)
        for count in [0,1]:yield Case(name,'apply','tiny',count=count)
        for operation in ['construct','inverse','compose','residuals','properties','array','pickle']:
            yield Case(name,operation,'nominal')
        for operation in ['estimate','from_estimate']:
            for dtype in ['float32','float64']:
                for label in ['fit','constant','collinear']:
                    yield Case(name,operation,label,dtype=dtype)
        for operation in ['bad-matrix','bad-coordinates','singular-inverse','failed-pickle']:
            yield Case(name,operation,'expected-error',expected_error=True)
        yield Case(name,'failed-access','nominal-private-helper-error',expected_error=True,nominal_helper_error=True)


def points(case):
    values=np.linspace(-1.5,2.,case.count*case.dimensions,dtype=np.float64).reshape(case.count,case.dimensions).astype(case.dtype)
    if case.layout=='f':values=np.asfortranarray(values)
    elif case.layout=='reverse':values=values[::-1,::-1]
    elif case.layout=='broadcast':values=np.broadcast_to(values[:1],values.shape)
    return values


def action(namespace,case):
    cls=getattr(namespace,case.model);d=case.dimensions
    matrix=np.eye(d+1);matrix[0,0]=1.1;matrix[0,-1]=.3
    if case.model=='ProjectiveTransform':matrix[-1,0]=.01
    model=cls(matrix);src=points(case);dst=src+.25
    op=case.operation
    if op=='construct':return lambda:cls(matrix.copy())
    if op=='apply':return lambda:model(src)
    if op=='residuals':return lambda:model.residuals(src,dst)
    if op=='inverse':return lambda:model.inverse
    if op=='compose':return lambda:model+cls()
    if op=='array':return lambda:np.asarray(model)
    if op=='pickle':return lambda:pickle.loads(pickle.dumps(model))
    if op=='properties':return lambda:{name:getattr(model,name) for name in ['dimensionality','rotation','translation','scale','shear'] if hasattr(model,name)}
    if op in ('estimate','from_estimate'):
        # Independent nondegenerate coordinates for stable fitting coverage.
        src=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[2.,-.5],[-1.,2.],[.3,.7]],dtype=case.dtype);dst=src+[.25,-.5]
        if case.label=='constant':src=np.zeros_like(src);dst=np.ones_like(dst)
        elif case.label=='collinear':src[:,1]=0;dst=src+[.25,-.5]
        if op=='from_estimate':return lambda:cls.from_estimate(src,dst)
        def estimate():
            target=cls();success=target.estimate(src,dst);return success,target
        return estimate
    if op=='bad-matrix':return lambda:cls(np.zeros((2,3)))
    if op=='bad-coordinates':return lambda:model(np.zeros((3,3)))
    if op=='singular-inverse':return lambda:cls(np.zeros((3,3))).inverse
    failed=cls.from_estimate(np.zeros((5,2)),np.ones((5,2)))
    if op=='failed-pickle':return lambda:pickle.loads(pickle.dumps(failed))
    if op=='failed-access':return lambda:failed.params
    raise AssertionError(op)


def observe(call):
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:value,error=call(),None
        except Exception as caught:value,error=None,caught
    return value,error,[(type(w.message).__name__,str(w.message)) for w in notices]


def normalize(message):
    for name in NAMES:message=re.sub(r'(?:skimage|skimagexx)[A-Za-z0-9_.]*\.'+name,name,message)
    return re.sub(r'0x[0-9a-fA-F]+','0xADDRESS',message)


def compare(actual,expected):
    if isinstance(expected,np.ndarray):
        assert isinstance(actual,np.ndarray) and actual.dtype==expected.dtype and actual.shape==expected.shape
        assert (actual.strides,actual.flags.owndata,actual.flags.writeable)==(expected.strides,expected.flags.owndata,expected.flags.writeable)
        np.testing.assert_allclose(actual,expected,rtol=5e-11,atol=5e-11,equal_nan=True)
    elif type(expected).__name__=='FailedEstimation':
        assert type(actual).__name__=='FailedEstimation' and not actual and actual.message==expected.message
    elif hasattr(expected,'params'):
        assert type(actual).__name__==type(expected).__name__;compare(actual.params,expected.params)
    elif isinstance(expected,dict):
        assert set(actual)==set(expected)
        for key in expected:compare(actual[key],expected[key])
    elif isinstance(expected,tuple):
        assert type(actual) is tuple and len(actual)==len(expected)
        for a,e in zip(actual,expected):compare(a,e)
    else:
        assert type(actual) is type(expected)
        if isinstance(expected,(float,np.floating)):np.testing.assert_allclose(actual,expected,rtol=5e-11,atol=5e-11,equal_nan=True)
        else:assert actual==expected


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true',required=True)
    parser.add_argument('--reference-only',action='store_true')
    parser.add_argument('--threads',type=int,nargs='+',default=[1])
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args(argv)
    if not args.threads or 1 not in args.threads or min(args.threads)<1 or len(set(args.threads))!=len(args.threads):parser.error('include one; positive distinct workers required')
    root=Path(__file__).resolve().parents[1];cases=list(corpus());specs=[asdict(c) for c in cases]
    kinds={operation:('cpp-coordinate-public-call' if operation in ('apply','residuals') else 'runtime-fit-or-matrix-provider' if operation in ('estimate','from_estimate','inverse','compose') else 'state-metadata-or-serialization') for operation in {c.operation for c in cases}}
    report=dict(status='preflight',metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),**cpu_metadata(),numpy=np.__version__,scipy=scipy.__version__,skimage=skimage.__version__,source_revision=source_revision(root),**source_fingerprint(root),binary_sha256=binary_hashes(),python_sha256=python_hashes(),corpus_version=CORPUS_VERSION,case_count=len(cases),corpus_sha256=hashlib.sha256(json.dumps(specs,sort_keys=True).encode()).hexdigest(),case_specs=specs,operation_kinds=kinds,workers=args.threads,reference_only=args.reference_only,check_only=True,measurement_scope='none; this tool has no timers',providers='Own C++ coordinates; NumPy/SciPy fitting/matrix providers, external BLAS/OMP pools must be constrained separately'),checks=[],failures=[],results=[])
    report['metadata']['harness_sha256']={name:hashlib.sha256((root/'benchmarks'/name).read_bytes()).hexdigest() for name in ('transform_classes.py','resize.py','provenance.py')}
    report['metadata']['provider_thread_environment']={name:os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS','OMP_WAIT_POLICY')}
    previous=xx.set_fallback_enabled(False);workers=xx.get_num_threads()
    try:
        assert skimage.__version__=='0.26.0'
        if not args.reference_only:
            missing=set(APIS)-set(native_classes())
            if missing:raise RuntimeError('public classes not registered: '+', '.join(sorted(missing)))
            report['metadata']['class_contracts']={api:class_contract(api) for api in APIS}
        for index,case in enumerate(cases):
            row=dict(index=index,**asdict(case))
            try:
                expected,error,notices=observe(action(original,case));assert (error is not None)==case.expected_error
                row['warnings']=notices
                if error is not None:row['expected_exception']=dict(type=type(error).__name__,message=normalize(str(error)),nominal_helper=case.nominal_helper_error)
                if not args.reference_only:
                    for count in args.threads:
                        xx.set_num_threads(count);assert xx.get_num_threads()==count
                        actual,ae,an=observe(action(xx.transform,case));assert an==notices
                        if error is not None:
                            if case.nominal_helper_error:assert type(ae).__name__==type(error).__name__ and isinstance(ae,AttributeError)
                            else:assert type(ae) is type(error)
                            assert normalize(str(ae))==normalize(str(error))
                        else:
                            if ae is not None:raise ae
                            compare(actual,expected)
                row['status']='passed'
            except Exception as caught:
                row.update(status='failed',message=str(caught),traceback=traceback.format_exc());report['failures'].append(row)
            report['checks'].append(row);save(report,args.output)
        report['status']='failed-parity' if report['failures'] else 'complete-reference-check' if args.reference_only else 'complete-public-class-check'
        return int(bool(report['failures']))
    except Exception as caught:
        report.update(status='failed');report['failures'].append(dict(phase='preflight',message=str(caught),traceback=traceback.format_exc()));return 1
    finally:
        xx.set_num_threads(workers);xx.set_fallback_enabled(previous);save(report,args.output)


if __name__=='__main__':raise SystemExit(main())
