"""IGG-specific ownership/preflight over admitted MARK validators and sampler entry.

No measurement engine: actual sampling and mandatory entry-stop are reused.
The exact common source is a pending selective-integration dependency.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import warnings
ROOT=Path(__file__).resolve().parents[1]
CONTEXT='coverage/contracts/inverse-gaussian-gradient-benchmark-execution-v4.json'
digest=lambda raw:hashlib.sha256(raw).hexdigest()
source_bytes=lambda p:Path(p).read_bytes().replace(b'\r\n',b'\n')


def boundary(root=ROOT,dependency_root=None):
    context=json.loads((root/CONTEXT).read_bytes());path=(Path(dependency_root)if dependency_root is not None else root)/'benchmarks/mark_boundaries_execution.py'
    assert path.is_file(),'Pending exact admitted MARK common dependency selective integration'
    assert digest(source_bytes(path))==context['common_boundary_canonical_sha256']
    spec=importlib.util.spec_from_file_location('_igg_admitted_boundary',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def require_sdk_dependencies(binding):
    assert {'include/skimagexx/inverse_gaussian_gradient.hpp','include/skimagexx/crop.hpp'}<=set(binding['SDK_members']),'Complete IGG/crop SDK header dependency required'


@contextmanager
def candidate_providers(np):
    # Only the non-timed preflight is instrumented. NumPy sqrt is permitted
    # solely for the explicitly declared scalar policy probe, never pixels.
    sqrt=np.sqrt
    def scalar_only(value,*a,**k):
        assert np.asarray(value).size<=1,'IGG image sqrt provider forbidden'
        return sqrt(value,*a,**k)
    try:np.sqrt=scalar_only;yield
    finally:np.sqrt=sqrt


def execute(module):
    context=json.loads((ROOT/CONTEXT).read_bytes());api=context['route']['api'];specs=module['case_specs']()
    assert specs==context['cases']and len(specs)==context['case_count']==8
    assert all(module['case_bounds'](s)for s in specs)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',action='store_true');parser.add_argument('--root-binding',type=Path)
    parser.add_argument('--first-cohort',action='store_true');parser.add_argument('--measurement-go',type=Path)
    parser.add_argument('--api',choices=[api],default=api);parser.add_argument('--case')
    parser.add_argument('--threads',type=int,choices=[1],default=1);parser.add_argument('--repeat',type=int,choices=[7],default=7)
    parser.add_argument('--output',type=Path,default=Path(context['default_output']));cli=parser.parse_args()
    if cli.plan:print(json.dumps(module['plan'](),indent=2));return
    if cli.root_binding is None:raise SystemExit('IGG_UNBOUND: genuine source/core/SDK/runtime/helper/root authority required before imports or bodies')
    shared=boundary(ROOT);b,context,go,binding_pin,go_pin=shared.validate_binding(cli.root_binding,root=ROOT,context_path=CONTEXT)
    require_sdk_dependencies(b)
    assert not(cli.first_cohort and cli.measurement_go)
    if cli.first_cohort:
        assert cli.case==specs[0]['id'];specs=specs[:1];measurement_pin=None;deadline=go['deadline_utc']
    else:
        assert cli.case is None,'All eight workloads and every loss must be retained'
        measurement_pin,deadline=shared.validate_measurement(cli.measurement_go,b,context,binding_pin,context_path=CONTEXT)
    assert not cli.output.exists()and not cli.output.with_suffix('.batches.jsonl').exists()
    output_root=Path(b['output_root']);assert output_root.is_absolute()and cli.output.resolve().is_relative_to(output_root.resolve())
    output_root.mkdir(parents=True,exist_ok=True)
    claim=output_root/('igg-first-cohort.claim.json'if cli.first_cohort else'igg-measurement.claim.json')
    with claim.open('x',encoding='utf-8')as stream:
        json.dump(dict(status='STARTED_NO_AUTOMATIC_RETRY',binding_sha256=binding_pin,cases=[s['id']for s in specs],measurement_sha256=measurement_pin),stream);stream.flush();os.fsync(stream.fileno())
    common=shared.support(ROOT,context);oldpath=list(sys.path);xx=None;old_fallback=None
    try:
        sys.path[:0]=[b['installed_site'],str(ROOT/'tools'),str(ROOT/'benchmarks')]
        from montage_sustained import measure
        from provenance import source_fingerprint
        assert digest(inspect.getsource(measure).encode())==context['method']['method_sha256']
        import numpy as np
        import skimage
        import skimagexx as xx
        assert np.__version__==b['numpy']and skimage.__version__=='0.26.0'and all(os.environ.get(n)=='1'for n in shared.LIMITS)
        core=Path(xx._core.__file__).resolve();assert xx._core.__name__=='skimagexx._core'and core==Path(b['core_path']).resolve()
        assert core.is_relative_to(Path(b['installed_site']).resolve())and digest(core.read_bytes())==b['core_sha256']
        upstream=importlib.import_module(context['original_module']);owned=importlib.import_module(context['route']['owned_module'])
        reference=skimage.segmentation.inverse_gaussian_gradient;native=xx.segmentation.inverse_gaussian_gradient
        assert reference is upstream.inverse_gaussian_gradient and native is owned.inverse_gaussian_gradient and owned._core is xx._core
        assert str(inspect.signature(reference))==str(inspect.signature(native))==context['signature']
        assert digest(source_bytes(upstream.__file__))==context['original_source_canonical_sha256']
        assert Path(owned.__file__).resolve().is_relative_to(Path(b['installed_site']).resolve())and digest(source_bytes(owned.__file__))==context['source_files_canonical_sha256'][context['owned_source']]
        fallback=importlib.import_module('skimagexx._compat');old_fallback=fallback._fallback_enabled
        xx.set_fallback_enabled(False);assert xx.get_num_threads()==1
        save,restore,flags=common.fp_controller(Path(b['source_root']),context)
        source_before=source_fingerprint(Path(b['source_root']));policy=np.geterr(),np.geterrcall(),np.getbufsize();rows=[]
        def hardware():
            raw=bytes(save());return dict(control=int.from_bytes(raw[:2],'little'),status=int.from_bytes(raw[4:6],'little'),mxcsr=int.from_bytes(raw[28:32],'little'))
        for index,spec in enumerate(specs):
            assert datetime.now(timezone.utc)<datetime.fromisoformat(deadline)
            args,options=module['fixture'](spec,np);before=[common.array_state(a,np)for a in args]
            requirements=module['native_requirements'](spec);entries={n:getattr(xx._core,n)for n in requirements}
            assert all(inspect.isbuiltin(e)and e.__module__=='skimagexx._core'for e in entries.values())
            returns=[];accepted=[];prior=sys.getprofile();caller=save()
            def profile(frame,event,value):
                if event=='call'and(str(frame.f_globals.get('__name__','')).startswith(('skimage.','scipy.ndimage'))or frame.f_code is fallback.fallback.__code__):raise AssertionError('Candidate original/helper body forbidden')
                if event=='c_call'and str(getattr(value,'__module__','')).startswith(('skimage.','scipy.ndimage')):raise AssertionError('Candidate compiled original/helper forbidden')
                if event=='c_return':
                    for key,entry in entries.items():
                        if value is entry:returns.append(key)
                if event=='return'and frame.f_code is native.__code__:accepted.append(value is not None)
            try:
                restore(caller);reference_before=hardware()
                with warnings.catch_warnings(record=True)as reference_notices:
                    warnings.simplefilter('always');expected=reference(*args,**options);reference_FP=hardware()
                assert [common.array_state(a,np)for a in args]==before and (np.geterr(),np.geterrcall(),np.getbufsize())==policy and xx.get_num_threads()==1
                restore(caller);native_before=hardware()
                with candidate_providers(np),warnings.catch_warnings(record=True)as native_notices:
                    warnings.simplefilter('always')
                    try:sys.setprofile(profile);actual=native(*args,**options);native_FP=hardware()
                    finally:sys.setprofile(prior)
                assert reference_before==native_before and reference_FP==native_FP
                assert not reference_notices and not native_notices and accepted==[True]
                assert {k:returns.count(k)for k in requirements}==requirements and len(returns)==sum(requirements.values())
                assert type(expected)is type(actual)is np.ndarray and expected.shape==actual.shape==args[0].shape
                assert expected.dtype==actual.dtype and expected.tobytes()==actual.tobytes()
                assert common.array_state(expected,np,args[0])==common.array_state(actual,np,args[0])
                assert actual.flags.owndata and not any(np.shares_memory(actual,a)for a in args)
                assert [common.array_state(a,np)for a in args]==before and (np.geterr(),np.geterrcall(),np.getbufsize())==policy and xx.get_num_threads()==1
            finally:sys.setprofile(prior);restore(caller)
            sample_state=save()
            try:timing=shared.sample_entry(measure,lambda:reference(*args,**options),lambda:native(*args,**options),cli,index,spec,deadline,common)
            finally:restore(sample_state)
            assert [common.array_state(a,np)for a in args]==before and (np.geterr(),np.geterrcall(),np.getbufsize())==policy and xx.get_num_threads()==1
            rows.append(dict(case_id=spec['id'],api=api,actual_C_returns=returns,accepted=accepted,FP=native_FP,providers=context['providers'],**timing))
        assert source_fingerprint(Path(b['source_root']))==source_before and digest(core.read_bytes())==b['core_sha256']
        assert digest(cli.root_binding.read_bytes())==binding_pin and digest(Path(b['root_go_path']).read_bytes())==go_pin and digest((ROOT/CONTEXT).read_bytes())==b['base_contract_sha256']
        if cli.measurement_go:assert digest(cli.measurement_go.read_bytes())==measurement_pin
        with cli.output.open('x',encoding='utf-8')as stream:json.dump(dict(metadata=dict(source_before=source_before,source_revision=b['source_revision'],producer=b['producer'],runtime_sha256=b['runtime_sha256'],core_sha256=b['core_sha256'],SDK_members=b['SDK_members'],root_binding_sha256=binding_pin,measurement_sha256=measurement_pin,contract_sha256=b['base_contract_sha256'],method_sha256=context['method']['method_sha256'],threads=1,repeat=7,target_ms=100,warmup_min_ms=250,pass_index=0,retain_all_losses=True),results=rows),stream,indent=2)
    finally:
        if old_fallback is not None:xx.set_fallback_enabled(old_fallback)
        sys.path[:]=oldpath
