"""Eight fresh public pairs; current log2/matmul, natural caller FP, no timing."""
import ctypes
import importlib
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import warnings
import pytest

ROOT=Path(__file__).resolve().parents[1]


def call_once(call,args,options,env,guard,outputs,role,diagnostics):
    prior=sys.getprofile();primary=None
    try:
        sys.setprofile(guard);before=env()
        try:
            value=call(*args,**options);outputs[role]=value
        except BaseException as error:primary=error;raise
        finally:
            try:after=env()
            except BaseException as error:
                diagnostics.append(dict(role=role,stage='post-target-FP',type=type(error).__name__,error=str(error)))
                if primary is None:raise
        return value,before,after
    finally:sys.setprofile(prior)


def ownership_profile(reference,native,prefix,fallback,entries,phase,counts,returned,accepted,suffix):
    def guard(frame,event,value):
        if event=='call':
            if frame.f_code is reference.__code__:
                assert phase[0]=='reference','Original call outside reference phase';counts['original']+=1
            if frame.f_code is native.__code__:
                assert phase[0]=='native';counts['native']+=1
            if phase[0]=='native':
                assert not str(frame.f_globals.get('__name__','')).startswith(('skimage.','scipy.ndimage')),'Original/helper body in native phase'
                assert frame.f_code is not fallback.__code__,'Fallback in native phase'
        if phase[0]=='native':
            if event=='c_call':assert not str(getattr(value,'__module__','')).startswith(('skimage.','scipy.ndimage'))
            if event=='c_return':
                for name,entry in entries.items():
                    if value is entry:returned.append(name)
            if event=='return'and frame.f_code is native.__code__:accepted.append(value is not None)
            if event=='return'and frame.f_code is prefix.__code__:suffix.append(frame.f_locals.get('native_result')is not None)
    return guard


def retain_record(path,record,outputs,encode):
    # Raw words are captured independently before any optional output metadata.
    for role,value in outputs.items():
        if role not in record:
            try:record[role]=encode(value,output=True)
            except BaseException as error:record[role]=dict(unavailable=type(error).__name__,error=str(error))
    with Path(path).open('x',encoding='utf-8',newline='\n')as stream:
        stream.write(json.dumps(record,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())


@pytest.mark.parametrize('index',range(4))
@pytest.mark.parametrize('workers',(1,4))
def test_current_public_large_original_pair(index,workers):
    directory=Path(os.environ['VI_PUBLIC_LOG2_LARGE_RECORD_DIR']);assert directory.is_absolute()
    directory.mkdir(parents=True,exist_ok=True);path=directory/(str(index)+'-workers'+str(workers)+'.json');assert not path.exists()
    assert not os.environ.get('VI_COMPONENT_DIR'),'Current PUBLIC package required'
    from label_pair_metrics_observer import observation,reference_runtime
    from util_views_fenv import FEnv,backend
    np,functions,runtime=reference_runtime()
    import scipy.sparse as sparse
    import skimage
    import skimagexx as xx
    owned=importlib.import_module('skimagexx._variation_of_information');compat=importlib.import_module('skimagexx._compat')
    reference=functions['metrics.variation_of_information'];native=xx.metrics.variation_of_information;core=xx._core
    assert reference is skimage.metrics.variation_of_information is importlib.import_module('skimage.metrics._variation_of_information').variation_of_information
    assert native is owned.variation_of_information and owned._core is core
    package=Path(xx.__file__).resolve().parent;assert core.__name__=='skimagexx._core'
    assert Path(core.__file__).resolve().is_relative_to(package)and Path(owned.__file__).resolve().is_relative_to(package)
    assert str(inspect.signature(native))==str(inspect.signature(reference))=='(image0=None, image1=None, *, table=None, ignore_labels=())'
    assert isinstance(np.log2,np.ufunc)and isinstance(np.matmul,np.ufunc)
    true_provider=np.log2;true_matmul=np.matmul
    libc,helper=backend()
    def env():
        state=FEnv();assert libc.fegetenv(ctypes.byref(state))==0;return state
    def restore(state):
        assert helper.reference_fenv_restore(ctypes.byref(state),32)==0
        current=env();assert bytes(current)==bytes(state);return current
    def semantic(state):return {k:getattr(state,k)for k in('control','status','mxcsr')}
    caller=env();policy=np.geterr(),np.geterrcall(),np.getbufsize();initial_workers=core.get_num_threads();assert initial_workers==1
    initial_fallback=compat._fallback_enabled;prior=sys.getprofile();primary=None;outputs={};diagnostics=[];record=dict(index=index,workers=workers,runtime=runtime,timing=False,primary=None,diagnostics=diagnostics)
    record.update(public_signature=str(inspect.signature(native)),native_core_path=str(Path(core.__file__).resolve()),native_frontend_path=str(Path(owned.__file__).resolve()),provider_type=type(true_provider).__name__,software_context=dict(policy=policy[0],handler_type=type(policy[1]).__name__,handler_identity=id(policy[1]),buffer=policy[2]))
    encode=runpy.run_path(str(ROOT/'tools/vi_cohort_value_diagnostic.py'))['array_words']
    try:
        recipes=runpy.run_path(str(ROOT/'benchmarks/variation_of_information_cases_v1.py'))
        spec=recipes['case_specs']()[index];recipes['case_bounds'](spec);record['case']=spec
        args,options=recipes['fixture'](spec,np);buffers=dict(truth=args[0],test=args[1])
        before=observation((args,options,buffers),np);pointers={k:v.ctypes.data for k,v in buffers.items()}
        requirements=recipes['native_requirements'](spec);assert sum(requirements.values())==11
        entries={name:getattr(core,name)for name in requirements}
        assert all(inspect.isbuiltin(v)and v.__module__==core.__name__ for v in entries.values())
        phase=['setup'];counts=dict(original=0,native=0);returned=[];accepted=[];suffix=[]
        guard=ownership_profile(reference,native,owned.contingency_table,compat.fallback,entries,phase,counts,returned,accepted,suffix)
        record.update(counts=counts,returned=returned,requirements=requirements,accepted=accepted,suffix=suffix)
        providers=runpy.run_path(str(ROOT/'benchmarks/vi_public_matmul_provider_execution.py'))['candidate_providers']
        core.set_num_threads(workers);xx.set_fallback_enabled(False)
        def pipeline(backend):return backend.metrics.variation_of_information(*args,**options)
        natural=env();record['natural_caller_raw32']=bytes(natural).hex()
        restore(natural);phase[0]='reference';record['phase']='reference'
        with warnings.catch_warnings(record=True)as reference_notices:
            warnings.simplefilter('always')
            expected,reference_before,reference_after=call_once(lambda:pipeline(skimage),(),{},env,guard,outputs,'expected',diagnostics)
        record['reference_FP']=dict(before=semantic(reference_before),after=semantic(reference_after),before_raw32=bytes(reference_before).hex(),after_raw32=bytes(reference_after).hex())
        record['expected']=encode(expected,output=True)
        expected_metadata=observation(expected,np)
        assert observation((args,options,buffers),np)==before
        assert np.geterr()==policy[0]and np.geterrcall()is policy[1]and np.getbufsize()==policy[2]
        restore(natural);phase[0]='native';record['phase']='native'
        with providers(np,sparse)as batches,warnings.catch_warnings(record=True)as native_notices:
            record['provider_batches']=batches
            warnings.simplefilter('always')
            actual,native_before,native_after=call_once(lambda:pipeline(xx),(),{},env,guard,outputs,'actual',diagnostics)
            record['native_FP']=dict(before=semantic(native_before),after=semantic(native_after),before_raw32=bytes(native_before).hex(),after_raw32=bytes(native_after).hex())
            record['actual']=encode(actual,output=True)
        record.update(counts=counts,returned=returned,requirements=requirements,accepted=accepted,suffix=suffix,provider_batches=batches,provider='current true numpy.log2 and numpy.matmul ufunc computation',provider_types=dict(log2=type(true_provider).__name__,matmul=type(true_matmul).__name__))
        assert type(expected)is type(actual)is np.ndarray and expected.shape==actual.shape==(2,)
        assert expected.dtype==actual.dtype==np.dtype('float64')and expected.tobytes()==actual.tobytes()
        assert observation(actual,np)==expected_metadata
        assert semantic(reference_before)==semantic(native_before)and semantic(reference_after)==semantic(native_after)
        assert semantic(reference_before)==semantic(natural)and semantic(native_before)==semantic(natural)
        assert not reference_notices and not native_notices
        assert counts==dict(original=1,native=1)and accepted==suffix==[True]
        assert {k:returned.count(k)for k in requirements}==requirements and len(returned)==11
        assert [row['provider']for row in batches]==['numpy.log2','numpy.matmul','numpy.log2','numpy.matmul']
        assert len(batches)==4 and np.log2 is true_provider and np.matmul is true_matmul
        assert isinstance(np.log2,np.ufunc)and isinstance(np.matmul,np.ufunc)
        assert expected.flags.owndata and actual.flags.owndata and not np.shares_memory(expected,actual)
        assert not any(np.shares_memory(value,a)for value in(expected,actual)for a in args)
        assert pointers=={k:v.ctypes.data for k,v in buffers.items()}and observation((args,options,buffers),np)==before
        assert np.geterr()==policy[0]and np.geterrcall()is policy[1]and np.getbufsize()==policy[2]
        assert core.get_num_threads()==workers and xx.metrics.variation_of_information is native and skimage.metrics.variation_of_information is reference
        record.update(input_before=before,input_after=observation((args,options,buffers),np),pointers_before=pointers,pointers_after={k:v.ctypes.data for k,v in buffers.items()},pointers_stable=True,output_metadata=expected_metadata,output_input_aliases=False,output_pair_alias=False,software_context_unchanged=True,namespace_identity_unchanged=True)
    except BaseException as error:
        primary=error;record['primary']=dict(type=type(error).__name__,module=type(error).__module__,error=str(error));raise
    finally:
        sys.setprofile(prior)
        try:
            core.set_num_threads(initial_workers);xx.set_fallback_enabled(initial_fallback)
            np.seterr(**policy[0]);np.seterrcall(policy[1]);np.setbufsize(policy[2])
            restored=restore(caller)
            record['caller32']=dict(saved=bytes(caller).hex(),actual=bytes(restored).hex(),exact=True)
            assert core.get_num_threads()==initial_workers and compat._fallback_enabled==initial_fallback
            try:retain_record(path,record,outputs,encode)
            except BaseException as error:
                if primary is None:raise
                sys.stderr.write('VI_LARGE_DIAGNOSTIC_SECONDARY '+str(error)+'\n')
        except BaseException as error:
            if primary is None:primary=error;raise
            sys.stderr.write('VI_LARGE_CLEANUP_SECONDARY '+str(error)+'\n')
        finally:
            try:
                restore(caller)
                assert np.geterr()==policy[0]and np.geterrcall()is policy[1]and np.getbufsize()==policy[2]
                assert core.get_num_threads()==initial_workers and compat._fallback_enabled==initial_fallback
                assert xx.metrics.variation_of_information is native and skimage.metrics.variation_of_information is reference and np.log2 is true_provider and np.matmul is true_matmul
            except BaseException as error:
                if primary is None:raise
                sys.stderr.write('VI_LARGE_RESTORE_SECONDARY '+str(error)+'\n')
