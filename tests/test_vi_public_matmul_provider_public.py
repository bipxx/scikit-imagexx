"""Fresh current-runtime pairs for all eight previously assessed VI recipes."""
from contextlib import nullcontext
import ctypes
import importlib
import inspect
import os
from pathlib import Path
import runpy
import sys
import pytest
from test_variation_of_information_public import target,workers,SPECS
from test_vi_public_log2_provider_large import call_once,ownership_profile,retain_record

ROOT=Path(__file__).resolve().parents[1]


def retain_target_FP(record,snapshots):
    # Encode only states already captured around the targets. No new FP read.
    record['target_FP']={role:[dict(semantic={key:getattr(value,key)for key in('control','status','mxcsr')},
        raw32=bytes(value).hex())for value in states]for role,states in snapshots.items()}


@pytest.mark.parametrize('spec',SPECS,ids=lambda s:s['id'])
def test_current_public_contract_original_pair(target,workers,spec):
    from label_pair_metrics_observer import observe,reference_runtime
    from util_views_fenv import FEnv,backend
    candidate_providers=runpy.run_path(str(ROOT/'benchmarks/vi_public_matmul_provider_execution.py'))['candidate_providers']
    array_words=runpy.run_path(str(ROOT/'tools/vi_cohort_value_diagnostic.py'))['array_words']
    np,functions,runtime=reference_runtime()
    import scipy.sparse as sparse
    import skimage
    import skimagexx as xx
    owned=importlib.import_module('skimagexx._variation_of_information')
    compat=importlib.import_module('skimagexx._compat')
    reference=functions['metrics.variation_of_information'];native=target[0].variation_of_information;core=target[1]
    assert native is xx.metrics.variation_of_information is owned.variation_of_information
    assert reference is skimage.metrics.variation_of_information and owned._core is core is xx._core
    assert str(inspect.signature(native))==str(inspect.signature(reference))=='(image0=None, image1=None, *, table=None, ignore_labels=())'
    assert isinstance(np.log2,np.ufunc)and isinstance(np.matmul,np.ufunc)
    providers=np.log2,np.matmul;policy=np.geterr(),np.geterrcall(),np.getbufsize()
    directory=Path(os.environ['VI_PUBLIC_MATMUL_SMALL_RECORD_DIR']);assert directory.is_absolute()
    directory.mkdir(parents=True,exist_ok=True);path=directory/(spec['id'].split('/')[-1]+'-workers'+str(workers)+'.json');assert not path.exists()
    libc,helper=backend()
    def env():
        value=FEnv();assert libc.fegetenv(ctypes.byref(value))==0;return value
    def restore(value):
        assert helper.reference_fenv_restore(ctypes.byref(value),32)==0
        actual=env();assert bytes(actual)==bytes(value);return actual
    def semantic(value):return {key:getattr(value,key)for key in('control','status','mxcsr')}
    fallback=compat._fallback_enabled;prior=sys.getprofile();primary=None
    outputs={};diagnostics=[];snapshots={};events=[]
    record=dict(case=spec,workers=workers,runtime=runtime,timing=False,primary=None,diagnostics=diagnostics)
    entries={key:getattr(core,key)for key in('get_num_threads','label_pair_check_storage','label_pair_counts_preweighted','variation_of_information_csr')}
    assert all(inspect.isbuiltin(value)and value.__module__==core.__name__ for value in entries.values())
    phase=['setup'];counts=dict(original=0,native=0);returned=[];accepted=[];suffix=[]
    record.update(provider_events=events,counts=counts,returned=returned,accepted=accepted,suffix=suffix)
    guard=ownership_profile(reference,native,owned.contingency_table,compat.fallback,entries,phase,counts,returned,accepted,suffix)
    def checked(call,role):
        def once(*args,**options):
            phase[0]=role;observations=[]
            def snapshot():
                state=env();observations.append(state);return state
            try:
                with candidate_providers(np,sparse)if role=='native'else nullcontext()as current:
                    if role=='native':events.append(current)
                    return call_once(call,args,options,snapshot,guard,outputs,
                        'actual'if role=='native'else'expected',diagnostics)[0]
            finally:snapshots[role]=observations;phase[0]='setup'
        return once
    caller=env()
    try:
        xx.set_fallback_enabled(False);natural=env();record['natural_caller_raw32']=bytes(natural).hex()
        restore(natural);expected=observe(checked(reference,'reference'),spec,np);record['reference_observation']=expected
        restore(natural);actual=observe(checked(native,'native'),spec,np);record['native_observation']=actual
        record['provider_events']=events
        assert counts==dict(original=1,native=1)
        assert actual==expected  # All words/metadata/errors/warnings/callbacks/alias/input fields.
        assert len(snapshots['reference'])==len(snapshots['native'])==2
        assert [semantic(v)for v in snapshots['reference']]==[semantic(v)for v in snapshots['native']]
        if spec.get('images_none'):
            assert events==[[]]and returned==[]and suffix==[]and accepted==[False]
            assert actual['error']==dict(type='AttributeError',module='builtins',message="'NoneType' object has no attribute 'shape'")
        else:
            requirements=(dict(get_num_threads=1,label_pair_check_storage=3,variation_of_information_csr=1)
                if 'table'in spec else dict(get_num_threads=2,label_pair_check_storage=7,label_pair_counts_preweighted=1,variation_of_information_csr=1))
            assert {key:returned.count(key)for key in requirements}==requirements
            assert len(returned)==sum(requirements.values())and accepted==[True]
            assert suffix==([]if 'table'in spec else[True])
            assert [row['provider']for row in events[0]]==['numpy.log2','numpy.matmul','numpy.log2','numpy.matmul']
            assert type(outputs['expected'])is type(outputs['actual'])is np.ndarray
            assert outputs['expected'].tobytes()==outputs['actual'].tobytes()
            assert not np.shares_memory(outputs['expected'],outputs['actual'])
        assert core.get_num_threads()==workers and np.log2 is providers[0]and np.matmul is providers[1]
        assert np.geterr()==policy[0]and np.geterrcall()is policy[1]and np.getbufsize()==policy[2]
        record.update(counts=counts,returned=returned,accepted=accepted,suffix=suffix)
    except BaseException as error:
        primary=error;record['primary']=dict(type=type(error).__name__,error=str(error));raise
    finally:
        sys.setprofile(prior)
        try:
            xx.set_fallback_enabled(fallback)
            np.seterr(**policy[0]);np.seterrcall(policy[1]);np.setbufsize(policy[2])
            actual=restore(caller);record['caller32']=dict(saved=bytes(caller).hex(),actual=bytes(actual).hex(),exact=True)
        except BaseException as error:
            if primary is None:primary=error;raise
            sys.stderr.write('VI_SMALL_SECONDARY '+str(error)+'\n')
        finally:
            try:
                retain_target_FP(record,snapshots)
                retain_record(path,record,outputs,array_words)
            except BaseException as error:
                if primary is None:primary=error;raise
                sys.stderr.write('VI_SMALL_DIAGNOSTIC_SECONDARY '+str(error)+'\n')
            finally:
                try:
                    restore(caller)
                    assert np.geterr()==policy[0]and np.geterrcall()is policy[1]and np.getbufsize()==policy[2]
                    assert np.log2 is providers[0]and np.matmul is providers[1]and core.get_num_threads()==workers
                    assert xx.metrics.variation_of_information is native and skimage.metrics.variation_of_information is reference
                except BaseException as error:
                    if primary is None:raise
                    sys.stderr.write('VI_SMALL_RESTORE_SECONDARY '+str(error)+'\n')
