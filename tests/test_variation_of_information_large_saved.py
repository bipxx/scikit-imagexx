"""Legacy scalar SDK-gateway records/retained FP only; not current public VI parity."""
import ast
import ctypes
import hashlib
import importlib
from types import SimpleNamespace
import json
from pathlib import Path
import warnings
import numpy as np
import pytest
from test_variation_of_information_public import target as public_target
from variation_of_information_source_context import provider_predecessor_source
from label_pair_metrics_observer import observation
from util_views_fenv import FEnv, backend

ROOT=Path(__file__).resolve().parents[1]
SAVED=json.loads((ROOT/'tests/data/variation_of_information_large_saved_v1.json').read_bytes())


LEGACY_FRONTEND_SHA256='f42ca166318d77900e4df9ee00a72a5ddec7309a3098fb83287a7ecf621ca5fa'


def legacy_scalar_gateway(module,core):
    """Exact32e Python prefix -> unchanged scalar SDK/core ABI, provider omitted/None."""
    source=provider_predecessor_source(ROOT,'python/skimagexx/_variation_of_information.py')
    assert hashlib.sha256(source).hexdigest()==LEGACY_FRONTEND_SHA256
    tree=ast.parse(source);definitions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    namespace=dict(__name__='_vi_legacy_scalar_sdk',np=np,sparse=importlib.import_module('scipy.sparse'),
        _core=core,contingency_table=module.contingency_table)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),'<exact32e legacy scalar SDK gateway>','exec'),namespace)
    return SimpleNamespace(variation_of_information=namespace['variation_of_information'],_core=core)


@pytest.fixture(scope='module')
def target(public_target):
    module,core=public_target
    return legacy_scalar_gateway(module,core),core


@pytest.mark.parametrize('index',range(4))
@pytest.mark.parametrize('workers',(1,4))
def test_admitted_large_legacy_scalar_words_and_state(target,index,workers):
    reference=SAVED['profiles'][np.__version__]['rows'][index]
    module,core=target
    # Existing frozen benchmark recipe, including its allocation/layout policy.
    import runpy
    recipes=runpy.run_path(str(ROOT/'benchmarks/variation_of_information.py'))
    spec=recipes['case_specs']()[index]
    args,options=recipes['fixture'](spec,np)
    assert spec['id']==reference['id']
    buffers=dict(truth=args[0],test=args[1])
    before=observation((args,options,buffers),np)
    pointers={k:v.ctypes.data for k,v in buffers.items()}
    libc,helper=backend()
    def env():
        value=FEnv();assert libc.fegetenv(ctypes.byref(value))==0
        return value
    def restore(value):assert helper.reference_fenv_restore(ctypes.byref(value),32)==0
    def semantic(value):return {k:getattr(value,k)for k in('control','status','mxcsr')}
    caller=env();policy=np.geterr(),np.geterrcall(),np.getbufsize()
    events=[]
    def callback(message,flags):events.append(dict(event='FP-handler',message=message,flags=int(flags)))
    try:
        core.set_num_threads(workers);np.seterrcall(callback)
        with warnings.catch_warnings(record=True)as notices:
            warnings.simplefilter('always')
            incoming=env()
            for k,v in reference['target_immediate']['before'].items():
                if k!='raw':setattr(incoming,k,v)
            restore(incoming)
            entry=env()
            result=module.variation_of_information(*args,**options)
            outgoing=env()
        assert semantic(entry)=={k:reference['target_immediate']['before'][k]for k in semantic(entry)}
        assert semantic(outgoing)=={k:reference['target_immediate']['after'][k]for k in semantic(outgoing)}
        assert observation(result,np)==reference['result']
        assert before==reference['before']
        assert observation((args,options,buffers),np)==reference['after']
        assert events==reference['events']
        assert [dict(category=type(w.message).__name__,message=str(w.message))for w in notices]==reference['warnings']
        assert reference['error']is None
        assert {'0':{k:bool(np.shares_memory(result,v))for k,v in buffers.items()}}==reference['output_input_aliases']
        assert pointers=={k:v.ctypes.data for k,v in buffers.items()}
        assert np.geterr()==policy[0]and np.geterrcall()is callback and np.getbufsize()==policy[2]
        assert core.get_num_threads()==workers
    finally:
        core.set_num_threads(1)
        np.seterr(**policy[0]);np.seterrcall(policy[1]);np.setbufsize(policy[2])
        restore(caller);assert bytes(env())==bytes(caller)
