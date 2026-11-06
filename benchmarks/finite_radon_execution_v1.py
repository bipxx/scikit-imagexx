"""Finite Radon allocated public-call comparison over the admitted Montage sampler.

Only source/metadata can run without external whole-product/root authority.
"""
import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import struct
import sys
import warnings
from finite_radon_cases_v1 import case_specs,case_bounds,fixture

ROOT=Path(__file__).resolve().parents[1]
CONTEXT='coverage/contracts/finite-radon-benchmark-execution-v2.json'


def digest(raw):return hashlib.sha256(raw).hexdigest()


def common_adapter(root=ROOT):
    # A private module instance reuses the reviewed validators/F1/entry-stop
    # unchanged. Only its explicit family context constant differs; no shared
    # module or source mutation, wrapper validator, or new benchmark engine.
    context=json.loads((root/CONTEXT).read_bytes());name='benchmarks/next_util_execution.py'
    assert digest((root/name).read_bytes().replace(b'\r\n',b'\n'))==context['sources'][name]['current_canonical_sha256']
    spec=importlib.util.spec_from_file_location('_finite_radon_common',root/name)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.CONTEXT=CONTEXT
    return module


def hardware(state):
    raw=bytes(state);assert len(raw)==32
    return dict(control=struct.unpack_from('<H',raw,0)[0],status=struct.unpack_from('<H',raw,4)[0],mxcsr=struct.unpack_from('<I',raw,28)[0])


def sample(common,measure,reference,native,cli,spec,index,go,expected,actual,returns,accepted,FP,*,api=None,args=None,options=None,entry=None,workers=None):
    # Actual real-entry observer sees these stable caller locals. A non-stopping
    # profiler cannot authorize a sampler body under first-cohort authority.
    previous=sys.getprofile()
    try:
        if cli.first_cohort:sys.setprofile(common.first_cohort_profile(measure,previous))
        return measure(reference,native,7,100,journal=cli.output.with_suffix('.batches.jsonl'),case_index=index,case_id=spec['id'],pass_index=cli.pass_index,deadline=go['deadline_utc'])
    finally:
        if cli.first_cohort:sys.setprofile(previous)


def execute(api):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',action='store_true');parser.add_argument('--root-binding',type=Path)
    parser.add_argument('--first-cohort',action='store_true');parser.add_argument('--measurement-go',type=Path)
    parser.add_argument('--output',type=Path,default=Path('build/finite-radon-'+api+'.json'))
    parser.add_argument('--case');parser.add_argument('--pass-index',type=int,choices=range(1,9))
    parser.add_argument('--threads',type=int,choices=[1],default=1);parser.add_argument('--repeat',type=int,choices=[7],default=7)
    cli=parser.parse_args();assert api in ('frt2','ifrt2')
    context=json.loads((ROOT/CONTEXT).read_bytes());specs=case_specs(api)
    assert specs==context['routes'][api]['cases'];bounds=[case_bounds(s)for s in specs]
    if cli.case is not None:specs=[s for s in specs if s['id']==cli.case]
    assert specs,'Unknown or empty case selection'
    if cli.plan:
        print(json.dumps(dict(state=context['state'],cases=specs,bounds=[case_bounds(s)for s in specs],method=context['method'],unbound=context['unbound']),indent=2));return
    common=common_adapter()
    binding,context,go,binding_sha256=common.load_binding(cli.root_binding,ROOT,api)
    assert cli.pass_index is not None,'Explicit pass-index required before science'
    assert not(cli.first_cohort and cli.measurement_go)
    measurement_sha256=None if cli.first_cohort else common.load_measurement(cli.measurement_go,binding,context,api,specs)
    assert not cli.output.exists()and not cli.output.with_suffix('.batches.jsonl').exists()
    sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'benchmarks')]
    from montage_sustained import measure
    from provenance import source_fingerprint
    assert digest(inspect.getsource(measure).encode())==context['method']['method_sha256']
    # Explicit requests remain interceptable even with cached science modules.
    import numpy as np
    import skimage
    import skimagexx as xx
    assert skimage.__version__=='0.26.0'and np.__version__==binding['numpy']
    assert all(os.environ[n]=='1'for n in common.LIMITS)and xx.get_num_threads()==1
    assert xx._core.__name__=='skimagexx._core'
    core=Path(xx._core.__file__).resolve();assert core==Path(binding['core_path']).resolve()
    assert core.is_relative_to(Path(binding['installed_site']).resolve())and digest(core.read_bytes())==binding['core_sha256']
    xx.set_fallback_enabled(False)
    target=context['apis'][api];original_module=importlib.import_module(target['original_module']);owned=importlib.import_module(target['owned_module'])
    reference=getattr(skimage.transform,api);native=getattr(xx.transform,api);entry=getattr(xx._core,api)
    assert reference is getattr(original_module,api)and native is getattr(owned,api)and owned._core is xx._core
    assert str(inspect.signature(reference))==str(inspect.signature(native))=='(a)'
    assert inspect.isbuiltin(entry)and entry.__module__=='skimagexx._core'
    assert digest(common.source_bytes(original_module.__file__))==target['original_source_canonical_sha256']
    # Receipt/source/backend validation once at genuine execute-entry boundary.
    save,restore,flags=common.fp_controller(Path(binding['source_root']),context)
    source_before=source_fingerprint(Path(binding['source_root']));rows=[]
    policy=np.geterr(),np.geterrcall(),np.getbufsize()
    fallback=importlib.import_module('skimagexx._compat').fallback
    for index,spec in enumerate(specs):
        args,options=fixture(spec,np);a=args[0];before=common.array_state(a,np)
        caller=save();returns=[];accepted=[];prior=sys.getprofile()
        def guard(frame,event,value):
            if event=='call'and frame.f_code in (reference.__code__,fallback.__code__):raise AssertionError('candidate entered original/fallback')
            if event=='c_call'and getattr(value,'__name__',None)=='sum'and isinstance(getattr(value,'__self__',None),np.ndarray):raise AssertionError('candidate entered provider sum')
            if event=='c_return'and value is entry:returns.append(api)
            if event=='return'and frame.f_code is native.__code__:accepted.append(True)
        try:
            restore(caller);reference_before=hardware(save())
            with warnings.catch_warnings(record=True)as expected_warnings:
                warnings.simplefilter('always');expected=reference(*args,**options);reference_FP=hardware(save())
            assert common.array_state(a,np)==before
            restore(caller);native_before=hardware(save())
            with warnings.catch_warnings(record=True)as actual_warnings:
                warnings.simplefilter('always')
                try:sys.setprofile(guard);actual=native(*args,**options);native_FP=hardware(save())
                finally:sys.setprofile(prior)
            assert reference_before==native_before and reference_FP==native_FP
            assert not expected_warnings and not actual_warnings
            assert returns==[api]and accepted==[True]
            assert type(expected)is type(actual)is np.ndarray
            assert common.array_state(expected,np,a)==common.array_state(actual,np,a)
            assert expected.flags.owndata and actual.flags.owndata and expected.flags.c_contiguous and actual.flags.c_contiguous
            assert not np.shares_memory(expected,a)and not np.shares_memory(actual,a)and not np.shares_memory(actual,expected)
            assert common.array_state(a,np)==before and xx.get_num_threads()==1
            assert (np.geterr(),np.geterrcall(),np.getbufsize())==policy
        finally:sys.setprofile(prior);restore(caller)
        timing=sample(common,measure,lambda:reference(*args,**options),lambda:native(*args,**options),cli,spec,index,go,expected,actual,returns,accepted,native_FP,api=api,args=args,options=options,entry=entry,workers=xx.get_num_threads())
        assert common.array_state(a,np)==before and xx.get_num_threads()==1
        assert (np.geterr(),np.geterrcall(),np.getbufsize())==policy
        rows.append(dict(case_id=spec['id'],api='transform.'+api,actual_C_returns=returns,native_accepted=True,FP=native_FP,input_bounds=case_bounds(spec),providers=target['providers'],**timing))
    assert source_fingerprint(Path(binding['source_root']))==source_before
    assert digest(core.read_bytes())==binding['core_sha256']and digest(Path(cli.root_binding).read_bytes())==binding_sha256
    assert digest((ROOT/CONTEXT).read_bytes())==binding['base_contract_sha256']
    if cli.measurement_go:assert digest(cli.measurement_go.read_bytes())==measurement_sha256
    cli.output.parent.mkdir(parents=True,exist_ok=True)
    cli.output.write_text(json.dumps(dict(metadata=dict(source_revision=binding['source_revision'],source_before=source_before,root_binding_sha256=binding_sha256,measurement_sha256=measurement_sha256,contract_sha256=binding['base_contract_sha256'],producer=binding['producer'],runtime_sha256=binding['runtime_sha256'],core_sha256=binding['core_sha256'],threads=1,repeat=7,target_ms=100,warmup_min_ms=250,pass_index=cli.pass_index,method_sha256=context['method']['method_sha256'],retain_all_losses=True),results=rows),indent=2)+'\n')
