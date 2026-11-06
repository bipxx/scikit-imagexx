"""Six family public-call adapters over the existing sustained Montage sampler.

No target import until authenticated external whole-product/root authority.
The unified controller may stop the real measure call at its Python entry.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings

CONTEXT='coverage/contracts/next-util-benchmark-execution-v2.json'
LIMITS=('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')


def digest(raw):return hashlib.sha256(raw).hexdigest()


def source_bytes(path):return Path(path).read_bytes().replace(b'\r\n',b'\n')


def historical_bytes(root,name):
    context=json.loads((Path(root)/CONTEXT).read_bytes());row=context['sources'][name]
    raw=source_bytes(Path(root)/name);assert digest(raw)==row['current_canonical_sha256'],name
    for fragment in row['inverse_fragments']:
        a,b=fragment['current'].encode(),fragment['predecessor'].encode()
        assert a and raw.count(a)==1,name
        raw=raw.replace(a,b,1)
    assert digest(raw)==row['predecessor_canonical_sha256'],name
    return raw


def load_binding(path,controller_root,route):
    if path is None:
        raise SystemExit('PROSPECTIVE_UNBOUND: PROSPECTIVE: genuine product/root measurement binding required; no target imports/calls')
    raw=Path(path).read_bytes();binding=json.loads(raw)
    allowed={'status','source_root','source_revision','git_argv','genuine_root_admission_path','genuine_root_admission_sha256','product_kind','interpreter','source_files_sha256','installed_site','numpy','core_path','core_sha256','output_root','routes','prepared_source','producer','runtime_path','runtime_sha256','shared_CMake_sha256','genuine_root_stager_path','genuine_root_stager_sha256','base_contract_sha256','root_go_path'}
    assert not set(binding)-allowed
    assert binding['status']=='ROOT_AUTHORIZED_GENUINE_FIRST_COHORTS'and binding['product_kind']=='GENUINE_INSTALLED_WHOLE_PRODUCT'
    context_raw=(Path(controller_root)/CONTEXT).read_bytes();assert binding['base_contract_sha256']==digest(context_raw)
    context=json.loads(context_raw);root=Path(binding['source_root']).resolve()
    assert (root/CONTEXT).read_bytes()==context_raw
    source=binding['prepared_source'];producer=binding['producer']
    assert set(source)=={'revision','files_sha256'}and source['files_sha256']
    assert set(producer)=={'run','revision','artifact','sha256'}
    revision=source['revision'];assert len(revision)==40 and set(revision)<=set('0123456789abcdef')
    assert revision==binding['source_revision']==producer['revision']
    git=binding['git_argv'];assert type(git)is list and len(git)==3 and git[1]=='-C'and all(type(x)is str for x in git)
    assert git[2].replace('\\','/').casefold()==str(root).replace('\\','/').replace('/mnt/c/','C:/',1).casefold()
    assert subprocess.check_output([*git,'rev-parse','HEAD'],text=True).strip()==revision
    assert not subprocess.check_output([*git,'status','--porcelain'],text=True).strip()
    def authenticated(field):
        payload=Path(binding[field+'_path']).read_bytes();assert digest(payload)==binding[field+'_sha256'];return json.loads(payload)
    runtime=authenticated('runtime');assert runtime['source']==source
    assert runtime['runtime']['packages_sha256']['skimagexx'][Path(binding['core_path']).name]==binding['core_sha256']
    admission=authenticated('genuine_root_admission')
    assert admission['root_accepted']is True and admission['product_kind']=='GENUINE_INSTALLED_WHOLE_PRODUCT'
    assert admission['source']==revision and str(admission['run'])==str(producer['run'])
    stager=authenticated('genuine_root_stager')
    assert str(stager['run'])==str(producer['run'])and str(stager['artifact'])==str(producer['artifact'])
    assert stager['revision']==revision and stager['artifact_sha256']==producer['sha256']
    assert len(producer['sha256'])==64 and set(producer['sha256'])<=set('0123456789abcdef')
    go=json.loads(Path(binding['root_go_path']).read_bytes())
    assert go['binding_sha256']==digest(raw)and route in go['routes']
    assert go['timed_batches']==0 and datetime.now(timezone.utc)<datetime.fromisoformat(go['deadline_utc'])
    assert source['files_sha256']['CMakeLists.txt']==binding['shared_CMake_sha256']==digest(source_bytes(root/'CMakeLists.txt'))
    for name,row in context['sources'].items():
        assert digest(source_bytes(Path(controller_root)/name))==row['current_canonical_sha256'],name
        assert digest(source_bytes(root/name))==row['current_canonical_sha256'],name
    for name,pin in context['production_source_canonical_sha256'].items():assert digest(source_bytes(root/name))==pin,name
    for name,pin in context['FENV_dependency']['source_files_raw_sha256'].items():assert digest((root/name).read_bytes())==pin,name
    assert Path(sys.executable).resolve()==Path(binding['interpreter']).resolve()
    assert str(binding['numpy'])in context['supported_numpy']
    return binding,context,go,digest(raw)


def load_measurement(path,binding,context,route,specs):
    # Reuse the existing controller's root measurement manifest, distinct from
    # its0timed first-cohort envelope. No local deadline or inferred authority.
    assert path is not None,'Root measurement manifest required; first-cohort GO permits0timed batches'
    raw=Path(path).read_bytes();authority=json.loads(raw)
    assert authority['schema']=='next-metrics-measurement-v1'
    assert authority['status']=='ROOT_AUTHORIZED_ONE_INITIAL_COMPARISON'
    assert authority['prepared_source']==binding['prepared_source']
    assert {k:str(v)for k,v in authority['producer'].items()}=={k:str(v)for k,v in binding['producer'].items()}
    assert authority['core_sha256']==binding['core_sha256']and authority['runtime_sha256']==binding['runtime_sha256']
    cohort=authority['first_cohort']
    assert cohort['status']=='ROOT_AUTHORIZED_GENUINE_FIRST_COHORTS'and cohort['source_revision']==binding['source_revision']
    assert cohort['root_admission_sha256']==binding['genuine_root_admission_sha256']
    assert cohort['root_stager_sha256']==binding['genuine_root_stager_sha256']
    assert cohort['source_files_raw_sha256'][CONTEXT]==binding['base_contract_sha256']
    family=authority['families'][Path(context['routes'][route]['driver']).stem]
    assert family['contract']==CONTEXT and family['cases']==len(specs)
    if len(specs)!=len(context['routes'][route]['cases']):assert family['case_ids']==[s['id']for s in specs]
    else:assert family.get('case_ids',[s['id']for s in specs])==[s['id']for s in specs]
    scope=authority['execution_scope']
    assert scope['workers']==1 and scope['repeat']==7 and scope['target_ms']==100 and scope['warmup_min_ms']==250
    assert scope['single_host_same_installed_runtime']is scope['no_automatic_retry']is scope['retain_all_losses']is True
    return digest(raw)


def make_case(route,module,spec,np):
    if route=='crop':
        image,options=module['fixture'](spec,np);return (image,spec['width']),options
    return module['fixture'](spec,np)


def array_state(array,np,input_array=None):
    # Inputs and ordinary POD outputs are bounded by frozen case_bounds. Object
    # payloads are fixed strings; preserve identity separately, never raw pointers.
    value=dict(type=[type(array).__module__,type(array).__name__],dtype=array.dtype.str,shape=list(array.shape),strides=list(array.strides),own=bool(array.flags.owndata),writeable=bool(array.flags.writeable),c=bool(array.flags.c_contiguous),f=bool(array.flags.f_contiguous),base_type=[type(array.base).__module__,type(array.base).__name__])
    value['payload']=array.tolist()if array.dtype.hasobject else digest(array.tobytes())
    if input_array is not None:
        sharing=bool(np.shares_memory(array,input_array));value['shares_input']=sharing
        value['pointer_delta']=int(array.ctypes.data-input_array.ctypes.data)if sharing else None
    return value


def output_state(value,api,args,np):
    if api=='util.regular_grid':
        def field(v):
            if v is None:return None
            assert isinstance(v,np.integer)
            return dict(type=[type(v).__module__,type(v).__name__],dtype=v.dtype.str,raw=v.tobytes().hex())
        assert type(value)is tuple and all(type(s)is slice for s in value)
        return [[field(v)for v in(s.start,s.stop,s.step)]for s in value]
    if api=='segmentation.join_segmentations':
        if type(value)is not tuple:return array_state(value,np)
        labels,first,second=value
        assert first.in_values is second.in_values
        assert first._array is None and second._array is None
        assert first._max_str_lines==second._max_str_lines==4
        # Exact upstream ArrayMap class identity remains an explicitly open
        # frontier; do not normalize it into a claimed equal class.
        return dict(labels=array_state(labels,np),maps=[dict(keys=array_state(m.in_values,np),values=array_state(m.out_values,np))for m in(first,second)],shared_keys=True,cache=None,max_str_lines=4)
    assert type(value)is np.ndarray
    return array_state(value,np,args[0])


def fenv_source(root,context):
    # Source-bound reviewed F1 interface; never import a cached or unpinned copy.
    for name,pin in context['FENV_dependency']['source_files_raw_sha256'].items():assert digest((Path(root)/name).read_bytes())==pin,name
    spec=importlib.util.spec_from_file_location('_next_util_fenv',Path(root)/'tests/util_views_fenv.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def fp_controller(root,context):
    # Genuine benchmarks accept only reviewed F1 source-built mode. Legacy
    # private mode remains exclusively in F1's historical interface/scope.
    assert os.environ.get('UTIL_VIEWS_FENV_MODE')=='source-built-v1','Source-built FENV mode required before target'
    assert not os.environ.get('UTIL_VIEWS_FENV_HELPER'),'Legacy helper forbidden in benchmark'
    assert os.environ.get('UTIL_VIEWS_FENV_RECEIPT')and os.environ.get('UTIL_VIEWS_FENV_RECEIPT_SHA256'),'Source-built receipt and external digest required before target'
    import ctypes
    module=fenv_source(root,context)
    # backend authenticates receipt/source/context/current emitted library,
    # compiler/header/control/ABI identities BEFORE configuring the native helper.
    libc,helper=module.backend()
    def save():
        state=(ctypes.c_uint32*8)();assert libc.fegetenv(ctypes.byref(state))==0;return state
    def restore(state):
        assert helper.reference_fenv_restore(ctypes.byref(state),32)==0
        assert bytes(save())==bytes(state)
    def flags():return dict(flags=libc.fetestexcept(0x3d),rounding=libc.fegetround())
    return save,restore,flags


def first_cohort_profile(measure,observer):
    # Real entry only, same caller frame. An unrelated/non-stopping observer
    # cannot turn0timing authority into permission to enter the sampler body.
    def profile(frame,event,value):
        if event=='call'and frame.f_code is measure.__code__:
            if observer is not None:observer(frame,event,value)
            raise AssertionError('First-cohort observer did not stop real sampler entry; body forbidden')
        if observer is not None:observer(frame,event,value)
    return profile


def expected_returns(api,spec):
    if api=='segmentation.join_segmentations':return {'join_encode':1,'join_divmod':int(spec['mapping'])}
    if api=='util.crop':return {'crop_view':1,'crop_copy_into':int(spec['copy']and not spec['payload_provider'])}
    if api=='util.regular_grid':
        import math
        return {'regular_grid_prepare':1,'regular_grid_finish':int(math.prod(spec['shape'])>spec['points'])}
    if api=='util.slice_along_axes':return {'util_slice_view':1,'util_slice_copy_into':int(spec['copy'])}
    return {'util_'+api.split('.')[1]:1}


def execute(route,module):
    parser=argparse.ArgumentParser(description=module.get('__doc__'))
    parser.add_argument('--plan',action='store_true');parser.add_argument('--root-binding',type=Path)
    parser.add_argument('--output',type=Path,default=Path('build/next-util-'+route+'.json'))
    parser.add_argument('--first-cohort',action='store_true');parser.add_argument('--measurement-go',type=Path);parser.add_argument('--api');parser.add_argument('--case');parser.add_argument('--threads',type=int,choices=[1],default=1)
    parser.add_argument('--repeat',type=int,choices=[7],default=7)
    cli=parser.parse_args();controller_root=Path(module['__file__']).resolve().parents[1]
    context=json.loads((controller_root/CONTEXT).read_bytes());description=context['routes'][route]
    specs=module['case_specs']();assert specs==description['cases']
    bounds=module.get('case_bounds',module.get('source_safe'))
    assert all(bounds(s)for s in specs)
    if cli.api is not None:assert cli.api in description['apis'];specs=[s for s in specs if s.get('api',description['apis'][0])==cli.api]
    if cli.case is not None:specs=[s for s in specs if s['id']==cli.case]
    assert specs,'Unknown or empty case selection'
    if cli.plan:
        print(json.dumps(dict(state='PROSPECTIVE_UNBOUND_NO_MEASUREMENT',cases=specs,bounds=[bounds(s)for s in specs],method=context['method'],unbound=context['unbound']),indent=2));return
    binding,context,go,binding_sha256=load_binding(cli.root_binding,controller_root,route)
    assert not(cli.first_cohort and cli.measurement_go),'Choose first-cohort or measurement authority'
    measurement_sha256=None if cli.first_cohort else load_measurement(cli.measurement_go,binding,context,route,specs)
    assert not cli.output.exists()and not cli.output.with_suffix('.batches.jsonl').exists()
    sys.path[:0]=[str(controller_root/'tools'),str(controller_root/'benchmarks')]
    from montage_sustained import measure
    from provenance import source_fingerprint
    assert digest(inspect.getsource(measure).encode())==context['method']['method_sha256']
    # Import requests remain explicit so cached modules cannot bypass refusal.
    import numpy as np
    import skimage
    import skimagexx as xx
    assert skimage.__version__=='0.26.0'and np.__version__==binding['numpy']
    assert all(os.environ[n]=='1'for n in LIMITS)and xx.get_num_threads()==1
    core=Path(xx._core.__file__);assert core.resolve()==Path(binding['core_path']).resolve()
    assert core.resolve().is_relative_to(Path(binding['installed_site']).resolve())
    assert digest(core.read_bytes())==binding['core_sha256']and xx._core.__name__=='skimagexx._core'
    xx.set_fallback_enabled(False)
    source_before=source_fingerprint(Path(binding['source_root']));policy=np.geterr(),np.geterrcall(),np.getbufsize()
    # Source/artifact admission is one stable execute-entry trust boundary.
    # No per-case compiler/header rehash or fresh module defeats F1's cache.
    save,restore,flags=fp_controller(Path(binding['source_root']),context);rows=[]
    for index,spec in enumerate(specs):
        api=spec.get('api',description['apis'][0]);family,name=api.split('.')
        target=context['apis'][api];original_module=importlib.import_module(target['original_module'])
        owned=importlib.import_module(target['owned_module']);reference=getattr(importlib.import_module('skimage.'+family),name);native=getattr(getattr(xx,family),name)
        assert reference is getattr(original_module,name)and native is getattr(owned,name)
        assert str(inspect.signature(reference))==str(inspect.signature(native))==target['signature']
        assert digest(source_bytes(original_module.__file__))==target['original_source_canonical_sha256']
        requirements=expected_returns(api,spec);entries={key:getattr(xx._core,key)for key in requirements}
        assert all(inspect.isbuiltin(v)and v.__module__=='skimagexx._core'for v in entries.values())
        args,options=make_case(route,module,spec,np)
        inputs=[v for v in args if type(v)is np.ndarray];before=[array_state(v,np)for v in inputs]
        caller=save();returns=[];accepted=[];prior=sys.getprofile();notices=[]
        denied={reference.__code__}
        for value in reference.__globals__.values():
            if inspect.isfunction(value)and value.__module__.startswith('skimage.'):denied.add(value.__code__)
        fallback=importlib.import_module('skimagexx._compat').fallback;denied.add(fallback.__code__)
        def profile(frame,event,value):
            if event=='call'and frame.f_code in denied:raise AssertionError('candidate entered whole original/helper/fallback')
            if event=='c_return':
                for key,entry in entries.items():
                    if value is entry:returns.append(key)
            if event=='return'and frame.f_code is native.__code__:accepted.append(True)
        try:
            restore(caller)
            with warnings.catch_warnings(record=True)as expected_warnings:
                warnings.simplefilter('always');expected=reference(*args,**options);reference_FP=flags()
            restore(caller)
            with warnings.catch_warnings(record=True)as actual_warnings:
                warnings.simplefilter('always')
                try:sys.setprofile(profile);actual=native(*args,**options);native_FP=flags()
                finally:sys.setprofile(prior)
            assert reference_FP==native_FP
            notices=lambda ws:[(type(w.message).__module__,type(w.message).__name__,str(w.message))for w in ws]
            assert notices(expected_warnings)==notices(actual_warnings)==[]
            assert accepted==[True]and all(returns.count(k)==n for k,n in requirements.items())
            expected_state=output_state(expected,api,args,np);actual_state=output_state(actual,api,args,np)
            assert expected_state==actual_state
            if api=='segmentation.join_segmentations':
                expected_labels=expected[0]if type(expected)is tuple else expected
                actual_labels=actual[0]if type(actual)is tuple else actual
                assert not any(np.shares_memory(actual_labels,a)for a in args)and not np.shares_memory(actual_labels,expected_labels)
            if api.startswith('util.view_as_'):assert np.shares_memory(actual,args[0])
            if (api in ('util.crop','util.slice_along_axes')or api.startswith('util.view_as_'))and actual.dtype.hasobject:
                assert all(a is b for a,b in zip(actual.flat,expected.flat))
            assert [array_state(v,np)for v in inputs]==before
            assert xx.get_num_threads()==1 and (np.geterr(),np.geterrcall(),np.getbufsize())==policy
        finally:sys.setprofile(prior);restore(caller)
        # Stable controller interface: caller locals at the real measure entry
        # include spec/api/args/options/expected/actual/returns/accepted and FP.
        # First-cohort observer sees the real entry/caller locals. If it does
        # not stop there, our chained fallback refuses BEFORE any sampler body.
        sampler_previous=sys.getprofile()
        try:
            if cli.first_cohort:sys.setprofile(first_cohort_profile(measure,sampler_previous))
            timing=measure(lambda:reference(*args,**options),lambda:native(*args,**options),7,100,journal=cli.output.with_suffix('.batches.jsonl'),case_index=index,case_id=spec['id'],pass_index=0,deadline=go['deadline_utc'])
        finally:
            if cli.first_cohort:sys.setprofile(sampler_previous)
        rows.append(dict(case_id=spec['id'],api=api,actual_C_returns=returns,native_accepted=True,FP=native_FP,providers=target['providers'],return_class_frontier=([type(expected[1]).__module__,type(actual[1]).__module__]if api=='segmentation.join_segmentations'and spec['mapping']else None),**timing))
        assert [array_state(v,np)for v in inputs]==before and xx.get_num_threads()==1
        assert (np.geterr(),np.geterrcall(),np.getbufsize())==policy
    assert source_fingerprint(Path(binding['source_root']))==source_before
    assert digest(core.read_bytes())==binding['core_sha256']and digest(Path(cli.root_binding).read_bytes())==binding_sha256
    assert digest((controller_root/CONTEXT).read_bytes())==binding['base_contract_sha256']
    if cli.measurement_go:assert digest(cli.measurement_go.read_bytes())==measurement_sha256
    cli.output.parent.mkdir(parents=True,exist_ok=True)
    cli.output.write_text(json.dumps(dict(metadata=dict(**source_before,source_before=source_before,source_revision=binding['source_revision'],root_binding_sha256=binding_sha256,measurement_sha256=measurement_sha256,contract_sha256=binding['base_contract_sha256'],numpy=np.__version__,reference=skimage.__version__,python=sys.version,core_path=str(core),prepared_run=binding['producer']['run'],prepared_artifact=binding['producer']['artifact'],producer=binding['producer'],runtime_sha256=binding['runtime_sha256'],core_sha256=binding['core_sha256'],threads=1,repeat=7,target_ms=100,method_sha256=context['method']['method_sha256'],retain_all_losses=True),results=rows),indent=2)+'\n')
