"""ROLLING_BALL public-call adapter; existing Montage sampler, no authority at import.

Reuse next_util_execution's F1 backend, snapshots and mandatory real-entry stop.
The family-specific envelope binds genuine source/product/SDK/runtime authority.
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

ROOT=Path(__file__).resolve().parents[1]
CONTEXT='coverage/contracts/rolling-ball-benchmark-execution-v3.json'
API='restoration.rolling_ball'
LIMITS=('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')
digest=lambda raw:hashlib.sha256(raw).hexdigest()
source_bytes=lambda path:Path(path).read_bytes().replace(b'\r\n',b'\n')


def historical_bytes(root,name):
    row=json.loads((Path(root)/CONTEXT).read_bytes())['historical_sources'][name]
    raw=source_bytes(Path(root)/name);assert digest(raw)==row['current_canonical_sha256'],name
    for fragment in row['inverse_fragments']:
        current,old=fragment['current'].encode(),fragment['predecessor'].encode()
        assert current and raw.count(current)==1,name;raw=raw.replace(current,old,1)
    assert digest(raw)==row['predecessor_canonical_sha256'],name
    return raw


def support(root,context):
    path=Path(root)/'benchmarks/next_util_execution.py'
    assert digest(source_bytes(path))==context['reused_executor_canonical_sha256']
    spec=importlib.util.spec_from_file_location('_rolling_ball_existing_executor',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def validate_binding(path,root=ROOT):
    if path is None:raise SystemExit('ROLLING_BALL_BENCHROLLING_BALL_UNBOUND: genuine product/root binding required before imports or bodies')
    raw=Path(path).read_bytes();b=json.loads(raw);context_raw=(Path(root)/CONTEXT).read_bytes();c=json.loads(context_raw)
    required={'status','product_kind','source_root','source_revision','git_argv','prepared_source','producer','interpreter','installed_site','numpy','core_path','core_sha256','SDK_members','runtime_path','runtime_sha256','genuine_root_admission_path','genuine_root_admission_sha256','genuine_root_stager_path','genuine_root_stager_sha256','helper_binding_path','helper_binding_sha256','base_contract_sha256','root_go_path','routes','output_root'}
    assert set(b)==required,'Unknown/missing binding fields'
    assert b['status']=='ROOT_AUTHORIZED_GENUINE_FIRST_COHORTS'and b['product_kind']=='GENUINE_INSTALLED_WHOLE_PRODUCT'
    assert b['base_contract_sha256']==digest(context_raw)
    prepared=b['prepared_source'];producer=b['producer'];revision=b['source_revision'];snapshot=Path(b['source_root']).resolve()
    assert set(prepared)=={'revision','files_sha256'}and prepared['files_sha256']
    assert set(producer)=={'run','revision','artifact','sha256'}and str(producer['run']).isdigit()and str(producer['artifact']).isdigit()
    assert len(revision)==40 and set(revision)<=set('0123456789abcdef')and revision==prepared['revision']==producer['revision']
    assert len(producer['sha256'])==64 and set(producer['sha256'])<=set('0123456789abcdef')
    assert not os.environ.get('GIT_DIR')and not os.environ.get('GIT_WORK_TREE')
    git=b['git_argv'];assert type(git)is list and len(git)==3 and git[1]=='-C'and all(type(x)is str for x in git)
    canonical=lambda p:str(p).replace('\\','/').replace('/mnt/c/','C:/',1).casefold()
    assert canonical(git[2])==canonical(snapshot)
    assert subprocess.check_output([*git,'rev-parse','HEAD'],text=True).strip()==revision
    assert not subprocess.check_output([*git,'status','--porcelain'],text=True).strip()
    assert (snapshot/CONTEXT).read_bytes()==context_raw
    assert prepared['files_sha256'][CONTEXT]==digest(source_bytes(snapshot/CONTEXT))
    assert set(b['routes'])=={'rolling-ball'}and b['routes']['rolling-ball']==c['route']
    def authenticated(stem):
        data=Path(b[stem+'_path']).read_bytes();assert digest(data)==b[stem+'_sha256'],stem;return json.loads(data)
    runtime=authenticated('runtime');assert runtime['schema']=='ci-offline-measurement-v1'and runtime['source']==prepared
    assert runtime['runtime']['versions']['numpy']==b['numpy']and b['numpy']in c['supported_numpy']
    assert runtime['runtime']['python']==sys.version and Path(sys.executable).resolve()==Path(b['interpreter']).resolve()
    assert runtime['runtime']['packages_sha256']['skimagexx'][Path(b['core_path']).name]==b['core_sha256']
    assert b['SDK_members']==runtime['runtime']['sdk_sha256']and 'include/skimagexx/rolling_ball.hpp'in b['SDK_members']
    assert any(n in b['SDK_members']for n in('lib/libskimagexx.a','lib/skimagexx.lib'))
    admission=authenticated('genuine_root_admission');assert admission['passed']is admission['root_accepted']is True and admission['profiles']==4
    assert admission['source']==revision and str(admission['run'])==str(producer['run'])and admission['product_kind']==b['product_kind']
    stage=authenticated('genuine_root_stager');assert stage['revision']==revision and str(stage['run'])==str(producer['run'])and str(stage['artifact'])==str(producer['artifact'])and stage['artifact_sha256']==producer['sha256']
    helper=authenticated('helper_binding')
    assert set(helper['env'])=={'UTIL_VIEWS_FENV_MODE','UTIL_VIEWS_FENV_RECEIPT','UTIL_VIEWS_FENV_RECEIPT_SHA256'}
    assert helper['env']['UTIL_VIEWS_FENV_MODE']=='source-built-v1'and not os.environ.get('UTIL_VIEWS_FENV_HELPER')
    assert all(os.environ.get(k)==v for k,v in helper['env'].items())
    helper_root=Path(helper['env']['UTIL_VIEWS_FENV_RECEIPT']).resolve().parent
    receipt=Path(helper['env']['UTIL_VIEWS_FENV_RECEIPT']);assert helper['files']and digest(receipt.read_bytes())==helper['env']['UTIL_VIEWS_FENV_RECEIPT_SHA256']==helper['files'][receipt.name]
    for name,pin in helper['files'].items():
        p=(helper_root/name).resolve();assert p.is_relative_to(helper_root)and digest(p.read_bytes())==pin
    for name,pin in c['source_files_canonical_sha256'].items():
        assert digest(source_bytes(Path(root)/name))==digest(source_bytes(snapshot/name))==prepared['files_sha256'][name]==pin,name
    assert prepared['files_sha256']['CMakeLists.txt']==digest(source_bytes(snapshot/'CMakeLists.txt'))
    for name,pin in c['FENV_dependency']['source_files_raw_sha256'].items():assert digest((snapshot/name).read_bytes())==pin,name
    go_raw=Path(b['root_go_path']).read_bytes();go=json.loads(go_raw)
    assert go['binding_sha256']==digest(raw)and go['routes']==['rolling-ball']and go['timed_batches']==0
    assert go['maximum_original_topcalls']==go['maximum_native_topcalls']==1
    assert go['helper_binding_sha256']==b['helper_binding_sha256']
    assert datetime.now(timezone.utc)<datetime.fromisoformat(go['deadline_utc'])
    return b,c,go,digest(raw),digest(go_raw)


def validate_measurement(path,binding,context,binding_pin):
    assert path is not None,'Separate root measurement GO required before any target body'
    raw=Path(path).read_bytes();m=json.loads(raw)
    assert m['schema']=='next-metrics-measurement-v1'and m['status']=='ROOT_AUTHORIZED_ONE_INITIAL_COMPARISON'
    for name in('prepared_source','producer','core_sha256','runtime_sha256'):assert m[name]==binding[name],name
    family=m['families']['rolling_ball_native']
    assert family==dict(contract=CONTEXT,cases=8,case_ids=[s['id']for s in context['cases']],untimed_calls_per_case=1)
    scope=m['execution_scope'];assert scope['workers']==1 and scope['repeat']==7 and scope['target_ms']==100 and scope['warmup_min_ms']==250
    assert scope['single_host_same_installed_runtime']is scope['no_automatic_retry']is scope['retain_all_losses']is True
    cohort=m['completed_first_cohort'];payload=Path(cohort['path']).read_bytes();assert digest(payload)==cohort['sha256'];done=json.loads(payload)
    assert done['status']=='PASS_REAL_CLI_STOP_BEFORE_BODY'and done['route']=='rolling-ball'and done['source']==binding['source_revision']
    assert done['root_binding_sha256']==cohort['binding_sha256']==binding_pin
    assert done['counts']==dict(original=1,native=1)and done['sampler_body_calls']==done['timed_batches']==0
    assert done['outer_state_restored']is done['poststop_source_authority_unchanged']is True
    assert datetime.now(timezone.utc)<datetime.fromisoformat(m['deadline_utc'])
    return digest(raw),m['deadline_utc']


def fixture(spec,np):
    count=1
    for n in spec['shape']:count*=n
    image=(np.arange(count,dtype=np.int64)%251).astype(spec['dtype']).reshape(spec['shape'])
    if spec['layout']=='F':image=np.asfortranarray(image)
    elif spec['layout']=='stride2':
        backing=np.zeros(tuple(2*n for n in spec['shape']),dtype=image.dtype)
        view=backing[(slice(None,None,2),)*image.ndim];view[...] = image;image=view
    if spec['nansafe']:image.flat[count//2]=np.nan
    return (image,),dict(radius=spec['radius'],nansafe=spec['nansafe'],workers=1)


def native_requirements(spec):
    return dict(get_num_threads=1,ball_kernel_apply=1,rolling_ball_apply=1)


def sample_entry(measure,reference,native,cli,index,spec,deadline,common):
    # The actual sampler receives the public callbacks. In cohort mode install
    # the admitted mandatory entry guard even if the outer observer returns or
    # replaces the profiler. This helper has no sampler/clock implementation.
    previous=sys.getprofile()
    journal=cli.output.with_suffix('.batches.jsonl')
    def entry_only(frame,event,value):
        # An outer observer cannot remove the guard on a preceding unrelated
        # Python event. At measure entry it runs first; even replacement there
        # cannot bypass the shared mandatory raise if the observer returns.
        if previous is not None and event=='call'and frame.f_code is measure.__code__:previous(frame,event,value)
    try:
        if cli.first_cohort:sys.setprofile(common.first_cohort_profile(measure,entry_only))
        return measure(reference,native,7,100,journal=journal,case_index=index,case_id=spec['id'],pass_index=0,deadline=deadline)
    finally:
        if cli.first_cohort:sys.setprofile(previous)


def execute(module):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',action='store_true');parser.add_argument('--root-binding',type=Path);parser.add_argument('--measurement-go',type=Path);parser.add_argument('--first-cohort',action='store_true')
    parser.add_argument('--api',choices=[API],default=API);parser.add_argument('--case');parser.add_argument('--threads',type=int,choices=[1],default=1);parser.add_argument('--repeat',type=int,choices=[7],default=7);parser.add_argument('--output',type=Path,default=Path('build/rolling-ball-performance.json'));cli=parser.parse_args()
    context=json.loads((ROOT/CONTEXT).read_bytes());specs=module['case_specs']();assert specs==context['cases']and len(specs)==8
    assert all(module['case_bounds'](s)for s in specs)
    if cli.case is not None:assert cli.first_cohort and cli.case==specs[0]['id'],'Only exact first-cohort case selection is permitted'
    if cli.plan:print(json.dumps(module['plan'](),indent=2));return
    b,context,go,binding_pin,go_pin=validate_binding(cli.root_binding)
    assert not(cli.first_cohort and cli.measurement_go)
    if cli.first_cohort:
        assert cli.case==specs[0]['id'],'First cohort is exact first frozen case only';specs=specs[:1];measurement_pin=None;deadline=go['deadline_utc']
    else:
        assert cli.case is None,'Measurement must retain all8 cases and all losses';measurement_pin,deadline=validate_measurement(cli.measurement_go,b,context,binding_pin)
    assert not cli.output.exists()and not cli.output.with_suffix('.batches.jsonl').exists()
    output_root=Path(b['output_root']).resolve();assert Path(b['output_root']).is_absolute()and cli.output.resolve().is_relative_to(output_root)
    output_root.mkdir(parents=True,exist_ok=True)
    claim=output_root/('rolling-ball-first-cohort.claim.json'if cli.first_cohort else'rolling-ball-measurement.claim.json')
    with claim.open('x',encoding='utf-8')as stream:
        json.dump(dict(status='STARTED_NO_AUTOMATIC_RETRY',root_binding_sha256=binding_pin,measurement_sha256=measurement_pin,cases=[s['id']for s in specs]),stream);stream.flush();os.fsync(stream.fileno())
    common=support(ROOT,context);sys.path[:0]=[b['installed_site'],str(ROOT/'tools'),str(ROOT/'benchmarks')]
    from montage_sustained import measure
    from provenance import source_fingerprint
    assert digest(inspect.getsource(measure).encode())==context['method']['method_sha256']
    import numpy as np
    import skimage
    import skimagexx as xx
    assert np.__version__==b['numpy']and skimage.__version__=='0.26.0'and all(os.environ.get(n)=='1'for n in LIMITS)
    assert Path(xx.__file__).resolve().is_relative_to(Path(b['installed_site']).resolve())and xx._core.__name__=='skimagexx._core'
    core=Path(xx._core.__file__).resolve();assert core==Path(b['core_path']).resolve()and core.is_relative_to(Path(b['installed_site']).resolve())and digest(core.read_bytes())==b['core_sha256']
    upstream=importlib.import_module('skimage.restoration._rolling_ball');owned=importlib.import_module('skimagexx._rolling_ball');public=importlib.import_module('skimage.restoration')
    reference=public.rolling_ball;native=xx.restoration.rolling_ball
    assert reference is upstream.rolling_ball and native is owned.rolling_ball and str(inspect.signature(native))==str(inspect.signature(reference))==context['signature']
    assert digest(source_bytes(upstream.__file__))==context['original_source_canonical_sha256']
    assert Path(owned.__file__).resolve().is_relative_to(Path(b['installed_site']).resolve())and digest(source_bytes(owned.__file__))==context['source_files_canonical_sha256']['python/skimagexx/_rolling_ball.py']
    assert xx.get_num_threads()==1;fallback=importlib.import_module('skimagexx._compat');old_fallback=fallback._fallback_enabled
    save,restore,flags=common.fp_controller(Path(b['source_root']),context);source_before=source_fingerprint(Path(b['source_root']));policy=np.geterr(),np.geterrcall(),np.getbufsize();rows=[]
    def hardware():
        raw=bytes(save());return dict(control=int.from_bytes(raw[:2],'little'),status=int.from_bytes(raw[4:6],'little'),mxcsr=int.from_bytes(raw[28:32],'little'))
    xx.set_fallback_enabled(False)
    try:
        for index,spec in enumerate(specs):
            assert datetime.now(timezone.utc)<datetime.fromisoformat(deadline)
            args,options=fixture(spec,np);inputs=list(args);before=[common.array_state(v,np)for v in inputs];requirements=native_requirements(spec);entries={n:getattr(xx._core,n)for n in requirements}
            assert all(inspect.isbuiltin(v)and v.__module__=='skimagexx._core'for v in entries.values())
            returns=[];accepted=[];prior=sys.getprofile();caller=save()
            def profile(frame,event,value):
                if event=='call'and(str(frame.f_globals.get('__name__','')).startswith(('skimage.','scipy.ndimage'))or frame.f_code is fallback.fallback.__code__):raise AssertionError('Candidate entered original/helper/fallback')
                if event=='c_call'and str(getattr(value,'__module__','')).startswith(('skimage.','scipy.ndimage')):raise AssertionError('Candidate entered compiled original/helper')
                if event=='c_return':
                    for key,entry in entries.items():
                        if value is entry:returns.append(key)
                if event=='return'and frame.f_code is native.__code__ and value is not None:accepted.append(True)
            try:
                restore(caller);reference_before=hardware()
                assert datetime.now(timezone.utc)<datetime.fromisoformat(deadline)
                with warnings.catch_warnings(record=True)as expected_notices:
                    warnings.simplefilter('always');expected=reference(*args,**options);reference_FP=hardware()
                restore(caller);native_before=hardware()
                assert datetime.now(timezone.utc)<datetime.fromisoformat(deadline)
                with warnings.catch_warnings(record=True)as actual_notices:
                    warnings.simplefilter('always')
                    try:sys.setprofile(profile);actual=native(*args,**options);native_FP=hardware()
                    finally:sys.setprofile(prior)
                assert reference_before==native_before and reference_FP==native_FP
                notices=lambda ws:[(type(w.message).__module__,type(w.message).__name__,str(w.message))for w in ws]
                assert notices(expected_notices)==notices(actual_notices)==[]
                assert accepted==[True]and {k:returns.count(k)for k in requirements}==requirements and len(returns)==sum(requirements.values())
                expected_state=common.array_state(expected,np,args[0]);actual_state=common.array_state(actual,np,args[0]);assert expected_state==actual_state
                assert not np.shares_memory(actual,args[0])
                assert [common.array_state(v,np)for v in inputs]==before and (np.geterr(),np.geterrcall(),np.getbufsize())==policy and xx.get_num_threads()==1
            finally:sys.setprofile(prior);restore(caller)
            # Future outer rolling-ball probe must inspect this caller plus sample_entry
            # locals; current9API controller has no ROLLING_BALL cohort and is unchanged.
            sample_state=save()
            try:timing=sample_entry(measure,lambda:reference(*args,**options),lambda:native(*args,**options),cli,index,spec,deadline,common)
            finally:restore(sample_state)
            assert [common.array_state(v,np)for v in inputs]==before
            assert (np.geterr(),np.geterrcall(),np.getbufsize())==policy and xx.get_num_threads()==1
            rows.append(dict(case_id=spec['id'],api=API,actual_C_returns=returns,accepted=accepted,FP=native_FP,providers=context['providers'],**timing))
        assert source_fingerprint(Path(b['source_root']))==source_before
        assert digest(Path(cli.root_binding).read_bytes())==binding_pin and digest(Path(b['root_go_path']).read_bytes())==go_pin
        assert digest((ROOT/CONTEXT).read_bytes())==b['base_contract_sha256']
        if cli.measurement_go:assert digest(cli.measurement_go.read_bytes())==measurement_pin
        cli.output.parent.mkdir(parents=True,exist_ok=True)
        with cli.output.open('x',encoding='utf-8')as stream:json.dump(dict(metadata=dict(**source_before,source_before=source_before,source_revision=b['source_revision'],producer=b['producer'],prepared_run=b['producer']['run'],prepared_artifact=b['producer']['artifact'],runtime_sha256=b['runtime_sha256'],core_sha256=b['core_sha256'],core_path=str(core),SDK_members=b['SDK_members'],root_binding_sha256=binding_pin,measurement_sha256=measurement_pin,contract_sha256=b['base_contract_sha256'],method_sha256=context['method']['method_sha256'],numpy=np.__version__,reference=skimage.__version__,python=sys.version,threads=1,repeat=7,target_ms=100,warmup_min_ms=250,retain_all_losses=True),results=rows),stream,indent=2)
    finally:xx.set_fallback_enabled(old_fallback)
