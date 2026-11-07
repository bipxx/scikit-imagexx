"""Root-bound one-API actual CLI observer; no authority at import or in UNBOUND mode."""
import argparse
import ast
from datetime import datetime,timezone
import hashlib,importlib,importlib.util,inspect,json,os
from pathlib import Path
import runpy,subprocess,sys,traceback

CONTRACT='coverage/contracts/rolling-ball-measurement-v4.json'
DRIVER_CONTRACT='coverage/contracts/rolling-ball-benchmark-execution-v3.json'
ROUTE='rolling-ball'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')

def prepare_host_helper(snapshot,results,python,manifest):
    # Called once outside targets/cohorts/sampler after actual producer admission.
    assert manifest['status']=='ROOT_AUTHORIZED_ONE_INITIAL_COMPARISON'and manifest['first_cohort']['status']=='ROOT_AUTHORIZED_GENUINE_FIRST_COHORTS'
    source=manifest['helper']['source_sha256']
    for name,pin in source.items():assert sha(Path(snapshot)/name)==pin
    assert manifest['helper']['mode']=='source-built-v1'and manifest['helper']['prepare_once_on_actual_host']is True
    assert not os.environ.get('UTIL_VIEWS_FENV_HELPER')
    bundle=Path(results)/'util-views-fenv'
    assert not bundle.exists(),'Helper preparation cannot be replayed'
    command=[str(python),str(Path(snapshot)/'tests/util_views_fenv_build.py'),'--output',str(bundle.resolve()),'--control-python',str(python)]
    write_json(Path(results)/'helper-command.json',command)
    with (Path(results)/'helper-build.log').open('w')as log:
        done=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    write_json(Path(results)/'helper-exit.json',dict(exit=done.returncode))
    assert done.returncode==0,'Helper setup failed; no cohorts authorized'
    receipt=bundle/'receipt.json';assert receipt.is_file()
    # The approved builder ran full32 positive/negative controls on this host.
    # Drivers independently validate this external digest+receipt before targets.
    env=dict(UTIL_VIEWS_FENV_MODE='source-built-v1',UTIL_VIEWS_FENV_RECEIPT=str(receipt.resolve()),UTIL_VIEWS_FENV_RECEIPT_SHA256=sha(receipt))
    write_json(Path(results)/'helper-binding.json',dict(env=env,files={f.relative_to(bundle).as_posix():sha(f)for f in bundle.rglob('*')if f.is_file()}))
    return env

def run_cohorts(controller,results,manifest,python,authorities):
    results=Path(results);terminals=[]
    manifest_path=Path(controller)/CONTRACT;manifest_pin=sha(manifest_path)
    for cohort in manifest['first_cohort']['routes']:
        authority=authorities[cohort]
        command=[str(python),str(Path(controller)/manifest['first_cohort']['probe_path']),'--binding',authority['binding'],'--root-go',authority['go'],'--route',cohort]
        write_json(results/('first-cohort-'+cohort+'-command.json'),command)
        with (results/('first-cohort-'+cohort+'.log')).open('w')as log:done=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        write_json(results/('first-cohort-'+cohort+'-exit.json'),dict(exit=done.returncode))
        assert done.returncode==0,'Stop remaining cohorts/timing; no retry: '+cohort
        terminal=json.loads((results/'first-cohort'/cohort/'terminal.json').read_bytes())
        assert terminal['status']=='PASS_REAL_CLI_STOP_BEFORE_BODY'and terminal['route']==cohort
        assert terminal['source']==manifest['first_cohort']['source_revision']and terminal['counts']==dict(original=1,native=1)
        assert terminal['timed_batches']==terminal['sampler_body_calls']==0 and terminal['outer_state_restored']is True
        assert terminal['poststop_source_authority_unchanged']is True
        assert terminal['root_binding_sha256']==sha(authority['binding'])
        terminals.append(terminal)
    try:
        assert len(terminals)==1
        assert sha(manifest_path)==manifest_pin,'Controller authority changed during cohorts'
        # Large installed core/runtime/SDK closure is verified once for this batch.
        # No extra per-route large-binary hash pass is introduced after the stop.
        runtime=Path(controller).parent/'stage/runtime.json'
        assert sha(runtime)==manifest['runtime_sha256'],'Postcohort runtime manifest changed'
        verifier=[str(python),str(Path(controller).parent/'snapshot/tools/ci_measurement_artifacts.py'),'verify','--wheel-dir',str(Path(controller).parent/'stage/wheels'),'--artifact',str(runtime)]
        with (results/'first-cohort-identity-after.log').open('w')as log:
            checked=subprocess.run(verifier,stdout=log,stderr=subprocess.STDOUT)
        write_json(results/'first-cohort-identity-after-exit.json',dict(exit=checked.returncode))
        assert checked.returncode==0,'Postcohort installed core/runtime/SDK verifier failed'
        assert sha(runtime)==manifest['runtime_sha256'],'Runtime manifest changed during verification'
        assert sha(manifest_path)==manifest_pin,'Controller authority changed during cohorts'
    except BaseException as error:
        write_json(results/'first-cohort-failure.json',dict(phase='postbatch-identity',type=type(error).__name__,error=str(error),retry=False,completed_routes=len(terminals)))
        raise
    write_json(results/'first-cohort-summary.json',dict(status='PASS',original_public_topcalls=1,native_public_topcalls=1,timed_batches=0,sampler_body_calls=0,poststop_source_authority_unchanged=True,postbatch_offline_identity_verified=True,routes=terminals))



def read_manifest(controller):
    path=Path(controller)/CONTRACT;manifest=json.loads(path.read_bytes())
    assert manifest['schema']=='next-metrics-measurement-v1'
    assert manifest['status']=='ROOT_AUTHORIZED_ONE_INITIAL_COMPARISON','UNBOUND: no helper/target/sampler execution'
    assert manifest['first_cohort']['status']=='ROOT_AUTHORIZED_GENUINE_FIRST_COHORTS'
    assert set(manifest['first_cohort']['routes'])=={ROUTE}
    assert type(manifest['SDK_members'])is int and manifest['SDK_members']>0
    scope=manifest['execution_scope']
    assert tuple(scope[k]for k in('canonical_cases','canonical_raw_samples','warmup_groups','workers','repeat','target_ms','warmup_min_ms'))==(8,112,16,1,7,100,250)
    assert scope['untimed_topcalls']==dict(original=1,native=1)
    assert scope['single_host_same_installed_runtime']is scope['no_automatic_retry']is scope['retain_all_losses']is True
    context=json.loads((Path(controller)/DRIVER_CONTRACT).read_bytes())
    assert manifest['first_cohort']['routes'][ROUTE]==context['route']
    assert manifest['cases']==context['cases']and len(context['cases'])==8
    assert manifest['families']['rolling_ball_native']==dict(contract=DRIVER_CONTRACT,cases=8,case_ids=[v['id']for v in context['cases']],untimed_calls_per_case=1)
    return path,manifest


def driver_adapter(root):
    path=Path(root)/'benchmarks/rolling_ball_execution.py'
    spec=importlib.util.spec_from_file_location('_outer_rolling_ball_driver',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def verify_source_state(binding,manifest):
    root=Path(binding['source_root']).resolve();git=binding['git_argv']
    assert not os.environ.get('GIT_DIR')and not os.environ.get('GIT_WORK_TREE')
    assert subprocess.check_output([*git,'rev-parse','HEAD'],text=True).strip()==binding['source_revision']
    assert not subprocess.check_output([*git,'status','--porcelain'],text=True).strip()
    assert binding['source_revision']==manifest['first_cohort']['source_revision']==manifest['producer']['revision']==manifest['prepared_source']['revision']
    assert binding['prepared_source']==manifest['prepared_source']and binding['producer']==manifest['producer']
    assert binding['runtime_sha256']==manifest['runtime_sha256']and binding['core_sha256']==manifest['core_sha256']
    assert len(binding['SDK_members'])==manifest['SDK_members']and type(manifest['SDK_members'])is int and manifest['SDK_members']>0
    for name,pin in manifest['first_cohort']['source_files_raw_sha256'].items():assert sha(root/name)==pin,name


def verify_helper_state(binding,go):
    path=Path(binding['helper_binding_path']);assert sha(path)==binding['helper_binding_sha256']==go['helper_binding_sha256']
    helper=json.loads(path.read_bytes());env=helper['env']
    assert set(env)=={'UTIL_VIEWS_FENV_MODE','UTIL_VIEWS_FENV_RECEIPT','UTIL_VIEWS_FENV_RECEIPT_SHA256'}
    assert env['UTIL_VIEWS_FENV_MODE']=='source-built-v1'and not os.environ.get('UTIL_VIEWS_FENV_HELPER')
    assert all(os.environ.get(k)==v for k,v in env.items())
    receipt=Path(env['UTIL_VIEWS_FENV_RECEIPT']);root=receipt.resolve().parent
    assert sha(receipt)==env['UTIL_VIEWS_FENV_RECEIPT_SHA256']==helper['files'][receipt.name]
    for name,pin in helper['files'].items():
        member=(root/name).resolve();assert member.is_relative_to(root)and sha(member)==pin


def finish_probe(folder,stopped,binding,binding_path,go_path,go,binding_pin,go_pin,restored,manifest):
    # The admitted mandatory sampler-entry stop bypasses driver epilogues.
    # Restore first, then authenticate the small mutable authority boundary.
    folder=Path(folder)
    try:
        assert stopped and restored,'Outer state not restored'
        assert sha(binding_path)==binding_pin and sha(go_path)==go_pin,'Poststop authority changed'
        assert sha(__file__)==go['probe_sha256'],'Poststop probe changed'
        assert datetime.now(timezone.utc)<datetime.fromisoformat(go['deadline_utc']),'Poststop GO expired'
        verify_source_state(binding,manifest)
        route=binding['routes'][stopped['route']]
        assert route==manifest['first_cohort']['routes'][stopped['route']]
        assert sha(Path(binding['source_root'])/DRIVER_CONTRACT)==binding['base_contract_sha256']
        verify_helper_state(binding,go)
    except BaseException as error:
        write_json(folder/'failure.json',dict(phase='poststop-source-authority',type=type(error).__name__,error=str(error),counts=stopped.get('counts'),retry=False,outer_state_restored=restored))
        raise
    write_json(folder/'terminal.json',dict(status='PASS_REAL_CLI_STOP_BEFORE_BODY',**stopped,root_binding_sha256=binding_pin,outer_state_restored=True,poststop_source_authority_unchanged=True,core_runtime_SDK_scope='Pending once-after-batch offline verifier before summaryPASS'))


def driver_argv(root,route,binding,output,deadline=None):
    # Frozen v1 route includes its driver as argv[0]. Consume it exactly once.
    declared=route['argv'];assert declared[0]==route['driver']=='benchmarks/rolling_ball_native.py'
    assert route['api']=='restoration.rolling_ball'
    argv=[str(Path(root)/route['driver']),*declared[1:],'--threads','1','--repeat','7','--output',str(output),'--root-binding',str(binding)]
    assert '--first-cohort'in argv and '--api'in argv and '--case'in argv and '--deadline-utc'not in argv
    assert argv[argv.index('--api')+1]==route['api']
    assert argv.count(route['driver'])==0 or argv[0]==route['driver']
    return argv


def validate_native_entry(frame,route,root,counts,accepted,input_state,inputs_before):
    # Actual c821 execute -> sample_entry -> measure, without invented locals.
    root=Path(root);bridge=frame.f_back;execute=bridge.f_back
    path=root/'benchmarks/rolling_ball_execution.py'
    assert Path(frame.f_code.co_filename).resolve()==root/'tools/montage_sustained.py'and frame.f_code.co_name=='measure'
    assert Path(bridge.f_code.co_filename).resolve()==path and bridge.f_code.co_name=='sample_entry'
    assert Path(execute.f_code.co_filename).resolve()==path and execute.f_code.co_name=='execute'
    c=execute.f_locals;s=bridge.f_locals
    assert counts==dict(original=1,native=1)and accepted==[True]
    assert c['cli'].first_cohort is True and c['cli'].measurement_go is None
    context=json.loads((root/DRIVER_CONTRACT).read_bytes());first=context['cases'][0]
    assert c['spec']==first==s['spec']and execute.f_globals['API']==route['api']=='restoration.rolling_ball'
    assert execute.f_globals['CONTEXT']==DRIVER_CONTRACT and c['context']['route']==route
    assert s['cli']is c['cli']and s['index']==c['index']==0
    assert c['accepted']==[True]and not c['rows']
    requirements=execute.f_globals['native_requirements'](c['spec'])
    assert c['requirements']==requirements==dict(get_num_threads=1,ball_kernel_apply=1,rolling_ball_apply=1)
    assert {k:c['returns'].count(k)for k in requirements}==requirements and len(c['returns'])==sum(requirements.values())
    assert c['expected_state']==c['actual_state']and c['reference_FP']==c['native_FP']and c['reference_before']==c['native_before']
    assert bytes(c['save']())==bytes(c['caller'])==bytes(c['sample_state'])and len(bytes(c['caller']))==32
    assert inputs_before==[input_state((c['args'],c['options']))]
    assert [c['common'].array_state(v,c['np'])for v in c['inputs']]==c['before']
    assert not {'samples','functions','events','loops'}.intersection(frame.f_locals)
    return c


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding',type=Path,required=True);parser.add_argument('--root-go',type=Path,required=True);parser.add_argument('--route',choices=(ROUTE,),required=True)
    args=parser.parse_args();controller=Path(__file__).resolve().parents[1];_,manifest=read_manifest(controller)
    binding_pin,go_pin=sha(args.binding),sha(args.root_go);binding=json.loads(args.binding.read_bytes());go=json.loads(args.root_go.read_bytes())
    route=manifest['first_cohort']['routes'][args.route]
    assert binding['status']=='ROOT_AUTHORIZED_GENUINE_FIRST_COHORTS'and binding['routes']=={args.route:route}
    assert go['binding_sha256']==binding_pin and go['probe_sha256']==sha(__file__)==manifest['first_cohort']['probe_sha256']
    assert go['routes']==[args.route]and go['timed_batches']==0 and go['maximum_original_topcalls']==go['maximum_native_topcalls']==1
    assert datetime.now(timezone.utc)<datetime.fromisoformat(go['deadline_utc'])and Path(binding['root_go_path']).resolve()==args.root_go.resolve()
    root=Path(binding['source_root']).resolve();verify_source_state(binding,manifest);verify_helper_state(binding,go)
    adapter=driver_adapter(root);validated,context,driver_go,bp,gp=adapter.validate_binding(args.binding,root=root)
    assert validated==binding and driver_go==go and(bp,gp)==(binding_pin,go_pin)
    assert Path(sys.executable).resolve()==Path(binding['interpreter']).resolve()
    assert all(os.environ[k]=='1'for k in('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS'))
    assert os.environ['PYTHONDONTWRITEBYTECODE']=='1'and not os.environ.get('SKIMAGEXX_DISABLE_SIMD')
    folder=Path(binding['output_root'])/args.route;folder.mkdir(parents=True,exist_ok=False)
    output=folder/'never-written.json';oldpath=list(sys.path);sys.path[:0]=[binding['installed_site'],str(root/'benchmarks'),str(root/'tools'),str(root)]
    import numpy as np
    import skimage
    import skimagexx as xx
    from skimagexx import _compat
    assert np.__version__==binding['numpy']and skimage.__version__=='0.26.0'
    assert Path(xx._core.__file__).resolve()==Path(binding['core_path']).resolve()
    assert Path(xx.__file__).resolve().is_relative_to(Path(binding['installed_site']).resolve())
    family,name=route['api'].split('.');public=importlib.import_module('skimage.'+family);owned=importlib.import_module('skimagexx._rolling_ball');native=getattr(getattr(xx,family),name);original=getattr(public,name)
    assert native is getattr(owned,name)and owned._core is xx._core
    assert str(inspect.signature(native))==str(inspect.signature(original))==context['signature']
    policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize();workers,fallback=xx.get_num_threads(),_compat._fallback_enabled;assert workers==1
    common=adapter.support(root,context);save,restore,_=common.fp_controller(root,context);outer=save()
    oldtrace,oldprofile,oldargv=sys.gettrace(),sys.getprofile(),sys.argv
    counts=dict(original=0,native=0);accepted=[];inputs_before=[];stopped={};native_active=False;restored=False
    def input_state(v):
        if isinstance(v,np.ndarray):return(id(v),v.dtype.str,repr(v.dtype.metadata),v.shape,v.strides,v.ctypes.data,v.flags.writeable,v.tobytes())
        if isinstance(v,(tuple,list)):return(type(v).__name__,tuple(input_state(x)for x in v))
        if isinstance(v,dict):return tuple((k,input_state(x))for k,x in sorted(v.items()))
        return v
    def journal(event,role):
        with(folder/'calls.jsonl').open('a',encoding='utf-8',newline='\n')as stream:
            stream.write(json.dumps(dict(event=event,role=role,counts=dict(counts)))+'\n');stream.flush();os.fsync(stream.fileno())
    class StopBeforeBody(BaseException):pass
    def trace(frame,event,value):
        nonlocal native_active
        role='original'if frame.f_code is original.__code__ else'native'if frame.f_code is native.__code__ else None
        if event=='call'and native_active and frame.f_globals.get('__name__','').startswith(('skimage.','scipy.ndimage')):raise AssertionError('Native entered original/provider algorithm')
        if event=='call'and role:
            assert counts[role]==0 and not native_active;counts[role]+=1;journal('before',role)
            if role=='original':
                parent=frame.f_back;assert Path(parent.f_code.co_filename).resolve()==root/'benchmarks/rolling_ball_execution.py'
                inputs_before.append(input_state((parent.f_locals['args'],parent.f_locals['options'])))
            else:native_active=True
        if event=='return'and role:
            if role=='native':native_active=False;accepted.append(value is not None)
            journal('after',role)
        return trace
    def profile(frame,event,value):
        if event=='call'and frame.f_code.co_name=='measure'and Path(frame.f_code.co_filename).resolve()==root/'tools/montage_sustained.py':
            c=validate_native_entry(frame,route,root,counts,accepted,input_state,inputs_before)
            assert(np.geterr(),np.geterrcall(),np.getbufsize())==(policy,handler,buffer)and xx.get_num_threads()==1 and not _compat._fallback_enabled
            stopped.update(source=binding['source_revision'],route=args.route,api=route['api'],counts=dict(counts),native_result_accepted=accepted,real_sampler=str(root/'tools/montage_sustained.py'),sampler_body_calls=0,timed_batches=0,public_driver_checks_completed=True,caller_full32_restored=True,native_returns=c['returns'],native_requirements=c['requirements'],marker_evidence='Actual execute inner observer: one Python get_num_threads, owned ball kernel and rolling min-plus; outer profiler is paused during native.')
            write_json(folder/'sampler-entry.json',stopped);raise StopBeforeBody()
    argv=driver_argv(root,route,args.binding,output)
    write_json(folder/'precall.json',dict(source=binding['source_revision'],argv=argv,route=route,product_kind=binding['product_kind'],binding_sha256=binding_pin))
    try:
        try:sys.settrace(trace);sys.setprofile(profile);sys.argv=argv;runpy.run_path(str(root/route['driver']),run_name='__main__')
        except StopBeforeBody:pass
        assert stopped and not output.exists()and not output.with_suffix('.batches.jsonl').exists()
    except BaseException as error:
        write_json(folder/'failure.json',dict(type=type(error).__name__,error=str(error),counts=counts,traceback=traceback.format_exc(),retry=False));raise
    finally:
        sys.settrace(oldtrace);sys.setprofile(oldprofile);sys.argv=oldargv;sys.path[:]=oldpath
        xx.set_num_threads(workers);xx.set_fallback_enabled(fallback);np.seterr(**policy);np.seterrcall(handler);np.setbufsize(buffer);restore(outer)
        restored=xx.get_num_threads()==workers and _compat._fallback_enabled==fallback and(np.geterr(),np.geterrcall(),np.getbufsize())==(policy,handler,buffer)and bytes(save())==bytes(outer)
    finish_probe(folder,stopped,binding,args.binding,args.root_go,go,binding_pin,go_pin,restored,manifest)


def prepare_cohorts(controller,snapshot,results,manifest,python,site,deadline):
    assert manifest['deadline_utc']is None and manifest['execution_clock']=='ROOT_DISPATCH_INPUT_BOUND'
    assert datetime.now(timezone.utc)<datetime.fromisoformat(deadline)
    controller=Path(controller);snapshot=Path(snapshot);results=Path(results);first=manifest['first_cohort'];routes=first['routes']
    assert manifest['status']=='ROOT_AUTHORIZED_ONE_INITIAL_COMPARISON'and first['status']=='ROOT_AUTHORIZED_GENUINE_FIRST_COHORTS'
    assert set(routes)=={ROUTE}and manifest['execution_scope']['untimed_topcalls']==dict(original=1,native=1)
    assert first['source_revision']==manifest['producer']['revision']==manifest['prepared_source']['revision']
    probe=controller/first['probe_path'];assert sha(probe)==first['probe_sha256']
    admission=controller/first['root_admission_path'];stager=controller/first['root_stager_path'];runtime=controller.parent/'stage/runtime.json'
    assert sha(admission)==first['root_admission_sha256']and sha(stager)==first['root_stager_sha256']and sha(runtime)==manifest['runtime_sha256']
    actual=json.loads(runtime.read_bytes());SDK=actual['runtime']['sdk_sha256'];assert len(SDK)==manifest['SDK_members']and type(manifest['SDK_members'])is int and manifest['SDK_members']>0
    core=Path(site)/first['core_member'];assert core.resolve().is_relative_to(Path(site).resolve())and sha(core)==manifest['core_sha256']
    helper=results/'helper-binding.json';assert helper.is_file()
    common=dict(status='ROOT_AUTHORIZED_GENUINE_FIRST_COHORTS',product_kind='GENUINE_INSTALLED_WHOLE_PRODUCT',source_root=str(snapshot.resolve()),source_revision=first['source_revision'],git_argv=['git','-C',str(snapshot.resolve())],prepared_source=manifest['prepared_source'],producer=manifest['producer'],interpreter=str(python),installed_site=str(site),numpy=actual['runtime']['versions']['numpy'],core_path=str(core.resolve()),core_sha256=manifest['core_sha256'],SDK_members=SDK,runtime_path=str(runtime.resolve()),runtime_sha256=manifest['runtime_sha256'],genuine_root_admission_path=str(admission.resolve()),genuine_root_admission_sha256=first['root_admission_sha256'],genuine_root_stager_path=str(stager.resolve()),genuine_root_stager_sha256=first['root_stager_sha256'],helper_binding_path=str(helper.resolve()),helper_binding_sha256=sha(helper),output_root=str((results/'first-cohort').resolve()))
    verify_source_state(common,manifest);index={}
    for cohort,route in routes.items():
        context=json.loads((snapshot/DRIVER_CONTRACT).read_bytes());assert cohort==ROUTE and context['route']==route
        folder=results/('authority-'+cohort);folder.mkdir(exist_ok=False);binding_path=folder/'binding.json';go_path=folder/'root-go.json'
        value=dict(common,routes={cohort:route},base_contract_sha256=sha(snapshot/DRIVER_CONTRACT),root_go_path=str(go_path.resolve()))
        write_json(binding_path,value)
        write_json(go_path,dict(binding_sha256=sha(binding_path),probe_sha256=first['probe_sha256'],routes=[cohort],timed_batches=0,maximum_original_topcalls=1,maximum_native_topcalls=1,deadline_utc=deadline,helper_binding_sha256=sha(helper)))
        index[cohort]=dict(binding=str(binding_path.resolve()),go=str(go_path.resolve()),measurement=str((folder/'measurement.json').resolve()))
    write_json(results/'cohort-authorities.json',index);return index


def scoped_measurement(manifest,route,parent_sha256,authority,terminal_path,summary):
    # The root-bound parent measurement GO preauthorizes only this fixed scope,
    # conditional on the real cohort and the postbatch verifier. Never create
    # a timing envelope from an unbound manifest or a cohort-only GO.
    assert manifest['status']=='ROOT_AUTHORIZED_ONE_INITIAL_COMPARISON'
    assert summary['status']=='PASS'and summary['postbatch_offline_identity_verified']is True and len(summary['routes'])==1
    raw=Path(terminal_path).read_bytes();done=json.loads(raw);binding_pin=sha(authority['binding'])
    assert done in summary['routes']and done['status']=='PASS_REAL_CLI_STOP_BEFORE_BODY'
    assert done['api']==route['api']and done['root_binding_sha256']==binding_pin
    assert done['counts']==dict(original=1,native=1)and done['timed_batches']==done['sampler_body_calls']==0
    assert done['outer_state_restored']is done['poststop_source_authority_unchanged']is True
    family=Path(route['driver']).stem;admitted=manifest['families'][family]
    assert admitted==dict(contract=DRIVER_CONTRACT,cases=len(manifest['cases']),case_ids=[s['id']for s in manifest['cases']],untimed_calls_per_case=1)
    value=json.loads(json.dumps(manifest));value['completed_first_cohort']=dict(path=str(Path(terminal_path).resolve()),sha256=hashlib.sha256(raw).hexdigest(),binding_sha256=binding_pin)
    value['scoped_parent_sha256']=parent_sha256;value['scoped_api']=route['api'];return value


def measurement_commands(snapshot,results,manifest,python,authorities,deadline):
    return {name:[str(python),str(Path(snapshot)/route['driver']),'--threads','1','--repeat','7','--api',route['api'],'--output',str(Path(results)/'first-cohort'/(name+'.json')),'--root-binding',authorities[name]['binding'],'--measurement-go',authorities[name]['measurement']]for name,route in manifest['first_cohort']['routes'].items()}


def run_measurements(snapshot,results,manifest,python,authorities,deadline):
    assert manifest['deadline_utc']is None and manifest['execution_clock']=='ROOT_DISPATCH_INPUT_BOUND'
    assert datetime.now(timezone.utc)<datetime.fromisoformat(deadline)
    results=Path(results);controller=Path(snapshot).parent/'controller';parent=controller/CONTRACT;parent_pin=sha(parent)
    assert json.loads(parent.read_bytes())==manifest and manifest['status']=='ROOT_AUTHORIZED_ONE_INITIAL_COMPARISON'
    summary=json.loads((results/'first-cohort-summary.json').read_bytes())
    for name,route in manifest['first_cohort']['routes'].items():
        target=Path(authorities[name]['measurement']);assert not target.exists()
        scoped=scoped_measurement(manifest,route,parent_pin,authorities[name],results/'first-cohort'/name/'terminal.json',summary);scoped['deadline_utc']=deadline;write_json(target,scoped)
    verifier=[str(python),str(Path(snapshot)/'tools/ci_measurement_artifacts.py'),'verify','--wheel-dir',str(Path(snapshot).parent/'stage/wheels'),'--artifact',str(Path(snapshot).parent/'stage/runtime.json')]
    for cohort,command in measurement_commands(snapshot,results,manifest,python,authorities,deadline).items():
        assert sha(parent)==parent_pin
        with(results/(cohort+'-identity-before.log')).open('w')as log:subprocess.run(verifier,stdout=log,stderr=subprocess.STDOUT,check=True)
        write_json(results/(cohort+'-command.json'),command)
        with(results/(cohort+'.log')).open('w')as log:done=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        write_json(results/(cohort+'-exit.json'),dict(exit=done.returncode));assert done.returncode==0,'Measurement failed; no automatic retry: '+cohort
        with(results/(cohort+'-identity-after.log')).open('w')as log:subprocess.run(verifier,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert sha(parent)==parent_pin


def sampler_order_scope(snapshot):
    # Authenticate the actual admitted caller and sampler formula, source only.
    tree=ast.parse((Path(snapshot)/'benchmarks/rolling_ball_execution.py').read_bytes())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='sample_entry')
    calls=[n for n in ast.walk(function)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='measure'];assert len(calls)==1
    keywords={v.arg:v.value for v in calls[0].keywords}
    assert isinstance(keywords['case_index'],ast.Name)and keywords['case_index'].id=='index'
    value=keywords['pass_index'];assert isinstance(value,ast.Constant)and type(value.value)is int
    execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
    loop=next(n for n in ast.walk(execute)if isinstance(n,ast.For)and ast.unparse(n.target)=='(index, spec)')
    assert ast.unparse(loop.iter)=='enumerate(specs)'
    sampler=ast.parse((Path(snapshot)/'tools/montage_sustained.py').read_bytes())
    measure=next(n for n in sampler.body if isinstance(n,ast.FunctionDef)and n.name=='measure')
    order=next(n.value for n in ast.walk(measure)if isinstance(n,ast.Assign)and any(isinstance(v,ast.Name)and v.id=='order'for v in n.targets))
    expected=ast.parse('[0,1] if (round_id+case_index+pass_index)%2==0 else [1,0]',mode='eval').body
    assert ast.dump(order)==ast.dump(expected)
    return value.value


def validate_reports(snapshot,results,manifest,authorities):
    # Existing Montage journal/median/order reconstruction; no timing engine.
    import math,statistics
    snapshot=Path(snapshot);results=Path(results);report_root=results/'first-cohort';summaries={};adapter=driver_adapter(snapshot);pass_index=sampler_order_scope(snapshot)
    for cohort,route in manifest['first_cohort']['routes'].items():
        context=json.loads((snapshot/DRIVER_CONTRACT).read_bytes());specs=manifest['cases'];count=len(specs)
        assert specs==context['cases']and context['method']['repeat']==7
        report=json.loads((report_root/(cohort+'.json')).read_bytes());meta=report['metadata']
        assert {k:meta[k]for k in manifest['benchmark_fingerprint']}==manifest['benchmark_fingerprint']
        assert meta['core_sha256']==manifest['core_sha256']and meta['runtime_sha256']==manifest['runtime_sha256']
        assert meta['SDK_members']==json.loads(Path(authorities[cohort]['binding']).read_bytes())['SDK_members']
        assert meta['threads']==1 and meta['repeat']==7 and meta['target_ms']==100 and meta['warmup_min_ms']==250
        assert meta['contract_sha256']==sha(snapshot/DRIVER_CONTRACT)and len(report['results'])==count
        assert meta['root_binding_sha256']==sha(authorities[cohort]['binding'])and meta['measurement_sha256']==sha(authorities[cohort]['measurement'])
        assert meta['producer']==manifest['producer']and meta['source_revision']==manifest['producer']['revision']
        assert meta['method_sha256']==context['method']['method_sha256']and meta['retain_all_losses']is True
        assert meta['source_before']==manifest['benchmark_fingerprint']
        journal=[json.loads(line)for line in(report_root/(cohort+'.batches.jsonl')).read_text().splitlines()]
        assert {event['case_id']for event in journal}=={spec['id']for spec in specs}
        assert [row['case_id']for row in report['results']]==[spec['id']for spec in specs]
        journal=[json.loads(line)for line in (report_root/(cohort+'.batches.jsonl')).read_text().splitlines()]
        assert {event['case_id']for event in journal}=={spec['id']for spec in specs}
        for case_index,(row,spec) in enumerate(zip(report['results'],specs)):
            assert row['accepted']==[True]and row['api']==route['api']
            assert row['providers']==context['providers']and set(row['FP'])=={'control','status','mxcsr'}
            requirements=adapter.native_requirements(spec)
            assert {k:row['actual_C_returns'].count(k)for k in requirements}==requirements
            assert len(row['actual_C_returns'])==sum(requirements.values())
            assert list(map(len,row['samples_ms']))==[7,7]
            assert all(math.isfinite(x)and x>0 for arm in row['samples_ms']for x in arm)
            assert list(map(len,row['loops']))==[7,7] and all(type(x)is int and x>0 for arm in row['loops']for x in arm)
            assert row['reference_ms']==statistics.median(row['samples_ms'][0])
            assert row['cpp_ms']==statistics.median(row['samples_ms'][1])
            assert row['speedup']==row['reference_ms']/row['cpp_ms']
            events=[event for event in journal if event['case_id']==row['case_id']]
            batches=row['batch_records']
            assert len(events)==2*len(batches)+1
            assert all(event['case_index']==case_index for event in events)
            timing_keys=('reference_ms','cpp_ms','speedup','samples_ms','loops','sample_labels','warmup_elapsed_ns','sample_order','batch_records')
            assert events[-1]['event']=='measure_returned' and events[-1]['timing']=={key:row[key]for key in timing_keys}
            warm=[0,0];accepted=[[],[]];counts=[[],[]];order=[]
            for before,after,batch in zip(events[0:-1:2],events[1:-1:2],batches):
                assert before['event']=='batch_before' and after==batch and after['event']=='batch_after'
                assert all(before[key]==after[key]for key in ('case_index','case_id','function','round','ordinal','loops'))
                assert after['complete'] is True and after['failure'] is None
                arm=after['function'];elapsed=after['elapsed_ns'];n=after['loops']
                assert arm in (0,1) and type(elapsed)is int and elapsed>0 and type(n)is int and n>0
                if after['phase']=='warmup':
                    assert before['requested_phase']=='warmup' and not after['accepted'] and after['round']==-1
                    warm[arm]+=elapsed
                elif after['phase']=='underlength-calibration':
                    assert before['requested_phase']=='sample' and not after['accepted'] and elapsed<100000000
                else:
                    assert before['requested_phase']=='sample' and after['phase']=='sample' and after['accepted'] and elapsed>=100000000
                    accepted[arm].append(elapsed/n/1e6);counts[arm].append(n)
                    order.append((after['round'],after['ordinal'],arm))
            assert warm==row['warmup_elapsed_ns'] and min(warm)>=250000000
            assert accepted==row['samples_ms'] and counts==row['loops']
            expected_order=[[0,1]if(round_id+case_index+pass_index)%2==0 else[1,0]for round_id in range(7)]
            assert row['sample_order']==expected_order
            assert order==[(round_id,ordinal,arm)for round_id,arms in enumerate(expected_order)for ordinal,arm in enumerate(arms)]
        summaries[cohort]=dict(raw_samples=count*14,
            wins=[row['case_id']for row in report['results']if row['speedup']>1],
            losses=[row['case_id']for row in report['results']if row['speedup']<1],
            ties=[row['case_id']for row in report['results']if row['speedup']==1])
    assert sum(v['raw_samples']for v in summaries.values())==manifest['execution_scope']['canonical_raw_samples']
    write_json(results/'summary.json',dict(families=summaries,scope='Current one-thread public original comparison only; all losses retained; no whole-product qualification or prior-C++ speed claim.'))


if __name__=='__main__':main()
