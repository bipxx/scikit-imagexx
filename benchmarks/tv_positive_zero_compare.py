"""TV zero-scan successor: immutable af7 versus root-admitted741 products.

Only reviewed artifact-only Actions may execute the fixed comparison. Different
complete cores/links remain explicit; this is not a helper-causality experiment.
"""
from __future__ import annotations
import argparse,ast,copy,dataclasses,hashlib,importlib,importlib.util,inspect,json,math,os,platform,random,secrets,statistics,subprocess,sys,time,traceback,types,uuid,warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import tv_positive_zero_artifacts as a
require=a.require;sha=a.sha;load_json=a.read_json;write_new=a.save_new;ROLES=a.ROLES

# Reuse mature one-use inherited-stdio transport in this controller's globals.
# The sole transport change clears poisoned child process environment selectors;
# the complete byte inverse below preserves the original authentication protocol.
transport_source=(ROOT/'benchmarks/rank_statistics_compare.py').read_bytes()
require(a.text_sha(ROOT/'benchmarks/rank_statistics_compare.py')==a.RANK_VALIDATOR_SHA256,'historical transport source changed')
transport_names={'startup_plan','startup_frame','authenticate_startup','Worker','SerialScheduler'}
transport_environment_old=b"key in ('PYTHONPATH','PYTHONHOME')"
transport_environment_new=b"key in ('PYTHONPATH','PYTHONHOME','PYTHONOPTIMIZE','PYTHONSTARTUP','PYTEST_ADDOPTS','PYTEST_PLUGINS')"
require(transport_source.count(transport_environment_old)==1,'historical worker environment boundary changed')
transport_derived=transport_source.replace(transport_environment_old,transport_environment_new)
require(transport_derived.replace(transport_environment_new,transport_environment_old)==transport_source,'transport environment inverse differs')
transport_nodes=[n for n in ast.parse(transport_derived).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in transport_names]
require(len(transport_nodes)==len(transport_names),'transport source incomplete')
exec(compile(ast.Module(body=transport_nodes,type_ignores=[]),'<exact-frozen-inherited-stdio-transport>','exec'),globals())


def factory(product,xx):
    """No fixture imports before actual product/public/runtime admission."""
    source=(ROOT/'benchmarks/tv_chambolle.py').read_bytes();require(a.text_sha(ROOT/'benchmarks/tv_chambolle.py')==a.FROZEN_FACTORY_SHA256,'frozen400 source changed')
    names={'contracts','timing_profiles','corpus','inventory','fixture','case_spec','observe','comparison','native_observe','timing_callback','digest','public_admission','resolve'}
    nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names or isinstance(n,ast.ClassDef) and n.name=='Case']
    module=types.ModuleType('_tv_compare_exact_frozen_factory');sys.modules[module.__name__]=module
    module.__dict__.update(dataclass=dataclasses.dataclass,Path=Path,ROOT=ROOT,API='restoration.denoise_tv_chambolle',LIMIT=8*1024*1024,_contracts=None,hashlib=hashlib,json=json,math=math,inspect=inspect,importlib=importlib,sys=sys,os=os,artifacts=types.SimpleNamespace(sha=sha))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<exact20b-TV-factory-observer>','exec'),module.__dict__)
    # Reuse the complete original public guard with its qualified SOURCE root;
    # fixture functions keep the separate immutable controller/contract root.
    provenance=importlib.util.spec_from_file_location('_tv_compare_provenance',ROOT/'benchmarks/provenance.py');p=importlib.util.module_from_spec(provenance);provenance.loader.exec_module(p)
    guard_ns=dict(module.__dict__,ROOT=Path(product)/'source',source_fingerprint=p.source_fingerprint)
    node=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name=='public_admission');exec(compile(ast.Module(body=[node],type_ignores=[]),'<exact20b-public-admission>','exec'),guard_ns)
    return module,guard_ns['public_admission'],p.source_fingerprint


def experiment(args):return a.load_experiment(args.experiment,args.experiment_sha256)
def generation():return os.environ['GITHUB_RUN_ID'],os.environ['GITHUB_RUN_ATTEMPT']
def rotating_triads(index):
    order=list(ROLES);random.Random(81+index).shuffle(order)
    return [order[r%3:]+order[:r%3] for r in range(7)]

def sample_admission(validated,measure,actions,workers,scalar,index,*,default_validated=False):
    require(validated and measure and actions=='true' and type(workers) is int and workers==1 and scalar is False,'timing before strict validation/primary worker forbidden')
    require(type(index) is int and 330<=index<400,'only frozen70 raster IDs may time')
    require(validated is True and default_validated is True,'fresh default70 path not validated before timers')

def verify_environment(product):
    spec=importlib.util.spec_from_file_location('_tv_compare_exact_runtime',ROOT/'tools/ci_measurement_artifacts.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    runtime=a.read_json(Path(product)/'runtime.json')
    require(m.installed_state(Path(product)/'wheels',runtime['wheels'])==runtime['runtime'],'actual installed runtime/SDK/library closure differs')
    return runtime

def worker_permit(path,profile,cfg,role=None,stream=None):
    require(path and os.environ.get('GITHUB_ACTIONS')=='true','direct worker timing forbidden')
    permit=a.read_json(path);run,attempt=generation();require(permit['profile']==profile and permit['tool_sha256']==sha(__file__),'wrong controller permit')
    receipt_rows=verify_barrier(permit['receipts'],cfg,permit['experiment_sha256'],run,attempt)
    require(sha(permit['claim'])==permit['claim_sha256'],'claim bytes differ');claim=a.read_json(permit['claim'])
    require(claim['profile']==profile and claim['generation']==run and claim['attempt']==attempt and claim['receipt_hashes']==[sha(p) for p in permit['receipts']],'claim run/receipts differ')
    return authenticate_startup(permit,claim,permit['claim'],role,profile,run,attempt,stream or sys.stdin)


def validate_authentication_index(report):
    records=report.get('authentication_artifacts')
    require(isinstance(records,dict) and set(records)=={'schema','original_raw_upload','claim','startup_receipts','permit'} and records['schema']=='tv-admission-original-auth-artifacts-v1' and records['original_raw_upload'] is True,'original claim/startup artifacts required')
    require(set(records['startup_receipts'])==set(ROLES),'all three original startup receipts required')
    names={'claim':'authentication/claim.json','permit':'worker-permit.json'}
    for name,record in [('claim',records['claim']),('permit',records['permit'])]+[(role,records['startup_receipts'][role]) for role in ROLES]:
        expected=names[name] if name in names else 'authentication/claim-'+name+'-stdio-startup.json'
        require(set(record)=={'path','sha256','bytes'} and record['path']==expected and type(record['bytes']) is int and record['bytes']>0 and isinstance(record['sha256'],str) and len(record['sha256'])==64 and all(c in '0123456789abcdef' for c in record['sha256']),'original authentication file record malformed')
        if name in ROLES:require(record['sha256']==report['worker_identities']['normal'][name]['startup_authentication']['startup_receipt_sha256'],'startup receipt not bound to actual worker')
    return records


def verify_authentication_artifacts(output,report):
    """Retain original one-use receipts in the uploaded output, no reconstruction."""
    output=Path(output);permit_path=output/'worker-permit.json';permit=a.read_json(permit_path)
    claim_path=output/'authentication'/'claim.json';require(Path(permit['claim']).resolve()==claim_path.resolve(),'claim outside original artifact output')
    require(sha(claim_path)==permit['claim_sha256'],'original claim SHA differs')
    claim=a.read_json(claim_path);auth=claim['authentication'];require({key:permit.get(key) for key in auth}==auth,'original permit/claim authentication differs')
    require(claim['profile']==report['profile'] and permit['profile']==report['profile'] and permit['tool_sha256']==sha(__file__) and permit['experiment_sha256']==report['experiment_sha256'],'original authentication source/profile/experiment differs')
    require(claim['receipt_hashes']==[sha(p) for p in permit['receipts']],'original claim barrier receipt binding differs')
    def record(path):return dict(path=path.relative_to(output).as_posix(),sha256=sha(path),bytes=len(path.read_bytes()))
    startup={}
    for role in ROLES:
        path=claim_path.with_name('claim-'+role+'-stdio-startup.json');r=a.read_json(path);identity=report['worker_identities']['normal'][role];observed=identity['startup_authentication']
        require(r['mechanism']==observed['mechanism']=='one-use-inherited-stdio-v1' and r['role']==role and r['profile']==report['profile'],'original startup role/profile/mechanism differs')
        require(r['generation']==claim['generation'] and r['attempt']==claim['attempt'] and r['tool_sha256']==auth['tool_sha256']==permit['tool_sha256'] and r['nonce_sha256']==auth['startup_sha256'][role],'original startup run/source/nonce binding differs')
        require(all(r[key]==observed[key] for key in ('controller_pid','launcher_pid','worker_pid','worker_parent_pid')) and r['worker_pid']==identity['pid'] and r['controller_pid']==auth['controller_pid'],'original startup PID binding differs')
        startup[role]=record(path)
    result=dict(schema='tv-admission-original-auth-artifacts-v1',original_raw_upload=True,claim=record(claim_path),startup_receipts=startup,permit=record(permit_path))
    validate_authentication_index(dict(report,authentication_artifacts=result));return result

def default_state(np,xx,wrapper,workers,*,validate=True):
    state=dict(policy=np.geterr(),handler_is_none=np.geterrcall() is None,buffer_elements=np.getbufsize(),
               providers_pristine=wrapper._providers_pristine(),workers=xx.get_num_threads())
    if validate:a.validate_default_state(state,workers)
    return state


class DefaultWitness:
    """Passive profile events; logs contain primitives, never frames/arrays.

    Only immutable code/builtin identities are retained. A frame-id maps to
    primitive call metadata until its return, then is removed. No function,
    allocation provider, NumPy policy/handler, or numerical call is replaced.
    """
    def __init__(self,np,core,wrapper,role,original,inner):
        self.array_type=np.ndarray;self.role=role;self.finite=core.tv_chambolle_finite_run;self.helper=wrapper._finite_call.__code__
        require(inspect.isbuiltin(self.finite) and self.finite.__name__=='tv_chambolle_finite_run' and self.finite.__module__==core.__name__,'real owned finite builtin required')
        require(inspect.isfunction(wrapper._finite_call) and wrapper._finite_call.__module__=='skimagexx._tv_chambolle' and wrapper._core is core,'real owned finite helper required')
        self.original=original;self.inner=inner;self.outer_code=inspect.unwrap(original).__code__;self.inner_code=inner.__code__
        self.stages={name:getattr(core,'tv_chambolle_'+name+'_into') for name in ('divergence','gradient','norm','dual')}
        require(all(inspect.isbuiltin(fn) and fn.__name__=='tv_chambolle_'+name+'_into' and fn.__module__==core.__name__ for name,fn in self.stages.items()),'owned materialized stage builtins required')
        self.active={}
        self.data=dict(finite=dict(helper_calls=0,c_call=0,c_return=0,c_exception=0,accepted=0,rejected=0,returns=[]),
            stages={n:dict(c_call=0,c_return=0,c_exception=0) for n in self.stages},
            original=dict(outer=0,inner=0,denied=0),helper_marker='owned-_finite_call-real-builtin-v1')

    def __call__(self,frame,event,arg):
        source=self.data['original'];finite=self.data['finite']
        if event=='call' and frame.f_code in (self.outer_code,self.inner_code):
            if self.role!='original':source['denied']+=1;raise AssertionError('pristine original algorithm forwarding forbidden')
            source['outer' if frame.f_code is self.outer_code else 'inner']+=1
        if event=='c_call' and (arg is self.original or arg is self.inner) and self.role!='original':
            source['denied']+=1;raise AssertionError('pristine original compiled forwarding forbidden')
        if event=='call' and frame.f_code is self.helper:
            arguments=frame.f_locals.get('arguments');function=frame.f_locals.get('function')
            require(function is self.finite and type(arguments) is tuple and len(arguments)==9,'finite helper must invoke exact real builtin')
            image=arguments[0];require(type(image) is self.array_type,'finite helper input must be plain ndarray')
            finite['helper_calls']+=1
            self.active[id(frame)]=dict(dtype=image.dtype.name,shape=[int(v) for v in image.shape],weight=arguments[4],eps=arguments[5],max_iterations=arguments[6],profile=arguments[7],buffer_elements=arguments[8],called=0,returned=0,exceptions=0)
        if event in ('c_call','c_return','c_exception'):
            if arg is self.finite:
                require(frame.f_code is self.helper and id(frame) in self.active,'finite builtin bypassed owned helper')
                finite[event]+=1;context=self.active[id(frame)];context[{'c_call':'called','c_return':'returned','c_exception':'exceptions'}[event]]+=1
            for name,function in self.stages.items():
                if arg is function:self.data['stages'][name][event]+=1
        if event=='return' and frame.f_code is self.helper:
            context=self.active.pop(id(frame));valid=(context['called']==context['returned']==1 and context['exceptions']==0 and type(arg) is tuple and len(arg)==4)
            if valid:
                value,iterations,alias,trace=arg
                valid=(type(value) is self.array_type and type(iterations) is int and type(alias) is bool and type(trace) is list and not trace
                       and value.dtype.name==context['dtype'] and list(value.shape)==context['shape']
                       and (value is frame.f_locals['arguments'][0])==alias)
            if valid:
                finite['accepted']+=1;record={k:v for k,v in context.items() if k not in ('called','returned','exceptions')};record.update(iterations=iterations,input_alias=alias);finite['returns'].append(record)
            else:finite['rejected']+=1

    def finish(self):
        require(not self.active,'unfinished finite helper frames');return self.data


def default_bases(value,np,h):
    result=[];seen=set()
    while isinstance(value,np.ndarray) and value.base is not None:
        base=value.base;require(id(base) not in seen and len(result)<32,'bounded base chain required');seen.add(id(base))
        result.append(dict(array=h.contracts().array(base),offset=None if not isinstance(base,np.ndarray) else int(value.ctypes.data-base.ctypes.data)))
        value=base
    return result


def observe_default(function,case,h,np,xx,wrapper,role,workers,*,prepared=None):
    # Setup/serialization is untimed. Unlike frozen legacy observe(), this
    # routine never sets NumPy policy/handler or installs allocation providers.
    before_state=default_state(np,xx,wrapper,workers)
    passed,image,owner,options,events=h.fixture(case) if prepared is None else prepared;require(not events,'default timed fixture contains custom protocols')
    require(default_state(np,xx,wrapper,workers)==before_state,'fixture changed default state')
    before=dict(image=h.contracts().array(image),owner=h.contracts().array(owner));pointer=image.ctypes.data
    bases_before=dict(image=default_bases(image,np,h),owner=default_bases(owner,np,h))
    module=importlib.import_module('skimage.restoration._denoise');original=h.resolve(True)
    witness=DefaultWitness(np,xx._core,wrapper,role,original,module._denoise_tv_chambolle_nd)
    previous=sys.getprofile();require(previous is None,'pristine default profiling state required')
    value=None;error=None
    try:
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            try:
                sys.setprofile(witness);value=function(passed,**options)
            except Exception as caught:error=dict(type=type(caught).__name__,message=str(caught))
            finally:sys.setprofile(previous)
        result=None
        if value is not None:
            result=dict(array=h.contracts().array(value),identity_image=value is image,identity_owner=value is owner,
                shares_image=bool(np.shares_memory(value,image)),shares_owner=bool(np.shares_memory(value,owner)),
                base_is_image=value.base is image,base_is_owner=value.base is owner)
        observation=dict(result=result,error=error,warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in notices],events=events,
            before=before,after=dict(image=h.contracts().array(image),owner=h.contracts().array(owner)),pointer_stable=image.ctypes.data==pointer,
            bases=dict(before=bases_before,after=dict(image=default_bases(image,np,h),owner=default_bases(owner,np,h)),result=default_bases(value,np,h)))
        after_state=default_state(np,xx,wrapper,workers,validate=False)
        # These70 are declared finite. Preserve raw words rather than silently
        # using the legacy NaN-only normalization to admit a nonfinite result.
        return dict(observation_digest=a.digest(observation),observation_payload=a.pack_default_observation(observation),state_before=before_state,state_after=after_state,dispatch=witness.finish())
    finally:
        require(sys.getprofile() is previous,'default profiler was not restored')


def retain_observation(phase, value):
    """Original worker stderr is exclusive and uploaded even when checks fail."""
    sys.stderr.write(json.dumps(dict(schema='tv-zero-observation-journal-v1',
        phase=phase, value=value), separators=(',', ':'), allow_nan=False)+'\n')
    sys.stderr.flush()


def prepare_timing_callback(h,np,xx,wrapper,role,profile,case,index,expected,records):
    """Validate the actual callback's one fixture before warmup or any timer."""
    require(case.timed and case.source_id is None,'only frozen timed fixtures')
    function=h.resolve(role=='original')
    prepared=h.fixture(case);passed,image,owner,options,events=prepared
    require(not events and all(isinstance(v,(int,float)) for k,v in options.items() if k!='channel_axis'),
            'timing fixture protocols changed')
    row=observe_default(function,case,h,np,xx,wrapper,role,1,prepared=prepared)
    row['case_index']=index;records.append(row);retain_observation('actual-timing-fixture',row)
    a.validate_default_state(row['state_after'],1)
    require(row['state_before']==row['state_after'],'actual timing fixture changed default state')
    a.validate_default_dispatch(row['dispatch'],role,profile,a.default_rule(index,h.inventory(case)))
    require(row['observation_digest']==expected and a.unpack_default_observation(row['observation_payload'])['error'] is None,
            'actual timing fixture differs from fresh whole-default sweep')
    contracts=h.contracts
    before=dict(image=contracts().array(image),owner=contracts().array(owner))
    # These closure bodies are exact copies of the frozen timing_callback.
    def call():return function(passed,**options)
    return call,lambda:dict(image=contracts().array(image),owner=contracts().array(owner))==before


def check_default_worker(h,np,xx,wrapper,role,profile,scalar,cases,value):
    value.update(schema=a.DEFAULT_SCHEMA,role=role,profile=profile,mode='scalar' if scalar else 'normal',checks=[],mismatches=[])
    try:
        for workers in (1,4):
            xx.set_num_threads(workers);records=[]
            check=dict(workers=workers,records=records,digests=[]);value['checks'].append(check)
            for index in range(330,400):
                row=observe_default(h.resolve(role=='original'),cases[index],h,np,xx,wrapper,role,workers);row['case_index']=index
                records.append(row);check['digests'].append(row['observation_digest']);retain_observation('default-sweep',row)
                observation=a.unpack_default_observation(row['observation_payload'])
                a.validate_default_state(row['state_after'],workers);require(row['state_before']==row['state_after'],'default policy/handler/buffer/provider/globalworker changed')
                require(h.comparison(observation)==observation,'default finite result unexpectedly needs NaN normalization')
                a.validate_default_dispatch(row['dispatch'],role,profile,a.default_rule(index,h.inventory(cases[index])))
                require(observation['error'] is None,'default finite timed fixture failed')
        require(value['checks'][0]['digests']==value['checks'][1]['digests'],'default workers1/4 output differs')
        require([r['dispatch'] for r in value['checks'][0]['records']]==[r['dispatch'] for r in value['checks'][1]['records']],'default workers1/4 dispatch differs')
        return value
    finally:xx.set_num_threads(1)


def worker_main(args):
    cfg=experiment(args);product=Path(args.product);product_role='before' if args.role=='original' else args.role
    startup=worker_permit(args.permit,args.profile,cfg,args.role) if args.measure else None
    seal=a.verify_product(product,product_role,args.profile,cfg) # BEFORE any numerical imports.
    import skimagexx as xx
    require(Path(xx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()),'not ordinary installed worker prefix')
    frame=cfg['products'][product_role][args.profile];require(sha(xx._core.__file__)==frame['core_sha256'],'wrong/borrowed core')
    runtime=verify_environment(product);previous=xx.get_num_threads();require(previous==1 and type(previous) is int,'default workers must be1');xx.set_fallback_enabled(False)
    h,guard,fingerprint=factory(product,xx);M=a.read_json(ROOT/'coverage/contracts/tv-chambolle-benchmark.json')
    guard(M,dict(core_sha256=frame['core_sha256'],product_source_sha256=fingerprint(product/'source')['source_sha256']))
    cases=h.corpus();require(len(cases)==400 and [h.inventory(v) for v in cases]==M['case_inventory'],'frozen400 inventory changed')
    # The exact public guard precedes contracts/fixtures; characterize source
    # helpers only now. Counterparts stay in separate ordinary-name processes.
    np=h.contracts().np
    require(h.contracts().profile()==args.profile and h.contracts().skimage.__version__=='0.26.0','wrong original runtime profile')
    cpu_source=(ROOT/'benchmarks/resize.py').read_bytes();cpu_node=next(n for n in ast.parse(cpu_source).body if isinstance(n,ast.FunctionDef) and n.name=='cpu_metadata')
    cpu_ns=dict(np=np,os=os,platform=platform,Path=Path);exec(compile(ast.Module(body=[cpu_node],type_ignores=[]),'<exact-shared-CPU-metadata>','exec'),cpu_ns)
    cpu=cpu_ns['cpu_metadata']()
    identity=dict(role=args.role,pid=os.getpid(),source=seal['revision'],product_sha256=seal['product_sha256'],core_sha256=sha(xx._core.__file__),core_path=str(Path(xx._core.__file__).resolve()),profile=args.profile,cpu=cpu,python=sys.version,numpy=np.__version__,scalar=os.environ.get('SKIMAGEXX_DISABLE_SIMD')=='1',default_workers=previous,has_avx2=xx.has_avx2(),fallback_enabled=False,startup_authentication=startup,guard_environment=a.execution_environment())
    print(json.dumps(dict(ok=True,identity=identity)),flush=True);validated=False;default_validated=False;callbacks={};timing_fixture_checks=[]
    wrapper=importlib.import_module('skimagexx._tv_chambolle');default_state(np,xx,wrapper,1)
    for line in sys.stdin:
        request=json.loads(line)
        try:
            operation=request['operation']
            if operation=='validate':
                require(request['workers']==[1,4],'both workers required');checks=[];mismatches=[]
                for workers in (1,4):
                    xx.set_num_threads(workers);digests=[]
                    for index,case in enumerate(cases):
                        value=h.observe(h.resolve(True),case) if args.role=='original' else h.native_observe(h.resolve(),case)
                        observed=h.digest(h.comparison(value));digests.append(observed);expected=M['reference_profiles'][args.profile]['observations'][str(index)]
                        if observed!=expected:
                            mismatch=dict(index=index,id=case.id,worker=workers,expected=expected,actual=value)
                            mismatches.append(mismatch);retain_observation('legacy-mismatch',mismatch)
                    checks.append(dict(workers=workers,digests=digests))
                xx.set_num_threads(1);require(checks[0]['digests']==checks[1]['digests'] and not mismatches,'frozen400 exact observation differs');validated=True;reply=dict(checks=checks,mismatches=mismatches)
            elif operation=='validate-default':
                require(validated and request['workers']==[1,4],'legacy400 validation required before default70')
                default_checks={};check_default_worker(h,np,xx,wrapper,args.role,args.profile,identity['scalar'],cases,default_checks);default_validated=True;reply=default_checks
            elif operation in ('warm','sample'):
                index=request['case_index'];sample_admission(validated,args.measure,os.environ.get('GITHUB_ACTIONS'),xx.get_num_threads(),identity['scalar'],index,default_validated=default_validated)
                default_state(np,xx,wrapper,1)
                if operation=='warm':require(index not in callbacks,'duplicate timing fixture warmup')
                if index not in callbacks:
                    require(operation=='warm','actual timing fixture must be checked before samples')
                    require(index==330+len(callbacks),'complete ordered timing fixture sequence required')
                    expected=default_checks['checks'][0]['digests'][index-330]
                    callbacks[index]=prepare_timing_callback(h,np,xx,wrapper,args.role,args.profile,
                        cases[index],index,expected,timing_fixture_checks)
                call,unchanged=callbacks[index]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    if operation=='warm':result=call();del result;reply=dict(warmed=True,timing_fixture=timing_fixture_checks[-1])
                    else:
                        loops=request['loops'];require(type(loops) is int and 1<=loops<=1000,'invalid loops')
                        time.sleep(.02);start=time.perf_counter_ns()
                        for _ in range(loops):result=call();del result
                        elapsed=time.perf_counter_ns()-start;reply=dict(ms=elapsed/loops/1e6,elapsed_ns=elapsed,loops=loops,identity=identity)
                require(unchanged(),'timed input/owner state changed')
                default_state(np,xx,wrapper,1)
            elif operation=='integrity':
                require(a.verify_product(product,product_role,args.profile,cfg)==seal,'product source/identity changed');verify_environment(product);require(cpu_ns['cpu_metadata']()==cpu,'worker CPU identity changed');reply=dict(seal=seal)
            elif operation=='close':print(json.dumps(dict(ok=True,result={})),flush=True);return
            else:raise RuntimeError('unknown worker operation')
            print(json.dumps(dict(ok=True,result=reply)),flush=True)
        except BaseException as error:print(json.dumps(dict(ok=False,error=type(error).__name__,message=str(error),traceback=traceback.format_exc(),operation=request,last_case=locals().get('index'),partial_checks=locals().get('checks'),partial_default_checks=locals().get('default_checks'),partial_timing_fixture_checks=timing_fixture_checks,mismatches=locals().get('mismatches'),last_observation=locals().get('value'))),flush=True)


def summarize(rows):
    require(len(rows)==70 and [r['case_index'] for r in rows]==list(range(330,400)),'complete70 summary required')
    def group(values):
        return dict(case_count=len(values),geometric_mean_before_speedup=math.exp(math.fsum(math.log(r['before_speedup']) for r in values)/len(values)),geometric_mean_after_speedup=math.exp(math.fsum(math.log(r['after_speedup']) for r in values)/len(values)),geometric_mean_improvement=math.exp(math.fsum(math.log(r['improvement']) for r in values)/len(values)),slower_than_before=[r['case_index'] for r in values if r['slower_than_before']],slower_than_original=[r['case_index'] for r in values if r['slower_than_original']])
    cohorts=sorted({r['inventory']['cohort'] for r in rows})
    return dict(overall=group(rows),cohorts={name:group([r for r in rows if r['inventory']['cohort']==name]) for name in cohorts})


def validate_timing_fixtures(report):
    checks=report.get('timing_fixture_checks')
    if report['check_only']:
        require(checks=={},'preparation cannot construct timing callbacks')
        return
    require(type(checks) is dict and set(checks)==set(ROLES),'all actual timing fixture records required')
    for role,rows in checks.items():
        require([r.get('case_index') for r in rows]==list(range(330,400)),
                'all70 ordered actual timing fixtures required')
        expected=report['default_checks']['normal'][role]['checks'][0]['records']
        require(rows==expected,'actual timing fixture/state/dispatch differs from fresh default sweep')


def validate_report(report,e,*,preflight_only=False):
    require(report.get('schema')==a.REPORT_SCHEMA and report.get('status') in ('complete-parity-check','complete-controlled-measurement') and not report.get('failures'),'failed/partial controlled report')
    require(report['controller_revision']==e['controller_revision'] and report['before_product_source']==a.BEFORE_PRODUCT and report['after_product_source']==a.AFTER_PRODUCT,'report source labels differ')
    require(type(report['check_only']) is bool,'check-only flag must be literal bool');modes=('normal','scalar') if report['check_only'] else ('normal',);require(set(report['checks'])==set(modes),'missing normal/scalar preparation')
    M=a.read_json(ROOT/'coverage/contracts/tv-chambolle-benchmark.json');expected=[M['reference_profiles'][report['profile']]['observations'][str(i)] for i in range(400)]
    for mode in modes:
        identities=report['worker_identities'][mode];require(set(identities)==set(ROLES) and len({v['pid'] for v in identities.values()})==3,'workers not three isolated processes')
        require(all(i['cpu']==identities['original']['cpu'] and i['default_workers']==1 and type(i['default_workers']) is int and i['scalar']==(mode=='scalar') for i in identities.values()),'CPU/worker/scalar identity differs')
        for role,identity in identities.items():
            product_role='before' if role=='original' else role;frame=e['products'][product_role][report['profile']]
            require(identity['role']==role and identity['profile']==report['profile'] and type(identity['pid']) is int and identity['pid']>0 and identity['source']==frame['pin']['revision'] and identity['product_sha256']==frame['pin']['product_sha256'] and identity['core_sha256']==frame['core_sha256'],'worker product/source identity differs')
            require(identity['has_avx2'] is (mode=='normal') and identity.get('fallback_enabled') is False,'worker SIMD/fallback policy differs')
            require(identity.get('guard_environment')==dict(optimize=0,custom_startup=False),
                    'worker assertion/startup environment not authenticated')
        require(identities['before']['core_path']!=identities['after']['core_path'] and identities['before']['core_sha256']!=identities['after']['core_sha256'],'same core used for before/after')
        require(set(report['checks'][mode])==set(ROLES),'backend observations missing')
        for role,value in report['checks'][mode].items():
            require(not value['mismatches'] and len(value['checks'])==2 and [v['workers'] for v in value['checks']]==[1,4] and all(v['digests']==expected for v in value['checks']),'complete exact400 workers1/4 observations required')
    a.validate_default_checks(report,M['case_inventory'])
    pair=report.get('source_comparison',{})
    require(pair.get('schema')=='tv-positive-zero-source-comparison-v1'
            and pair.get('whole_core_equivalent') is False and pair.get('helper_causal_attribution') is False
            and pair.get('attribution')==a.PLAN['attribution'], 'unreviewed attribution claim')
    if not preflight_only:validate_timing_fixtures(report)
    if not preflight_only:require(bool(report['results'])!=report['check_only'],'preparation cannot time')
    if not report['check_only'] and not preflight_only:
        require([r['case_index'] for r in report['results']]==list(range(330,400)),'all70 original canonical indices required')
        require(report.get('summary')==summarize(report['results']),'all-case/cohort summary differs')
        for row in report['results']:
            require(row['inventory']==M['case_inventory'][row['case_index']],'timing fixture changed')
            require(set(row.get('calibration',{}))==set(ROLES),'all calibration records required')
            for role,v in row['calibration'].items():require(type(v['ms']) is float and math.isfinite(v['ms']) and v['ms']>0 and v['loops']==1 and type(v['loops']) is int and type(v['elapsed_ns']) is int and v['elapsed_ns']>0 and v['ms']==v['elapsed_ns']/1e6 and v['identity']==report['worker_identities']['normal'][role],'calibration identity/elapsed differs')
            require(set(row['samples_ms'])==set(ROLES) and all(len(v)==7 and all(type(x) is float and math.isfinite(x) and x>0 for x in v) for v in row['samples_ms'].values()),'exact seven samples per backend required')
            require(row['triad_order']==rotating_triads(row['case_index']) and len(row['raw_samples'])==21,'paired rotation/raw rounds differ')
            expected_order=[(i,role) for i,order in enumerate(row['triad_order']) for role in order]
            require([(v['round'],v['role']) for v in row['raw_samples']]==expected_order,'raw sample order differs')
            for role in ROLES:
                raw=[v for v in row['raw_samples'] if v['role']==role]
                require([v['ms'] for v in raw]==row['samples_ms'][role],'raw samples differ')
                require(all(type(v['loops']) is int and 1<=v['loops']<=1000 and type(v['elapsed_ns']) is int and v['elapsed_ns']>0 and v['ms']==v['elapsed_ns']/v['loops']/1e6 and v['identity']==report['worker_identities']['normal'][role] for v in raw),'raw elapsed/loop/CPU/source differs')
            medians={r:statistics.median(v) for r,v in row['samples_ms'].items()};require(row['medians_ms']==medians and row['before_speedup']==medians['original']/medians['before'] and row['after_speedup']==medians['original']/medians['after'] and row['improvement']==medians['before']/medians['after'],'sample/ratio arithmetic differs')
            require(row['slower_than_before']==(medians['after']>medians['before']) and row['slower_than_original']==(medians['after']>medians['original']),'regression flag differs')
    if not report['check_only'] and not preflight_only:validate_authentication_index(report)
    return report

def verify_barrier(paths,e,experiment_sha,run,attempt):
    rows=a.barrier(paths,experiment_sha,run,attempt)
    for path in paths:
        receipt=a.read_json(path);report_path=Path(path).parent/'report.json';require(sha(report_path)==receipt['report_sha256'],'preparation report bytes differ');r=validate_report(a.read_json(report_path),e)
        require(r['check_only'] is True and receipt['controller_revision']==e['controller_revision'] and r['experiment_sha256']==experiment_sha and r['profile']==receipt['profile'],'receipt/report provenance differs')
        require(receipt['common_build_sha256']==r['source_comparison']['common_sha256'] and receipt['reports']=={mode:a.digest(r['checks'][mode]) for mode in ('normal','scalar')},'receipt/check/common build binding differs')
        require(receipt['default_reports']=={mode:a.digest(dict(checks=r['default_checks'][mode],assets=r['default_observation_assets'])) for mode in ('normal','scalar')},'receipt/default70 dispatch/state/raw observation binding differs')
    return rows


def controller(args):
    cfg=experiment(args);require(args.output and not Path(args.output).exists(),'fresh output required');require(os.environ.get('GITHUB_ACTIONS')=='true','actual workers and operation timings require isolated Actions')
    products={r:Path(getattr(args,r)) for r in ('before','after')};pair=a.verify_pair(products['before'],products['after'],args.profile,cfg);measure=not args.check_only
    if measure:verify_barrier(args.receipts,cfg,args.experiment_sha256,*generation())
    else:require(not args.receipts,'check-only cannot consume measurement barrier')
    output=Path(args.output);output.mkdir(parents=True);report=dict(schema=a.REPORT_SCHEMA,controller_revision=cfg['controller_revision'],before_product_source=a.BEFORE_PRODUCT,after_product_source=a.AFTER_PRODUCT,session_id=str(uuid.uuid4()),profile=args.profile,check_only=args.check_only,experiment_sha256=args.experiment_sha256,source_comparison=pair,worker_identities={},checks={},default_validation_schema=a.DEFAULT_SCHEMA,default_checks={},default_observation_assets={},timing_fixture_checks={} if args.check_only else {r:[] for r in ROLES},results=[],failures=[],status='preflight')
    permit=None;nonces={}
    if measure:
        nonces,authentication=startup_plan();run,attempt=generation();claim=output/'authentication'/'claim.json'
        write_new(claim,dict(profile=args.profile,generation=run,attempt=attempt,receipt_hashes=[sha(p) for p in args.receipts],authentication=authentication))
        permit=output/'worker-permit.json';write_new(permit,dict(profile=args.profile,experiment_sha256=args.experiment_sha256,**authentication,claim=str(claim),claim_sha256=sha(claim),receipts=[str(Path(p).resolve()) for p in args.receipts]))
    scheduler=SerialScheduler();workers={}
    try:
        for mode in (('normal','scalar') if args.check_only else ('normal',)):
            workers={role:Worker(getattr(args,('after' if role=='after' else 'before')+'_python'),products['after' if role=='after' else 'before'],role,args.profile,output/(role+'-'+mode+'.stderr'),measure,mode=='scalar',permit,args.experiment,args.experiment_sha256,nonces.get(role)) for role in ROLES}
            report['worker_identities'][mode]={r:w.identity for r,w in workers.items()}
            report['checks'][mode]={r:scheduler.request(w,dict(operation='validate',workers=[1,4])) for r,w in workers.items()}
            report['default_checks'][mode]={r:a.ingest_default_reply(report,scheduler.request(w,dict(operation='validate-default',workers=[1,4]))) for r,w in workers.items()}
            # Complete legacy400 AND fresh default70/dispatch/allthree before any warm/calibration path.
            partial=dict(report,status='complete-parity-check',check_only=True,results=[]);validate_report(partial,cfg) if mode=='scalar' else None
            if measure:
                validate_report(dict(report,status='complete-controlled-measurement'),cfg,preflight_only=True)
                require(len({w.identity['pid'] for w in workers.values()})==3 and all(v['checks'][0]['digests']==report['checks'][mode]['original']['checks'][0]['digests'] and not v['mismatches'] for v in report['checks'][mode].values()),'all backend parity required before timers')
                M=a.read_json(ROOT/'coverage/contracts/tv-chambolle-benchmark.json')
                for index in range(330,400):
                    for role,w in workers.items():
                        warm=scheduler.request(w,dict(operation='warm',case_index=index))
                        row=warm['timing_fixture']
                        require(warm['warmed'] is True and row['case_index']==index,'actual fixture warmup incomplete')
                        a.ingest_default_reply(report,dict(checks=[dict(records=[row])]))
                        report['timing_fixture_checks'][role].append(row)
                        require(row==report['default_checks'][mode][role]['checks'][0]['records'][index-330],
                                'actual callback fixture differs before calibration')
                    calibration={r:scheduler.request(w,dict(operation='sample',case_index=index,loops=1)) for r,w in workers.items()};loops={r:max(1,min(1000,int(15/max(calibration[r]['ms'],.001)))) for r in ROLES};samples={r:[] for r in ROLES};raw=[];orders=rotating_triads(index)
                    for round_index,order in enumerate(orders):
                        for role in order:
                            sample=scheduler.request(workers[role],dict(operation='sample',case_index=index,loops=loops[role]));samples[role].append(sample['ms']);raw.append(dict(round=round_index,role=role,**sample))
                    medians={r:statistics.median(v) for r,v in samples.items()};report['results'].append(dict(case_index=index,inventory=M['case_inventory'][index],triad_order=orders,calibration=calibration,samples_ms=samples,raw_samples=raw,medians_ms=medians,before_speedup=medians['original']/medians['before'],after_speedup=medians['original']/medians['after'],improvement=medians['before']/medians['after'],slower_than_before=medians['after']>medians['before'],slower_than_original=medians['after']>medians['original']))
                    (output/'progress.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
            for w in workers.values():scheduler.request(w,dict(operation='integrity'));w.close()
            workers={}
        require(a.verify_pair(products['before'],products['after'],args.profile,cfg)==pair,'source/common identity changed');report['status']='complete-parity-check' if args.check_only else 'complete-controlled-measurement';
        if measure:
            report['summary']=summarize(report['results']);report['authentication_artifacts']=verify_authentication_artifacts(output,report)
        validate_report(report,cfg)
    except BaseException as error:report['status']='failed';report['failures'].append(dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()));raise
    finally:
        for w in workers.values():
            if w.process.poll() is None:
                try:w.close()
                except BaseException as error:report['failures'].append(dict(phase='close',type=type(error).__name__,message=str(error)))
        write_new(output/'report.json',report)
    if args.check_only:
        run,attempt=generation();write_new(output/'preparation.json',dict(schema=a.RECEIPT_SCHEMA,passed=True,profile=args.profile,controller_revision=cfg['controller_revision'],generation=run,attempt=attempt,experiment_sha256=args.experiment_sha256,report_sha256=sha(output/'report.json'),modes=['normal','scalar'],workers=[1,4],case_count=400,timed_count=70,common_build_sha256=pair['common_sha256'],reports={mode:a.digest(report['checks'][mode]) for mode in ('normal','scalar')},default_validation_schema=a.DEFAULT_SCHEMA,default_timed_count=70,default_reports={mode:a.digest(dict(checks=report['default_checks'][mode],assets=report['default_observation_assets'])) for mode in ('normal','scalar')}))
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['selectors','fetch','verify-pair','compare','barrier','worker'])
    parser.add_argument('--experiment',type=Path,required=True);parser.add_argument('--experiment-sha256',required=True);parser.add_argument('--profile',choices=a.PROFILES);parser.add_argument('--before',type=Path);parser.add_argument('--after',type=Path);parser.add_argument('--before-python');parser.add_argument('--after-python');parser.add_argument('--output',type=Path);parser.add_argument('--check-only',action='store_true');parser.add_argument('--receipts',nargs='*',type=Path,default=[]);parser.add_argument('--product',type=Path);parser.add_argument('--role',choices=ROLES);parser.add_argument('--measure',action='store_true');parser.add_argument('--permit',type=Path)
    args=parser.parse_args(argv);e=experiment(args)
    if args.stage!='barrier':require(args.profile in a.PROFILES,'explicit runtime profile required')
    if args.stage=='selectors':
        for role in ('before','after'):
            f=e['products'][role][args.profile];print(role+'_run='+f['pin']['artifact_run']);print(role+'_artifact='+f['pin']['artifact_name'])
    elif args.stage=='fetch':
        require(os.environ.get('GITHUB_ACTIONS')=='true','artifact download restricted to preparation host');require(args.output is not None,'output required')
        for role in ('before','after'):a.fetch_product(e['products'][role][args.profile],args.output/role)
    elif args.stage=='verify-pair':print(json.dumps(a.verify_pair(args.before,args.after,args.profile,e)))
    elif args.stage=='barrier':print(json.dumps(verify_barrier(args.receipts,e,args.experiment_sha256,*generation())))
    elif args.stage=='worker':worker_main(args)
    else:controller(args)

if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc();raise
