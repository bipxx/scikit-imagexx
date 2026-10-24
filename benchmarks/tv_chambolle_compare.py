"""Three isolated TV workers; only reviewed artifact-only Actions may time calls."""
from __future__ import annotations
import argparse,ast,copy,dataclasses,hashlib,importlib,importlib.util,inspect,json,math,os,platform,random,secrets,statistics,subprocess,sys,time,traceback,types,uuid,warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import tv_chambolle_compare_artifacts as a
require=a.require;sha=a.sha;load_json=a.read_json;write_new=a.save_new;ROLES=a.ROLES

# Reuse exact mature one-use inherited-stdio authentication/transport ASTs in
# THIS controller's globals, without altering any historical module or guard.
transport_source=(ROOT/'benchmarks/rank_statistics_compare.py').read_bytes()
require(a.text_sha(ROOT/'benchmarks/rank_statistics_compare.py')==a.RANK_VALIDATOR_SHA256,'historical transport source changed')
transport_names={'startup_plan','startup_frame','authenticate_startup','Worker','SerialScheduler'}
transport_nodes=[n for n in ast.parse(transport_source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in transport_names]
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

def sample_admission(validated,measure,actions,workers,scalar,index):
    require(validated and measure and actions=='true' and type(workers) is int and workers==1 and scalar is False,'timing before strict validation/primary worker forbidden')
    require(type(index) is int and 330<=index<400,'only frozen70 raster IDs may time')

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

def worker_main(args):
    cfg=experiment(args);product=Path(args.product);product_role='before' if args.role=='original' else args.role
    startup=worker_permit(args.permit,args.profile,cfg,args.role) if args.measure else None
    c=a.rank_controller();validation=copy.deepcopy(c.config());validation['accepted_products']={r:{p:cfg['products'][r][p]['pin'] for p in a.PROFILES} for r in ('before','after')}
    seal=c.verify_product(product,product_role,args.profile,validation) # BEFORE any numerical imports.
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
    identity=dict(role=args.role,pid=os.getpid(),source=seal['revision'],product_sha256=seal['product_sha256'],core_sha256=sha(xx._core.__file__),core_path=str(Path(xx._core.__file__).resolve()),profile=args.profile,cpu=cpu,python=sys.version,numpy=np.__version__,scalar=os.environ.get('SKIMAGEXX_DISABLE_SIMD')=='1',default_workers=previous,has_avx2=xx.has_avx2(),fallback_enabled=False,startup_authentication=startup)
    print(json.dumps(dict(ok=True,identity=identity)),flush=True);validated=False;callbacks={}
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
                        if observed!=expected:mismatches.append(dict(index=index,id=case.id,worker=workers,expected=expected,actual=value))
                    checks.append(dict(workers=workers,digests=digests))
                xx.set_num_threads(1);require(checks[0]['digests']==checks[1]['digests'] and not mismatches,'frozen400 exact observation differs');validated=True;reply=dict(checks=checks,mismatches=mismatches)
            elif operation in ('warm','sample'):
                index=request['case_index'];sample_admission(validated,args.measure,os.environ.get('GITHUB_ACTIONS'),xx.get_num_threads(),identity['scalar'],index)
                if index not in callbacks:callbacks[index]=h.timing_callback(h.resolve(args.role=='original'),cases[index])
                call,unchanged=callbacks[index]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    if operation=='warm':result=call();del result;reply=dict(warmed=True)
                    else:
                        loops=request['loops'];require(type(loops) is int and 1<=loops<=1000,'invalid loops')
                        time.sleep(.02);start=time.perf_counter_ns()
                        for _ in range(loops):result=call();del result
                        elapsed=time.perf_counter_ns()-start;reply=dict(ms=elapsed/loops/1e6,elapsed_ns=elapsed,loops=loops,identity=identity)
                require(unchanged(),'timed input/owner state changed')
            elif operation=='integrity':
                require(c.verify_product(product,product_role,args.profile,validation)==seal,'product source/identity changed');verify_environment(product);require(cpu_ns['cpu_metadata']()==cpu,'worker CPU identity changed');reply=dict(seal=seal)
            elif operation=='close':print(json.dumps(dict(ok=True,result={})),flush=True);return
            else:raise RuntimeError('unknown worker operation')
            print(json.dumps(dict(ok=True,result=reply)),flush=True)
        except BaseException as error:print(json.dumps(dict(ok=False,error=type(error).__name__,message=str(error),traceback=traceback.format_exc(),operation=request,last_case=locals().get('index'),partial_checks=locals().get('checks'),mismatches=locals().get('mismatches'),last_observation=locals().get('value'))),flush=True)


def summarize(rows):
    require(len(rows)==70 and [r['case_index'] for r in rows]==list(range(330,400)),'complete70 summary required')
    def group(values):
        return dict(case_count=len(values),geometric_mean_before_speedup=math.exp(math.fsum(math.log(r['before_speedup']) for r in values)/len(values)),geometric_mean_after_speedup=math.exp(math.fsum(math.log(r['after_speedup']) for r in values)/len(values)),geometric_mean_improvement=math.exp(math.fsum(math.log(r['improvement']) for r in values)/len(values)),slower_than_before=[r['case_index'] for r in values if r['slower_than_before']],slower_than_original=[r['case_index'] for r in values if r['slower_than_original']])
    cohorts=sorted({r['inventory']['cohort'] for r in rows})
    return dict(overall=group(rows),cohorts={name:group([r for r in rows if r['inventory']['cohort']==name]) for name in cohorts})


def validate_report(report,e,*,preflight_only=False):
    require(report.get('schema')=='tv-cursor-controlled-v1' and report.get('status') in ('complete-parity-check','complete-controlled-measurement') and not report.get('failures'),'failed/partial controlled report')
    require(report['controller_revision']==e['controller_revision'] and report['before_product_source']==a.BEFORE_PRODUCT and report['intended_before_source']==a.BEFORE_ANCHOR and report['after_product_source']==a.AFTER_PRODUCT,'report source labels differ')
    require(type(report['check_only']) is bool,'check-only flag must be literal bool');modes=('normal','scalar') if report['check_only'] else ('normal',);require(set(report['checks'])==set(modes),'missing normal/scalar preparation')
    M=a.read_json(ROOT/'coverage/contracts/tv-chambolle-benchmark.json');expected=[M['reference_profiles'][report['profile']]['observations'][str(i)] for i in range(400)]
    for mode in modes:
        identities=report['worker_identities'][mode];require(set(identities)==set(ROLES) and len({v['pid'] for v in identities.values()})==3,'workers not three isolated processes')
        require(all(i['cpu']==identities['original']['cpu'] and i['default_workers']==1 and type(i['default_workers']) is int and i['scalar']==(mode=='scalar') for i in identities.values()),'CPU/worker/scalar identity differs')
        for role,identity in identities.items():
            product_role='before' if role=='original' else role;frame=e['products'][product_role][report['profile']]
            require(identity['role']==role and identity['profile']==report['profile'] and type(identity['pid']) is int and identity['pid']>0 and identity['source']==frame['pin']['revision'] and identity['product_sha256']==frame['pin']['product_sha256'] and identity['core_sha256']==frame['core_sha256'],'worker product/source identity differs')
            require(identity['has_avx2'] is (mode=='normal') and identity.get('fallback_enabled') is False,'worker SIMD/fallback policy differs')
        require(identities['before']['core_path']!=identities['after']['core_path'] and identities['before']['core_sha256']!=identities['after']['core_sha256'],'same core used for before/after')
        require(set(report['checks'][mode])==set(ROLES),'backend observations missing')
        for role,value in report['checks'][mode].items():
            require(not value['mismatches'] and len(value['checks'])==2 and [v['workers'] for v in value['checks']]==[1,4] and all(v['digests']==expected for v in value['checks']),'complete exact400 workers1/4 observations required')
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
    return report

def verify_barrier(paths,e,experiment_sha,run,attempt):
    rows=a.barrier(paths,experiment_sha,run,attempt)
    for path in paths:
        receipt=a.read_json(path);report_path=Path(path).parent/'report.json';require(sha(report_path)==receipt['report_sha256'],'preparation report bytes differ');r=validate_report(a.read_json(report_path),e)
        require(r['check_only'] is True and receipt['controller_revision']==e['controller_revision'] and r['experiment_sha256']==experiment_sha and r['profile']==receipt['profile'],'receipt/report provenance differs')
        require(receipt['common_build_sha256']==r['source_comparison']['common_sha256'] and receipt['reports']=={mode:a.digest(r['checks'][mode]) for mode in ('normal','scalar')},'receipt/check/common build binding differs')
    return rows


def controller(args):
    cfg=experiment(args);require(args.output and not Path(args.output).exists(),'fresh output required');require(os.environ.get('GITHUB_ACTIONS')=='true','actual workers and operation timings require isolated Actions')
    products={r:Path(getattr(args,r)) for r in ('before','after')};pair=a.verify_pair(products['before'],products['after'],args.profile,cfg);measure=not args.check_only
    if measure:verify_barrier(args.receipts,cfg,args.experiment_sha256,*generation())
    else:require(not args.receipts,'check-only cannot consume measurement barrier')
    output=Path(args.output);output.mkdir(parents=True);report=dict(schema='tv-cursor-controlled-v1',controller_revision=cfg['controller_revision'],before_product_source=a.BEFORE_PRODUCT,intended_before_source=a.BEFORE_ANCHOR,after_product_source=a.AFTER_PRODUCT,session_id=str(uuid.uuid4()),profile=args.profile,check_only=args.check_only,experiment_sha256=args.experiment_sha256,source_comparison=pair,worker_identities={},checks={},results=[],failures=[],status='preflight')
    permit=None;nonces={}
    if measure:
        nonces,authentication=startup_plan();run,attempt=generation();claim=Path(os.environ['RUNNER_TEMP'])/('tv-cursor-'+run+'-'+attempt+'-'+args.profile+'.json')
        write_new(claim,dict(profile=args.profile,generation=run,attempt=attempt,receipt_hashes=[sha(p) for p in args.receipts],authentication=authentication))
        permit=output/'worker-permit.json';write_new(permit,dict(profile=args.profile,experiment_sha256=args.experiment_sha256,**authentication,claim=str(claim),claim_sha256=sha(claim),receipts=[str(Path(p).resolve()) for p in args.receipts]))
    scheduler=SerialScheduler();workers={}
    try:
        for mode in (('normal','scalar') if args.check_only else ('normal',)):
            workers={role:Worker(getattr(args,('after' if role=='after' else 'before')+'_python'),products['after' if role=='after' else 'before'],role,args.profile,output/(role+'-'+mode+'.stderr'),measure,mode=='scalar',permit,args.experiment,args.experiment_sha256,nonces.get(role)) for role in ROLES}
            report['worker_identities'][mode]={r:w.identity for r,w in workers.items()}
            report['checks'][mode]={r:scheduler.request(w,dict(operation='validate',workers=[1,4])) for r,w in workers.items()}
            # Complete all400/allthree before touching any warm/calibration path.
            partial=dict(report,status='complete-parity-check',check_only=True,results=[]);validate_report(partial,cfg) if mode=='scalar' else None
            if measure:
                validate_report(dict(report,status='complete-controlled-measurement'),cfg,preflight_only=True)
                require(len({w.identity['pid'] for w in workers.values()})==3 and all(v['checks'][0]['digests']==report['checks'][mode]['original']['checks'][0]['digests'] and not v['mismatches'] for v in report['checks'][mode].values()),'all backend parity required before timers')
                M=a.read_json(ROOT/'coverage/contracts/tv-chambolle-benchmark.json')
                for index in range(330,400):
                    for w in workers.values():scheduler.request(w,dict(operation='warm',case_index=index))
                    calibration={r:scheduler.request(w,dict(operation='sample',case_index=index,loops=1)) for r,w in workers.items()};loops={r:max(1,min(1000,int(15/max(calibration[r]['ms'],.001)))) for r in ROLES};samples={r:[] for r in ROLES};raw=[];orders=rotating_triads(index)
                    for round_index,order in enumerate(orders):
                        for role in order:
                            sample=scheduler.request(workers[role],dict(operation='sample',case_index=index,loops=loops[role]));samples[role].append(sample['ms']);raw.append(dict(round=round_index,role=role,**sample))
                    medians={r:statistics.median(v) for r,v in samples.items()};report['results'].append(dict(case_index=index,inventory=M['case_inventory'][index],triad_order=orders,calibration=calibration,samples_ms=samples,raw_samples=raw,medians_ms=medians,before_speedup=medians['original']/medians['before'],after_speedup=medians['original']/medians['after'],improvement=medians['before']/medians['after'],slower_than_before=medians['after']>medians['before'],slower_than_original=medians['after']>medians['original']))
                    (output/'progress.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
            for w in workers.values():scheduler.request(w,dict(operation='integrity'));w.close()
            workers={}
        require(a.verify_pair(products['before'],products['after'],args.profile,cfg)==pair,'source/common identity changed');report['status']='complete-parity-check' if args.check_only else 'complete-controlled-measurement';
        if measure:report['summary']=summarize(report['results'])
        validate_report(report,cfg)
    except BaseException as error:report['status']='failed';report['failures'].append(dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()));raise
    finally:
        for w in workers.values():
            if w.process.poll() is None:
                try:w.close()
                except BaseException as error:report['failures'].append(dict(phase='close',type=type(error).__name__,message=str(error)))
        write_new(output/'report.json',report)
    if args.check_only:
        run,attempt=generation();write_new(output/'preparation.json',dict(schema='tv-cursor-preparation-v1',passed=True,profile=args.profile,controller_revision=cfg['controller_revision'],generation=run,attempt=attempt,experiment_sha256=args.experiment_sha256,report_sha256=sha(output/'report.json'),modes=['normal','scalar'],workers=[1,4],case_count=400,timed_count=70,common_build_sha256=pair['common_sha256'],reports={mode:a.digest(report['checks'][mode]) for mode in ('normal','scalar')}))
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
