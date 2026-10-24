"""Qualified public max-tree vs original0.26; no local operation timing admission.

One ordinary installed-product process owns the genuine core and two actual
public callables. Source/provider observers are UNTImed validation only. Timed
callbacks contain only the complete public call; result destruction is timed.
"""
from __future__ import annotations
import argparse,ast,contextlib,hashlib,importlib,inspect,json,math,os,platform,random,secrets,statistics,subprocess,sys,time,traceback,types,uuid,warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import max_tree_compare_artifacts as a
require=a.require
COUNTS={'reference':351,'numeric':233,'metadata':118,'candidates':252}
STAGES={'_max_tree':'build','_compute_area':'area','_compute_extension':'extension','_direct_filter':'filter','_max_tree_local_maxima':'maxima'}


def baseline():
    a.frozen_sources();b=a.module('_immutable_max_tree_benchmark',ROOT/'benchmarks/max_tree.py')
    m=a.read(b.MANIFEST);b.verify_manifest(m);return b,m


def disable_fallback(policy):
    # The authentic setter returns the PREVIOUS policy, initially True.
    policy.set_fallback_enabled(False)
    require(policy.set_fallback_enabled(False) is False,'fallback disabling failed')


def public_admission(product,profile,cert):
    """No reference helper or fixture is imported before this real-core gate."""
    import skimagexx as xx
    wrapper=importlib.import_module('skimagexx._max_tree');coverage=importlib.import_module('skimagexx.coverage');core=xx._core
    require(core.__name__=='skimagexx._core' and sys.modules.get('skimagexx._core') is core,'sole ordinary core required')
    package=Path(xx.__file__).resolve().parent
    require(package.is_relative_to(Path(sys.prefix).resolve()) and Path(wrapper.__file__).resolve().parent==Path(core.__file__).resolve().parent==package,'installed package ownership required; no source/private adapter')
    require(a.sha(core.__file__)==cert['profiles'][profile]['core_sha256'],'wrong actual core')
    require(wrapper._core is core and wrapper._require_core() is core,'wrapper/core identity differs')
    require(core.max_tree_interface=='max-tree-prepared-v1' and type(core.max_tree_type_capabilities) is int and core.max_tree_type_capabilities==1023,'typed native interface differs')
    require(type(core.max_tree_operation_capabilities) is int and core.max_tree_operation_capabilities==63 and core.max_tree_parallel_traversal is False,'operation/dependency contract differs')
    require(core.max_tree_stage_interface=='max-tree-stages-v1' and type(core.max_tree_stage_capabilities) is int and core.max_tree_stage_capabilities==31,'actual numerical stages missing')
    for name in ('max_tree_run','max_tree_counts')+wrapper._STAGE_ENTRIES:
        fn=getattr(core,name,None);require(inspect.isbuiltin(fn) and fn.__module__==core.__name__ and fn.__name__==name,'borrowed/Python native entry forbidden')
    expected=a.read(Path(product)/'qualification.json')['expected_registry'];names=tuple(coverage.native_functions())
    require(len(names)==len(set(names))==202 and set(names)==set(expected['functions']),'actual199image+3metadata registry differs')
    for name in a.NAMES:
        fn=getattr(xx.morphology,name);require(fn is getattr(wrapper,name) and inspect.isfunction(fn) and fn.__module__=='skimagexx._max_tree' and fn.__globals__.get('_core') is core,'actual owned public alias required')
        require(names.count('morphology.'+name)==1,'public path unregistered')
    require(xx.set_num_threads is core.set_num_threads and xx.get_num_threads is core.get_num_threads,'sole shared worker state required')
    require(type(xx.get_num_threads()) is int and xx.get_num_threads()==1,'default global workers must be1')
    for n in (4,1):xx.set_num_threads(n);require(core.get_num_threads()==n,'shared worker limit differs')
    disable_fallback(xx)
    require(not any(k.startswith('SKIMAGEXX_') and k!='SKIMAGEXX_DISABLE_SIMD' for k in os.environ),'private selectors forbidden')
    require(all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS')),'provider pools must be1')
    return xx,wrapper,core


@contextlib.contextmanager
def no_forward(reference):
    """Guard actual original Python/Cython identities; restore ALL state."""
    original=reference.original;saved={n:getattr(original._max_tree,n) for n in reference.ENTRIES}
    codes={getattr(original,n).__code__ for n in reference.NAMES};attempts=[];previous=sys.getprofile()
    def forbidden(*args,**kwargs):attempts.append('cython');raise AssertionError('upstream max-tree numerical forwarding forbidden')
    def profile(frame,event,arg):
        if event=='call' and frame.f_code in codes or event=='c_call' and any(arg is value for value in saved.values()):
            attempts.append(frame.f_code.co_name);raise AssertionError('upstream numerical identity reached')
    try:
        for n in saved:setattr(original._max_tree,n,forbidden)
        sys.setprofile(profile);yield attempts
    finally:
        sys.setprofile(previous)
        for n,fn in saved.items():setattr(original._max_tree,n,fn)
        require(not attempts,'upstream attempt caught by observer: '+str(attempts))


def counter_check(counts,*,entries=None,minimum=0,aggregate=None):
    require(set(counts)=={'aggregate',*STAGES.values()},'real counter set differs')
    for row in counts.values():require(set(row)=={'attempted','completed'} and all(type(v) is int and v>=0 for v in row.values()) and row['attempted']>=row['completed'],'invalid attempted/completed counters')
    if entries is not None:
        require(counts['aggregate']==dict(attempted=0,completed=0),'observable provider route must use stages')
        for source,stage in STAGES.items():
            n=sum(e['name']==source and 'after' in e for e in entries)
            require(counts[stage]==dict(attempted=n,completed=n),'real staged completion count differs')
    require(sum(r['completed'] for r in counts.values())>=minimum,'no completed owned numerical work')
    if aggregate is not None:require(counts['aggregate']==dict(attempted=aggregate,completed=aggregate) and all(v==dict(attempted=0,completed=0) for k,v in counts.items() if k!='aggregate'),'pristine call did not use actual aggregate C++')


def old_fixture(reference,matrix,profile,spec,actual_provided_tree=False):
    passed,image,owner,options,events=reference.fixture(spec,False)
    if actual_provided_tree and spec.get('reuse')=='both' and spec['mode']=='numeric':
        expected=matrix['observations'][matrix['profiles'][profile]['snapshots'][spec['id']]]
        for key,row in expected['provided_before'].items():
            value=reference.np.empty(tuple(row['shape']),reference.np.dtype(row['dtype']))
            value.reshape(-1)[...]=reference.np.frombuffer(bytes.fromhex(row['bytes']),dtype=value.dtype);options[key]=value
    return passed,image,owner,options,events


def old_outcome(result,image,options,reference):
    np=reference.np
    def one(value):return dict(type=type(value).__name__,array=reference.array(value),shares_input=bool(np.shares_memory(value,image)),
        base=reference.array(value.base) if isinstance(value.base,np.ndarray) else None,
        shares_provided={k:bool(np.shares_memory(value,options[k])) for k in ('parent','tree_traverser') if isinstance(options.get(k),np.ndarray)})
    return [one(v) for v in result] if isinstance(result,tuple) else one(result)


def candidate_outcome(result,image,options,b,np,blobs=None):
    def one(v):return dict(array=b.array_record(v,np,blobs),type=type(v).__name__,is_input=v is image,
        shares_input=bool(np.shares_memory(v,image)),base=b.array_record(v.base,np,blobs) if isinstance(v.base,np.ndarray) else None,
        shares_provided={k:bool(np.shares_memory(v,options[k])) for k in ('parent','tree_traverser') if k in options})
    return [one(v) for v in result] if isinstance(result,tuple) else one(result)


@contextlib.contextmanager
def certify_original(reference,b,shape):
    saved={name:getattr(reference.original._max_tree,name) for name in reference.ENTRIES}
    def wrap(name,fn):
        def call(*args):b.certify_entry(name,args,shape,reference.np);return fn(*args)
        return call
    try:
        for name,fn in saved.items():setattr(reference.original._max_tree,name,wrap(name,fn))
        yield
    finally:
        for name,fn in saved.items():setattr(reference.original._max_tree,name,fn)


def callback_pair(case,b,reference,wrapper):
    """Immutable inputs/authentic original provided tree prepared OUTSIDE timers."""
    np=reference.np;image,owner=b.fixture(case,np);options=dict(connectivity=case['connectivity'])
    if case['function'] not in ('max_tree','max_tree_local_maxima'):
        options['area_threshold' if case['function'].startswith('area') else 'diameter_threshold']=case['threshold']
    if case['provided']:
        work=reference.original.invert(image) if case['function'].endswith('closing') else image
        with certify_original(reference,b,image.shape):parent,order=reference.original.max_tree(work,case['connectivity'])
        options.update(parent=parent,tree_traverser=order)
    def state():return dict(image=b.array_record(image,np),owner=b.array_record(owner,np),provided={k:b.array_record(options[k],np) for k in ('parent','tree_traverser') if k in options})
    initial=state();callbacks={}
    for backend,module in (('original',reference.original),('native',wrapper)):
        function=getattr(module,case['function'])
        # THIS is the whole measured unit. No observer, certificate, copy,
        # digest, reset, IPC or source/profile tracing is hidden inside it.
        def call(function=function):return function(image,**options)
        callbacks[backend]=call
    source=inspect.getsource(callback_pair);node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='call')
    metadata=dict(case=case,input_state=initial,callback_source=ast.get_source_segment(source,node),
        callback_AST=ast.dump(node,include_attributes=False),callback_source_sha256=hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest(),
        actual_targets={k:dict(module=getattr(reference.original if k=='original' else wrapper,case['function']).__module__,name=case['function']) for k in callbacks})
    return callbacks,state,initial,metadata,image,options


def validate_all(b,M,reference,matrix,wrapper,core,profile,workers,blobs):
    expected=M['profiles'][profile];rows=dict(source351=[],pristine233=[],source252=[],pristine252=[])
    def fixture(spec,actual_provided_tree=False):return old_fixture(reference,matrix,profile,spec,actual_provided_tree)
    # Literal frozen source observer, actual INSTALLED owned module. The core,
    # public wrapper and namespace are never source-loaded or substituted.
    observer=types.FunctionType(reference.observe.__code__,dict(reference.observe.__globals__,original=wrapper,fixture=fixture))
    native_context=types.SimpleNamespace(np=reference.np,original=wrapper,scalar=reference.scalar)
    for spec in reference.CASES:
        core.max_tree_counts(True)
        with no_forward(reference):actual=observer(spec)
        counts=core.max_tree_counts();target=matrix['observations'][matrix['profiles'][profile]['snapshots'][spec['id']]]
        require(actual==target,'source351 observation differs '+spec['id']+' '+json.dumps(actual));counter_check(counts,entries=actual['entries'],minimum=int(spec['mode']=='numeric'))
        rows['source351'].append(dict(id=spec['id'],mode=spec['mode'],digest=reference.digest(actual),counts=counts))
        if spec['mode']!='numeric':continue
        passed,image,owner,options,_=fixture(spec,True);before=(reference.array(image),reference.array(owner));provided={k:reference.array(v) for k,v in options.items() if k in ('parent','tree_traverser')}
        core.max_tree_counts(True)
        with no_forward(reference),warnings.catch_warnings(record=True) as caught,reference.np.errstate(all='warn'):
            warnings.simplefilter('always');result=getattr(wrapper,spec['function'])(passed,**options)
        value=old_outcome(result,image,options,reference);del result;counts=core.max_tree_counts()
        require(value==target['result'] and not caught and before==(reference.array(image),reference.array(owner)) and provided=={k:reference.array(options[k]) for k in provided},'pristine233 ownership/value/mutation differs '+spec['id']+' '+json.dumps(dict(actual=value,counts=counts)))
        counter_check(counts,minimum=1);rows['pristine233'].append(dict(id=spec['id'],digest=a.digest(value),counts=counts))
    for index,case in enumerate(b.CASES):
        core.max_tree_counts(True)
        with no_forward(reference):actual=b.observe_case(case,native_context,blobs)
        counts=core.max_tree_counts();require(actual==expected['cases'][index]['observation'],'source252 observation differs '+case['id']+' '+json.dumps(actual))
        counter_check(counts,entries=actual['entries'],minimum=1);rows['source252'].append(dict(id=case['id'],digest=b.digest(actual),counts=counts))
        callbacks,state,before,metadata,image,options=callback_pair(case,b,reference,wrapper);core.max_tree_counts(True)
        with no_forward(reference),warnings.catch_warnings(record=True) as caught,reference.np.errstate(all='warn'):
            warnings.simplefilter('always');result=callbacks['native']()
        value=candidate_outcome(result,image,options,b,reference.np,blobs);del result;counts=core.max_tree_counts()
        require(value==expected['cases'][index]['observation']['result'] and not caught and state()==before,'pristine252 output/owner/state differs '+case['id']+' '+json.dumps(dict(actual=value,counts=counts,before=before,after=state())))
        counter_check(counts,aggregate=1);rows['pristine252'].append(dict(id=case['id'],digest=a.digest(value),counts=counts))
    return dict(workers=workers,**rows)


def startup(permit_path,stream):
    require(permit_path and not stream.isatty() and os.environ.get('GITHUB_ACTIONS')=='true','direct worker denied')
    permit=a.read(permit_path);line=stream.readline();require(bool(line),'startup handshake absent');frame=json.loads(line)
    require(frame.get('nonce') and hashlib.sha256(frame['nonce'].encode()).hexdigest()==permit['nonce_sha256'],'wrong inherited-stdio secret')
    for key in ('controller_pid','controller_sha256','generation','attempt','profile','certificate_sha256'):require(frame.get(key)==permit.get(key),'startup source/run/profile differs')
    require(permit['controller_sha256']==a.sha(__file__) and permit['generation']==os.environ['GITHUB_RUN_ID'] and permit['attempt']==os.environ['GITHUB_RUN_ATTEMPT'],'stale/wrong-source worker permit')
    launcher=frame.get('launcher_pid');require(type(launcher) is int and launcher>0 and type(permit['controller_pid']) is int and permit['controller_pid']>0,'launcher/controller invalid')
    require((os.getppid()==permit['controller_pid'] and launcher==os.getpid()) or (os.getppid()==launcher and launcher!=os.getpid()),'not exact controller/launcher child')
    receipt=a.save(Path(permit_path).with_suffix('.started.json'),dict(pid=os.getpid(),parent=os.getppid(),launcher=launcher,permit_sha256=a.sha(permit_path),nonce_sha256=permit['nonce_sha256']))
    return permit,dict(worker_pid=os.getpid(),worker_parent=os.getppid(),launcher_pid=launcher,receipt_sha256=a.sha(receipt))


def sample_admission(validated,measure,workers,scalar,index):
    require(validated and measure and os.environ.get('GITHUB_ACTIONS')=='true' and type(workers) is int and workers==1 and scalar is False,'timing requires completed barrier/validation, worker1 normal Actions')
    require(type(index) is int and 0<=index<252,'only all252 frozen candidate IDs may time')


def worker_main(args):
    permit,authentication=startup(args.permit,sys.stdin)
    cert=a.certificate(args.certificate,args.certificate_sha256,a.controller_revision())
    require(permit['certificate_sha256']==args.certificate_sha256 and permit['profile']==args.profile and type(permit['measure']) is bool and permit['measure']==args.measure,'permit arguments differ')
    if args.measure:
        verify_barrier(permit['receipts'],cert,args.certificate_sha256)
        claim=Path(permit['claim']);expected=Path(os.environ['RUNNER_TEMP'])/('max-tree-'+os.environ['GITHUB_RUN_ID']+'-'+os.environ['GITHUB_RUN_ATTEMPT']+'-'+args.profile+'.json')
        require(claim.resolve()==expected.resolve() and a.sha(claim)==permit['claim_sha256'],'exclusive measurement claim differs')
        claimed=a.read(claim)
        require(claimed.get('schema')=='max-tree-measurement-claim-v1' and claimed.get('profile')==args.profile and claimed.get('generation')==os.environ['GITHUB_RUN_ID'] and claimed.get('attempt')==os.environ['GITHUB_RUN_ATTEMPT'],'measurement claim run/profile schema differs')
        require(claimed['nonce_sha256']==permit['nonce_sha256'] and claimed['controller_pid']==permit['controller_pid'] and claimed['controller_sha256']==a.sha(__file__) and claimed['certificate_sha256']==args.certificate_sha256 and claimed['receipt_sha256']==[a.sha(p) for p in permit['receipts']],'claim/source/receipt/startup binding differs')
    seal=a.verify_product(args.product,args.profile,cert);a.verify_installed(args.product)
    xx,wrapper,core=public_admission(args.product,args.profile,cert)
    require(sys.version.split()[0]==cert['profiles'][args.profile]['python_version'],'interpreter patch differs')
    b,M=baseline();reference,matrix=b.load_reference();require(reference.profile()==args.profile,'wrong original reference profile')
    for name in a.NAMES:require(str(inspect.signature(getattr(wrapper,name)))==matrix['signatures'][name] and getattr(wrapper,name) is not getattr(reference.original,name),'public signature/original forwarding differs')
    cpu_node=next(n for n in ast.parse((ROOT/'benchmarks/resize.py').read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='cpu_metadata')
    ns=dict(np=reference.np,os=os,platform=platform,Path=Path);exec(compile(ast.Module(body=[cpu_node],type_ignores=[]),'<shared-exact-CPU-metadata>','exec'),ns);cpu=ns['cpu_metadata']()
    identity=dict(pid=os.getpid(),profile=args.profile,source=cert['source'],product_sha256=seal['product_sha256'],core_sha256=a.sha(core.__file__),core_path=str(Path(core.__file__).resolve()),cpu=cpu,python=sys.version,numpy=reference.np.__version__,scalar=os.environ.get('SKIMAGEXX_DISABLE_SIMD')=='1',default_workers=1,fallback_enabled=False,has_avx2=bool(xx.has_avx2()),authentication=authentication)
    require(not identity['scalar'] or identity['has_avx2'] is False,'scalar dispatch was not actually disabled')
    print(json.dumps(dict(ok=True,identity=identity)),flush=True);validated=args.measure;prepared=None
    for line in sys.stdin:
        request=json.loads(line)
        try:
            op=request['operation']
            if op=='validate':
                require(not args.measure,'full correctness belongs to preparation hosts');original351=[];original252=[]
                blobs=Path(args.permit).parent/'observed-raster';blobs.mkdir(exist_ok=False)
                for spec in reference.CASES:
                    actual=reference.observe(spec);require(actual==matrix['observations'][matrix['profiles'][args.profile]['snapshots'][spec['id']]],'original351 differs');original351.append(reference.digest(actual))
                for index,case in enumerate(b.CASES):
                    actual=b.observe_case(case,reference,blobs);require(actual==M['profiles'][args.profile]['cases'][index]['observation'],'original252 differs '+case['id']+' '+json.dumps(actual));original252.append(b.digest(actual))
                checks=[]
                for workers in (1,4):xx.set_num_threads(workers);checks.append(validate_all(b,M,reference,matrix,wrapper,core,args.profile,workers,blobs))
                xx.set_num_threads(1);validated=True;reply=dict(original351=original351,original252=original252,checks=checks)
            elif op=='prepare-case':
                index=request['index'];sample_admission(validated,args.measure,xx.get_num_threads(),identity['scalar'],index)
                prepared=callback_pair(b.CASES[index],b,reference,wrapper);callbacks,state,before,metadata,image,options=prepared
                target=M['profiles'][args.profile]['cases'][index]['observation'];expected=target['result']
                require({k:before[k] for k in ('image','owner')}==target['before'] and before['provided']==target['provided_before'],'timed input/provided-tree data differs from frozen source fixture')
                with certify_original(reference,b,image.shape):result=callbacks['original']()
                require(candidate_outcome(result,image,options,b,reference.np)==expected,'original callback certification differs');del result
                core.max_tree_counts(True)
                with no_forward(reference):result=callbacks['native']()
                require(candidate_outcome(result,image,options,b,reference.np)==expected and state()==before,'native callback certification differs');del result
                counter_check(core.max_tree_counts(),aggregate=1);reply=metadata
            elif op=='sample':
                index=request['index'];sample_admission(validated,args.measure,xx.get_num_threads(),identity['scalar'],index)
                require(prepared is not None and prepared[3]['case']==b.CASES[index] and request['backend'] in ('original','native'),'wrong prepared callback')
                callbacks,state,before,metadata,image,options=prepared;require(state()==before,'input changed before sample')
                loops=request['loops'];require(type(loops) is int and 1<=loops<=1000,'invalid sample loop count')
                core.max_tree_counts(True);call=callbacks[request['backend']]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore');old=reference.np.seterr(all='warn')
                    try:
                        start=time.perf_counter_ns()
                        for _ in range(loops):result=call();del result
                        elapsed=time.perf_counter_ns()-start
                    finally:reference.np.seterr(**old)
                require(state()==before,'input changed during sample');counts=core.max_tree_counts()
                counter_check(counts,aggregate=loops if request['backend']=='native' else 0)
                reply=dict(elapsed_ns=elapsed,loops=loops,ms=elapsed/loops/1e6,identity=identity,callback_source_sha256=metadata['callback_source_sha256'],counts=counts)
            elif op=='release':prepared=None;callbacks=state=before=metadata=image=options=call=None;reply={}
            elif op=='integrity':
                require(a.verify_product(args.product,args.profile,cert)==seal,'product/source changed');a.verify_installed(args.product);public_admission(args.product,args.profile,cert);require(ns['cpu_metadata']()==cpu,'CPU identity changed');reply=dict(seal=seal)
            elif op=='close':print(json.dumps(dict(ok=True,result={})),flush=True);return
            else:raise RuntimeError('unknown worker operation')
            print(json.dumps(dict(ok=True,result=reply)),flush=True)
        except BaseException as error:print(json.dumps(dict(ok=False,error=type(error).__name__,message=str(error),traceback=traceback.format_exc(),operation=request,partial_checks=locals().get('checks'),last_observation=locals().get('actual'))),flush=True)


class Worker:
    def __init__(self,args,output,scalar=False,measure=False,nonce=None,claim=None):
        nonce=nonce or secrets.token_hex(32);permit=output/'worker-permit.json';run=os.environ['GITHUB_RUN_ID'];attempt=os.environ['GITHUB_RUN_ATTEMPT']
        values=dict(controller_pid=os.getpid(),controller_sha256=a.sha(__file__),generation=run,attempt=attempt,profile=args.profile,certificate_sha256=args.certificate_sha256,nonce_sha256=hashlib.sha256(nonce.encode()).hexdigest(),measure=measure,receipts=[str(Path(p).resolve()) for p in args.receipts])
        if measure:require(claim is not None,'measurement claim absent');values.update(claim=str(Path(claim).resolve()),claim_sha256=a.sha(claim))
        a.save(permit,values);env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1');env.pop('PYTHONPATH',None)
        if scalar:env['SKIMAGEXX_DISABLE_SIMD']='1'
        else:env.pop('SKIMAGEXX_DISABLE_SIMD',None)
        command=[sys.executable,str(Path(__file__).resolve()),'worker','--profile',args.profile,'--product',str(Path(args.product).resolve()),'--certificate',str(Path(args.certificate).resolve()),'--certificate-sha256',args.certificate_sha256,'--permit',str(permit.resolve())]
        if measure:command.append('--measure')
        self.error=(output/'worker-stderr.log').open('x',encoding='utf-8');self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.error,text=True,env=env,cwd=ROOT)
        self.process.stdin.write(json.dumps(dict(values,nonce=nonce,launcher_pid=self.process.pid))+'\n');self.process.stdin.flush()
        row=json.loads(self.process.stdout.readline());require(row.get('ok'),str(row));self.identity=row['identity']
    def request(self,value):
        self.process.stdin.write(json.dumps(value)+'\n');self.process.stdin.flush();line=self.process.stdout.readline()
        require(bool(line),'worker ended without response');row=json.loads(line);require(row.get('ok'),json.dumps(row));return row['result']
    def close(self):
        self.request(dict(operation='close'));require(self.process.wait()==0,'worker exit failure');self.error.close()


def expected_digests(profile):
    b,M=baseline();matrix=a.read(ROOT/'coverage/contracts/max-tree.json')
    return M,matrix,[matrix['profiles'][profile]['snapshots'][c['id']] for c in matrix['cases']]


def validate_report(report,cert):
    require(report.get('schema')=='max-tree-public-comparison-v1' and report.get('passed') is True and not report.get('failures'),'failed/partial report')
    profile=report['profile'];M,matrix,old=expected_digests(profile);frame=cert['profiles'][profile]
    require(report['source']==cert['source'] and report['product_sha256']==frame['pin']['product_sha256'] and report['counts']==COUNTS,'wrong source/product/scope')
    require(type(report['check_only']) is bool and report['controller_sha256']==a.sha(__file__),'wrong controller/check-only source')
    require(set(report['identities'])==({'normal','scalar'} if report['check_only'] else {'normal'}),'worker modes missing')
    for mode,identity in report['identities'].items():
        require(type(identity['pid']) is int and identity['pid']>0 and identity['source']==cert['source'] and identity['product_sha256']==frame['pin']['product_sha256'] and identity['core_sha256']==frame['core_sha256'] and type(identity['default_workers']) is int and identity['default_workers']==1 and identity['fallback_enabled'] is False and type(identity['scalar']) is bool and identity['scalar']==(mode=='scalar'),'actual worker source/core/default policy differs')
        require(type(identity['has_avx2']) is bool and (mode!='scalar' or identity['has_avx2'] is False),'actual scalar dispatch differs')
        auth=identity['authentication'];require(auth['worker_pid']==identity['pid'] and all(type(auth[k]) is int and auth[k]>0 for k in ('worker_pid','worker_parent','launcher_pid')) and len(auth['receipt_sha256'])==64,'one-use startup identity missing')
        require(type(identity['cpu']) is dict and isinstance(identity['cpu'].get('cpu_model'),str) and identity['cpu']['cpu_model'],'actual CPU metadata missing')
    if report['check_only']:
        require(set(report['checks'])=={'normal','scalar'} and not report['results'],'normal+scalar preparation cannot time')
        for mode,row in report['checks'].items():
            identity=report['identities'][mode];require(identity['default_workers']==1 and identity['scalar']==(mode=='scalar') and identity['fallback_enabled'] is False and identity['core_sha256']==frame['core_sha256'],'worker identity differs')
            require(row['original351']==old and row['original252']==[r['digest'] for r in M['profiles'][profile]['cases']],'all original observations required')
            require([r['workers'] for r in row['checks']]==[1,4] and all(type(r['workers']) is int for r in row['checks']),'both real workers required')
            for workers in row['checks']:
                for key,n in (('source351',351),('pristine233',233),('source252',252),('pristine252',252)):
                    require(len(workers[key])==n and len({r['id'] for r in workers[key]})==n,'partial/duplicate native observation rows')
                require([r['digest'] for r in workers['source351']]==old and [r['digest'] for r in workers['source252']]==[r['digest'] for r in M['profiles'][profile]['cases']],'native source observation differs')
                require([(r['id'],r['mode']) for r in workers['source351']]==[(c['id'],c['mode']) for c in matrix['cases']],'reference351 ID/metadata partition differs')
                require([r['id'] for r in workers['source252']]==[c['id'] for c in M['cases']],'candidate252 ID order differs')
                for row,spec in zip(workers['source351'],matrix['cases']):
                    expected=matrix['observations'][matrix['profiles'][profile]['snapshots'][spec['id']]]
                    counter_check(row['counts'],entries=expected['entries'],minimum=int(spec['mode']=='numeric'))
                for row,expected in zip(workers['source252'],M['profiles'][profile]['cases']):counter_check(row['counts'],entries=expected['observation']['entries'],minimum=1)
                numeric=[c for c in matrix['cases'] if c['mode']=='numeric']
                require([r['id'] for r in workers['pristine233']]==[c['id'] for c in numeric],'pristine233 ID partition changed')
                for row,spec in zip(workers['pristine233'],numeric):
                    expected=matrix['observations'][matrix['profiles'][profile]['snapshots'][spec['id']]]['result'];require(row['digest']==a.digest(expected),'pristine233 value differs');counter_check(row['counts'],minimum=1)
                for row,expected in zip(workers['pristine252'],M['profiles'][profile]['cases']):
                    require(row['id']==expected['id'] and row['digest']==a.digest(expected['observation']['result']),'pristine252 differs');counter_check(row['counts'],aggregate=1)
    else:
        require(report['checks']=={} and [r['index'] for r in report['results']]==list(range(252)),'all252 IDs must be measured, no filtering')
        for row,case in zip(report['results'],M['cases']):
            require(type(row['index']) is int and row['case']==case and len(row['raw_samples'])==14 and row['orders']==orders(row['index']),'fixed7paired rounds differ')
            source=inspect.getsource(callback_pair);node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='call');callback_source=ast.get_source_segment(source,node)
            require(row['callback']['case']==case and row['callback']['callback_source']==callback_source and row['callback']['callback_source_sha256']==hashlib.sha256(callback_source.encode()).hexdigest() and row['callback']['callback_AST']==ast.dump(node,include_attributes=False),'actual timed callback source/AST differs')
            require(row['callback']['actual_targets']=={k:dict(module='skimage.morphology.max_tree' if k=='original' else 'skimagexx._max_tree',name=case['function']) for k in ('original','native')},'timed public target differs')
            state=row['callback']['input_state'];expected=M['profiles'][profile]['cases'][row['index']]['observation']
            require({k:state[k] for k in ('image','owner')}==expected['before'] and state['provided']==expected['provided_before'],'raw timed fixture/provided-tree data differs')
            require([(v['round'],v['backend']) for v in row['raw_samples']]==[(i,backend) for i,order in enumerate(row['orders']) for backend in order],'raw paired order differs')
            for value in row['raw_samples']:
                require(type(value['loops']) is int and 1<=value['loops']<=1000 and type(value['elapsed_ns']) is int and value['elapsed_ns']>0 and type(value['ms']) is float and math.isfinite(value['ms']) and value['ms']>0 and value['ms']==value['elapsed_ns']/value['loops']/1e6 and value['identity']==report['identities']['normal'] and value['callback_source_sha256']==row['callback']['callback_source_sha256'],'sample source/CPU/callback/elapsed differs')
                counter_check(value['counts'],aggregate=value['loops'] if value['backend']=='native' else 0)
            require(set(row['calibration'])=={'original','native'},'calibration raw samples missing')
            for backend,value in row['calibration'].items():
                require(type(value['loops']) is int and value['loops']==1 and type(value['elapsed_ns']) is int and value['elapsed_ns']>0 and type(value['ms']) is float and value['ms']==value['elapsed_ns']/1e6 and value['identity']==report['identities']['normal'] and value['callback_source_sha256']==row['callback']['callback_source_sha256'],'calibration identity/elapsed differs')
                counter_check(value['counts'],aggregate=1 if backend=='native' else 0)
            med={k:statistics.median([v['ms'] for v in row['raw_samples'] if v['backend']==k]) for k in ('original','native')}
            require(row['medians_ms']==med and row['speedup']==med['original']/med['native'] and row['slower']==(med['native']>med['original']),'paired statistics/regression flag differs')
        require(report['summary']==summarize(report['results']),'complete all-case summaries differ')
    return report


def verify_barrier(paths,cert,certificate_sha):
    rows=a.barrier(paths,certificate_sha,os.environ['GITHUB_RUN_ID'],os.environ['GITHUB_RUN_ATTEMPT'],cert)
    for path in paths:
        receipt=a.read(path);report=validate_report(a.read(Path(path).parent/'report.json'),cert)
        require(report['check_only'] is True and report['controller_sha256']==receipt['controller_sha256']==a.sha(__file__) and report['certificate_sha256']==certificate_sha,'preparation report/receipt source differs')
    return rows


def orders(index):
    rng=random.Random(151+index);rows=[]
    for _ in range(7):row=['original','native'];rng.shuffle(row);rows.append(row)
    return rows


def summarize(rows):
    require(len(rows)==252,'complete252 summary required')
    def group(values):return dict(cases=len(values),geomean=math.exp(math.fsum(math.log(v['speedup']) for v in values)/len(values)),slower=[v['index'] for v in values if v['slower']])
    return dict(overall=group(rows),rebuild=group([r for r in rows if not r['case']['provided']]),provided_tree=group([r for r in rows if r['case']['provided']]),functions={n:group([r for r in rows if r['case']['function']==n]) for n in a.NAMES})


def controller(args):
    require(os.environ.get('GITHUB_ACTIONS')=='true','actual product workers/timers restricted to isolated Actions')
    cert=a.certificate(args.certificate,args.certificate_sha256,a.controller_revision())
    nonce=None;claim=None
    if args.stage=='measure':
        verify_barrier(args.receipts,cert,args.certificate_sha256);nonce=secrets.token_hex(32)
        claim,_=a.claim(args.receipts,args.certificate_sha256,os.environ['GITHUB_RUN_ID'],os.environ['GITHUB_RUN_ATTEMPT'],cert,args.profile,nonce)
    else:require(not args.receipts,'preparation cannot consume a measurement receipt')
    # Product/receipt verification precedes numerical imports and worker startup.
    seal=a.verify_product(args.product,args.profile,cert);output=Path(args.output);require(not output.exists(),'fresh output required');output.mkdir(parents=True)
    retained_claim=a.retain_claim(claim,output) if claim is not None else None
    report=dict(schema='max-tree-public-comparison-v1',profile=args.profile,source=cert['source'],product_sha256=seal['product_sha256'],certificate_sha256=args.certificate_sha256,controller_sha256=a.sha(__file__),counts=COUNTS,check_only=args.stage=='prepare',identities={},checks={},results=[],failures=[],passed=False,measurement_claim=retained_claim)
    worker=None
    try:
        for mode in (('normal','scalar') if args.stage=='prepare' else ('normal',)):
            prefix=output/mode;prefix.mkdir();worker=Worker(args,prefix,scalar=mode=='scalar',measure=args.stage=='measure',nonce=nonce,claim=claim);report['identities'][mode]=worker.identity
            if args.stage=='prepare':report['checks'][mode]=worker.request(dict(operation='validate'))
            else:
                for index in range(252):
                    callback=worker.request(dict(operation='prepare-case',index=index));samples=[]
                    calibration={backend:worker.request(dict(operation='sample',index=index,backend=backend,loops=1)) for backend in ('original','native')}
                    loops={backend:max(1,min(1000,int(20/max(v['ms'],.001)))) for backend,v in calibration.items()}
                    for round_index,order in enumerate(orders(index)):
                        for backend in order:samples.append(dict(round=round_index,backend=backend,**worker.request(dict(operation='sample',index=index,backend=backend,loops=loops[backend]))))
                    med={k:statistics.median([v['ms'] for v in samples if v['backend']==k]) for k in ('original','native')}
                    report['results'].append(dict(index=index,case=callback['case'],callback=callback,orders=orders(index),calibration=calibration,raw_samples=samples,medians_ms=med,speedup=med['original']/med['native'],slower=med['native']>med['original']))
                    worker.request(dict(operation='release'));(output/'progress.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
            worker.request(dict(operation='integrity'));worker.close();worker=None
        require(a.verify_product(args.product,args.profile,cert)==seal,'product changed after execution');report['passed']=True
        if args.stage=='measure':report['summary']=summarize(report['results'])
        validate_report(report,cert)
    except BaseException as error:report['passed']=False;report['failures'].append(dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()));raise
    finally:
        if worker is not None and worker.process.poll() is None:
            try:worker.close()
            except BaseException as error:report['failures'].append(dict(phase='close',message=str(error)))
        a.save(output/'report.json',report)
    if args.stage=='prepare':a.save(output/'preparation.json',dict(schema='max-tree-preparation-v1',passed=True,profile=args.profile,certificate_sha256=args.certificate_sha256,generation=os.environ['GITHUB_RUN_ID'],attempt=os.environ['GITHUB_RUN_ATTEMPT'],modes=['normal','scalar'],workers=[1,4],counts=COUNTS,report_sha256=a.sha(output/'report.json'),product_sha256=seal['product_sha256'],controller_sha256=a.sha(__file__)))
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('fetch','selectors','prepare','measure','worker','barrier'))
    parser.add_argument('--certificate',type=Path,required=True);parser.add_argument('--certificate-sha256',required=True);parser.add_argument('--profile',choices=a.PROFILES)
    parser.add_argument('--product',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--receipts',nargs='*',type=Path,default=[]);parser.add_argument('--permit',type=Path);parser.add_argument('--measure',action='store_true')
    args=parser.parse_args(argv)
    if args.stage=='worker':return worker_main(args)
    cert=a.certificate(args.certificate,args.certificate_sha256,a.controller_revision())
    if args.stage=='fetch':return a.fetch(cert['profiles'][args.profile],args.output)
    if args.stage=='selectors':print('python='+cert['profiles'][args.profile]['python_version']);return
    if args.stage=='barrier':print(json.dumps(verify_barrier(args.receipts,cert,args.certificate_sha256)));return
    return controller(args)


if __name__=='__main__':main()
