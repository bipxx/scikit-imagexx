"""Source/inert controls for four fixed wide-outer reference calls; originals0."""
import ast,copy,hashlib,importlib.util,json,os,subprocess,sys,types
from datetime import datetime,timedelta,timezone
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
FILE=ROOT/'tests/mark_wide_outer_capture.py'
S=importlib.util.spec_from_file_location('igg_large_test',FILE);C=importlib.util.module_from_spec(S);S.loader.exec_module(C)
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def test_exact_four_cases_and_corrected_source_bounds():
    c=json.loads((ROOT/'coverage/contracts/mark-wide-outer-capture-v2.json').read_bytes())
    assert c['cases']==C.case_specs()and len(c['cases'])==4
    assert [C.bounds(s)['result_bytes_upper']for s in c['cases']]==[12,12,288,144]
    assert sum(C.bounds(s)['result_bytes_upper']for s in c['cases'])==456
    assert c['bounds']['max_result_bytes']==288 and c['bounds']['all_four_return_backing_bytes_upper']==456
    assert max(c['cases'][1]['labels'])==(1<<64)-1
    for name,pin in c['frozen_source_pins'].items():assert H(ROOT/name)==pin,name
    assert c['actual_original_GO']is None and c['actual_original_budget']==0


@pytest.mark.parametrize('field,value',[('label_dtype','object'),('background',1),('mode','inner'),('label_shape',[1000000,1000000]),('layout','other')])
def test_source_bounds_refuse_changed_frozen_case(field,value):
    spec=copy.deepcopy(C.case_specs()[0]);spec[field]=value
    with pytest.raises((AssertionError,StopIteration)):C.bounds(spec)


def envelope(tmp_path,monkeypatch):
    root=tmp_path/'root';root.mkdir();monkeypatch.setattr(C,'R',root)
    helpers={}
    for name in ('source','compiler_receipt','library'):
        path=root/(name+'.txt');path.write_text('inert metadata');helpers[name]=dict(path=path.name,sha256=H(path))
    a=root/'admission.json';a.write_text('{"passed":true}');helpers['admission']=dict(path=a.name,sha256=H(a));helpers['ABI']=[32,0,4,8,12,20,28,8]
    specs=C.case_specs();ids=[s['id']for s in specs]
    q=dict(schema='mark-wide-outer-capture-binding-v2',status='UNBOUND_CAPTURE_CAPABLE_ZERO_BUDGET',source='a'*40,frozen_case_source='3106c063a0d34af0af48a2f2f3928fade46c322d',original_budget=0,native_image_budget=0,sampler_budget=0,observation_budget=0,root_original_GO=None,case_ids=ids,api_case_ids={api:[s['id']for s in specs if s['api']=='segmentation.'+api]for api in('find_boundaries','mark_boundaries')},profile_case_ids={p:ids for p in('minimum','modern')},controller_raw_sha256=H(FILE),profiles={'minimum':{},'modern':{}},git_argv=['git','-C','inert'],source_files_raw_sha256={},case_contract=dict(path='coverage/contracts/mark-wide-outer-capture-v2.json',sha256=H(ROOT/'coverage/contracts/mark-wide-outer-capture-v2.json')),helper=helpers,cases_sha256=hashlib.sha256(json.dumps(specs,sort_keys=True).encode()).hexdigest(),prospective_calls_per_profile=4,prospective_total_calls=8,calls_per_recipe=1,prefix_factory_invocations_per_process=8,prefix_factory_invocations_total=64,prefix_inert_return_arrays_per_process=1)
    for n in C.LIMITS:monkeypatch.setenv(n,'1')
    for n in ('GIT_DIR','GIT_WORK_TREE','LD_PRELOAD','LD_AUDIT','OPENBLAS_CORETYPE','NPY_DISABLE_CPU_FEATURES','MKL_DEBUG_CPU_TYPE'):monkeypatch.delenv(n,raising=False)
    monkeypatch.setattr(C.subprocess,'check_output',lambda argv,**kw:'a'*40+'\n'if argv[-1]=='HEAD'else'')
    names=set(json.loads((ROOT/'coverage/contracts/mark-wide-outer-capture-v2.json').read_bytes())['frozen_source_pins'])|{'tests/mark_wide_outer_capture.py','tests/test_mark_wide_outer_capture.py','coverage/contracts/mark-wide-outer-capture-v2.json'}
    q['source_files_raw_sha256']={n:H(ROOT/n)for n in names}
    path=tmp_path/'binding.json';path.write_text(json.dumps(q));return q,path,root


@pytest.mark.parametrize('fault',('none','budget','cases','schema','controller','GO-field'))
def test_actual_binding_metadata_boundary_is_zero_authority(tmp_path,monkeypatch,fault):
    q,path,_=envelope(tmp_path,monkeypatch)
    if fault=='budget':q['original_budget']=8
    elif fault=='cases':q['case_ids']=q['case_ids'][::-1]
    elif fault=='schema':q['schema']='inverse-gaussian-gradient-capture-v1'
    elif fault=='controller':q['controller_raw_sha256']='0'*64
    elif fault=='GO-field':q['root_original_GO']='not-authority'
    path.write_text(json.dumps(q))
    if fault=='none':assert C.validate(path,H(path),'minimum')==q
    else:
        with pytest.raises(AssertionError):C.validate(path,H(path),'minimum')


def test_actual_capture_missing_GO_refuses_without_runtime(tmp_path,monkeypatch):
    q,path,_=envelope(tmp_path,monkeypatch);C.validate(path,H(path),'minimum')
    args=types.SimpleNamespace(root_go=None,root_go_sha256=None)
    with pytest.raises(AssertionError,match='NO_ROOT_GO'):C.validate_go(args,q)


def nested_target_call(namespace):
    tree=ast.parse(FILE.read_bytes());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main');node=copy.deepcopy(next(n for n in ast.walk(main)if isinstance(n,ast.FunctionDef)and n.name=='target_call'))
    # Same counter binding in a module namespace instead of the runtime closure.
    for i,n in enumerate(node.body):
        if isinstance(n,ast.Nonlocal):assert n.names==['public_attempts','total_public'];node.body[i]=ast.Global(names=n.names)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<actual target wrapper>','exec'),namespace)
    return namespace['target_call']


def test_consumed_immediate_raw_survives_encoder_failure():
    class EncError(Exception):pass
    samples=[object(),object()];calls=[];diagnostics={'immediate_raw':[]}
    namespace=dict(args=types.SimpleNamespace(mode='capture'),capture_allowed=True,current_case='inert',public_attempts=0,total_public=0,check_deadline=lambda go:None,go={},np_state=lambda np:{},np=None,env=lambda:samples.pop(0),targets={'find_boundaries':lambda *a,**k:calls.append(1)},active_api='find_boundaries',immediate_calls=[],_DIAGNOSTICS=diagnostics,state=lambda value:(_ for _ in()).throw(EncError('encoding failed')))
    with pytest.raises(EncError):nested_target_call(namespace)(None)
    assert calls==[1] and namespace['total_public']==diagnostics['public_attempts']==1
    assert len(diagnostics['immediate_raw'])==1 and len(samples)==0


def fenv():
    x=C.FEnv();x.control=0x37f;x.tags=0xffff;x.mxcsr=0x1f80;return x


@pytest.mark.parametrize('fault',('completed-writer','observer-then-failure-writer','caller-restore-and-failure-writer'))
def test_actual_one_shot_loop_failure_preserves_consumption_and_ack(tmp_path,fault):
    main=next(n for n in ast.parse(FILE.read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='main')
    branch=next(n for n in ast.walk(main)if isinstance(n,ast.If)and ast.unparse(n.test)=="args.mode == 'capture'"and any(isinstance(x,ast.For)for x in n.body))
    events=[];np=types.SimpleNamespace(geterr=lambda:{},geterrcall=lambda:None,getbufsize=lambda:8192,get_printoptions=lambda:{},seterr=lambda **kw:None,seterrcall=lambda x:None,setbufsize=lambda x:None,set_printoptions=lambda **kw:None)
    env=dict(args=types.SimpleNamespace(output=tmp_path,root_go=tmp_path/'GO',root_go_sha256='pin',binding=tmp_path/'binding',binding_sha256='pin'),go={'runtime_identity':{}},identity={},dump=lambda *a:None,check_deadline=lambda go:None,H=lambda p:'pin',specs=[{'id':str(i),'api':'segmentation.find_boundaries'}for i in range(4)],env=fenv,restore=lambda x:None,np=np,counts={'target_attempted':0},capture_rows=[],total_public=0,targets={},EntryRefused=C.EntryRefused,datetime=datetime,timezone=timezone,traceback=C.traceback,state=lambda x:C._raw_fenv(x),_DIAGNOSTICS={'immediate_raw':[]})
    def observer(call,spec,np):
        env.update(public_attempts=1,call_count=1,immediate_calls=[{'before':{},'after':{}}],total_public=1)
        env['counts']['target_attempted']+=1
        if fault.startswith('observer'):raise ValueError('primary observer encoder failure')
        return {'id':spec['id']}
    def append(path,row):
        events.append(row['event'])
        if row['event']=='completed':raise OSError('completed write failure')
        if row['event']=='failed_case_diagnostics'and(fault.startswith('observer')or fault.startswith('caller')):raise RuntimeError('secondary writer failure')
    env.update(observe=observer,target_call=lambda:None,append=append,q={'source':'inert','case_ids':[str(i)for i in range(4)]},Path=Path,__file__=str(FILE))
    if fault.startswith('caller'):
        calls=[]
        def failed_restore_state():
            x=fenv();x.status=1 if calls else 0;calls.append(1);return x
        env['env']=failed_restore_state
    with pytest.raises((OSError,ValueError,AssertionError)):
        exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(branch.body),type_ignores=[])),'<actual capture loop>','exec'),env)
    assert env['total_public']==env['counts']['target_attempted']==1 and env['capture_rows']==[]
    assert env['_DIAGNOSTICS']['caller_raw']['exact32']==(not fault.startswith('caller'))and events[0]=='started'
    if fault.startswith('caller'):
        assert env['_DIAGNOSTICS']['last_case_diagnostics']['failure']['type']=='AssertionError'
        assert env['_DIAGNOSTICS']['last_case_diagnostics']['failure_writer_error']['type']=='RuntimeError'
    elif fault.startswith('observer'):
        assert env['last_case_diagnostics']['failure']['type']=='ValueError'
        assert env['last_case_diagnostics']['failure_writer_error']['type']=='RuntimeError'
    else:assert events==['started','completed','failed_case_diagnostics']


def test_actual_terminal_fallback_preserves_primary_and_partial_state(tmp_path,monkeypatch):
    def broken():raise RuntimeError('primary setup failure')
    monkeypatch.setattr(C,'main',broken);monkeypatch.setattr(sys,'argv',['controller','--output',str(tmp_path)])
    monkeypatch.setattr(C,'_DIAGNOSTICS',dict(owns_output=True,public_attempts=1,rows=[],immediate_raw=[dict(before=fenv(),after=fenv())],caller_raw=dict(saved=fenv(),actual=fenv(),exact32=True)))
    with pytest.raises(RuntimeError):C.entry()
    row=json.loads((tmp_path/'terminal-fallback.json').read_bytes())
    assert row['public_attempts']==1 and row['completed_acknowledged']==0 and row['caller_restore']['exact32']
    assert row['process_restore']is None and row['primary_failure']['type']=='RuntimeError' and row['retry']is False


def test_actual_CLI_unbound_refuses_before_science(tmp_path):
    output=tmp_path/'fresh';args=[sys.executable,str(FILE),'--profile','minimum','--prefix-api','find_boundaries','--binding',str(tmp_path/'absent'),'--binding-sha256','0'*64,'--output',str(output),'--expect-prerequisite-refusal']
    p=subprocess.run(args,capture_output=True,text=True);assert p.returncode==0,p.stderr
    row=json.loads((output/'terminal.json').read_bytes());assert row['scientific_imports']==row['original_budget']==row['target_body_lines']==0 and not row['journal_created']

def go_envelope(tmp_path,monkeypatch,profile='minimum'):
    q,binding,root=envelope(tmp_path,monkeypatch);identity={'scope':'INERT_GO_ENVELOPE_ONLY'};pin=H(binding)
    review=root/'readiness.json';review.write_text(json.dumps(dict(passed=True,source=q['source'],controller_sha256=q['controller_raw_sha256'],binding_sha256=pin)))
    prefixes=[]
    for api,cache in [(a,c)for a in('find_boundaries','mark_boundaries')for c in('cold','cached')]:
        row=dict(status='PASS_ACTUAL_MARK_WIDE_OUTER_CONTROLLER_ZERO_BODY_PREFIX',profile=profile,source=q['source'],controller_raw_sha256=q['controller_raw_sha256'],binding_sha256=pin,recipe_observations=0,journal_created=False,full_initial_state_restored=True,counts=dict(target_body_lines=0,provider_body_lines=0,target_attempted=2,target_refused=2,provider_attempted=0,provider_refused=0,compiled_provider_attempted=0,compiled_provider_refused=0),controls=[dict(caller_restore=dict(exact32=True))for _ in range(3)],observer_target_immediate=[{}],module_request=dict(was_cached=cache=='cached'),cache_state=cache,case_ids=q['case_ids'],full_recipe_pool_count=4,factory_invocations=8,inert_return_array_allocations=1,runtime_identity=identity,helper=q['helper'],factory_controls=[dict(id=s,all_input_words_equal_frozen_recipe=True,caller_restore=dict(exact32=True))for s in q['case_ids']],observer_success_controls=[{}],codec_metadata_controls=[dict(caller_restore=dict(exact32=True))for _ in range(2)],prefix_api=api)
        path=root/(api+'-'+cache+'.json');path.write_text(json.dumps(row));prefixes.append(dict(path=path.name,sha256=H(path)))
    go=dict(status='ROOT_AUTHORIZED_MARK_WIDE_OUTER_ONCE',source=q['source'],binding_sha256=pin,controller_sha256=q['controller_raw_sha256'],source_files_raw_sha256=q['source_files_raw_sha256'],cases_sha256=q['cases_sha256'],case_ids=q['case_ids'],api_case_ids=q['api_case_ids'],profiles=['minimum','modern'],profile=profile,maximum_public_topcalls_per_profile=4,maximum_total_public_topcalls=8,calls_per_recipe=1,native_calls=0,sampler_bodies=0,timed_batches=0,builds=0,journal_directory=str(tmp_path/'capture'),no_retry=True,not_before_utc=(datetime.now(timezone.utc)-timedelta(minutes=1)).isoformat(),deadline_utc=(datetime.now(timezone.utc)+timedelta(minutes=5)).isoformat(),independent_readiness=dict(path=review.name,sha256=H(review)),prefixes=prefixes,runtime_identity=identity,minimum_terminal_review=None)
    if profile=='modern':
        path=root/'minimum-admission.json';path.write_text(json.dumps(dict(passed=True,source=q['source'],profile='minimum',public_topcalls=4,completed_rows=4,binding_sha256=pin)));go['minimum_terminal_review']=dict(path=path.name,sha256=H(path))
    path=tmp_path/'GO.json';path.write_text(json.dumps(go));args=types.SimpleNamespace(root_go=path,root_go_sha256=H(path),profile=profile,output=tmp_path/'capture',binding_sha256=pin)
    return args,q,go,path


@pytest.mark.parametrize('profile',('minimum','modern'))
@pytest.mark.parametrize('fault',('none','budget','other-profile','minimum-review','prefix-body','deadline'))
def test_actual_GO_bounds_profile_and_sequential_gate(tmp_path,monkeypatch,profile,fault):
    args,q,go,path=go_envelope(tmp_path,monkeypatch,profile)
    if fault=='budget':go['maximum_public_topcalls_per_profile']=9
    elif fault=='other-profile':go['profile']='modern'if profile=='minimum'else'minimum'
    elif fault=='minimum-review':go['minimum_terminal_review']=None if profile=='modern'else{'path':'unexpected','sha256':'0'*64}
    elif fault=='prefix-body':
        p=C.R/go['prefixes'][0]['path'];row=json.loads(p.read_bytes());row['counts']['target_body_lines']=1;p.write_text(json.dumps(row));go['prefixes'][0]['sha256']=H(p)
    elif fault=='deadline':go['deadline_utc']=(datetime.now(timezone.utc)-timedelta(minutes=1)).isoformat()
    path.write_text(json.dumps(go));args.root_go_sha256=H(path)
    if fault=='none':assert C.validate_go(args,q)==go
    else:
        with pytest.raises((AssertionError,FileNotFoundError)):C.validate_go(args,q)


@pytest.mark.parametrize('api',('find_boundaries','mark_boundaries'))
def test_actual_trace_refuses_first_frame_even_cached(api):
    code=compile('def '+api+'(*args,**kwargs):\n    body.append(1)\n', '<pinned inert source>', 'exec')
    dummy={'body':[]};exec(code,dummy)
    tree=ast.parse(FILE.read_bytes());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    nodes=[copy.deepcopy(next(n for n in ast.walk(main)if isinstance(n,ast.FunctionDef)and n.name==name))for name in('kind','trace')]
    for node in nodes:
        for i,part in enumerate(node.body):
            if isinstance(part,ast.Nonlocal):node.body[i]=ast.Global(names=part.names)
    env=dict(target_path='<pinned inert source>',active_api=api,targets={api:dummy[api]},protected={},args=types.SimpleNamespace(mode='prefix'),counts=dict(target_attempted=0,target_refused=0,target_body_lines=0),entries=[],state=lambda x:{},env=lambda:None,EntryRefused=C.EntryRefused)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'<actual entry trace>','exec'),env)
    prior=sys.gettrace();sys.settrace(env['trace'])
    try:
        with pytest.raises(C.EntryRefused,match='TARGET_REFUSED_BEFORE_BODY'):dummy[api](None)
    finally:sys.settrace(prior)
    assert dummy['body']==[]and env['counts']==dict(target_attempted=1,target_refused=1,target_body_lines=0)


def test_actual_terminal_writer_fallback_retains_primary_and_raw(tmp_path,monkeypatch):
    def broken():C.dump(tmp_path/'terminal.json',{'status':'primary consumed-action failure'})
    monkeypatch.setattr(C,'main',broken);monkeypatch.setattr(sys,'argv',['controller','--output',str(tmp_path)])
    monkeypatch.setattr(C,'_DIAGNOSTICS',dict(owns_output=True,public_attempts=1,rows=[],immediate_raw=[dict(before=fenv(),after=fenv())],caller_raw=dict(saved=fenv(),actual=fenv(),exact32=True),primary_failure={'type':'ValueError','message':'primary consumed-action failure'}))
    actual=C.dump
    def writer(path,value):
        if path.name=='terminal.json':raise OSError('terminal writer failed')
        return actual(path,value)
    monkeypatch.setattr(C,'dump',writer)
    with pytest.raises(OSError):C.entry()
    row=json.loads((tmp_path/'terminal-fallback.json').read_bytes())
    assert row['primary_failure']['type']=='ValueError'and row['terminal_failure']['type']=='OSError'
    assert row['public_attempts']==1 and row['completed_acknowledged']==0 and len(row['immediate_raw'])==1 and row['caller_restore']['exact32']and row['retry']is False


def test_reused_failure_and_fp_functions_same_interpreter_source():
    old=Path('C:/Users/HEQ/Projects/scikit-imagexx/build/worktrees/igg-large-accuracy-v1/tests/inverse_gaussian_gradient_large_accuracy.py')
    before=ast.parse(old.read_bytes());after=ast.parse(FILE.read_bytes())
    for name in ('dump','append','np_state','check_deadline','_raw_fenv'):
        a=next(n for n in before.body if isinstance(n,ast.FunctionDef)and n.name==name)
        b=next(n for n in after.body if isinstance(n,ast.FunctionDef)and n.name==name)
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)



def test_actual_four_factory_prefix_block_with_inert_arrays():
    # Storage stand-ins exercise actual factory/validator/receipt assembly;
    # this is not NumPy setup or output accuracy evidence.
    import struct,math
    class Dtype:
        metadata=None
        def __init__(self,name):self.name=name
        def __eq__(self,other):return self.name==other.name
    class Array:
        def __init__(self,values,dtype):
            self.dtype=Dtype(dtype);self.raw=struct.pack('<'+{'int64':'q','uint64':'Q','uint8':'B','float32':'f'}[dtype]*len(values),*values);self.nbytes=len(self.raw)
            self.flags=types.SimpleNamespace(c_contiguous=True,writeable=True)
        def reshape(self,*shape):self.shape=tuple(shape[0])if len(shape)==1 else shape;return self
        def tobytes(self):return self.raw
    np=types.SimpleNamespace(array=Array,ndarray=Array,dtype=Dtype,geterr=lambda:{},geterrcall=lambda:None,getbufsize=lambda:8192)
    main=next(n for n in ast.parse(FILE.read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='main')
    loop=next(n for n in ast.walk(main)if isinstance(n,ast.For)and ast.unparse(n.target)=='fixed'and ast.unparse(n.iter)=='specs')
    env=dict(specs=C.case_specs(),env=fenv,np=np,fixture=C.fixture,validate_factory=C.validate_factory,json=json,factory_controls=[],restore=lambda value:None,state=C._raw_fenv)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(loop)],type_ignores=[])),'<actual factory-prefix loop>','exec'),env)
    assert len(env['factory_controls'])==4
    assert [r['result_bytes_upper']for r in env['factory_controls']]==[12,12,288,144]
    assert all(r['all_input_words_equal_frozen_recipe']and r['caller_restore']['exact32']for r in env['factory_controls'])


@pytest.mark.parametrize('control,mxcsr',((0x37f,0x1f80),(0x77f,0x1f80),(0x37f,0x3f80)))
def test_actual_initial_rounding_scope_guard(control,mxcsr):
    main=next(n for n in ast.parse(FILE.read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='main')
    assertion=next(n for n in ast.walk(main)if isinstance(n,ast.Assert)and isinstance(n.msg,ast.Constant)and n.msg.value=='SOURCE_SCOPE_REQUIRES_NEAREST')
    env={'initial':types.SimpleNamespace(control=control,mxcsr=mxcsr)}
    code=compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(assertion)],type_ignores=[])),'<actual incoming-rounding guard>','exec')
    if control==0x37f and mxcsr==0x1f80:exec(code,env)
    else:
        with pytest.raises(AssertionError,match='SOURCE_SCOPE_REQUIRES_NEAREST'):exec(code,env)


@pytest.mark.parametrize('stage',('open','write','flush','fsync','close'))
def test_actual_entry_partial_terminal_io_requires_distinct_fallback(tmp_path,monkeypatch,stage):
    actual_open=Path.open;actual_fsync=C.os.fsync;faulted=[];terminal_fd=[]
    class Partial:
        def __init__(self,f):self.f=f;terminal_fd.append(f.fileno())
        def __enter__(self):return self
        def __exit__(self,*a):
            self.f.close()
            if stage=='close'and not faulted:faulted.append(stage);raise OSError('terminal close fault')
        def write(self,raw):
            if stage=='write'and not faulted:
                self.f.write(raw[:9]);faulted.append(stage);raise OSError('terminal partial write fault')
            return self.f.write(raw)
        def flush(self):
            self.f.flush()
            if stage=='flush'and not faulted:faulted.append(stage);raise OSError('terminal flush fault')
        def fileno(self):return self.f.fileno()
    def opening(path,*a,**kw):
        f=actual_open(path,*a,**kw)
        if path.name!='terminal.json':return f
        if stage=='open'and not faulted:f.close();faulted.append(stage);raise OSError('terminal partial open fault')
        return Partial(f)
    def syncing(fd):
        actual_fsync(fd)
        if stage=='fsync'and fd in terminal_fd and not faulted:faulted.append(stage);raise OSError('terminal fsync fault')
    monkeypatch.setattr(Path,'open',opening);monkeypatch.setattr(C.os,'fsync',syncing)
    monkeypatch.setattr(C,'_DIAGNOSTICS',dict(owns_output=True,public_attempts=1,rows=[],primary_failure={'type':'ValueError','message':'primary consumed error'},last_case_diagnostics={'failure_writer_error':{'type':'RuntimeError','message':'secondary writer'},'id':'inert'},immediate_raw=[dict(before=fenv(),after=fenv())],caller_raw=dict(saved=fenv(),actual=fenv(),exact32=True)))
    def actual_writer_boundary():
        C.dump(tmp_path/'terminal.json',{'status':'failure'})
        C._DIAGNOSTICS['terminal_acknowledged']=True
    monkeypatch.setattr(C,'main',actual_writer_boundary);monkeypatch.setattr(sys,'argv',['controller','--output',str(tmp_path)])
    with pytest.raises(OSError):C.entry()
    assert faulted==[stage]and (tmp_path/'terminal.json').exists()
    row=json.loads((tmp_path/'terminal-fallback.json').read_bytes())
    assert row['partial_terminal_created']and row['terminal_acknowledged']is False
    assert row['primary_failure']['type']=='ValueError'and row['last_case_diagnostics']['failure_writer_error']['type']=='RuntimeError'
    assert row['public_attempts']==1 and row['completed_acknowledged']==0 and len(row['immediate_raw'])==1
    assert row['caller_restore']['exact32']and C._DIAGNOSTICS['fallback_acknowledged']and not C._DIAGNOSTICS.get('terminal_acknowledged',False)


@pytest.mark.parametrize('prior_failure',(False,True))
def test_actual_late_validation_and_writer_failure_preserve_full_diagnostics(tmp_path,monkeypatch,prior_failure):
    # Execute the actual final trust-boundary try and terminal tail, with inert
    # state and deliberate writers. No runtime import, target, helper or body.
    main=next(n for n in ast.parse(FILE.read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='main')
    final=next(n for n in main.body if isinstance(n,ast.Try)and any(isinstance(x,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='final_failure'for t in x.targets)for h in n.handlers for x in h.body))
    tail=copy.deepcopy(main.body[main.body.index(final):])
    before=fenv();after=fenv();after.status=1
    last={'id':'inert-case','failure_writer_error':{'type':'OSError','message':'secondary case writer'},'caller_restore':{'exact32':False}}
    diagnostics=dict(owns_output=True,public_attempts=1,rows=[],immediate_raw=[dict(before=before,after=after)],caller_raw=dict(saved=before,actual=after,exact32=False),last_case_diagnostics=last)
    failure={'type':'AssertionError','message':'caller restore mismatch'}if prior_failure else None
    names={n.id for n in ast.walk(ast.Module(body=tail,type_ignores=[]))if isinstance(n,ast.Name)}
    import builtins
    env={n:None for n in names if not hasattr(builtins,n)};env.update(args=types.SimpleNamespace(mode='capture',binding=tmp_path/'binding',binding_sha256='pin',root_go=tmp_path/'GO',root_go_sha256='pin',output=tmp_path,profile='minimum',prefix_api=None,cache_state='cold'),failure=failure,last_case_diagnostics=last,_DIAGNOSTICS=diagnostics,FACTORY_INVOCATIONS=4,q=dict(source='inert',controller_raw_sha256='pin',source_files_raw_sha256={},helper={},git_argv=[],case_ids=['inert-case']),Path=Path,__file__=str(FILE),traceback=C.traceback,initial=before,final=before,state=C._raw_fenv,counts={'target_attempted':1},entries=[],controls=[],codec_metadata_controls=[],factory_controls=[],observer_success_controls=[],immediate_calls=[],total_public=1,capture_rows=[],LIMITS=(),os=os,sys=sys)
    hash_calls=[]
    def bad_hash(path):
        if not hash_calls:hash_calls.append(path);raise ValueError('late source validation failure')
        return 'pin'
    def bad_dump(path,row):
        path.write_text('{partial',encoding='utf-8');raise OSError('late terminal partial writer')
    env.update(H=bad_hash,dump=bad_dump)
    def run_tail():exec(compile(ast.fix_missing_locations(ast.Module(body=tail,type_ignores=[])),'<actual final source/terminal tail>','exec'),env)
    monkeypatch.setattr(C,'main',run_tail);monkeypatch.setattr(C,'_DIAGNOSTICS',diagnostics);monkeypatch.setattr(sys,'argv',['controller','--output',str(tmp_path)])
    with pytest.raises(OSError):C.entry()
    row=json.loads((tmp_path/'terminal-fallback.json').read_bytes())
    assert row['final_validation_failure']['type']=='ValueError'and row['terminal_failure']['type']=='OSError'
    assert row['primary_failure']['type']==('AssertionError'if prior_failure else'ValueError')
    assert row['last_case_diagnostics']['failure_writer_error']['type']=='OSError'and not row['caller_restore']['exact32']
    assert row['public_attempts']==1 and row['completed_acknowledged']==0 and row['partial_terminal_created']


@pytest.mark.parametrize('api',('find_boundaries','mark_boundaries'))
@pytest.mark.parametrize('cached',(False,True))
def test_actual_identity_trace_refuses_accepted_filename_alias(tmp_path,api,cached):
    origin=tmp_path/'boundaries.py';alias=str(tmp_path/'alias')+'/../boundaries.py'
    assert Path(alias).resolve()==origin.resolve()and alias!=str(origin)
    dummy={'body':[]};exec(compile('def '+api+'(*args,**kwargs):\n    body.append(1)\n',alias,'exec'),dummy)
    target=dummy[api];assert isinstance(target,C.FunctionType)
    assert Path(target.__code__.co_filename).resolve()==origin.resolve()
    # Cold/cached refers to the inert callable lookup, not a scientific import.
    cache={api:target}if cached else{};cache.setdefault(api,target)
    main=next(n for n in ast.parse(FILE.read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='main')
    nodes=[copy.deepcopy(next(n for n in ast.walk(main)if isinstance(n,ast.FunctionDef)and n.name==name))for name in('kind','trace')]
    for node in nodes:
        for i,part in enumerate(node.body):
            if isinstance(part,ast.Nonlocal):node.body[i]=ast.Global(names=part.names)
    env=dict(target_path=str(origin),active_api=api,targets=cache,protected={},args=types.SimpleNamespace(mode='prefix'),counts=dict(target_attempted=0,target_refused=0,target_body_lines=0),entries=[],state=lambda x:{},env=lambda:None,EntryRefused=C.EntryRefused)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'<actual alias entry trace>','exec'),env)
    old=sys.gettrace();sys.settrace(env['trace'])
    try:
        with pytest.raises(C.EntryRefused):cache[api](None)
    finally:sys.settrace(old)
    assert dummy['body']==[]and env['counts']['target_refused']==1 and env['counts']['target_body_lines']==0


def test_actual_fixture_counter_includes_observer_materializations(monkeypatch):
    monkeypatch.setattr(C,'FACTORY_INVOCATIONS',0)
    class Array:
        def __init__(self,*a,**k):pass
        def reshape(self,*a):return self
    np=types.SimpleNamespace(array=Array)
    specs=C.case_specs()
    for spec in specs:C.fixture(spec,np)  # Four explicit factory controls.
    for _ in range(4):C.fixture(specs[0],np)  # One observer refusal,2errors,1success.
    assert C.FACTORY_INVOCATIONS==8
    c=json.loads((ROOT/'coverage/contracts/mark-wide-outer-capture-v2.json').read_bytes())
    assert c['successor_v3']['total_prefix_fixture_invocations']==64 and c['successor_v3']['inert_return_bytes_per_process']==16



def test_durable_terminal_ack_prevents_duplicate_fallback(tmp_path,monkeypatch):
    monkeypatch.setattr(C,'_DIAGNOSTICS',dict(owns_output=True))
    def completed_failure_terminal():
        C.dump(tmp_path/'terminal.json',{'status':'FAILED_INERT_COMPLETE_TERMINAL'})
        C._DIAGNOSTICS['terminal_acknowledged']=True
        raise SystemExit(1)
    monkeypatch.setattr(C,'main',completed_failure_terminal);monkeypatch.setattr(sys,'argv',['controller','--output',str(tmp_path)])
    with pytest.raises(SystemExit):C.entry()
    assert C._DIAGNOSTICS['terminal_acknowledged']and not(tmp_path/'terminal-fallback.json').exists()
    assert json.loads((tmp_path/'terminal.json').read_bytes())['status']=='FAILED_INERT_COMPLETE_TERMINAL'
