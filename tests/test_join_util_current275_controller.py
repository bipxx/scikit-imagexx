"""Affected JOIN/util source/inert controller checks; no target/helper/sampler bodies."""
import ast,copy,hashlib,importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/join-util-current275-measurement-v3.json'
DATA=json.loads(CONTRACT.read_bytes());ROUTES=DATA['first_cohort']['routes']
spec=importlib.util.spec_from_file_location('join_util_current275',ROOT/'tools/join_util_current275_probe.py')
PROBE=importlib.util.module_from_spec(spec);spec.loader.exec_module(PROBE)
TOOL=PROBE
H=lambda b:hashlib.sha256(b).hexdigest()


@pytest.mark.parametrize('name',ROUTES)
def test_each_real_api_cli_interface(name):
    route=ROUTES[name]
    argv=PROBE.driver_argv(ROOT,route,'binding.json','never.json','UTC')
    assert argv[0]==str(ROOT/route['driver'])
    if route['kind']=='util-executor':
        assert '--deadline-utc'not in argv and '--first-cohort'in argv
        assert argv[argv.index('--api')+1]==route['api']
        assert argv[argv.index('--case')+1]==route['first_case']['id']
        assert argv[-2:]==['--root-binding','binding.json']
    else:assert argv[argv.index('--deadline-utc')+1]=='UTC'


def test_contexts_scoped_without_recipe_or_method_override(tmp_path):
    util=json.loads((ROOT/'coverage/contracts/next-util-benchmark-execution-v4.json').read_bytes())
    for name,route in ROUTES.items():
        binding=PROBE.scoped_binding({'source_revision':'source'},name,route,tmp_path/'go.json',hashlib.sha256((ROOT/route['contract']).read_bytes()).hexdigest())
        assert set(binding['routes'])=={name}and binding['routes'][name]is route
        assert binding['base_contract_sha256']==hashlib.sha256((ROOT/route['contract']).read_bytes()).hexdigest()
        if route['kind']=='util-executor':
            actual=util['routes'][route['execution_route']]
            assert route['cases']==[s for s in actual['cases']if s.get('api',actual['apis'][0])==route['api']]
            measurement=PROBE.scoped_measurement(DATA,route,'parent')
            family=measurement['families'][Path(route['driver']).stem]
            assert family['case_ids']==[s['id']for s in route['cases']]
            assert family['cases']==len(route['cases'])
            assert measurement['execution_scope']==DATA['execution_scope']
    assert ROUTES['blocks']['first_case']!=ROUTES['windows']['first_case']
    assert DATA['families']['array_views']['cases']==8
    assert len(ROUTES['blocks']['cases'])==len(ROUTES['windows']['cases'])==4
    bad=copy.deepcopy(ROUTES['blocks']);bad['cases'][0]['id']='unadmitted'
    with pytest.raises(AssertionError):PROBE.scoped_measurement(DATA,bad,'parent')


def frame_fixture(name):
    route=ROUTES[name];args=('input',);options={'exact':True};state=('unchanged',)
    caller=dict(cli=SimpleNamespace(first_cohort=True,measurement_go=None),spec=copy.deepcopy(route['first_case']),api=route['api'],accepted=[True],rows=[],returns=[k for k,v in route['native_return_counts'].items()for _ in range(v)],requirements=dict(route['native_return_counts']),expected_state={'alias':'same'},actual_state={'alias':'same'},reference_FP={'flags':32,'rounding':0},native_FP={'flags':32,'rounding':0},save=lambda:bytes(32),caller=bytes(32),args=args,options=options,inputs=['input'],before=[state],np=None)
    parent=SimpleNamespace(f_code=SimpleNamespace(co_filename=str(ROOT/'benchmarks/next_util_execution.py')),f_locals=caller,f_globals={'array_state':lambda v,np:state})
    frame=SimpleNamespace(f_back=parent)
    input_state=lambda v:v
    return route,frame,dict(counts=dict(original=1,native=1),accepted=[True],returns=[],getters=[],input_state=input_state,inputs_before=[(args,options)])


@pytest.mark.parametrize('name',[k for k,v in ROUTES.items()if v['kind']=='util-executor'])
def test_real_execute_frame_adapter_positive(name):
    route,frame,kwargs=frame_fixture(name)
    PROBE.validate_util_entry(frame,route,ROOT,**kwargs)


def test_actual_root_refusal_stops_before_science(monkeypatch,tmp_path):
    binding=tmp_path/'binding.json';binding.write_text(json.dumps({'status':'UNBOUND'}))
    monkeypatch.setattr(PROBE.sys,'argv',['probe','--binding',str(binding),'--root-go',str(tmp_path/'never.json'),'--route','blocks'])
    with pytest.raises(AssertionError,match='UNBOUND'):PROBE.main()
    assert list(tmp_path.iterdir())==[binding]


def test_measurement_commands_cannot_replay_first_cohort_or_pass_deadline_to_util(tmp_path):
    authorities={name:{'binding':name+'.binding','measurement':name+'.measurement'}for name in ROUTES}
    commands=PROBE.measurement_commands(ROOT,tmp_path,DATA,'python',authorities,'UTC')
    assert len(commands)==5 and 'metrics-contingency'not in commands
    for name,argv in commands.items():
        assert '--first-cohort'not in argv and '--case'not in argv
        if ROUTES[name]['kind']=='util-executor':
            assert '--deadline-utc'not in argv and '--measurement-go'in argv
            assert argv[argv.index('--api')+1]==ROUTES[name]['api']
        else:assert '--deadline-utc'in argv


@pytest.mark.parametrize('cached',(False,True))
@pytest.mark.parametrize('tamper',('none','helper-digest','helper-env','route','counter'))
def test_actual_bound_probe_metadata_reaches_import_request_refusal(monkeypatch,tmp_path,cached,tamper):
    import builtins,sys
    from datetime import datetime,timezone,timedelta
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    source=tmp_path/'source';source.mkdir();revision='a'*40
    admission=tmp_path/'admission.json';admission.write_text(json.dumps(dict(source=revision,passed=True,root_accepted=True,profiles=4)))
    bundle=tmp_path/'helper';bundle.mkdir();receipt=bundle/'receipt.json';receipt.write_text('{"synthetic_metadata_only":true}')
    env=dict(UTIL_VIEWS_FENV_MODE='source-built-v1',UTIL_VIEWS_FENV_RECEIPT=str(receipt),UTIL_VIEWS_FENV_RECEIPT_SHA256=sha(receipt))
    helper=tmp_path/'helper-binding.json';helper.write_text(json.dumps(dict(env=env,files={'receipt.json':sha(receipt)})))
    binding=tmp_path/'binding.json';go=tmp_path/'go.json'
    payload=dict(status='ROOT_AUTHORIZED_GENUINE_FIRST_COHORTS',routes={'blocks':ROUTES['blocks']},source_root=str(source),git_argv=['synthetic-git','-C',str(source)],source_revision=revision,genuine_root_admission_path=str(admission),genuine_root_admission_sha256=sha(admission),product_kind='GENUINE_INSTALLED_WHOLE_PRODUCT',interpreter=sys.executable,source_files_sha256={},output_root=str(tmp_path/'first-cohort'),installed_site=str(tmp_path/'never-installed'))
    binding.write_text(json.dumps(payload))
    go.write_text(json.dumps(dict(binding_sha256=sha(binding),probe_sha256=sha(PROBE.__file__),routes=['views'],timed_batches=0,maximum_original_topcalls=1,maximum_native_topcalls=1,deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),helper_binding_sha256=sha(helper))))
    def git(argv,**kwargs):
        assert argv[:3]==payload['git_argv']
        if argv[3:]==['rev-parse','HEAD']:return revision+'\n'
        assert argv[3:]==['status','--porcelain'];return ''
    monkeypatch.setattr(PROBE.subprocess,'check_output',git)
    for key,value in env.items():monkeypatch.setenv(key,value)
    for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS'):monkeypatch.setenv(key,'1')
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE','1')
    for key in ('GIT_DIR','GIT_WORK_TREE','UTIL_VIEWS_FENV_HELPER','SKIMAGEXX_DISABLE_SIMD'):monkeypatch.delenv(key,raising=False)
    monkeypatch.setattr(sys,'path',list(sys.path))
    if cached:monkeypatch.setitem(sys.modules,'numpy',SimpleNamespace(synthetic_cached_module=True))
    monkeypatch.setattr(sys,'argv',['probe','--binding',str(binding),'--root-go',str(go),'--route','blocks'])
    if tamper=='helper-digest':helper.write_text('{}')
    elif tamper=='helper-env':monkeypatch.setenv('UTIL_VIEWS_FENV_RECEIPT_SHA256','0'*64)
    elif tamper in ('route','counter'):
        changed=json.loads(go.read_text())
        if tamper=='route':changed['routes']=['blocks']
        else:changed['maximum_original_topcalls']=2
        go.write_text(json.dumps(changed))
    prior=builtins.__import__;requests=[]
    class RefusedBeforeScience(BaseException):pass
    def request(name,*args,**kwargs):
        if name=='numpy':requests.append(name);raise RefusedBeforeScience()
        return prior(name,*args,**kwargs)
    with monkeypatch.context()as patch:
        patch.setattr(builtins,'__import__',request)
        with pytest.raises(RefusedBeforeScience if tamper=='none'else AssertionError):PROBE.main()
    assert requests==(['numpy']if tamper=='none'else [])and builtins.__import__ is prior
    assert not (tmp_path/'first-cohort/blocks/calls.jsonl').exists()
    assert not (tmp_path/'first-cohort/blocks/never-written.json').exists()


@pytest.mark.parametrize('mutation',('none','source','HEAD','dirty','binding','GO','probe','contract','helper-binding','helper-receipt','helper-env','deadline','restoration'))
def test_actual_poststop_terminal_write_path_refuses_mutation(tmp_path,monkeypatch,mutation):
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    root=tmp_path/'source';root.mkdir();source=root/'small.py';source.write_text('# frozen')
    contract=root/'contract.json';contract.write_text('{}')
    bundle=tmp_path/'helper';bundle.mkdir();receipt=bundle/'receipt.json';receipt.write_text('{"synthetic_metadata_only":true}')
    env=dict(UTIL_VIEWS_FENV_MODE='source-built-v1',UTIL_VIEWS_FENV_RECEIPT=str(receipt),UTIL_VIEWS_FENV_RECEIPT_SHA256=sha(receipt))
    helper=tmp_path/'helper-binding.json';helper.write_text(json.dumps(dict(env=env,files={'receipt.json':sha(receipt)})))
    folder=tmp_path/'first-cohort/blocks';folder.mkdir(parents=True)
    binding=tmp_path/'binding.json';binding.write_text('{}');go_path=tmp_path/'go.json';go_path.write_text('{}')
    bp,gp=sha(binding),sha(go_path)
    value=dict(source_root=str(root),git_argv=['fixture-git'],source_revision='a'*40,source_files_sha256={'small.py':sha(source),'contract.json':sha(contract)},output_root=str(folder.parent),routes={'blocks':{'contract':'contract.json'}},base_contract_sha256=sha(contract))
    go=dict(probe_sha256=sha(PROBE.__file__),helper_binding_sha256=sha(helper),deadline_utc='2099-01-01T00:00:00+00:00')
    stopped=dict(route='blocks',counts=dict(original=1,native=1),timed_batches=0,sampler_body_calls=0)
    def git(argv,**kwargs):
        if argv[-2:]==['rev-parse','HEAD']:return ('b'*40 if mutation=='HEAD'else'a'*40)+'\n'
        assert argv[-2:]==['status','--porcelain'];return ' M small.py'if mutation=='dirty'else ''
    monkeypatch.setattr(PROBE.subprocess,'check_output',git)
    for k,v in env.items():monkeypatch.setenv(k,v)
    for k in ('GIT_DIR','GIT_WORK_TREE','UTIL_VIEWS_FENV_HELPER'):monkeypatch.delenv(k,raising=False)
    if mutation=='source':source.write_text('# changed')
    elif mutation=='binding':binding.write_text('{"changed":true}')
    elif mutation=='GO':go_path.write_text('{"changed":true}')
    elif mutation=='probe':go['probe_sha256']='0'*64
    elif mutation=='contract':contract.write_text('{"changed":true}')
    elif mutation=='helper-binding':helper.write_text('{}')
    elif mutation=='helper-receipt':receipt.write_text('{"changed":true}')
    elif mutation=='deadline':go['deadline_utc']='2000-01-01T00:00:00+00:00'
    elif mutation=='helper-env':monkeypatch.setenv('UTIL_VIEWS_FENV_RECEIPT_SHA256','0'*64)
    call=lambda:PROBE.finish_probe(folder,stopped,value,binding,go_path,go,bp,gp,mutation!='restoration')
    if mutation=='none':
        call();terminal=json.loads((folder/'terminal.json').read_text())
        assert terminal['poststop_source_authority_unchanged']is True
        assert terminal['outer_state_restored']is True
        assert not (folder/'failure.json').exists()
    else:
        with pytest.raises(AssertionError):call()
        assert not (folder/'terminal.json').exists()
        failure=json.loads((folder/'failure.json').read_text())
        assert failure['phase']=='poststop-source-authority'and failure['retry']is False
    assert not (folder/'never-written.json').exists()


@pytest.mark.parametrize('fault',('none','verifier','runtime','runtime-during-verifier','manifest'))
def test_actual_batch_summary_requires_one_postbatch_verifier(tmp_path,monkeypatch,fault):
    # Synthetic completed child receipts; neither child probe nor verifier runs.
    controller=tmp_path/'controller';manifest_path=controller/'coverage/contracts/join-util-current275-measurement-v3.json';manifest_path.parent.mkdir(parents=True);manifest_path.write_text('{}')
    stage=tmp_path/'stage';stage.mkdir();runtime=stage/'runtime.json';runtime.write_text('{"synthetic_only":true}')
    results=tmp_path/'results';results.mkdir();manifest=copy.deepcopy(DATA)
    manifest['runtime_sha256']=hashlib.sha256(runtime.read_bytes()).hexdigest()
    manifest['first_cohort']['source_revision']='source'
    authorities={name:dict(binding='binding',go='go')for name in ROUTES};seen=[]
    def completed(command,**kwargs):
        if 'verify'in command:
            seen.append('verify')
            assert not (results/'first-cohort-summary.json').exists()
            if fault=='runtime-during-verifier':runtime.write_text('{"changed":true}')
            return SimpleNamespace(returncode=int(fault=='verifier'))
        cohort=command[-1];seen.append(cohort);folder=results/'first-cohort'/cohort;folder.mkdir(parents=True)
        terminal=dict(status='PASS_REAL_CLI_STOP_BEFORE_BODY',route=cohort,source='source',counts=dict(original=1,native=1),timed_batches=0,sampler_body_calls=0,outer_state_restored=True,poststop_source_authority_unchanged=True)
        (folder/'terminal.json').write_text(json.dumps(terminal))
        if len(seen)==5:
            if fault=='runtime':runtime.write_text('{"changed":true}')
            elif fault=='manifest':manifest_path.write_text('{"changed":true}')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(PROBE.subprocess,'run',completed)
    call=lambda:PROBE.run_cohorts(controller,results,manifest,'python',authorities)
    if fault=='none':
        call();assert seen[-1]=='verify'and seen.count('verify')==1
        summary=json.loads((results/'first-cohort-summary.json').read_text());assert summary['postbatch_offline_identity_verified']is True
    else:
        with pytest.raises(AssertionError):call()
        assert not (results/'first-cohort-summary.json').exists()
        assert seen.count('verify')==(1 if fault in ('verifier','runtime-during-verifier')else 0)
        failure=json.loads((results/'first-cohort-failure.json').read_text())
        assert failure['phase']=='postbatch-identity'and failure['retry']is False
        if fault=='verifier':assert json.loads((results/'first-cohort-identity-after-exit.json').read_text())['exit']==1


def synthetic(tmp_path,kind,observed_offset=None):
    """Two cases exercise both case-index phases; times are invented metadata."""
    snapshot=tmp_path/'snapshot';results=tmp_path/'results';snapshot.mkdir();results.mkdir()
    def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    save(snapshot/'coverage/contracts/next-util-benchmark-execution-v4.json',{'method':{'method_sha256':'synthetic-method'}})
    specs=[{'id':'synthetic-0'},{'id':'synthetic-1'}];contract='synthetic-contract.json';save(snapshot/contract,{'cases':specs})
    route=dict(kind=kind,api='util.crop'if kind=='util-executor'else'draw.ellipse_perimeter',cases=specs,contract=contract)
    manifest=dict(first_cohort={'routes':{'synthetic':route}},benchmark_fingerprint={'source_sha256':'SYNTHETIC_NOT_GENUINE'},core_sha256='SYNTHETIC_NO_CORE',runtime_sha256='SYNTHETIC_NO_RUNTIME',producer={'run':'SYNTHETIC'},execution_scope={'canonical_raw_samples':28})
    binding=tmp_path/'binding.json';measurement=tmp_path/'measurement.json';save(binding,{});save(measurement,{})
    authorities={'synthetic':dict(binding=str(binding),measurement=str(measurement))}
    meta=dict(source_sha256='SYNTHETIC_NOT_GENUINE',core_sha256=manifest['core_sha256'],threads=1,repeat=7,target_ms=100,contract_sha256=TOOL.sha(snapshot/contract),root_binding_sha256=TOOL.sha(binding),measurement_sha256=TOOL.sha(measurement),runtime_sha256=manifest['runtime_sha256'],producer=manifest['producer'],method_sha256='synthetic-method',retain_all_losses=True,raw_samples=28,reference_pool_limits={'synthetic':'1'})
    rows=[];journal=[];correct=0 if kind=='util-executor'else 1;observed=correct if observed_offset is None else observed_offset
    for ci,case in enumerate(specs):
        # Explicit two-arm phase table, independently checked against frozenAST.
        first=[0,1]if(ci+observed)%2==0 else[1,0];second=list(reversed(first));orders=[first,second,first,second,first,second,first]
        row=dict(case_id=case['id'],api=route['api'],parameters=case,native_accepted=True,FP={'flags':0,'rounding':0},samples_ms=[[100.]*7,[200.]*7],loops=[[1]*7,[1]*7],reference_ms=100.,cpp_ms=200.,speedup=.5,sample_labels=['scikit-image','current'],warmup_elapsed_ns=[250000000,250000000],sample_order=orders,batch_records=[])
        def batch(arm,round_id,ordinal,phase,elapsed):
            common=dict(case_index=ci,case_id=case['id'],function=arm,round=round_id,ordinal=ordinal,loops=1)
            before=dict(common,event='batch_before',requested_phase=phase)
            after=dict(common,event='batch_after',phase=phase,elapsed_ns=elapsed,complete=True,accepted=phase=='sample',failure=None)
            journal.extend((before,after));row['batch_records'].append(after)
        for arm in(0,1):batch(arm,-1,arm,'warmup',250000000)
        for rid,order in enumerate(orders):
            for ordinal,arm in enumerate(order):batch(arm,rid,ordinal,'sample',100000000 if arm==0 else 200000000)
        keys=('reference_ms','cpp_ms','speedup','samples_ms','loops','sample_labels','warmup_elapsed_ns','sample_order','batch_records')
        journal.append(dict(event='measure_returned',case_index=ci,case_id=case['id'],timing={k:copy.deepcopy(row[k])for k in keys}))
        rows.append(row)
    save(results/'synthetic.json',dict(metadata=meta,results=rows))
    (results/'synthetic.batches.jsonl').write_text(''.join(json.dumps(x)+'\n'for x in journal))
    return snapshot,results,manifest,authorities


@pytest.mark.parametrize('kind',('util-executor','dedicated'))
def test_self_consistent_wrong_source_order_refused(tmp_path,monkeypatch,kind):
    args=synthetic(tmp_path,kind,1 if kind=='util-executor'else 0);writes=[];monkeypatch.setattr(TOOL,'write_json',lambda *args:writes.append(args))
    with pytest.raises(AssertionError):TOOL.validate_reports(*args)
    assert writes==[]


@pytest.mark.parametrize('fault',('case-index','ordinal','warmup','median','native','missing-event','unknown-kind'))
def test_remaining_strict_guards_not_weakened(tmp_path,monkeypatch,fault):
    snapshot,results,manifest,authority=synthetic(tmp_path,'util-executor');report=json.loads((results/'synthetic.json').read_bytes());events=[json.loads(s)for s in(results/'synthetic.batches.jsonl').read_text().splitlines()]
    if fault=='case-index':events[0]['case_index']=1
    elif fault=='ordinal':events[4]['ordinal']=1
    elif fault=='warmup':report['results'][0]['warmup_elapsed_ns'][0]=0
    elif fault=='median':report['results'][0]['reference_ms']=1.
    elif fault=='native':report['results'][0]['native_accepted']=False
    elif fault=='missing-event':events.pop()
    elif fault=='unknown-kind':manifest['first_cohort']['routes']['synthetic']['kind']='UNREVIEWED'
    (results/'synthetic.json').write_text(json.dumps(report));(results/'synthetic.batches.jsonl').write_text(''.join(json.dumps(x)+'\n'for x in events));writes=[];monkeypatch.setattr(TOOL,'write_json',lambda *args:writes.append(args))
    with pytest.raises((AssertionError,KeyError)):TOOL.validate_reports(snapshot,results,manifest,authority)
    assert writes==[]


def test_five_affected_routes_unbound_and_exact_frozen_methods():
    assert set(ROUTES)=={'join','crop','slice','blocks','windows'}
    assert all(r['kind']=='util-executor'for r in ROUTES.values())
    assert all(DATA[k]is None for k in('prepared_source','SDK_members','core_sha256','runtime_sha256','benchmark_fingerprint','deadline_utc'))
    assert all(v is None for v in DATA['producer'].values())
    assert all(DATA['first_cohort'][k]is None for k in('source_revision','probe_sha256','root_admission_path','root_admission_sha256','root_stager_path','root_stager_sha256','core_member','source_files_raw_sha256'))
    assert all(DATA['helper'][k]is None for k in('receipt_sha256','library_sha256','root_GO'))
    assert(DATA['execution_scope']['canonical_cases'],DATA['execution_scope']['canonical_raw_samples'],DATA['execution_scope']['warmup_groups'])==(32,448,64)
    with pytest.raises(AssertionError,match='UNBOUND'):PROBE.read_manifest(ROOT)
    for name,pin in DATA['benchmark_source_sha256'].items():assert H((ROOT/name).read_bytes().replace(b'\r\n',b'\n'))==pin,name
    for name,pin in DATA['helper']['source_sha256'].items():assert H((ROOT/name).read_bytes())==pin,name
    context=json.loads((ROOT/'coverage/contracts/join-util-current275-controller-v3.json').read_bytes())
    assert DATA['execution_scope']['untimed_topcalls']==dict(original=5,native=5)
    for name,route in ROUTES.items():assert route['cases']==context['prior_routes'][name]['cases']and route['argv']==context['prior_routes'][name]['argv']
    for name,row in context['sources'].items():assert H((ROOT/name).read_bytes())==row['current_raw_sha256'],name


def test_actual_pass0_and_historical_dedicated1_callers_are_source_closed():
    for name,expected in [('next_util_execution',0),('label_pair_metrics',1),('find_boundaries',1),('ellipse_perimeter',1)]:
        tree=ast.parse((ROOT/f'benchmarks/{name}.py').read_bytes());calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='measure']
        assert len(calls)==1;kw={k.arg:k.value for k in calls[0].keywords}
        assert isinstance(kw['pass_index'],ast.Constant)and kw['pass_index'].value==expected
        assert ast.unparse(kw['case_index'])=='index'
    assert all(r['kind']=='util-executor'for r in ROUTES.values())  # No dedicated cohort authorized.


def test_workflow_inline_python_and_only_five_cohorts():
    text=(ROOT/'.github/workflows/join-util-current275-initial.yml').read_text()
    assert text.count('coverage/contracts/join-util-current275-measurement-v3.json')==2
    lines=text.splitlines();blocks=[];current=None
    for line in lines:
        if line.strip().endswith("<<'PY'"):current=[]
        elif current is not None and line.strip()=='PY':blocks.append('\n'.join(x[10:]for x in current));current=None
        elif current is not None:current.append(line)
    assert len(blocks)>=5
    for body in blocks:compile(body,'<actual workflow inline Python>','exec')


@pytest.mark.parametrize('kind',('util-executor','dedicated'))
def test_corrected_actual_report_reader_accepts_both_historical_offsets_and_losses(tmp_path,monkeypatch,kind):
    args=synthetic(tmp_path,kind);writes=[];monkeypatch.setattr(PROBE,'write_json',lambda *a:writes.append(a))
    PROBE.validate_reports(*args)
    assert len(writes)==1 and writes[0][1]['families']['synthetic']['losses']==['synthetic-0','synthetic-1']
    assert not(args[1]/'summary.json').exists()


def synthetic_bound_manifest(tmp_path):
    # A metadata-only root-receipt fixture; these values are not genuine evidence.
    m=copy.deepcopy(DATA);revision='a'*40
    m.update(status='ROOT_AUTHORIZED_ONE_INITIAL_COMPARISON',producer=dict(run='123',revision=revision,artifact='456',sha256='b'*64),prepared_source=dict(revision=revision,files_sha256=dict(DATA['benchmark_source_sha256'])),SDK_members=105,core_sha256='d'*64,benchmark_fingerprint={'source_sha256':'e'*64},runtime_sha256='f'*64)
    stager=dict(revision=revision,run=123,artifact=456,artifact_sha256='b'*64,profile='minimum',status='AUTHENTICATED_STAGED_UNACCEPTED_SYNTHETIC')
    (tmp_path/'stager.json').write_text(json.dumps(stager));stager_sha=PROBE.sha(tmp_path/'stager.json')
    core=dict(core_sha256='d'*64,sole_core='skimagexx/_core.synthetic.so',source_files=len(m['prepared_source']['files_sha256']))
    admission=dict(passed=True,root_accepted=True,profiles=4,product_kind='GENUINE_INSTALLED_WHOLE_PRODUCT',source=revision,run=123,SDK_members=105,production_source_sha256='e'*64,stagers={'minimum':{'receipt_sha256':stager_sha}},qualified_profiles={'minimum':{'core':core}})
    (tmp_path/'admission.json').write_text(json.dumps(admission))
    probe=tmp_path/DATA['first_cohort']['probe_path'];probe.parent.mkdir(parents=True);probe.write_bytes((ROOT/DATA['first_cohort']['probe_path']).read_bytes())
    m['first_cohort'].update(status='ROOT_AUTHORIZED_GENUINE_FIRST_COHORTS',source_revision=revision,root_admission_path='admission.json',root_admission_sha256=PROBE.sha(tmp_path/'admission.json'),root_stager_path='stager.json',root_stager_sha256=stager_sha,probe_sha256=PROBE.sha(probe),core_member=core['sole_core'])
    return m


@pytest.mark.parametrize('tamper',('none','source','prepared','run','artifact','archive','admission-digest','stager-digest','core','core-member','SDK','fingerprint','source-pin','source-count','probe','routes','case-budget','sample-budget','worker','losses','helper-receipt','helper-library','helper-GO','runtime-format'))
def test_actual_bound_metadata_refuses_changed_authority_before_any_process(tmp_path,monkeypatch,tamper):
    m=synthetic_bound_manifest(tmp_path);first=m['first_cohort']
    if tamper=='source':m['producer']['revision']='0'*40
    elif tamper=='prepared':m['prepared_source']['revision']='0'*40
    elif tamper=='run':m['producer']['run']='1'
    elif tamper=='artifact':m['producer']['artifact']='1'
    elif tamper=='archive':m['producer']['sha256']='0'*64
    elif tamper=='admission-digest':first['root_admission_sha256']='0'*64
    elif tamper=='stager-digest':first['root_stager_sha256']='0'*64
    elif tamper=='core':m['core_sha256']='0'*64
    elif tamper=='core-member':first['core_member']='wrong.so'
    elif tamper=='SDK':m['SDK_members']=102
    elif tamper=='fingerprint':m['benchmark_fingerprint']['source_sha256']='0'*64
    elif tamper=='source-pin':m['prepared_source']['files_sha256']['CMakeLists.txt']='0'*64
    elif tamper=='source-count':m['prepared_source']['files_sha256'].pop('CMakeLists.txt')
    elif tamper=='probe':first['probe_sha256']='0'*64
    elif tamper=='routes':first['routes'].pop('blocks')
    elif tamper=='case-budget':m['execution_scope']['canonical_cases']=40
    elif tamper=='sample-budget':m['execution_scope']['canonical_raw_samples']=1
    elif tamper=='worker':m['execution_scope']['workers']=4
    elif tamper=='losses':m['execution_scope']['retain_all_losses']=False
    elif tamper=='helper-receipt':m['helper']['receipt_sha256']='0'*64
    elif tamper=='helper-library':m['helper']['library_sha256']='0'*64
    elif tamper=='helper-GO':m['helper']['root_GO']={}
    elif tamper=='runtime-format':m['runtime_sha256']='unknown'
    output=tmp_path/'coverage/contracts/join-util-current275-measurement-v3.json';output.parent.mkdir(parents=True);output.write_text(json.dumps(m));calls=[]
    def forbidden(*a,**kw):calls.append(a);raise AssertionError('BODY_OR_PROCESS_REACHED')
    monkeypatch.setattr(PROBE.subprocess,'run',forbidden)
    if tamper=='none':assert PROBE.read_manifest(tmp_path)[1]==m
    else:
        with pytest.raises(AssertionError):PROBE.read_manifest(tmp_path)
    assert not calls


def test_current280_projection_preserves_exact_affected_subset_and_predecessors():
    assert DATA['prospective_source']['revision']=='ab7872a5483a5d5e3660ba20fdd873d78463a1b5'
    assert tuple(DATA['prospective_source'][k]for k in('image_count','functions','product_members','SDK_members','SDK_families','SDK_checks'))==(280,283,226,107,37,74)
    pred=DATA['predecessor_controller'];raw=(ROOT/pred['contract_path']).read_bytes();assert H(raw)==pred['contract_raw_sha256'];old=json.loads(raw)
    assert DATA['execution_scope']==old['execution_scope']and DATA['helper']==old['helper']
    ctx=json.loads((ROOT/'coverage/contracts/next-util-benchmark-execution-v4.json').read_bytes())
    assert set(ROUTES)=={'join','crop','slice','blocks','windows'}
    for key,r in ROUTES.items():
        assert r['cases']==old['first_cohort']['routes'][key]['cases']and r['argv']==old['first_cohort']['routes'][key]['argv']
        assert r['cases']==[v for v in ctx['routes'][r['execution_route']]['cases']if v.get('api',r['api'])==r['api']]
    for name,row in DATA['controller_inverse'].items():
        raw=(ROOT/name).read_bytes();assert H(raw)==row['current_raw_sha256']and raw.decode()==row['current']
        assert H(row['predecessor'].encode())==row['predecessor_raw_sha256']
