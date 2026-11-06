"""Source-only controller interface/authority controls: no targets/helper/sampler."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/next-metrics-measurement-v2.json'
DATA=json.loads(CONTRACT.read_bytes())
spec=importlib.util.spec_from_file_location('controller_v2',ROOT/'tools/next_metrics_first_cohort_probe.py')
PROBE=importlib.util.module_from_spec(spec);spec.loader.exec_module(PROBE)
ROUTES=DATA['first_cohort']['routes']


def test_unbound_actual_authority_and_fixed_proposal(tmp_path):
    assert DATA['schema']=='next-metrics-measurement-v1'
    assert DATA['status']=='PROSPECTIVE_UNBOUND_NO_MEASUREMENT'
    assert set(DATA['producer'].values())=={None}
    assert all(DATA[k]is None for k in ('prepared_source','SDK_members','core_sha256','runtime_sha256','benchmark_fingerprint'))
    first=DATA['first_cohort']
    assert all(first[k]is None for k in ('source_revision','probe_sha256','root_admission_path','root_admission_sha256','root_stager_path','root_stager_sha256','core_member'))
    assert all(DATA['helper'][k]is None for k in ('receipt_sha256','library_sha256','root_GO'))
    assert len(ROUTES)==9 and len({r['api']for r in ROUTES.values()})==9
    assert sum(v['cases']for v in DATA['families'].values())==56
    scope=DATA['execution_scope'];assert(scope['canonical_cases'],scope['canonical_raw_samples'],scope['warmup_groups'])==(56,784,112)
    with pytest.raises(AssertionError,match='UNBOUND'):PROBE.read_manifest(ROOT)


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
    util=json.loads((ROOT/'coverage/contracts/next-util-benchmark-execution-v2.json').read_bytes())
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


@pytest.mark.parametrize('failure',('case','api','count','acceptance','native-marker','FP','full32','input','alias','timing','getter','caller'))
def test_real_execute_frame_adapter_refuses_changed_scope(failure):
    route,frame,kwargs=frame_fixture('blocks');caller=frame.f_back.f_locals
    if failure=='case':caller['spec']={'id':'different'}
    elif failure=='api':caller['api']='util.view_as_windows'
    elif failure=='count':kwargs['counts']['original']=2
    elif failure=='acceptance':caller['accepted']=[False]
    elif failure=='native-marker':caller['returns']=[]
    elif failure=='FP':caller['native_FP']['flags']=0
    elif failure=='full32':caller['save']=lambda:bytes([1])+bytes(31)
    elif failure=='input':kwargs['inputs_before']=[]
    elif failure=='alias':caller['actual_state']={'alias':'changed'}
    elif failure=='timing':caller['cli'].first_cohort=False
    elif failure=='getter':kwargs['getters']=['unexpected-worker-lookup']
    elif failure=='caller':frame.f_back.f_code.co_filename='wrong.py'
    with pytest.raises(AssertionError):PROBE.validate_util_entry(frame,route,ROOT,**kwargs)


def test_actual_root_refusal_stops_before_science(monkeypatch,tmp_path):
    binding=tmp_path/'binding.json';binding.write_text(json.dumps({'status':'UNBOUND'}))
    monkeypatch.setattr(PROBE.sys,'argv',['probe','--binding',str(binding),'--root-go',str(tmp_path/'never.json'),'--route','blocks'])
    with pytest.raises(AssertionError,match='UNBOUND'):PROBE.main()
    assert list(tmp_path.iterdir())==[binding]


def test_measurement_commands_cannot_replay_first_cohort_or_pass_deadline_to_util(tmp_path):
    authorities={name:{'binding':name+'.binding','measurement':name+'.measurement'}for name in ROUTES}
    commands=PROBE.measurement_commands(ROOT,tmp_path,DATA,'python',authorities,'UTC')
    assert len(commands)==8 and 'metrics-contingency'not in commands
    for name,argv in commands.items():
        assert '--first-cohort'not in argv and '--case'not in argv
        if ROUTES[name]['kind']=='util-executor':
            assert '--deadline-utc'not in argv and '--measurement-go'in argv
            assert argv[argv.index('--api')+1]==ROUTES[name]['api']
        else:assert '--deadline-utc'in argv


def test_source_identity_helper_closure_and_no_hidden_sampler():
    for name,pin in DATA['benchmark_source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin,name
    for name,pin in DATA['first_cohort']['source_files_raw_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==pin,name
    for name,pin in DATA['helper']['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==pin,name
    tree=ast.parse((ROOT/'tools/next_metrics_first_cohort_probe.py').read_text());functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    prepare=ast.unparse(functions['prepare_host_helper']);assert 'util_views_fenv_build.py'in prepare and '--control-python'in prepare and 'rglob'in prepare
    assert "bundle.exists()"in prepare and "done.returncode == 0"in prepare
    main=ast.unparse(functions['main']);assert 'helper_binding_sha256'in ast.unparse(functions['verify_helper_state'])and 'StopBeforeBody'in main
    assert main.index('finish_probe(')>main.index('xx.set_fallback_enabled(fallback)')
    execute=ast.parse((ROOT/'benchmarks/next_util_execution.py').read_text())
    assert 'first_cohort_profile' in {n.name for n in execute.body if isinstance(n,ast.FunctionDef)}
    workflow=(ROOT/'.github/workflows/next-metrics-initial.yml').read_text()
    assert workflow.count("probe['prepare_host_helper']")==1
    assert workflow.index("probe['prepare_host_helper']")<workflow.index("probe['run_cohorts']")<workflow.index("probe['run_measurements']")
    assert 'path: results/'in workflow and 'if: always()'in workflow
    assert 'next-metrics-measurement-v1.json'not in workflow
    assert '--first-cohort'not in ast.unparse(functions['run_measurements'])


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
    controller=tmp_path/'controller';manifest_path=controller/'coverage/contracts/next-metrics-measurement-v2.json';manifest_path.parent.mkdir(parents=True);manifest_path.write_text('{}')
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
        if len(seen)==9:
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
