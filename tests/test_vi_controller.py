"""Source/setup-only one-API authority, real CLI/frame and C1 terminal controls."""
import ast,copy,hashlib,importlib.util,json,runpy,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/vi-measurement-v3.json'
DATA=json.loads(CONTRACT.read_bytes());ROUTES=DATA['first_cohort']['routes']
HISTORICAL_PATH='coverage/contracts/vi-measurement-v2.json'
HISTORICAL=json.loads((ROOT/HISTORICAL_PATH).read_bytes())
spec=importlib.util.spec_from_file_location('vi_outer',ROOT/'tools/vi_first_cohort_probe.py');PROBE=importlib.util.module_from_spec(spec);spec.loader.exec_module(PROBE)
H=lambda b:hashlib.sha256(b).hexdigest()

def synthetic(tmp_path,monkeypatch):
 from test_variation_of_information_benchmark_execution import synthetic as source_synthetic
 root,path,b,c=source_synthetic(tmp_path,monkeypatch);root=Path(b['source_root'])
 for name in DATA['first_cohort']['source_files_raw_sha256']:
  p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes());b['prepared_source']['files_sha256'][name]=H(p.read_bytes().replace(b'\r\n',b'\n'))
 runtime=Path(b['runtime_path']);r=json.loads(runtime.read_bytes());r['source']=b['prepared_source'];runtime.write_text(json.dumps(r));b['runtime_sha256']=PROBE.sha(runtime);path.write_text(json.dumps(b))
 go_path=Path(b['root_go_path']);go=json.loads(go_path.read_bytes());go.update(binding_sha256=PROBE.sha(path),probe_sha256=PROBE.sha(PROBE.__file__));go_path.write_text(json.dumps(go))
 return path,b,c


def test_current281_bound_source_and_historical_scope():
    assert DATA['status']=='ROOT_AUTHORIZED_ONE_INITIAL_COMPARISON'
    assert DATA['producer']==dict(run='42347579925',revision='dda9cc83d9434a10616f84cf1e1125d93f5f36fd',artifact='13565257857',sha256='5868729f1d010e56d66a23e50880a5f32d3a5c11b3d0b1ae4620d5b7394116c1')
    assert DATA['prepared_source']['revision']==DATA['producer']['revision']and len(DATA['prepared_source']['files_sha256'])==2954
    assert DATA['SDK_members']==108 and len(DATA['bound_SDK_sha256'])==108
    assert DATA['core_sha256']=='6a4a4a2630b6bf2947ccefd594ae3ac1958980f5087a61362cbdbbbebb25f1ae'
    assert DATA['runtime_sha256']=='d8a5ed760956f08613e56643df8a55fc874fe2f2c436d8f60e1e16fc2803d1f4'
    assert DATA['deadline_utc']is DATA['execution_created_utc']is None
    assert DATA['execution_clock']=='ROOT_DISPATCH_INPUT_BOUND'
    assert all(DATA['helper'][k]is None for k in('receipt_sha256','library_sha256','root_GO'))
    assert all(DATA['execution_authority'][k]is None for k in('controller_root_admission','created_utc','deadline_utc','cohort_GO','measurement_GO'))
    assert DATA['execution_scope']==HISTORICAL['execution_scope']
    for name,route in ROUTES.items():
        old=HISTORICAL['first_cohort']['routes'][name]
        for key in('cases','argv','measurement_argv','api'):
            if key in route:assert route[key]==old[key]
    assert PROBE.read_manifest(ROOT)[1]==DATA
    first=DATA['first_cohort']
    assert first['source_revision']==DATA['producer']['revision']and first['probe_sha256']==H((ROOT/first['probe_path']).read_bytes())
    assert first['root_admission_sha256']=='3724197bc0ce1f7ececdf39b7ad3e80f29c80477bb8067793fa54a5210c2ec8b'
    assert first['root_stager_sha256']=='9dd465cdf6888407dd18157cc5ea79457ceb4cc5b1885f24226127dbbcf7c860'
    assert H((ROOT/first['root_admission_path']).read_bytes())==first['root_admission_sha256']
    assert H((ROOT/first['root_stager_path']).read_bytes())==first['root_stager_sha256']
    for name,pin in DATA['benchmark_source_sha256'].items():
        assert H((ROOT/name).read_bytes().replace(b'\r\n',b'\n'))==pin==DATA['prepared_source']['files_sha256'][name]
    for name,pin in first['source_files_raw_sha256'].items():assert H((ROOT/name).read_bytes())==pin
    for name,pin in DATA['helper']['source_sha256'].items():assert H((ROOT/name).read_bytes())==pin


def frame_fixture(name):
 route=ROUTES[name];cases=runpy.run_path(str(ROOT/'benchmarks/variation_of_information_cases_v1.py'));requirements=cases['native_requirements'](route['first_case']);cli=SimpleNamespace(first_cohort=True,measurement_go=None)
 state=lambda value,np,input=None:('state',value)
 caller=dict(cli=cli,spec=copy.deepcopy(route['first_case']),api=route['api'],context={'route':route['driver_route'],'cases':route['cases']},accepted=[True],suffix=[True],rows=[],returns=[k for k,v in requirements.items()for _ in range(v)],requirements=requirements,expected='result',actual='result',reference_FP={'control':895,'status':32,'mxcsr':8096},native_FP={'control':895,'status':32,'mxcsr':8096},reference_before={'control':895,'status':0,'mxcsr':8096},native_before={'control':895,'status':0,'mxcsr':8096},save=lambda:bytes(32),caller=bytes(32),sample_state=bytes(32),args=('truth','test'),options={'ignore_labels':[0]},before=[state(v,None)for v in('truth','test')],np=None,index=0,common=SimpleNamespace(array_state=state),module={'native_requirements':cases['native_requirements']})
 execute=SimpleNamespace(f_code=SimpleNamespace(co_filename=str(ROOT/'benchmarks/variation_of_information_execution.py'),co_name='execute'),f_locals=caller,f_globals={})
 bridge=SimpleNamespace(f_code=SimpleNamespace(co_filename=str(ROOT/'benchmarks/mark_boundaries_execution.py'),co_name='sample_entry'),f_back=execute,f_locals=dict(cli=cli,spec=caller['spec'],index=0))
 frame=SimpleNamespace(f_code=SimpleNamespace(co_filename=str(ROOT/'tools/montage_sustained.py'),co_name='measure'),f_back=bridge,f_locals={})
 return route,frame,dict(counts=dict(original=1,native=1),accepted=[True],input_state=lambda v:v,inputs_before=[(caller['args'],caller['options'])])


@pytest.mark.parametrize('name',ROUTES)
@pytest.mark.parametrize('fault',('none','case','api','contract','bridge','execute','sampler','count','acceptance','suffix','marker','FP','before-FP','full32','input','alias','body'))
def test_real_frame_adapter_exact_roles_and_refusals(name,fault):
 route,frame,k=frame_fixture(name);c=frame.f_back.f_back.f_locals
 if fault=='case':c['spec']={'id':'changed'}
 elif fault=='api':c['api']='foreign'
 elif fault=='contract':c['context']['route']={}
 elif fault in('bridge','execute','sampler'):
  target={'bridge':frame.f_back,'execute':frame.f_back.f_back,'sampler':frame}[fault];target.f_code.co_filename='foreign.py'
 elif fault=='count':k['counts']['original']=2
 elif fault=='acceptance':c['accepted']=[]
 elif fault=='suffix':c['suffix']=[]
 elif fault=='marker':c['returns'].append('foreign')
 elif fault=='FP':c['native_FP']={'status':0}
 elif fault=='before-FP':c['native_before']={'status':32}
 elif fault=='full32':c['sample_state']=bytes([1])*32
 elif fault=='input':k['inputs_before']=[]
 elif fault=='alias':c['actual']='foreign-result'
 elif fault=='body':frame.f_locals['samples']=[]
 if fault=='none':assert PROBE.validate_native_entry(frame,route,ROOT,**k)is c
 else:
  with pytest.raises(AssertionError):PROBE.validate_native_entry(frame,route,ROOT,**k)


@pytest.mark.parametrize('name',ROUTES)
@pytest.mark.parametrize('cached',(False,True))
@pytest.mark.parametrize('tamper',('none','route','counter','helper-digest'))
def test_actual_bound_probe_reaches_import_request_refusal(monkeypatch,tmp_path,name,cached,tamper):
 import builtins
 path,b,c=synthetic(tmp_path,monkeypatch);go_path=Path(b['root_go_path']);go=json.loads(go_path.read_bytes())
 manifest=copy.deepcopy(DATA);manifest.update(status='ROOT_AUTHORIZED_ONE_INITIAL_COMPARISON',producer=b['producer'],prepared_source=b['prepared_source'],runtime_sha256=b['runtime_sha256'],core_sha256=b['core_sha256'],SDK_members=len(b['SDK_members']))
 manifest['first_cohort'].update(status='ROOT_AUTHORIZED_GENUINE_FIRST_COHORTS',source_revision=b['source_revision'],probe_sha256=PROBE.sha(PROBE.__file__))
 monkeypatch.setattr(PROBE,'read_manifest',lambda root:(CONTRACT,manifest));monkeypatch.setattr(sys,'argv',['probe','--binding',str(path),'--root-go',str(go_path),'--route',name]);monkeypatch.setattr(sys,'path',list(sys.path))
 if tamper=='route':go['routes']=['foreign'];go_path.write_text(json.dumps(go))
 elif tamper=='counter':go['maximum_original_topcalls']=2;go_path.write_text(json.dumps(go))
 elif tamper=='helper-digest':Path(b['helper_binding_path']).write_text('{}')
 if cached:monkeypatch.setitem(sys.modules,'numpy',SimpleNamespace(synthetic_cached=True))
 prior=builtins.__import__;requests=[]
 class Refused(BaseException):pass
 def request(n,*a,**k):
  if n.split('.')[0]in('numpy','skimage','skimagexx','scipy'):requests.append(n);raise Refused()
  return prior(n,*a,**k)
 with monkeypatch.context()as patch:
  patch.setattr(builtins,'__import__',request)
  with pytest.raises(Refused if tamper=='none'else AssertionError):PROBE.main()
 assert requests==(['numpy']if tamper=='none'else[])and builtins.__import__ is prior
 folder=Path(b['output_root'])/name;assert not(folder/'calls.jsonl').exists()and not(folder/'terminal.json').exists()


@pytest.mark.parametrize('mutation',('none','source','HEAD','dirty','binding','GO','probe','contract','helper-binding','helper-receipt','helper-env','deadline','restoration'))
def test_actual_poststop_terminal_write_path_refuses_mutation(tmp_path,monkeypatch,mutation):
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    root=tmp_path/'source';root.mkdir();source=root/'small.py';source.write_text('# frozen')
    contract=root/PROBE.DRIVER_CONTEXT;contract.parent.mkdir(parents=True);contract.write_text('{}')
    bundle=tmp_path/'helper';bundle.mkdir();receipt=bundle/'receipt.json';receipt.write_text('{"synthetic_metadata_only":true}')
    env=dict(UTIL_VIEWS_FENV_MODE='source-built-v1',UTIL_VIEWS_FENV_RECEIPT=str(receipt),UTIL_VIEWS_FENV_RECEIPT_SHA256=sha(receipt))
    helper=tmp_path/'helper-binding.json';helper.write_text(json.dumps(dict(env=env,files={'receipt.json':sha(receipt)})))
    folder=tmp_path/'first-cohort/vi';folder.mkdir(parents=True)
    binding=tmp_path/'binding.json';binding.write_text('{}');go_path=tmp_path/'go.json';go_path.write_text('{}')
    bp,gp=sha(binding),sha(go_path)
    value=dict(SDK_members=dict.fromkeys(range(2),'inert'),prepared_source={'revision':'a'*40},producer={'revision':'a'*40},runtime_sha256='runtime',core_sha256='core',helper_binding_path=str(helper),helper_binding_sha256=sha(helper),source_root=str(root),git_argv=['fixture-git'],source_revision='a'*40,output_root=str(folder.parent),routes={'vi':{'legacy':'inert'}},base_contract_sha256=sha(contract))
    go=dict(probe_sha256=sha(PROBE.__file__),helper_binding_sha256=sha(helper),deadline_utc='2099-01-01T00:00:00+00:00')
    stopped=dict(route='vi',counts=dict(original=1,native=1),timed_batches=0,sampler_body_calls=0)
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
    manifest=dict(prepared_source=value['prepared_source'],producer=value['producer'],runtime_sha256='runtime',core_sha256='core',SDK_members=2,first_cohort=dict(source_revision='a'*40,source_files_raw_sha256={'small.py':sha(source),PROBE.DRIVER_CONTEXT:value['base_contract_sha256']},routes={'vi':{'driver_route':value['routes']['vi']}}))
    if mutation=='source':manifest['first_cohort']['source_files_raw_sha256']['small.py']=H(b'# frozen')
    call=lambda:PROBE.finish_probe(folder,stopped,value,binding,go_path,go,bp,gp,mutation!='restoration',manifest)
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
    controller=tmp_path/'controller';manifest_path=controller/'coverage/contracts/vi-measurement-v3.json';manifest_path.parent.mkdir(parents=True);manifest_path.write_text('{}')
    stage=tmp_path/'stage';stage.mkdir();runtime=stage/'runtime.json';runtime.write_text('{"synthetic_only":true}')
    results=tmp_path/'results';results.mkdir();manifest=copy.deepcopy(DATA)
    manifest['runtime_sha256']=hashlib.sha256(runtime.read_bytes()).hexdigest()
    manifest['first_cohort']['source_revision']='source'
    authorities={name:dict(binding=str(tmp_path/(name+'-binding.json')),go='go')for name in ROUTES};seen=[]
    for row in authorities.values():Path(row['binding']).write_text('{}')
    def completed(command,**kwargs):
        if 'verify'in command:
            seen.append('verify')
            assert not (results/'first-cohort-summary.json').exists()
            if fault=='runtime-during-verifier':runtime.write_text('{"changed":true}')
            return SimpleNamespace(returncode=int(fault=='verifier'))
        cohort=command[-1];seen.append(cohort);folder=results/'first-cohort'/cohort;folder.mkdir(parents=True)
        terminal=dict(root_binding_sha256=H(Path(authorities[cohort]['binding']).read_bytes()),status='PASS_REAL_CLI_STOP_BEFORE_BODY',route=cohort,source='source',counts=dict(original=1,native=1),timed_batches=0,sampler_body_calls=0,outer_state_restored=True,poststop_source_authority_unchanged=True)
        (folder/'terminal.json').write_text(json.dumps(terminal))
        if len(seen)==1:
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


@pytest.mark.parametrize('fault',('none','status','cases','workers','losses'))
def test_root_parent_measurement_go_is_separate_and_fixed(tmp_path,fault):
    manifest=copy.deepcopy(DATA);manifest['status']='ROOT_AUTHORIZED_ONE_INITIAL_COMPARISON'
    route=ROUTES['vi'];binding=tmp_path/'binding.json';binding.write_text('{}');terminal=tmp_path/'terminal.json'
    done=dict(status='PASS_REAL_CLI_STOP_BEFORE_BODY',route='vi',api=route['api'],root_binding_sha256=PROBE.sha(binding),counts=dict(original=1,native=1),timed_batches=0,sampler_body_calls=0,outer_state_restored=True,poststop_source_authority_unchanged=True)
    terminal.write_text(json.dumps(done));summary=dict(status='PASS',postbatch_offline_identity_verified=True,routes=[done])
    if fault=='status':manifest['status']='PROSPECTIVE_UNBOUND_NO_MEASUREMENT'
    elif fault=='cases':manifest['families']['variation_of_information_consumer']['case_ids']=[]
    elif fault=='workers':summary['postbatch_offline_identity_verified']=False
    elif fault=='losses':summary['routes']=[]
    call=lambda:PROBE.scoped_measurement(manifest,route,'parent-sha',{'binding':str(binding)},terminal,summary)
    if fault=='none':
        actual=call();assert actual['execution_scope']==DATA['execution_scope']and actual['families']==DATA['families']
        assert actual['completed_first_cohort']==dict(path=str(terminal.resolve()),sha256=PROBE.sha(terminal),binding_sha256=PROBE.sha(binding))
    else:
        with pytest.raises(AssertionError):call()


def test_actual_workflow_and_restore_order_are_source_closed():
    text=(ROOT/'.github/workflows/vi-initial.yml').read_text();assert '== 104'not in text and "frozen['SDK_members']>0"in text
    assert text.count("probe['prepare_host_helper']")==1
    assert text.index("probe['prepare_host_helper']")<text.index("probe['run_cohorts']")<text.index("probe['run_measurements']")
    assert 'path: results/'in text and 'if: always()'in text
    assert 'next-metrics-measurement-v2.json'not in text
    tree=ast.parse((ROOT/'tools/vi_first_cohort_probe.py').read_bytes());functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    main=ast.unparse(functions['main']);assert main.index('shared.validate_binding')<main.index('import numpy as np')
    assert main.index('adapter.require_sdk_dependencies')<main.index('import numpy as np')
    assert main.index('finish_probe(')>main.index('restore(outer)')
    assert 'validate_native_entry'in main and 'StopBeforeBody'in main
    prepare=ast.unparse(functions['prepare_host_helper']);assert 'util_views_fenv_build.py'in prepare and 'rglob'in prepare
    assert 'bundle.exists()'in prepare and '--control-python'in prepare
    assert PROBE.sampler_order_scope(ROOT)==0
    assert '--first-cohort'not in ast.unparse(functions['measurement_commands'])
    # Compile the actual workflow's inline Python without executing any block.
    lines=text.splitlines();blocks=[];block=None
    for line in lines:
        if line.strip().endswith("<<'PY'"):block=[]
        elif block is not None:
            if line.strip()=='PY':blocks.append('\n'.join(block)+'\n');block=None
            else:block.append(line[10:])
    assert len(blocks)==5
    for body in blocks:compile(body,'<actual workflow Python>','exec')


def synthetic_reports(tmp_path):
 import math
 results=tmp_path/'results';results.mkdir();manifest=copy.deepcopy(DATA);manifest.update(benchmark_fingerprint={'synthetic':'not-genuine'},core_sha256='core',runtime_sha256='runtime',producer={'revision':'a'*40});authorities={};cases=runpy.run_path(str(ROOT/'benchmarks/variation_of_information_cases_v1.py'))
 for cohort,route in ROUTES.items():
  binding=tmp_path/(cohort+'-binding.json');SDK={'synthetic':'metadata'};binding.write_text(json.dumps({'SDK_members':SDK,'interpreter':'python'}));measurement=tmp_path/(cohort+'-measurement.json');measurement.write_text('{}');authorities[cohort]=dict(binding=str(binding),measurement=str(measurement));context=json.loads((ROOT/route['contract']).read_bytes());rows=[];journal=[]
  for i,spec in enumerate(route['cases']):
   events=[];loops=[1,1];samples=[[],[]];counts=[[],[]]
   def batch(arm,round_id,ordinal,elapsed,phase):
    n=loops[arm];common=dict(case_index=i,case_id=spec['id'],function=arm,round=round_id,ordinal=ordinal,loops=n);before=dict(common,event='batch_before',requested_phase='warmup'if phase=='warmup'else'sample');after=dict(common,event='batch_after',elapsed_ns=elapsed,phase=phase,accepted=phase=='sample',complete=True,failure=None);journal.extend([before,after]);events.append(after);loops[arm]=max(1,math.ceil(n*100000000/elapsed*1.15))
    if phase=='sample':samples[arm].append(elapsed/n/1e6);counts[arm].append(n)
   for arm in(0,1):batch(arm,-1,arm,250000000,'warmup')
   order=[[0,1]if(r+i)%2==0 else[1,0]for r in range(7)]
   for round_id,arms in enumerate(order):
    for ordinal,arm in enumerate(arms):batch(arm,round_id,ordinal,loops[arm]*(100000000 if arm==0 else 200000000),'sample')
   timing=dict(reference_ms=100.,cpp_ms=200.,speedup=.5,samples_ms=samples,loops=counts,sample_labels=['scikit-image','current'],warmup_elapsed_ns=[250000000]*2,sample_order=order,batch_records=events);requirements=cases['native_requirements'](spec)
   row=dict(case_id=spec['id'],api=route['api'],accepted=[True],counts_suffix_accepted=[True],actual_C_returns=[k for k,v in requirements.items()for _ in range(v)],FP=dict(control=895,status=32,mxcsr=8096),providers=context['providers'],**timing);rows.append(row);journal.append(dict(event='measure_returned',case_index=i,case_id=spec['id'],timing=timing))
  meta=dict(source_before=manifest['benchmark_fingerprint'],core_sha256='core',runtime_sha256='runtime',SDK_members=SDK,threads=1,repeat=7,target_ms=100,warmup_min_ms=250,pass_index=0,contract_sha256=PROBE.sha(ROOT/route['contract']),root_binding_sha256=PROBE.sha(binding),measurement_sha256=PROBE.sha(measurement),producer=manifest['producer'],source_revision='a'*40,method_sha256=context['method']['method_sha256'],retain_all_losses=True)
  (results/(cohort+'-command.json')).write_text(json.dumps(PROBE.measurement_commands(ROOT,results,manifest,'python',authorities,manifest['deadline_utc'])[cohort]));(results/(cohort+'.json')).write_text(json.dumps(dict(metadata=meta,results=rows)));(results/(cohort+'.batches.jsonl')).write_text(''.join(json.dumps(row)+'\n'for row in journal))
 return results,manifest,authorities


@pytest.mark.parametrize('fault',('none','old-pass1-order','drop-loss','marker','FP','warmup','median','binding','method','suffix','SDK-map','command','adaptive-loop'))
def test_actual_report_validator_uses_admitted_pass0_and_retains_losses(tmp_path,fault):
    results,manifest,authorities=synthetic_reports(tmp_path);path=results/'vi.json';report=json.loads(path.read_bytes());row=report['results'][0]
    if fault=='old-pass1-order':row['sample_order']=[[1,0]if i%2==0 else[0,1]for i in range(7)]
    elif fault=='drop-loss':report['results'].pop()
    elif fault=='marker':row['actual_C_returns'].append('foreign')
    elif fault=='FP':row['FP']['normalized-away']=True
    elif fault=='warmup':row['warmup_elapsed_ns']=[1,1]
    elif fault=='median':row['reference_ms']=0
    elif fault=='binding':report['metadata']['root_binding_sha256']='changed'
    elif fault=='method':report['metadata']['method_sha256']='changed'
    elif fault=='suffix':row['counts_suffix_accepted']=[]
    elif fault=='SDK-map':report['metadata']['SDK_members']={}
    elif fault=='command':
        command=results/'vi-command.json';argv=json.loads(command.read_bytes());argv[argv.index('--api')+1]='foreign';command.write_text(json.dumps(argv))
    elif fault=='adaptive-loop':row['loops'][0][0]+=1
    path.write_text(json.dumps(report));call=lambda:PROBE.validate_reports(ROOT,results,manifest,authorities)
    if fault=='none':
        call();summary=json.loads((results/'summary.json').read_bytes())
        assert len(summary['families']['vi']['losses'])==4
    else:
        with pytest.raises(AssertionError):call()
        assert not(results/'summary.json').exists()


def test_actual_current_driver_metadata_and_Montage_pass0_interfaces():
 source=(ROOT/'benchmarks/variation_of_information_execution.py').read_text();tree=ast.parse(source);execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute');report=next(n for n in ast.walk(execute)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='dict'and any(k.arg=='metadata'for k in n.keywords));metadata=next(k.value for k in report.keywords if k.arg=='metadata')
 assert any(k.arg=='source_before'and ast.unparse(k.value)=='source_before'for k in metadata.keywords)and all(k.arg is not None for k in metadata.keywords)
 assert PROBE.sampler_order_scope(ROOT)==0 and DATA['current_interface']['validate_binding_keywords']==['path','root','context_path']
 assert all(v is None for v in HISTORICAL['producer'].values())and HISTORICAL['prepared_source']is None;assert DATA['prepared_source']['revision']==DATA['producer']['revision']=='dda9cc83d9434a10616f84cf1e1125d93f5f36fd'
 adapter=PROBE.driver_adapter(ROOT);shared=adapter.boundary(ROOT);assert list(__import__('inspect').signature(shared.validate_binding).parameters)==['path','root','context_path']
 adapter.require_sdk_dependencies({'SDK_members':{'include/skimagexx/variation_of_information.hpp':'inert','include/skimagexx/label_pair_metrics.hpp':'inert'}})
 with pytest.raises(AssertionError):adapter.require_sdk_dependencies({'SDK_members':{'include/skimagexx/variation_of_information.hpp':'inert'}})


@pytest.mark.parametrize('cached',(False,True))
def test_actual_UNBOUND_probe_cli_before_cold_cached_requests(monkeypatch,cached):
    monkeypatch.setattr(PROBE,'CONTRACT',HISTORICAL_PATH)  # actual oldUNBOUND refusal, not current bound authority
    import builtins
    prior=builtins.__import__;requests=[]
    class Refused(BaseException):pass
    def refused(name,*a,**k):
        if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):requests.append(name);raise Refused()
        return prior(name,*a,**k)
    with monkeypatch.context()as patch:
        if cached:patch.setitem(sys.modules,'numpy',SimpleNamespace(inert_cached=True))
        patch.setattr(sys,'argv',['probe','--binding','UNBOUND.json','--root-go','UNBOUND-go.json','--route','vi'])
        patch.setattr(builtins,'__import__',refused)
        with pytest.raises(AssertionError,match='UNBOUND'):PROBE.main()
    assert not requests and builtins.__import__ is prior


@pytest.mark.parametrize('observer_mode',('stop','non-stopping','replacing'))
def test_actual_source_tail_and_real_sampler_frames_with_inert_prepared_locals(tmp_path,observer_mode):
 route,frame,k=frame_fixture('vi');prepared=dict(frame.f_back.f_back.f_locals);adapter=PROBE.driver_adapter(ROOT);shared=adapter.boundary(ROOT);common=shared.support(ROOT,json.loads((ROOT/PROBE.DRIVER_CONTEXT).read_bytes()));measure=runpy.run_path(str(ROOT/'tools/montage_sustained.py'))['measure'];callbacks=[];body=[];restores=[];accepted=[]
 prepared.update(measure=measure,shared=shared,reference=lambda:callbacks.append('original'),native=lambda:callbacks.append('native'),restore=lambda value:restores.append(value),deadline='2099-01-01T00:00:00+00:00');prepared['cli'].output=tmp_path/'never.json';prepared['common'].first_cohort_profile=common.first_cohort_profile
 source=(ROOT/'benchmarks/variation_of_information_execution.py').read_text();original=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='execute');assignment=next(n for n in ast.walk(original)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='sample_state'for t in n.targets));tail=next(n for n in ast.walk(original)if isinstance(n,ast.Try)and n.body and isinstance(n.body[0],ast.Assign)and isinstance(n.body[0].value,ast.Call)and isinstance(n.body[0].value.func,ast.Attribute)and n.body[0].value.func.attr=='sample_entry')
 prefix=[ast.Assign(targets=[ast.Name(id='_prepared',ctx=ast.Store())],value=ast.Name(id='module',ctx=ast.Load()))]+[ast.Assign(targets=[ast.Name(id=name,ctx=ast.Store())],value=ast.Subscript(value=ast.Name(id='_prepared',ctx=ast.Load()),slice=ast.Constant(value=name),ctx=ast.Load()))for name in prepared];node=copy.deepcopy(original);node.body=[*prefix,copy.deepcopy(assignment),copy.deepcopy(tail)];namespace={};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(ROOT/'benchmarks/variation_of_information_execution.py'),'exec'),namespace)
 class Stopped(BaseException):pass
 previous_profile,previous_trace=sys.getprofile(),sys.gettrace()
 def profile(frame,event,value):
  if event=='call'and frame.f_code is measure.__code__:
   PROBE.validate_native_entry(frame,route,ROOT,**k);accepted.append(True)
   if observer_mode=='stop':raise Stopped()
   if observer_mode=='replacing':sys.setprofile(None)
 def trace(frame,event,value):
  if frame.f_code is measure.__code__ and event=='line':body.append(frame.f_lineno)
  return trace
 try:
  sys.settrace(trace);sys.setprofile(profile)
  with pytest.raises(Stopped if observer_mode=='stop'else AssertionError):namespace['execute'](prepared)
 finally:sys.settrace(previous_trace);sys.setprofile(previous_profile)
 assert accepted==[True]and not callbacks and not body and restores==[bytes(32)]and not prepared['cli'].output.exists()


@pytest.mark.parametrize('cached',(False,True))
def test_normalized_route_runs_actual_driver_parser_until_request_refusal(tmp_path,monkeypatch,cached):
 import builtins
 path,binding,context=synthetic(tmp_path,monkeypatch);monkeypatch.syspath_prepend(str(ROOT/'benchmarks'));route=ROUTES['vi'];output=Path(binding['output_root'])/'never.json';argv=PROBE.driver_argv(ROOT,route,path,output);adapter=PROBE.driver_adapter(ROOT);module=runpy.run_path(str(ROOT/route['driver']));prior=builtins.__import__;requests=[]
 class Refused(BaseException):pass
 def request(name,*a,**k):
  if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):requests.append(name);raise Refused()
  return prior(name,*a,**k)
 with monkeypatch.context()as patch:
  if cached:patch.setitem(sys.modules,'numpy',SimpleNamespace(inert_cached=True))
  patch.setattr(sys,'argv',argv);patch.setattr(sys,'path',list(sys.path));patch.setattr(builtins,'__import__',request)
  with pytest.raises(Refused):adapter.execute(module)
 assert requests==['numpy']and not output.exists()and not output.with_suffix('.batches.jsonl').exists()
 assert (Path(binding['output_root'])/'vi-first-cohort.claim.json').exists()



def test_v2_current_213_interface_and_frozen_predecessor():
 DATA=HISTORICAL  # exact retained predecessor metadata, never actual current authority
 predecessor=DATA['controller_predecessor']
 assert predecessor['manifest_raw_sha256']=='269d8c17ac06674ecd3a7894fdff114b54d05065521dc8b9e314bdd31bc9a249'
 old=predecessor['frozen_scope'];context=json.loads((ROOT/PROBE.DRIVER_CONTEXT).read_bytes());route=ROUTES['vi']
 assert PROBE.DRIVER_CONTEXT=='coverage/contracts/variation-of-information-benchmark-execution-v5.json'
 assert route['driver_route']==context['route']and set(route['driver_route'])=={'api','driver','owned_module','contract'}
 assert route['driver_route']['contract']==route['contract']==PROBE.DRIVER_CONTEXT
 assert 'legacy_driver_route'not in route
 assert DATA['execution_scope']==old['execution_scope']
 assert route['cases']==old['first_cohort']['routes']['vi']['cases']and route['argv']==old['first_cohort']['routes']['vi']['argv']
 assert context['cases']==json.loads((ROOT/'coverage/contracts/variation-of-information-benchmark-execution-v4.json').read_bytes())['cases']
 assert context['method']==json.loads((ROOT/'coverage/contracts/variation-of-information-benchmark-execution-v4.json').read_bytes())['method']
 assert "CONTEXT='coverage/contracts/mark-boundaries-benchmark-execution-v6.json'"in(ROOT/'benchmarks/mark_boundaries_execution.py').read_text()
 assert DATA['prospective_source']['revision']=='213b986504b6f62f79a98b21be0581f1bc9b3559'and DATA['prospective_source']['native_images']==281
 assert DATA['prospective_source']['SDK_members']==108 and DATA['SDK_members']is None
 assert 'python/skimagexx/coverage.py'in DATA['benchmark_source_sha256']
 for name,row in DATA['controller_source_inverse'].items():
  raw=binding_predecessor_bytes(name,(ROOT/name).read_bytes()).replace(b'\r\n',b'\n');assert H(raw)==row['current_canonical_sha256']
  for part in row['inverse_fragments']:
   assert raw.count(part['current'].encode())==1;raw=raw.replace(part['current'].encode(),part['predecessor'].encode(),1)
  assert H(raw)==row['predecessor_canonical_sha256']


@pytest.mark.parametrize('fault',('none','legacy-v3-route','legacy-v4-route','legacy-active-v4'))
def test_v2_real_manifest_reader_rejects_previous_route_versions(tmp_path,monkeypatch,fault):
 path,b,c=synthetic(tmp_path,monkeypatch);source=Path(b['source_root']);m=copy.deepcopy(DATA)
 m['status']='ROOT_AUTHORIZED_ONE_INITIAL_COMPARISON';m['first_cohort']['status']='ROOT_AUTHORIZED_GENUINE_FIRST_COHORTS';m['SDK_members']=2
 if fault=='legacy-v3-route':m['first_cohort']['routes']['vi']['driver_route']['contract']='coverage/contracts/variation-of-information-benchmark-execution-v3.json'
 elif fault=='legacy-v4-route':m['first_cohort']['routes']['vi']['driver_route']['contract']='coverage/contracts/variation-of-information-benchmark-execution-v4.json'
 elif fault=='legacy-active-v4':m['first_cohort']['routes']['vi']['contract']='coverage/contracts/variation-of-information-benchmark-execution-v4.json'
 out=source/PROBE.CONTRACT;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(m))
 if fault=='none':assert PROBE.read_manifest(source)[1]==m
 else:
  with pytest.raises(AssertionError):PROBE.read_manifest(source)


def binding_predecessor_bytes(name,raw):
    row=DATA['binding_successor']['controller_source_inverse'].get(name)
    if row is None:return raw
    assert H(raw)==row['current_raw_sha256']and raw.decode()==row['current']
    previous=row['predecessor'].encode();assert H(previous)==row['predecessor_raw_sha256']
    return previous


def test_current_controller_inverse_keeps_critical_guards_byte_source_bound():
    assert H((ROOT/HISTORICAL_PATH).read_bytes())==DATA['binding_successor']['predecessor_raw_sha256']
    name=DATA['first_cohort']['probe_path'];current=(ROOT/name).read_bytes();old=binding_predecessor_bytes(name,current)
    a={n.name:ast.dump(n,include_attributes=False)for n in ast.parse(current).body if isinstance(n,ast.FunctionDef)}
    b={n.name:ast.dump(n,include_attributes=False)for n in ast.parse(old).body if isinstance(n,ast.FunctionDef)}
    for function in('main','validate_native_entry','finish_probe','verify_helper_state','verify_source_state','driver_adapter','driver_argv','validate_reports','sampler_order_scope','prepare_host_helper'):
        if function in a:assert a[function]==b[function],function


def test_dispatch_clock_expiry_refuses_before_process_or_source_import(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(PROBE.subprocess,'run',lambda *a,**k:calls.append(a))
    with pytest.raises(AssertionError):
        PROBE.prepare_cohorts(ROOT,ROOT,tmp_path,DATA,'python',tmp_path,'2000-01-01T00:00:00+00:00')
    with pytest.raises(AssertionError):
        PROBE.run_measurements(ROOT,tmp_path,DATA,'python',{},'2000-01-01T00:00:00+00:00')
    assert not calls
