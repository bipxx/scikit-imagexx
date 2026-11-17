"""Source/setup-only two-API authority, real CLI/frame and C1 terminal controls."""
import ast,copy,hashlib,importlib.util,json,runpy,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools.mark_lbp_fp_context import source_successor_predecessor_bytes
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/mark-lbp-measurement-v7.json'
DATA=json.loads(CONTRACT.read_bytes());ROUTES=DATA['first_cohort']['routes']
HISTORICAL_PATH='coverage/contracts/mark-lbp-measurement-v4.json'
HISTORICAL=json.loads((ROOT/HISTORICAL_PATH).read_bytes())
BOUND_PREDECESSOR_PATH='coverage/contracts/mark-lbp-measurement-v5.json'
BOUND_PREDECESSOR=json.loads((ROOT/BOUND_PREDECESSOR_PATH).read_bytes())
OUTPUT_ROOT_PREDECESSOR_PATH='coverage/contracts/mark-lbp-measurement-v6.json'
OUTPUT_ROOT_PREDECESSOR=json.loads((ROOT/OUTPUT_ROOT_PREDECESSOR_PATH).read_bytes())
spec=importlib.util.spec_from_file_location('mark_lbp_outer',ROOT/'tools/mark_lbp_first_cohort_probe.py');PROBE=importlib.util.module_from_spec(spec);spec.loader.exec_module(PROBE)
H=lambda b:hashlib.sha256(b).hexdigest()


def test_current16a5_bound_source_and_historical_scope():
    assert DATA['status']=='ROOT_AUTHORIZED_ONE_INITIAL_COMPARISON'
    assert DATA['producer']==dict(run='43706703018',revision='16a5b50193125199ad26cf5a89acec5babf5290c',artifact='14160352344',sha256='7410ecd5d2a1cc828005db459865056c9e19f66f9f1595545a9ebd42e3a9f4ec')
    assert DATA['prepared_source']['revision']==DATA['producer']['revision']and len(DATA['prepared_source']['files_sha256'])==2988
    assert DATA['SDK_members']==108 and len(DATA['bound_SDK_sha256'])==108
    assert DATA['core_sha256']=='fea7635f5b1a5bc69f4ce864284e21bfc61e6844667e8fa29d0cc809a0a1cefd'
    assert DATA['runtime_sha256']=='975b98daab73327391766fb5f10d411c724d245701352f22147895f0e1e7770d'
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
    assert first['root_admission_sha256']=='640a40fce735beea143a9f417d85d3f10cf68baa9a6c7fd6de69712d45853698'
    assert first['root_stager_sha256']=='6c97da8c848fb25e5d8d3a5306e39c98753485125513d8e4162f3ee667653fbd'
    assert H((ROOT/first['root_admission_path']).read_bytes())==first['root_admission_sha256']
    assert H((ROOT/first['root_stager_path']).read_bytes())==first['root_stager_sha256']
    for name,pin in DATA['benchmark_source_sha256'].items():
        raw=source_successor_predecessor_bytes(ROOT,name,(ROOT/name).read_bytes(),canonical=True).replace(b'\r\n',b'\n')
        assert H(raw)==pin==DATA['prepared_source']['files_sha256'][name]
    for name,pin in first['source_files_raw_sha256'].items():assert H(source_successor_predecessor_bytes(ROOT,name,(ROOT/name).read_bytes()))==pin
    for name,pin in DATA['helper']['source_sha256'].items():assert H(source_successor_predecessor_bytes(ROOT,name,(ROOT/name).read_bytes()))==pin


def frame_fixture(name):
    route=ROUTES[name];adapter=PROBE.driver_adapter(ROOT);lbp=runpy.run_path(str(ROOT/'benchmarks/draw_multiblock_lbp_execution.py'))
    requirements=(adapter.native_requirements if name=='mark'else lbp['native_requirements'])(route['first_case']);state=('same',)
    cli=SimpleNamespace(first_cohort=True,measurement_go=None)
    caller=dict(cli=cli,spec=copy.deepcopy(route['first_case']),api=route['api'],context_path=route['contract'],context={'route':route},accepted=[True],rows=[],returns=[k for k,v in requirements.items()for _ in range(v)],requirements=requirements,module={}if name=='mark'else{'native_requirements':lbp['native_requirements']},expected_state={'alias':'same'},actual_state={'alias':'same'},reference_FP={'flags':32},native_FP={'flags':32},reference_before={'flags':0},native_before={'flags':0},save=lambda:bytes(32),caller=bytes(32),sample_state=bytes(32),args=('input',),options={'exact':True},inputs=['input'],before=[state],np=None,common=SimpleNamespace(array_state=lambda v,np:state),index=0)
    execute=SimpleNamespace(f_code=SimpleNamespace(co_filename=str(ROOT/'benchmarks/mark_boundaries_execution.py'),co_name='execute'),f_locals=caller,f_globals={'native_requirements':adapter.native_requirements})
    bridge=SimpleNamespace(f_code=SimpleNamespace(co_filename=str(ROOT/'benchmarks/mark_boundaries_execution.py'),co_name='sample_entry'),f_back=execute,f_locals=dict(cli=cli,spec=caller['spec'],index=0))
    frame=SimpleNamespace(f_code=SimpleNamespace(co_filename=str(ROOT/route['sampler']),co_name='measure'),f_back=bridge,f_locals={})
    kwargs=dict(counts=dict(original=1,native=1),accepted=[True],input_state=lambda v:v,inputs_before=[(caller['args'],caller['options'])])
    return route,frame,kwargs


@pytest.mark.parametrize('name',ROUTES)
@pytest.mark.parametrize('fault',('none','case','api','contract','bridge','execute','sampler','count','acceptance','marker','FP','before-FP','full32','input','alias','body'))
def test_real_frame_adapter_exact_roles_and_refusals(name,fault):
    route,frame,k=frame_fixture(name);c=frame.f_back.f_back.f_locals
    if fault=='case':c['spec']={'id':'changed'}
    elif fault=='api':c['api']='foreign'
    elif fault=='contract':c['context_path']='foreign'
    elif fault in('bridge','execute','sampler'):
        target={'bridge':frame.f_back,'execute':frame.f_back.f_back,'sampler':frame}[fault];target.f_code.co_filename='foreign.py'
    elif fault=='count':k['counts']['original']=2
    elif fault=='acceptance':c['accepted']=[]
    elif fault=='marker':c['returns'].append('foreign')
    elif fault=='FP':c['native_FP']={'flags':0}
    elif fault=='before-FP':c['native_before']={'flags':32}
    elif fault=='full32':c['sample_state']=bytes([1])*32
    elif fault=='input':k['inputs_before']=[]
    elif fault=='alias':c['actual_state']={}
    elif fault=='body':frame.f_locals['samples']=[]
    if fault=='none':assert PROBE.validate_native_entry(frame,route,ROOT,**k)is c
    else:
        with pytest.raises(AssertionError):PROBE.validate_native_entry(frame,route,ROOT,**k)


@pytest.mark.parametrize('name',ROUTES)
@pytest.mark.parametrize('cached',(False,True))
@pytest.mark.parametrize('tamper',('none','route','counter','helper-digest'))
def test_actual_bound_probe_reaches_import_request_refusal(monkeypatch,tmp_path,name,cached,tamper):
    # Reuse the admitted actual driver envelope factory; descriptors only.
    import builtins
    from test_mark_boundaries_benchmark_execution import synthetic
    path,b,c=synthetic(tmp_path,monkeypatch,context_path=ROOT/ROUTES[name]['contract'])
    SDK=b['SDK_members'];SDK.update({f'fixture/{i}':'a'*64 for i in range(105-len(SDK))})
    runtime=Path(b['runtime_path']);r=json.loads(runtime.read_bytes());r['runtime']['sdk_sha256']=SDK;runtime.write_text(json.dumps(r));b['runtime_sha256']=PROBE.sha(runtime)
    b['SDK_members']=SDK;path.write_text(json.dumps(b));go_path=Path(b['root_go_path']);go=json.loads(go_path.read_bytes());go.update(binding_sha256=PROBE.sha(path),probe_sha256=PROBE.sha(PROBE.__file__));go_path.write_text(json.dumps(go))
    manifest=copy.deepcopy(DATA);manifest.update(status='ROOT_AUTHORIZED_ONE_INITIAL_COMPARISON',producer=b['producer'],prepared_source=b['prepared_source'],runtime_sha256=b['runtime_sha256'],core_sha256=b['core_sha256'],SDK_members=105)
    manifest['first_cohort'].update(status='ROOT_AUTHORIZED_GENUINE_FIRST_COHORTS',source_revision=b['source_revision'],probe_sha256=PROBE.sha(PROBE.__file__),source_files_raw_sha256={n:PROBE.sha(Path(b['source_root'])/n)for n in b['prepared_source']['files_sha256']})
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
    folder=Path(b['output_root'])/'first-cohort'/name
    assert folder.is_dir()==(tamper=='none')
    assert not(folder/'calls.jsonl').exists()and not(folder/'terminal.json').exists()


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
    value=dict(SDK_members=dict.fromkeys(range(105),'inert'),prepared_source={'revision':'a'*40},producer={'revision':'a'*40},runtime_sha256='runtime',core_sha256='core',helper_binding_path=str(helper),helper_binding_sha256=sha(helper),source_root=str(root),git_argv=['fixture-git'],source_revision='a'*40,output_root=str(folder.parent),routes={'blocks':{'contract':'contract.json'}},base_contract_sha256=sha(contract))
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
    manifest=dict(prepared_source=value['prepared_source'],producer=value['producer'],runtime_sha256='runtime',core_sha256='core',SDK_members=105,first_cohort=dict(source_revision='a'*40,source_files_raw_sha256={'small.py':sha(source),'contract.json':value['base_contract_sha256']},routes=value['routes']))
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
    controller=tmp_path/'controller';manifest_path=controller/CONTRACT.relative_to(ROOT);manifest_path.parent.mkdir(parents=True);manifest_path.write_text('{}')
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
        if len(seen)==2:
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
    route=ROUTES['mark'];binding=tmp_path/'binding.json';binding.write_text('{}');terminal=tmp_path/'terminal.json'
    done=dict(status='PASS_REAL_CLI_STOP_BEFORE_BODY',route='mark',api=route['api'],root_binding_sha256=PROBE.sha(binding),counts=dict(original=1,native=1),timed_batches=0,sampler_body_calls=0,outer_state_restored=True,poststop_source_authority_unchanged=True)
    terminal.write_text(json.dumps(done));summary=dict(status='PASS',postbatch_offline_identity_verified=True,routes=[done,{'second_cohort':True}])
    if fault=='status':manifest['status']='PROSPECTIVE_UNBOUND_NO_MEASUREMENT'
    elif fault=='cases':manifest['families']['mark_boundaries_consumer']['case_ids']=[]
    elif fault=='workers':summary['postbatch_offline_identity_verified']=False
    elif fault=='losses':summary['routes']=[]
    call=lambda:PROBE.scoped_measurement(manifest,route,'parent-sha',{'binding':str(binding)},terminal,summary)
    if fault=='none':
        actual=call();assert actual['execution_scope']==DATA['execution_scope']and actual['families']==DATA['families']
        assert actual['completed_first_cohort']==dict(path=str(terminal.resolve()),sha256=PROBE.sha(terminal),binding_sha256=PROBE.sha(binding))
    else:
        with pytest.raises(AssertionError):call()


def test_actual_workflow_and_restore_order_are_source_closed():
    text=(ROOT/'.github/workflows/mark-lbp-initial.yml').read_text();assert '== 104'not in text and "frozen['SDK_members']>0"in text
    assert text.count("probe['prepare_host_helper']")==1
    assert text.index("probe['prepare_host_helper']")<text.index("probe['run_cohorts']")<text.index("probe['run_measurements']")
    assert 'path: results/'in text and 'if: always()'in text
    assert 'next-metrics-measurement-v2.json'not in text
    tree=ast.parse((ROOT/'tools/mark_lbp_first_cohort_probe.py').read_bytes());functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    main=ast.unparse(functions['main']);assert main.index('adapter.validate_binding')<main.index('import numpy as np')
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
    results=tmp_path/'results';results.mkdir();manifest=copy.deepcopy(DATA);manifest.update(benchmark_fingerprint={'synthetic':'not-genuine'},core_sha256='core',runtime_sha256='runtime',producer={'revision':'a'*40})
    authorities={};adapter=PROBE.driver_adapter(ROOT);lbp=runpy.run_path(str(ROOT/'benchmarks/draw_multiblock_lbp_execution.py'))
    for cohort,route in ROUTES.items():
        binding=tmp_path/(cohort+'-binding.json');binding.write_text(json.dumps({'SDK_members':{'synthetic':'metadata'}}));measurement=tmp_path/(cohort+'-measurement.json');measurement.write_text('{}');authorities[cohort]=dict(binding=str(binding),measurement=str(measurement))
        context=json.loads((ROOT/route['contract']).read_bytes());rows=[];journal=[]
        for i,spec in enumerate(route['cases']):
            events=[]
            def batch(arm,round_id,ordinal,elapsed,phase):
                common=dict(case_index=i,case_id=spec['id'],function=arm,round=round_id,ordinal=ordinal,loops=1)
                before=dict(common,event='batch_before',requested_phase='warmup'if phase=='warmup'else'sample')
                after=dict(common,event='batch_after',elapsed_ns=elapsed,phase=phase,accepted=phase=='sample',complete=True,failure=None)
                journal.extend([before,after]);events.append(after)
            for arm in (0,1):batch(arm,-1,arm,250000000,'warmup')
            order=[[0,1]if(r+i)%2==0 else[1,0]for r in range(7)]
            for round_id,arms in enumerate(order):
                for ordinal,arm in enumerate(arms):batch(arm,round_id,ordinal,100000000 if arm==0 else 200000000,'sample')
            timing=dict(reference_ms=100.,cpp_ms=200.,speedup=.5,samples_ms=[[100.]*7,[200.]*7],loops=[[1]*7,[1]*7],sample_labels=['scikit-image','current'],warmup_elapsed_ns=[250000000]*2,sample_order=order,batch_records=events)
            requirements=(adapter.native_requirements if cohort=='mark'else lbp['native_requirements'])(spec)
            row=dict(case_id=spec['id'],api=route['api'],accepted=[True],actual_C_returns=[k for k,v in requirements.items()for _ in range(v)],FP=dict(control=895,status=32,mxcsr=8096),providers=context['providers'],**timing);rows.append(row)
            journal.append(dict(event='measure_returned',case_index=i,case_id=spec['id'],timing=timing))
        meta=dict(synthetic='not-genuine',core_sha256='core',runtime_sha256='runtime',SDK_members={'synthetic':'metadata'},threads=1,repeat=7,target_ms=100,warmup_min_ms=250,contract_sha256=PROBE.sha(ROOT/route['contract']),root_binding_sha256=PROBE.sha(binding),measurement_sha256=PROBE.sha(measurement),producer=manifest['producer'],source_revision='a'*40,method_sha256=context['method']['method_sha256'],retain_all_losses=True)
        (results/(cohort+'.json')).write_text(json.dumps(dict(metadata=meta,results=rows)))
        (results/(cohort+'.batches.jsonl')).write_text(''.join(json.dumps(row)+'\n'for row in journal))
    return results,manifest,authorities


@pytest.mark.parametrize('fault',('none','old-pass1-order','drop-loss','marker','FP','warmup','median','binding','method'))
def test_actual_report_validator_uses_admitted_pass0_and_retains_losses(tmp_path,fault):
    results,manifest,authorities=synthetic_reports(tmp_path);path=results/'mark.json';report=json.loads(path.read_bytes());row=report['results'][0]
    if fault=='old-pass1-order':row['sample_order']=[[1,0]if i%2==0 else[0,1]for i in range(7)]
    elif fault=='drop-loss':report['results'].pop()
    elif fault=='marker':row['actual_C_returns'].append('foreign')
    elif fault=='FP':row['FP']['normalized-away']=True
    elif fault=='warmup':row['warmup_elapsed_ns']=[1,1]
    elif fault=='median':row['reference_ms']=0
    elif fault=='binding':report['metadata']['root_binding_sha256']='changed'
    elif fault=='method':report['metadata']['method_sha256']='changed'
    path.write_text(json.dumps(report));call=lambda:PROBE.validate_reports(ROOT,results,manifest,authorities)
    if fault=='none':
        call();summary=json.loads((results/'summary.json').read_bytes())
        assert len(summary['families']['mark']['losses'])==8 and len(summary['families']['lbp']['losses'])==4
    else:
        with pytest.raises(AssertionError):call()
        assert not(results/'summary.json').exists()


def test_current280_source_projection_preserves_prior_controller_and_cases():
    DATA=HISTORICAL  # exact retained predecessor metadata, never actual current authority
    assert DATA['prospective_source']['revision']=='ab7872a5483a5d5e3660ba20fdd873d78463a1b5'
    assert tuple(DATA['prospective_source'][k]for k in('image_count','functions','product_members','SDK_members','SDK_families','SDK_checks'))==(280,283,226,107,37,74)
    predecessor=DATA['predecessor_controller'];raw=(ROOT/predecessor['contract_path']).read_bytes()
    assert H(raw)==predecessor['contract_raw_sha256'];old=json.loads(raw)
    assert DATA['execution_scope']==old['execution_scope']and DATA['helper']==old['helper']
    for name,route in ROUTES.items():
        assert route['cases']==old['first_cohort']['routes'][name]['cases']
        assert route['argv']==old['first_cohort']['routes'][name]['argv']
    for name,row in DATA['controller_inverse'].items():
        current=__import__('tools.join_source_context',fromlist=['rolling_predecessor_raw_bytes']).rolling_predecessor_raw_bytes(ROOT,name,binding_predecessor_bytes(name,(ROOT/name).read_bytes()))
        assert H(current)==row['current_raw_sha256']and current.decode()==row['current']
        assert H(row['predecessor'].encode())==row['predecessor_raw_sha256']
    assert all(DATA[k]is None for k in('prepared_source','SDK_members','core_sha256','runtime_sha256','deadline_utc','benchmark_fingerprint'))


def test_actual_current_driver_metadata_and_Montage_pass0_interfaces():
    source=(ROOT/'benchmarks/mark_boundaries_execution.py').read_text();tree=ast.parse(source)
    execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
    report=next(n for n in ast.walk(execute)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='dict'and any(k.arg=='metadata'for k in n.keywords))
    metadata=next(k.value for k in report.keywords if k.arg=='metadata')
    assert any(k.arg is None and ast.unparse(k.value)=='source_before'for k in metadata.keywords)
    assert any(k.arg=='source_before'and ast.unparse(k.value)=='source_before'for k in metadata.keywords)
    assert DATA['current_interface']['report_source']=='metadata flattened fingerprint fields plus metadata.source_before'
    assert PROBE.sampler_order_scope(ROOT)==0
    assert ROUTES['mark']['sampler_caller']==ROUTES['lbp']['sampler_caller']=='sample_entry'


def binding_predecessor_bytes(name,raw):
    row=BOUND_PREDECESSOR['binding_successor']['controller_source_inverse'].get(name)
    if row is None:return raw
    raw=source_successor_predecessor_bytes(ROOT,name,raw)
    assert H(raw)==DATA['binding_successor']['controller_source_pins'][name]
    assert H(row['current'].encode())==row['current_raw_sha256']
    previous=row['predecessor'].encode();assert H(previous)==row['predecessor_raw_sha256']
    return previous


def test_current_controller_inverse_keeps_critical_guards_byte_source_bound():
    assert H((ROOT/OUTPUT_ROOT_PREDECESSOR_PATH).read_bytes())==DATA['binding_successor']['predecessor_raw_sha256']
    assert H((ROOT/BOUND_PREDECESSOR_PATH).read_bytes())==OUTPUT_ROOT_PREDECESSOR['binding_successor']['predecessor_raw_sha256']
    assert H((ROOT/HISTORICAL_PATH).read_bytes())==BOUND_PREDECESSOR['binding_successor']['predecessor_raw_sha256']
    name=DATA['first_cohort']['probe_path'];current=(ROOT/name).read_bytes();old=binding_predecessor_bytes(name,current)
    approved=b"folder=Path(binding['output_root'])/'first-cohort'/args.route"
    assert current.count(approved)==1
    current=current.replace(approved,b"folder=Path(binding['output_root'])/args.route",1)
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


def test_current16a5_preserves_frozen_DDA_scope_and_exact_runtime_binding():
    assert DATA['execution_scope']==BOUND_PREDECESSOR['execution_scope']
    assert BOUND_PREDECESSOR['producer']['revision']=='dda9cc83d9434a10616f84cf1e1125d93f5f36fd'
    assert DATA['benchmark_fingerprint']['source_sha256']=='2c0102a64c720a7ca35d8c850cd6035f2c5ef5e869fd93817fb855d1087d6090'
    assert DATA['actual_bound_source']['source_files']==2988
    admission=json.loads((ROOT/DATA['first_cohort']['root_admission_path']).read_bytes())
    assert admission['product_kind']=='GENUINE_INSTALLED_WHOLE_PRODUCT'
    assert admission['source']==DATA['producer']['revision'] and admission['profiles']==4
    assert admission['passed'] is admission['root_accepted'] is True
    for name,route in ROUTES.items():
        prior=BOUND_PREDECESSOR['first_cohort']['routes'][name]
        current=json.loads((ROOT/route['contract']).read_bytes())
        previous=json.loads((ROOT/prior['contract']).read_bytes())
        assert route==current['route'] and route['cases']==prior['cases']
        assert current['cases']==previous['cases'] and current['method']==previous['method']
        assert route['contract'].endswith('-v7.json')
    assert DATA['scope_limits']['historical64_retiming'] is False
    assert DATA['scope_limits']['VI4']=='held'
    assert DATA['scope_limits']['LBP_double_measurement'] is False


@pytest.mark.parametrize('name',('tools/mark_lbp_first_cohort_probe.py','.github/workflows/mark-lbp-initial.yml'))
def test_controller_successor_changes_only_approved_metadata_expressions(name):
    row=BOUND_PREDECESSOR['binding_successor']['controller_source_inverse'][name]
    previous=row['current'].encode()
    assert H(previous)==row['current_raw_sha256']
    expected=previous.replace(BOUND_PREDECESSOR_PATH.encode(),CONTRACT.relative_to(ROOT).as_posix().encode())
    if name.endswith('.py'):
        expected=expected.replace(
            b"('mark','mark-boundaries-benchmark-execution-v6.json'),('lbp','draw-multiblock-lbp-benchmark-execution-v5.json')",
            b"('mark','mark-boundaries-benchmark-execution-v7.json'),('lbp','draw-multiblock-lbp-benchmark-execution-v7.json')")
        expected=expected.replace(b"output_root=str((results/'first-cohort').resolve())",b'output_root=str(results.resolve())')
        expected=expected.replace(b"folder=Path(binding['output_root'])/args.route",b"folder=Path(binding['output_root'])/'first-cohort'/args.route")
    assert (ROOT/name).read_bytes()==expected


def prepared_controller(tmp_path,monkeypatch):
    # Reuse the existing inert authority factory; no installed product is loaded.
    from test_mark_boundaries_benchmark_execution import synthetic,source_fixture_bytes
    _,binding,_=synthetic(tmp_path,monkeypatch,context_path=ROOT/ROUTES['mark']['contract'])
    snapshot=Path(binding['source_root']);prepared=binding['prepared_source']
    for route in ROUTES.values():
        context_path=route['contract'];context=json.loads((ROOT/context_path).read_bytes())
        names=set(context['source_files_canonical_sha256'])|set(context['FENV_dependency']['source_files_raw_sha256'])|{context_path}
        for name in names:
            target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source_fixture_bytes(name,ROOT/context_path))
            prepared['files_sha256'][name]=H(target.read_bytes().replace(b'\r\n',b'\n'))
    controller=tmp_path/'controller';manifest=copy.deepcopy(DATA);first=manifest['first_cohort']
    for name,source in ((first['probe_path'],PROBE.__file__),(first['root_admission_path'],binding['genuine_root_admission_path']),(first['root_stager_path'],binding['genuine_root_stager_path'])):
        target=controller/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(source).read_bytes())
    site=Path(binding['installed_site']);site.mkdir();core=site/'_core.so';core.write_bytes(b'inert fixture bytes, never imported')
    SDK=binding['SDK_members']
    for route in ROUTES.values():SDK[json.loads((ROOT/route['contract']).read_bytes())['SDK_header']]='c'*64
    SDK.update({f'fixture/{i}':'a'*64 for i in range(108-len(SDK))})
    runtime=json.loads(Path(binding['runtime_path']).read_bytes());runtime['source']=prepared
    runtime['runtime']['sdk_sha256']=SDK;runtime['runtime']['packages_sha256']['skimagexx']={'_core.so':H(core.read_bytes())}
    stage=tmp_path/'stage';stage.mkdir();runtime_path=stage/'runtime.json';PROBE.write_json(runtime_path,runtime)
    manifest.update(prepared_source=prepared,producer=binding['producer'],core_sha256=PROBE.sha(core),runtime_sha256=PROBE.sha(runtime_path),SDK_members=len(SDK))
    first.update(source_revision=binding['source_revision'],probe_sha256=PROBE.sha(controller/first['probe_path']),root_admission_sha256=PROBE.sha(controller/first['root_admission_path']),root_stager_sha256=PROBE.sha(controller/first['root_stager_path']),core_member='_core.so',source_files_raw_sha256={n:PROBE.sha(snapshot/n)for n in prepared['files_sha256']})
    results=tmp_path/'results';results.mkdir();(results/'helper-binding.json').write_bytes(Path(binding['helper_binding_path']).read_bytes())
    for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS'):monkeypatch.setenv(name,'1')
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE','1');monkeypatch.delenv('SKIMAGEXX_DISABLE_SIMD',raising=False)
    deadline='2099-01-01T00:00:00+00:00'
    authorities=PROBE.prepare_cohorts(controller,snapshot,results,manifest,sys.executable,site,deadline)
    commands=PROBE.measurement_commands(snapshot,results,manifest,sys.executable,authorities,deadline)
    return snapshot,results,manifest,authorities,commands,deadline


def test_actual_prepared_bindings_contain_generated_reports_and_preserve_cohort_collectors(tmp_path,monkeypatch):
    import builtins
    _,results,manifest,authorities,commands,_=prepared_controller(tmp_path,monkeypatch)
    monkeypatch.setattr(PROBE,'read_manifest',lambda root:(CONTRACT,manifest));monkeypatch.setattr(sys,'path',list(sys.path))
    prior=builtins.__import__;requests=[]
    class Refused(BaseException):pass
    def refuse(name,*a,**k):
        if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):requests.append(name);raise Refused()
        return prior(name,*a,**k)
    for name,authority in authorities.items():
        binding=json.loads(Path(authority['binding']).read_bytes());command=commands[name]
        output=Path(command[command.index('--output')+1]);root=Path(binding['output_root'])
        assert root.is_absolute()and root==results.resolve()and output==results/(name+'.json')
        assert output.resolve().is_relative_to(root)and command[command.index('--root-binding')+1]==authority['binding']
        assert '--first-cohort'not in command
        monkeypatch.setattr(sys,'argv',['probe','--binding',authority['binding'],'--root-go',authority['go'],'--route',name])
        with monkeypatch.context()as patch:
            patch.setattr(builtins,'__import__',refuse)
            with pytest.raises(Refused):PROBE.main()
        folder=results/'first-cohort'/name
        assert folder.is_dir()and not(folder/'terminal.json').exists()and not output.exists()
        assert not(results/name).exists()
    assert requests==['numpy','numpy']and builtins.__import__ is prior


@pytest.mark.parametrize('name',('mark','lbp'))
@pytest.mark.parametrize('destination',('inside','sibling','outside'))
def test_actual_prepared_measurement_driver_containment_before_scientific_import(tmp_path,monkeypatch,name,destination):
    import builtins
    snapshot,results,manifest,authorities,commands,deadline=prepared_controller(tmp_path,monkeypatch)
    # Only synthetic completed-cohort metadata is supplied; no cohort/target runs.
    done=[]
    for cohort,route in ROUTES.items():
        terminal=results/'first-cohort'/cohort/'terminal.json';terminal.parent.mkdir(parents=True)
        row=dict(status='PASS_REAL_CLI_STOP_BEFORE_BODY',route=cohort,api=route['api'],source=manifest['prepared_source']['revision'],root_binding_sha256=PROBE.sha(authorities[cohort]['binding']),counts=dict(original=1,native=1),sampler_body_calls=0,timed_batches=0,outer_state_restored=True,poststop_source_authority_unchanged=True)
        PROBE.write_json(terminal,row);done.append(row)
    summary=dict(status='PASS',postbatch_offline_identity_verified=True,routes=done)
    scoped=PROBE.scoped_measurement(manifest,ROUTES[name],'inert-parent-pin',authorities[name],results/'first-cohort'/name/'terminal.json',summary)
    scoped['deadline_utc']=deadline;PROBE.write_json(authorities[name]['measurement'],scoped)
    command=commands[name];output_index=command.index('--output')+1
    if destination=='sibling':command[output_index]=str(results.with_name(results.name+'-sibling')/(name+'.json'))
    elif destination=='outside':command[output_index]=str(tmp_path/'outside'/(name+'.json'))
    output=Path(command[output_index]);module=runpy.run_path(str(snapshot/ROUTES[name]['driver']))
    adapter=PROBE.driver_adapter(snapshot);prior=builtins.__import__;requests=[]
    class Refused(BaseException):pass
    def refuse(module_name,*a,**k):
        if module_name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):requests.append(module_name);raise Refused()
        return prior(module_name,*a,**k)
    monkeypatch.setattr(sys,'argv',command[1:]);monkeypatch.setattr(sys,'path',list(sys.path))
    with monkeypatch.context()as patch:
        patch.setattr(builtins,'__import__',refuse)
        with pytest.raises(Refused if destination=='inside'else AssertionError)as caught:adapter.execute(module,context_path=ROUTES[name]['contract'])
    if destination!='inside':
        assert caught.traceback[-1].name=='execute'
        assert 'cli.output.resolve().is_relative_to(output_root)'in str(caught.traceback[-1].statement)
    assert requests==(['numpy']if destination=='inside'else[])and builtins.__import__ is prior
    claim=results/(name+'-measurement.claim.json');assert claim.exists()==(destination=='inside')
    if claim.exists():assert json.loads(claim.read_bytes())['cases']==[s['id']for s in ROUTES[name]['cases']]
    assert not output.exists()and not output.with_suffix('.batches.jsonl').exists()


def test_output_root_context_preserves_qualified_binding_and_prior_scope():
    current=copy.deepcopy(DATA);old=copy.deepcopy(OUTPUT_ROOT_PREDECESSOR)
    metadata=current.pop('output_root_successor')
    assert metadata['source_execution_authority']==0 and metadata['parent']=='c25ee4592a8166e0fdb72475cab5a6e6aeec52c6'
    assert metadata['failed_run']=='43721720516'and metadata['driver_containment_unchanged']is True
    assert current['binding_successor']['predecessor_contract']==OUTPUT_ROOT_PREDECESSOR_PATH
    current['binding_successor']=old['binding_successor'];current['controller_reuse']=old['controller_reuse']
    current['first_cohort']['probe_sha256']=old['first_cohort']['probe_sha256']
    assert current==old


@pytest.mark.parametrize('name',('mark','lbp'))
def test_current_exact_route_guard_refuses_predecessor_context(tmp_path,name):
    changed=copy.deepcopy(DATA)
    changed['first_cohort']['routes'][name]['contract']=BOUND_PREDECESSOR['first_cohort']['routes'][name]['contract']
    for route in ROUTES.values():
        target=tmp_path/route['contract'];target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/route['contract']).read_bytes())
    target=tmp_path/CONTRACT.relative_to(ROOT);target.write_text(json.dumps(changed),encoding='utf-8')
    with pytest.raises(AssertionError):
        PROBE.read_manifest(tmp_path)
