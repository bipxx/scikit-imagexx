"""Inert source/authority tests only; no scientific/helper/sampler bodies."""
import ast,builtins,copy,hashlib,importlib.util,json,sys,types
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/finite-radon-measurement-v4.json'
DATA=json.loads(CONTRACT.read_bytes());ROUTES=DATA['first_cohort']['routes']
spec=importlib.util.spec_from_file_location('radon_controller',ROOT/'tools/finite_radon_first_cohort_probe.py')
PROBE=importlib.util.module_from_spec(spec);spec.loader.exec_module(PROBE)
H=lambda b:hashlib.sha256(b).hexdigest()


def test_frozen_dependencies_and_unbound_authority():
    assert DATA['status']=='PROSPECTIVE_UNBOUND_NO_MEASUREMENT'
    assert all(v is None for v in DATA['producer'].values())
    for key in ('prepared_source','SDK_members','core_sha256','runtime_sha256','deadline_utc','integration_status','benchmark_fingerprint'):assert DATA[key]is None
    for key in ('source_revision','core_member','source_files_raw_sha256','root_admission_path','root_admission_sha256','root_stager_path','root_stager_sha256'):assert DATA['first_cohort'][key]is None
    assert DATA['prospective_source']['revision']=='ab7872a5483a5d5e3660ba20fdd873d78463a1b5'and DATA['prospective_source']['image_count']==280
    assert DATA['prospective_source']['SDK_members']==107 and DATA['prospective_source']['actual_qualification']=='UNBOUND_NOT_GENUINE'
    assert DATA['helper']['receipt_sha256']is DATA['helper']['library_sha256']is DATA['helper']['root_GO']is None
    assert PROBE.sha(PROBE.__file__)==DATA['first_cohort']['probe_sha256']
    for name,pin in DATA['benchmark_source_sha256'].items():assert H(__import__('tools.join_source_context',fromlist=['rolling_predecessor_bytes']).rolling_predecessor_bytes(ROOT,name,(ROOT/name).read_bytes()))==pin,name
    with pytest.raises(AssertionError,match='UNBOUND'):PROBE.read_manifest(ROOT)


@pytest.mark.parametrize('failure',(False,True))
def test_actual_driver_adapter_restores_caller_import_path(failure,monkeypatch):
    before=list(sys.path);identity=sys.path
    if failure:
        original=PROBE.importlib.util.spec_from_file_location
        def failing_spec(*args,**kwargs):
            spec=original(*args,**kwargs)
            def refuse(module):raise ImportError('INERT_DRIVER_IMPORT_FAILURE')
            spec.loader.exec_module=refuse
            return spec
        monkeypatch.setattr(PROBE.importlib.util,'spec_from_file_location',failing_spec)
        with pytest.raises(ImportError,match='INERT_DRIVER_IMPORT_FAILURE'):PROBE.driver_adapter(ROOT)
    else:
        loaded=PROBE.driver_adapter(ROOT)
        assert Path(loaded.__file__).resolve()==ROOT/'benchmarks/finite_radon_execution_v1.py'
        assert callable(loaded.execute)
    assert sys.path is identity and sys.path==before


@pytest.mark.parametrize('route',ROUTES)
@pytest.mark.parametrize('cached',(False,True))
def test_actual_unbound_probe_before_science(route,cached,monkeypatch,tmp_path):
    seen=[];old=builtins.__import__
    def guard(name,*a,**k):
        if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):seen.append(name);raise AssertionError('BODY_FORBIDDEN')
        return old(name,*a,**k)
    if cached:monkeypatch.setitem(sys.modules,'numpy',types.ModuleType('numpy'))
    monkeypatch.setattr(builtins,'__import__',guard)
    monkeypatch.setattr(sys,'argv',['probe','--route',route,'--binding',str(tmp_path/'absent'),'--root-go',str(tmp_path/'absent-go')])
    with pytest.raises(AssertionError,match='UNBOUND'):PROBE.main()
    assert not seen and not list(tmp_path.iterdir())


def synthetic(tmp_path,monkeypatch,route):
    import next_util_benchmark_support as support
    monkeypatch.setattr(support,'CONTEXT',ROOT/DATA['driver_context'])
    path=support.synthetic_binding(tmp_path,route,monkeypatch);b=json.loads(path.read_bytes())
    runtime=Path(b['runtime_path']);v=json.loads(runtime.read_bytes());v['runtime']['sdk_sha256']={'inert.hpp':'a'*64};runtime.write_text(json.dumps(v));b['runtime_sha256']=PROBE.sha(runtime)
    b['routes']={route:ROUTES[route]};b['output_root']=str(tmp_path/'first-cohort');path.write_text(json.dumps(b))
    receipt=tmp_path/'receipt.json';receipt.write_text('{"inert":true}')
    env=dict(UTIL_VIEWS_FENV_MODE='source-built-v1',UTIL_VIEWS_FENV_RECEIPT=str(receipt),UTIL_VIEWS_FENV_RECEIPT_SHA256=PROBE.sha(receipt))
    helper=tmp_path/'helper.json';helper.write_text(json.dumps(dict(env=env,files={'receipt.json':PROBE.sha(receipt)})))
    for k,value in env.items():monkeypatch.setenv(k,value)
    for k in ('UTIL_VIEWS_FENV_HELPER','GIT_DIR','GIT_WORK_TREE'):monkeypatch.delenv(k,raising=False)
    gp=Path(b['root_go_path']);go=json.loads(gp.read_bytes());go.update(binding_sha256=PROBE.sha(path),probe_sha256=PROBE.sha(PROBE.__file__),maximum_original_topcalls=1,maximum_native_topcalls=1,helper_binding_path=str(helper),helper_binding_sha256=PROBE.sha(helper));gp.write_text(json.dumps(go))
    m=copy.deepcopy(DATA);m.update(status='ROOT_AUTHORIZED_ONE_INITIAL_COMPARISON',producer=b['producer'],prepared_source=b['prepared_source'],runtime_sha256=b['runtime_sha256'],core_sha256=b['core_sha256'],SDK_members=1)
    m['first_cohort'].update(status='ROOT_AUTHORIZED_GENUINE_FIRST_COHORTS',source_revision=b['source_revision'],source_files_raw_sha256={n:PROBE.sha(Path(b['source_root'])/n)for n in b['prepared_source']['files_sha256']})
    return path,b,gp,go,m


@pytest.mark.parametrize('route',ROUTES)
@pytest.mark.parametrize('cached',(False,True))
@pytest.mark.parametrize('fault',('none','counter','route','helper'))
def test_inert_bound_actual_probe_stops_at_request(route,cached,fault,monkeypatch,tmp_path):
    path,b,gp,go,m=synthetic(tmp_path,monkeypatch,route)
    if fault=='counter':go['maximum_original_topcalls']=2
    elif fault=='route':go['routes']=['foreign']
    elif fault=='helper':go['helper_binding_sha256']='0'*64
    gp.write_text(json.dumps(go));monkeypatch.setattr(PROBE,'read_manifest',lambda root:(CONTRACT,m))
    monkeypatch.setattr(sys,'argv',['probe','--binding',str(path),'--root-go',str(gp),'--route',route]);monkeypatch.setattr(sys,'path',list(sys.path))
    if cached:monkeypatch.setitem(sys.modules,'numpy',types.ModuleType('numpy'))
    seen=[];old=builtins.__import__
    class RequestRefused(BaseException):pass
    def guard(name,*a,**k):
        if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):seen.append(name);raise RequestRefused
        return old(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',guard)
    with pytest.raises(RequestRefused if fault=='none'else AssertionError):PROBE.main()
    assert seen==(['numpy']if fault=='none'else[])
    assert not(Path(b['output_root'])/route/'calls.jsonl').exists()


def frame_fixture(route):
    r=ROUTES[route];cli=SimpleNamespace(first_cohort=True,measurement_go=None,pass_index=1)
    state={'inert':'state'};entry=object()
    c=dict(cli=cli,spec=r['first_case'],api=route,context=json.loads((ROOT/DATA['driver_context']).read_bytes()),accepted=[True],rows=[],returns=[route],entry=entry,reference_FP={'control':895,'status':32,'mxcsr':8096},native_FP={'control':895,'status':32,'mxcsr':8096},reference_before={},native_before={},save=lambda:bytes(32),caller=bytes(32),args=('input',),options={},a='input',before=state,expected='e',actual='a',np=None,common=SimpleNamespace(array_state=lambda *a:state),index=0)
    execute=SimpleNamespace(f_code=SimpleNamespace(co_filename=str(ROOT/'benchmarks/finite_radon_execution_v1.py'),co_name='execute'),f_locals=c)
    bridge=SimpleNamespace(f_code=SimpleNamespace(co_filename=str(ROOT/'benchmarks/finite_radon_execution_v1.py'),co_name='sample'),f_back=execute,f_locals=dict(cli=cli,spec=c['spec'],index=0,entry=entry,workers=1))
    frame=SimpleNamespace(f_code=SimpleNamespace(co_filename=str(ROOT/r['sampler']),co_name='measure'),f_back=bridge,f_locals={})
    return r,frame,dict(counts={'original':1,'native':1},accepted=[True],input_state=lambda v:v,inputs_before=[(c['args'],c['options'])])


@pytest.mark.parametrize('route',ROUTES)
@pytest.mark.parametrize('fault',('none','case','count','marker','FP','before','full32','input','frame','body','pass','worker'))
def test_actual_radon_frame_layout_and_rejection(route,fault):
    r,f,k=frame_fixture(route);c=f.f_back.f_back.f_locals
    if fault=='case':c['spec']={}
    elif fault=='count':k['counts']['original']=2
    elif fault=='marker':c['returns'].append('foreign')
    elif fault=='FP':c['native_FP']={}
    elif fault=='before':c['native_before']={'masked':True}
    elif fault=='full32':c['caller']=bytes([1])*32
    elif fault=='input':k['inputs_before']=[]
    elif fault=='frame':f.f_back.f_code.co_name='sample_entry'
    elif fault=='body':f.f_locals['loops']=[]
    elif fault=='pass':c['cli'].pass_index=0
    elif fault=='worker':f.f_back.f_locals['workers']=4
    if fault=='none':assert PROBE.validate_native_entry(f,r,ROOT,**k)is c
    else:
        with pytest.raises(AssertionError):PROBE.validate_native_entry(f,r,ROOT,**k)


@pytest.mark.parametrize('route',ROUTES)
@pytest.mark.parametrize('fault',('none','source','HEAD','dirty','binding','GO','contract','helper','deadline','restore'))
def test_actual_C1_poststop_terminal_boundary(route,fault,monkeypatch,tmp_path):
    path,b,gp,go,m=synthetic(tmp_path,monkeypatch,route);root=Path(b['source_root']);folder=tmp_path/'poststop';folder.mkdir()
    bp,gpin=PROBE.sha(path),PROBE.sha(gp);stopped=dict(route=route,counts={'original':1,'native':1},timed_batches=0,sampler_body_calls=0)
    if fault=='source':(root/'src/finite_radon.cpp').write_text('changed')
    elif fault in ('HEAD','dirty'):monkeypatch.setattr(PROBE.subprocess,'check_output',lambda argv,**k:'foreign\n'if fault=='HEAD'else' M changed')
    elif fault=='binding':path.write_text('{}')
    elif fault=='GO':gp.write_text('{}')
    elif fault=='contract':(root/ROUTES[route]['contract']).write_text('{}')
    elif fault=='helper':Path(go['helper_binding_path']).write_text('{}')
    elif fault=='deadline':go['deadline_utc']='2000-01-01T00:00:00+00:00'
    call=lambda:PROBE.finish_probe(folder,stopped,b,path,gp,go,bp,gpin,fault!='restore',m)
    if fault=='none':call();assert json.loads((folder/'terminal.json').read_bytes())['poststop_source_authority_unchanged']
    else:
        with pytest.raises(AssertionError):call()
        assert not(folder/'terminal.json').exists()and json.loads((folder/'failure.json').read_bytes())['retry']is False


def synthetic_reports(tmp_path,pass_index=1):
    results=tmp_path/'results';results.mkdir();m=copy.deepcopy(DATA);m.update(benchmark_fingerprint={'synthetic':'not-genuine'},core_sha256='core',runtime_sha256='runtime',producer={'revision':'a'*40});authorities={}
    for name,route in ROUTES.items():
        binding=tmp_path/(name+'-binding');binding.write_text('{}');measurement=tmp_path/(name+'-measurement');measurement.write_text('{}');authorities[name]=dict(binding=str(binding),measurement=str(measurement))
        command=['python',str(ROOT/route['driver']),'--pass-index',str(pass_index)];(results/(name+'-command.json')).write_text(json.dumps(command))
        context=json.loads((ROOT/route['contract']).read_bytes());rows=[];journal=[]
        for i,case in enumerate(route['cases']):
            events=[];loops=[1,1];counts=[[],[]]
            def batch(arm,r,ordinal,phase):
                n=loops[arm];elapsed=250000000 if phase=='warmup'else n*(100000000 if arm==0 else 200000000)
                common=dict(case_index=i,case_id=case['id'],function=arm,round=r,ordinal=ordinal,loops=n)
                before=dict(common,event='batch_before',requested_phase=phase);after=dict(common,event='batch_after',elapsed_ns=elapsed,phase=phase,accepted=phase=='sample',complete=True,failure=None)
                journal.extend([before,after]);events.append(after)
                import math
                loops[arm]=max(1,math.ceil(n*100000000/elapsed*1.15))
                if phase=='sample':counts[arm].append(n)
            for arm in (0,1):batch(arm,-1,arm,'warmup')
            order=[[0,1]if(r+i+pass_index)%2==0 else[1,0]for r in range(7)]
            for r,arms in enumerate(order):
                for ordinal,arm in enumerate(arms):batch(arm,r,ordinal,'sample')
            timing=dict(reference_ms=100.,cpp_ms=200.,speedup=.5,samples_ms=[[100.]*7,[200.]*7],loops=counts,sample_labels=['scikit-image','current'],warmup_elapsed_ns=[250000000]*2,sample_order=order,batch_records=events)
            rows.append(dict(case_id=case['id'],api=route['api'],native_accepted=True,actual_C_returns=[name],FP=dict(control=895,status=32,mxcsr=8096),providers=context['apis'][name]['providers'],**timing));journal.append(dict(event='measure_returned',case_index=i,case_id=case['id'],timing=timing))
        meta=dict(source_before=m['benchmark_fingerprint'],core_sha256='core',runtime_sha256='runtime',threads=1,repeat=7,target_ms=100,warmup_min_ms=250,pass_index=pass_index,contract_sha256=PROBE.sha(ROOT/route['contract']),root_binding_sha256=PROBE.sha(binding),measurement_sha256=PROBE.sha(measurement),producer=m['producer'],source_revision='a'*40,method_sha256=context['method']['method_sha256'],retain_all_losses=True)
        (results/(name+'.json')).write_text(json.dumps(dict(metadata=meta,results=rows)));(results/(name+'.batches.jsonl')).write_text(''.join(json.dumps(x)+'\n'for x in journal))
    return results,m,authorities


@pytest.mark.parametrize('fault',('none','old-pass0','drop-loss','marker','FP','warmup','median','binding','method','loops','source'))
def test_source_derived_report_order_and_all_losses(tmp_path,fault):
    results,m,authorities=synthetic_reports(tmp_path);p=results/'frt2.json';report=json.loads(p.read_bytes());row=report['results'][0]
    if fault=='old-pass0':row['sample_order']=[[0,1]if i%2==0 else[1,0]for i in range(7)]
    elif fault=='drop-loss':report['results'].pop()
    elif fault=='marker':row['actual_C_returns'].append('foreign')
    elif fault=='FP':row['FP']['normalized']=True
    elif fault=='warmup':row['warmup_elapsed_ns']=[1,1]
    elif fault=='median':row['reference_ms']=0
    elif fault=='binding':report['metadata']['root_binding_sha256']='foreign'
    elif fault=='method':report['metadata']['method_sha256']='foreign'
    elif fault=='loops':row['loops'][0][0]=3
    elif fault=='source':report['metadata']['source_before']={'changed':True}
    p.write_text(json.dumps(report))
    if fault=='none':
        PROBE.validate_reports(ROOT,results,m,authorities);summary=json.loads((results/'summary.json').read_bytes());assert all(len(v['losses'])==8 for v in summary['families'].values())
    else:
        with pytest.raises(AssertionError):PROBE.validate_reports(ROOT,results,m,authorities)
        assert not(results/'summary.json').exists()


@pytest.mark.parametrize('value',range(1,9))
def test_all_explicit_pass_indices_and_actual_sampler_keyword(value,tmp_path):
    assert PROBE.sampler_order_scope(ROOT,['--pass-index',str(value)])==value


def test_workflow_source_order_and_inline_compile():
    text=(ROOT/'.github/workflows/finite-radon-initial.yml').read_text()
    assert text.count("probe['prepare_host_helper']")==1
    assert text.index("probe['prepare_host_helper']")<text.index("probe['run_cohorts']")<text.index("probe['run_measurements']")
    assert '== 104'not in text and "frozen['SDK_members']>0"in text
    blocks=[];block=None
    for line in text.splitlines():
        if line.strip().endswith("<<'PY'"):block=[]
        elif block is not None:
            if line.strip()=='PY':blocks.append('\n'.join(block)+'\n');block=None
            else:block.append(line[10:])
    assert len(blocks)==5
    for body in blocks:compile(body,'<workflow>','exec')
    s=(ROOT/'tools/finite_radon_first_cohort_probe.py').read_text();main=ast.unparse(next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)and n.name=='main'))
    assert main.index('common.load_binding')<main.index('import numpy as np')and main.index('restore(outer)')<main.index('finish_probe(')
    assert 'sample_state'not in s and "co_name=='sample'"in s


@pytest.mark.parametrize('fault',('none','verifier','runtime','runtime-during-verifier','manifest'))
def test_actual_batch_summary_requires_one_postbatch_verifier(tmp_path,monkeypatch,fault):
    # Synthetic completed child receipts; neither child probe nor verifier runs.
    controller=tmp_path/'controller';manifest_path=controller/'coverage/contracts/finite-radon-measurement-v4.json';manifest_path.parent.mkdir(parents=True);manifest_path.write_text('{}')
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
    route=ROUTES['frt2'];binding=tmp_path/'binding.json';binding.write_text('{}');terminal=tmp_path/'terminal.json'
    done=dict(status='PASS_REAL_CLI_STOP_BEFORE_BODY',route='frt2',api=route['api'],root_binding_sha256=PROBE.sha(binding),counts=dict(original=1,native=1),timed_batches=0,sampler_body_calls=0,outer_state_restored=True,poststop_source_authority_unchanged=True)
    terminal.write_text(json.dumps(done));summary=dict(status='PASS',postbatch_offline_identity_verified=True,routes=[done,{'second_cohort':True}])
    if fault=='status':manifest['status']='PROSPECTIVE_UNBOUND_NO_MEASUREMENT'
    elif fault=='cases':manifest['families']['finite_radon_frt2_v1']['case_ids']=[]
    elif fault=='workers':summary['postbatch_offline_identity_verified']=False
    elif fault=='losses':summary['routes']=[]
    call=lambda:PROBE.scoped_measurement(manifest,route,'parent-sha',{'binding':str(binding)},terminal,summary)
    if fault=='none':
        actual=call();assert actual['execution_scope']==DATA['execution_scope']and actual['families']==DATA['families']
        assert actual['completed_first_cohort']==dict(path=str(terminal.resolve()),sha256=PROBE.sha(terminal),binding_sha256=PROBE.sha(binding))
    else:
        with pytest.raises(AssertionError):call()


def test_current280_source_projection_preserves_prior_controller_and_cases():
    assert DATA['prospective_source']['revision']=='ab7872a5483a5d5e3660ba20fdd873d78463a1b5'
    assert tuple(DATA['prospective_source'][k]for k in('image_count','functions','product_members','SDK_members','SDK_families','SDK_checks'))==(280,283,226,107,37,74)
    predecessor=DATA['predecessor_controller'];raw=(ROOT/predecessor['contract_path']).read_bytes()
    assert H(raw)==predecessor['contract_raw_sha256'];old=json.loads(raw)
    assert DATA['execution_scope']==old['execution_scope']and DATA['helper']==old['helper']
    for name,route in ROUTES.items():
        assert route['cases']==old['first_cohort']['routes'][name]['cases']
        assert route['argv']==old['first_cohort']['routes'][name]['argv']
    for name,row in DATA['controller_inverse'].items():
        current=__import__('tools.join_source_context',fromlist=['rolling_predecessor_raw_bytes']).rolling_predecessor_raw_bytes(ROOT,name,(ROOT/name).read_bytes())
        assert H(current)==row['current_raw_sha256']and current.decode()==row['current']
        assert H(row['predecessor'].encode())==row['predecessor_raw_sha256']
    assert all(DATA[k]is None for k in('prepared_source','SDK_members','core_sha256','runtime_sha256','deadline_utc','benchmark_fingerprint'))


def test_actual_current_driver_nested_metadata_and_explicit_pass_interfaces():
    source=(ROOT/'benchmarks/finite_radon_execution_v1.py').read_text();tree=ast.parse(source)
    execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
    report=next(n for n in ast.walk(execute)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='dict'and any(k.arg=='metadata'for k in n.keywords))
    metadata=next(k.value for k in report.keywords if k.arg=='metadata')
    assert any(k.arg=='source_before'and ast.unparse(k.value)=='source_before'for k in metadata.keywords)
    assert all(k.arg is not None for k in metadata.keywords)
    assert DATA['current_interface']['report_source']=='metadata.source_before'
    assert DATA['driver_context']=='coverage/contracts/finite-radon-benchmark-execution-v4.json'
    for route in ROUTES.values():assert PROBE.sampler_order_scope(ROOT,route['argv'])==1
