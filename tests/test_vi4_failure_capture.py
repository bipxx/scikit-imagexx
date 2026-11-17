"""Complete inert VI4 catcher controls; no scientific imports or target calls."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import shutil
import sys
from types import SimpleNamespace
import pytest
from vi_public_matmul_source_view import _vi_2270_source_epoch

ROOT=Path(__file__).resolve().parents[1]
CONTEXT='coverage/contracts/vi4-failure-capture-v2.json'
DATA=json.loads((ROOT/CONTEXT).read_bytes())


def load(name='tools/vi4_failure_capture.py'):
    spec=importlib.util.spec_from_file_location('_inert_vi4_'+Path(name).stem,ROOT/name)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_unbound_and_exact_immutable_dependencies():
    module=load();assert module.context(ROOT)==DATA
    with pytest.raises(AssertionError,match='UNBOUND'):module.read_manifest(ROOT)
    assert DATA['producer']is DATA['prepared_source']is DATA['core_sha256']is DATA['runtime_sha256']is None
    assert DATA['capture']['failure_only_budget']['extra_fegetenv_max']==3
    assert DATA['capture']['failure_only_budget']['extra_restore_max']==2
    assert set(DATA['capture']['success_extra_budget'].values())=={0}


def test_v2_current_source_binding_preserves_frozen_capture_scope():
    module=load();capture=DATA['capture'];binding=capture['source_binding_successor']
    previous=binding['predecessor_context'];raw=(ROOT/previous['path']).read_bytes()
    assert previous==dict(path='coverage/contracts/vi4-failure-capture-v1.json',sha256='8dcde72d7f4f3b97782c31ca09690b5d2bd7963025069cc4735060519f64ed53')
    assert hashlib.sha256(raw).hexdigest()==previous['sha256'];old=json.loads(raw)
    assert binding['execution_authority']is None
    for key in('controller_root_admission','created_utc','deadline_utc','cohort_GO','measurement_GO'):
        assert DATA['execution_authority'][key]is None
    for key in('cases','phase_lines','phase_rule','envelopes','failure_only_budget','success_extra_budget','campaign_budget','failure_outputs','unavailable','cold_source_projection'):
        assert capture[key]==old['capture'][key],key
    provider=json.loads((ROOT/module.PROVIDER).read_bytes())
    assert module.PROVIDER=='coverage/contracts/vi-public-log2-provider-benchmark-v2.json'
    assert DATA['benchmark_source_sha256']==provider['source_files_canonical_sha256']
    for name,pin in DATA['benchmark_source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin,name
    for name,pin in DATA['first_cohort']['source_files_raw_sha256'].items():
        assert module.sha(ROOT/name)==pin,name
    assert capture['provider_context']==DATA['families'][capture['report_family']]['contract']==module.PROVIDER
    route=DATA['first_cohort']['routes']['vi']
    assert route['contract']==route['driver_route']['contract']==module.PROVIDER
    assert capture['isolated_probe_bindings']['CONTRACT']==CONTEXT
    assert capture['isolated_probe_bindings']['DRIVER_CONTEXT']==module.PROVIDER


def test_fresh_namespace_preserves_all_old_bodies_and_modules():
    module=load();old=load('tools/vi_first_cohort_probe.py');current=module.bound_probe(ROOT)
    assert old.CONTRACT=='coverage/contracts/vi-measurement-v5.json'
    assert old.DRIVER_CONTEXT=='coverage/contracts/variation-of-information-benchmark-execution-v6.json'
    assert current.CONTRACT==CONTEXT and current.DRIVER_CONTEXT==module.PROVIDER
    for name in('main','verify_source_state','verify_helper_state','validate_native_entry','finish_probe','prepare_cohorts','scoped_measurement','measurement_commands','validate_reports'):
        assert getattr(current,name).__code__.co_code==getattr(old,name).__code__.co_code,name
    oldtext=(ROOT/'tools/vi_first_cohort_probe.py').read_text()
    tree=ast.parse(oldtext);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='scoped_measurement')
    expected=ast.get_source_segment(oldtext,node).replace("family=Path(route['driver']).stem;admitted=manifest['families'][family]","family=manifest['capture']['report_family'];admitted=manifest['families'][family]")
    actual=(ROOT/'tools/vi4_failure_capture.py').read_text();node=next(n for n in ast.parse(actual).body if isinstance(n,ast.FunctionDef)and n.name=='scoped_measurement')
    assert ast.get_source_segment(actual,node)==expected


def test_three_cold_source_projections_reconstruct_exact_old_assertions():
    current=(ROOT/'tools/vi4_failure_capture.py').read_text();old=(ROOT/'tools/vi_first_cohort_probe.py').read_text()
    for name,row in DATA['capture']['cold_source_projection'].items():
        select=lambda text:ast.get_source_segment(text,next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name==name))
        value=select(current);assert hashlib.sha256(value.encode()).hexdigest()==row['current_source_sha256']
        for part in reversed(row['inverse']):
            assert value[part['start']:part['end']]==part['current'];value=value[:part['start']]+part['predecessor']+value[part['end']:]
        assert value==select(old)and hashlib.sha256(value.encode()).hexdigest()==row['predecessor_source_sha256']


def fake_array(label,raw,shape=(2,),kind='f',itemsize=8,events=None,bad=False,objects=False):
    def tobytes(self):
        if events is not None:events.append(label)
        if bad:raise ValueError('bad-output-codec')
        return self.raw
    cls=type('ndarray',(),dict(__module__='numpy',tobytes=tobytes))
    value=cls();value.raw=raw;value.dtype=SimpleNamespace(str='<'+kind+str(itemsize),kind=kind,itemsize=itemsize,hasobject=objects,descr=[('',kind+str(itemsize))],metadata=None)
    value.shape=shape;value.strides=(shape[-1]*itemsize,itemsize)if len(shape)==2 else(itemsize,)
    value.nbytes=len(raw);value.flags=SimpleNamespace(owndata=True,writeable=True,c_contiguous=True,f_contiguous=False);value.base=None
    return value


def observed_error(index,line,*,cohort=False,bad_expected=False,restore_failure=False,execute_root=None,probe_root=None):
    if execute_root is None:execute_root=ROOT
    if probe_root is None:probe_root=ROOT
    events=[];spec=DATA['capture']['cases'][index];n=spec['shape'][0]*spec['shape'][1];state=[b'O'*32 if cohort else(b'S'*32 if line>141 else b'C'*32)]
    def save():events.append('save');return state[0]
    def restore(saved):
        events.append('restore');state[0]=bytes(saved)
        if restore_failure:raise RuntimeError('restore-secondary')
        assert save()==bytes(saved)
    common=runpy.run_path(str(ROOT/'benchmarks/next_util_execution.py'))
    values=dict(spec=spec,index=index,api='metrics.variation_of_information',args=tuple(fake_array('input'+str(i),bytes(n*8),tuple(spec['shape']),'i',events=events)for i in range(2)),np=None,common=SimpleNamespace(array_state=common['array_state']),expected=fake_array('expected',bytes(16),events=events,bad=bad_expected),actual=fake_array('actual',bytes(16),events=events),reference_before={'status':0},native_before={'status':0},reference_FP={'status':32},native_FP={'status':32},reference_notices=[],native_notices=[],returns=['owned'],requirements={'owned':1},accepted=[True],suffix=[True],before=[{'prior':'hash'}],caller=b'C'*32,sample_state=b'S'*32,save=save,restore=restore,rows=[{'case_id':v['id']}for v in DATA['capture']['cases'][:index]])
    primary=AssertionError('strict-word-or-dtype');text=['def execute():']+['    '+key+'=payload['+repr(key)+']'for key in values]
    assert len(text)<line
    text+=['']*(line-len(text)-1)+['    raise primary']
    namespace=dict(payload=values,primary=primary);exec(compile('\n'.join(text),str(execute_root/'benchmarks/variation_of_information_execution.py'),'exec'),namespace)
    entry=namespace['execute']
    if cohort:
        code="def main():\n    outer=payload['outer']\n    save=payload['save']\n    restore=payload['restore']\n    counts={'original':1,'native':1}\n    restored=False\n    try:entry()\n    finally:\n        restore(outer)\n        restored=True\n"
        ns=dict(payload=dict(outer=b'O'*32,save=save,restore=restore),entry=entry);exec(compile(code,str(probe_root/'tools/vi_first_cohort_probe.py'),'exec'),ns);entry=ns['main']
    try:entry()
    except BaseException as error:
        events.clear();return error,events,values
    raise AssertionError('Fixture must raise')


@pytest.mark.parametrize('index',range(4))
@pytest.mark.parametrize('fault',('dtype','word'))
def test_all_four_case_outputs_precede_large_input_hashes(tmp_path,index,fault):
    module=load();error,events,values=observed_error(index,136)
    if fault=='dtype':
        values['actual'].dtype.str='<f4';values['actual'].dtype.itemsize=4;values['actual'].nbytes=8;values['actual'].raw=bytes(8)
    else:values['actual'].raw=b'\x01'+bytes(15)
    path=tmp_path/'failure.json';module.capture_failure(error,ROOT,'measurement',path,DATA['capture']);record=json.loads(path.read_bytes())
    assert record['expected']['words']['raw_hex']==bytes(16).hex()
    assert record['actual']['words']['raw_hex']==values['actual'].raw.hex()
    assert record['actual']['words']['raw_hex']!=record['expected']['words']['raw_hex']
    assert events.index('expected')<events.index('actual')<events.index('input0')<events.index('input1')
    assert record['case']==DATA['capture']['cases'][index]and record['index']==index
    assert len(record['input_after'])==2 and values['args'][0].nbytes in(131072,524288)
    assert events.count('save')==3 and events.count('restore')==2
    assert record['FP']['before_codecs']is True and record['FP']['after_codecs']is True


@pytest.mark.parametrize('line',(110,124,125,128,130))
def test_later_case_early_failure_never_attributes_stale_outputs(tmp_path,line):
    module=load();error,events,_=observed_error(3,line);path=tmp_path/'failure.json'
    module.capture_failure(error,ROOT,'measurement',path,DATA['capture']);record=json.loads(path.read_bytes())
    assert 'words'not in record['actual']
    assert ('words'in record['expected'])==(line>125)
    assert ('input_after'in record)==(line>110)
    assert record['completed_case_ids']==[s['id']for s in DATA['capture']['cases'][:3]]
    if line<=125:assert 'reference_FP'not in record
    assert 'native_FP'not in record


@pytest.mark.parametrize('phase,line,boundary',(('cohort',136,'cohort-outer'),('measurement',136,'caller'),('measurement',142,'sample')))
def test_applicable_envelope_and_timed_value_label(tmp_path,phase,line,boundary):
    module=load();error,events,_=observed_error(1,line,cohort=phase=='cohort');path=tmp_path/'failure.json'
    module.capture_failure(error,ROOT,phase,path,DATA['capture']);record=json.loads(path.read_bytes())
    assert record['FP']['boundary']==boundary and record['FP']['before_codecs']is True
    assert record['execute_caller_raw32']==(b'C'*32).hex()
    if phase=='cohort':assert record['probe_restored']is True and record['FP']['saved_raw32']==(b'O'*32).hex()
    if line==142:assert record['output_scope']=='earlier-untimed-pair'and record['execute_sample_raw32']==(b'S'*32).hex()


def test_real_split_controller_and_prepared_roots_keep_outer_and_values(tmp_path):
    module=load();controller=tmp_path/'controller';snapshot=tmp_path/'snapshot';codec=snapshot/module.CODEC;codec.parent.mkdir(parents=True);shutil.copyfile(ROOT/module.CODEC,codec)
    error,events,_=observed_error(2,136,cohort=True,execute_root=snapshot,probe_root=controller)
    configuration=dict(DATA['capture'],controller_root=str(controller),controller_context_sha256='a'*64);path=tmp_path/'failure.json'
    module.capture_failure(error,snapshot,'cohort',path,configuration);record=json.loads(path.read_bytes())
    assert record['value_frame_found']and record['probe_restored']and record['FP']['boundary']=='cohort-outer'
    assert record['capture_context_sha256']=='a'*64 and record['expected']['words']['raw_hex']==bytes(16).hex()


def test_one_bad_codec_retains_counterpart_and_mask(tmp_path,monkeypatch):
    monkeypatch.setenv('NPY_DISABLE_CPU_FEATURES','AVX512F');module=load();error,events,_=observed_error(2,136,bad_expected=True);path=tmp_path/'failure.json'
    module.capture_failure(error,ROOT,'measurement',path,DATA['capture']);record=json.loads(path.read_bytes())
    assert 'unavailable'in record['expected']and'words'in record['actual']
    assert record['NPY_DISABLE_CPU_FEATURES']=='AVX512F'and module.os.environ['NPY_DISABLE_CPU_FEATURES']=='AVX512F'


@pytest.mark.parametrize('fault',('oversize','object','non-array'))
def test_bad_output_descriptor_does_not_hide_the_other_result(tmp_path,fault):
    module=load();error,events,values=observed_error(2,136)
    if fault=='oversize':values['actual'].raw=bytes(64);values['actual'].nbytes=64
    elif fault=='object':values['actual'].dtype.hasobject=True
    else:values['actual'].__class__.__module__='unsupported'
    path=tmp_path/'failure.json';module.capture_failure(error,ROOT,'measurement',path,DATA['capture']);record=json.loads(path.read_bytes())
    assert 'words'in record['expected']and'unavailable'in record['actual']
    assert 'descriptor'in record['actual']and'actual'not in events


def test_failed_probe_restore_is_explicit_and_never_claims_C1(tmp_path,capsys):
    module=load();error,events,_=observed_error(0,136,cohort=True,restore_failure=True);path=tmp_path/'failure.json'
    module.capture_failure(error,ROOT,'cohort',path,DATA['capture']);record=json.loads(path.read_bytes())
    assert record['probe_restored']is False and any(n.get('stage')=='FP-after-codecs'for n in record['notes'])
    assert 'VI4_CAPTURE_FINAL_RESTORE_FAIL'in capsys.readouterr().err
    assert events.count('restore')==2


@pytest.mark.parametrize('fault',('writer','restore','both'))
def test_secondary_failures_preserve_original_exception_and_restore_budget(tmp_path,monkeypatch,fault,capsys):
    module=load();primary,events,_=observed_error(0,136,restore_failure=fault!='writer')
    def invoke(*a,**k):raise primary
    monkeypatch.setattr(module,'bound_probe',lambda root:SimpleNamespace(main=invoke))
    if fault!='restore':monkeypatch.setattr(module,'write_json',lambda *a:(_ for _ in()).throw(OSError('writer-secondary')))
    argv=sys.argv;path=list(sys.path)
    with pytest.raises(AssertionError)as found:module.run_catching(ROOT/'tools/vi_first_cohort_probe.py',['probe'],ROOT,'cohort',tmp_path/'failure.json',DATA['capture'])
    assert found.value is primary and sys.argv is argv and sys.path==path
    assert events.count('restore')==2 and events.count('save')<=3
    assert 'VI4_CAPTURE_'in capsys.readouterr().err


def test_success_has_no_capture_FP_codec_or_profiler_work(tmp_path,monkeypatch):
    module=load();events=[];argv=sys.argv;path=list(sys.path);script=tmp_path/'driver.py'
    def run(name,**kwargs):events.append((name,list(sys.argv),sys.path[0],kwargs));return {'success':True}
    monkeypatch.setattr(module.runpy,'run_path',run)
    monkeypatch.setattr(module,'capture_failure',lambda *a:pytest.fail('Success entered capture'))
    monkeypatch.setattr(sys,'setprofile',lambda *a:pytest.fail('Success changed profiler'))
    assert module.run_catching(script,[str(script),'--threads','1'],ROOT,'measurement',tmp_path/'never.json',DATA['capture'])=={'success':True}
    assert len(events)==1 and events[0][1]==[str(script),'--threads','1']and events[0][2]==str(script.parent)
    assert not(tmp_path/'never.json').exists()and sys.argv is argv and sys.path==path


@pytest.mark.parametrize('code,post_error,writer_error',((0,False,False),(7,False,False),(7,True,False),(0,True,False),(7,False,True)))
def test_actual_cold_child_always_postchecks_and_preserves_nonzero(tmp_path,monkeypatch,code,post_error,writer_error):
    module=load();results=tmp_path/'results';results.mkdir();binding=results/'binding.json';binding.write_text(json.dumps({'output_root':str(results)}));events=[]
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:(events.append('child')or SimpleNamespace(returncode=code)))
    def post(*a):
        events.append('post')
        if post_error:raise ValueError('identity-failed')
    monkeypatch.setattr(module,'_post_identity',post)
    if writer_error:
        writer=module.write_json
        def write(path,value):
            if str(path).endswith('vi-exit.json'):raise OSError('writer-failed')
            writer(path,value)
        monkeypatch.setattr(module,'write_json',write)
    call=lambda:module._child(ROOT,tmp_path,results,{},'python','vi','measurement',['python','driver.py','--root-binding',str(binding)])
    if code:
        with pytest.raises(SystemExit)as found:call()
        assert found.value.code==code
    elif post_error:
        with pytest.raises(ValueError):call()
    else:call()
    assert events==['child','post']


@pytest.mark.parametrize('fault',('none','parent-bytes','runtime-bytes','verifier'))
def test_actual_postidentity_checks_raw_parent_runtime_and_verifier_once(tmp_path,monkeypatch,fault):
    module=load();controller=tmp_path/'controller';snapshot=tmp_path/'snapshot';results=tmp_path/'results';results.mkdir();(tmp_path/'stage').mkdir()
    runtime=tmp_path/'stage/runtime.json';runtime.write_text('{}');manifest={'runtime_sha256':module.sha(runtime)}
    parent=controller/CONTEXT;parent.parent.mkdir(parents=True);parent.write_text(json.dumps(manifest));pin=module.sha(parent);calls=[]
    if fault=='parent-bytes':parent.write_bytes(parent.read_bytes()+b' ')
    elif fault=='runtime-bytes':runtime.write_bytes(runtime.read_bytes()+b' ')
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:(calls.append(a)or SimpleNamespace(returncode=7 if fault=='verifier'else 0)))
    invoke=lambda:module._post_identity(controller,snapshot,results,manifest,'python','vi',pin)
    if fault=='none':invoke()
    else:
        with pytest.raises(AssertionError):invoke()
    assert len(calls)==(0 if fault in('parent-bytes','runtime-bytes')else 1)
    if calls:assert json.loads((results/'vi-identity-after-exit.json').read_bytes())['exit']==(7 if fault=='verifier'else 0)


@pytest.mark.parametrize('fault',('none','status','route','source','counts','timed_batches','sampler_body_calls','outer_state_restored','poststop_source_authority_unchanged','root_binding_sha256'))
def test_actual_cohort_acceptance_keeps_every_old_terminal_gate(tmp_path,monkeypatch,fault):
    module=load();binding=tmp_path/'binding.json';binding.write_text('{}');folder=tmp_path/'first-cohort/vi';folder.mkdir(parents=True)
    manifest=json.loads(json.dumps(DATA));manifest['first_cohort']['source_revision']='a'*40
    terminal=dict(status='PASS_REAL_CLI_STOP_BEFORE_BODY',route='vi',source='a'*40,counts=dict(original=1,native=1),timed_batches=0,sampler_body_calls=0,outer_state_restored=True,poststop_source_authority_unchanged=True,root_binding_sha256=module.sha(binding))
    if fault!='none':terminal[fault]={'counts':{'original':2,'native':1},'outer_state_restored':False,'poststop_source_authority_unchanged':False}.get(fault,'wrong')
    (folder/'terminal.json').write_text(json.dumps(terminal));calls=[];monkeypatch.setattr(module,'_child',lambda *a:calls.append(a))
    authority={'vi':{'binding':str(binding),'go':'inert-root-go'}}
    if fault=='none':module.run_cohorts(ROOT,tmp_path,manifest,'python',authority)
    else:
        with pytest.raises(AssertionError):module.run_cohorts(ROOT,tmp_path,manifest,'python',authority)
    assert len(calls)==1
    assert (tmp_path/'first-cohort-summary.json').exists()==(fault=='none')


def test_actual_complete4_cold_route_reuses_old_scope_and_refuses_replay(tmp_path,monkeypatch):
    module=load();probe=module.bound_probe(ROOT);controller=tmp_path/'controller';snapshot=tmp_path/'snapshot';results=tmp_path/'results';results.mkdir();(controller/'coverage/contracts').mkdir(parents=True)
    manifest=json.loads(json.dumps(DATA));manifest['status']='ROOT_AUTHORIZED_ONE_INITIAL_COMPARISON';(controller/CONTEXT).write_text(json.dumps(manifest))
    binding=results/'binding.json';binding.write_text('{}');authority={'binding':str(binding),'measurement':str(results/'measurement.json')}
    terminal=dict(status='PASS_REAL_CLI_STOP_BEFORE_BODY',api='metrics.variation_of_information',root_binding_sha256=module.sha(binding),counts=dict(original=1,native=1),timed_batches=0,sampler_body_calls=0,outer_state_restored=True,poststop_source_authority_unchanged=True)
    folder=results/'first-cohort/vi';folder.mkdir(parents=True);(folder/'terminal.json').write_text(json.dumps(terminal));(results/'first-cohort-summary.json').write_text(json.dumps(dict(status='PASS',postbatch_offline_identity_verified=True,routes=[terminal])))
    monkeypatch.setattr(module,'bound_probe',lambda root:probe);events=[]
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:events.append('before-verifier'))
    monkeypatch.setattr(module,'_child',lambda *a:events.append(a[-1]))
    module.run_measurements(snapshot,results,manifest,'python',{'vi':authority},'2099-01-01T00:00:00+00:00')
    command=events[1];assert events[0]=='before-verifier'and '--case'not in command and '--first-cohort'not in command
    assert command[command.index('--threads')+1]=='1'and command[command.index('--repeat')+1]=='7'
    assert command[1]==str(snapshot/'benchmarks/vi_public_log2_provider_consumer.py')
    scoped=json.loads(Path(authority['measurement']).read_bytes());assert scoped['families']==manifest['families']and scoped['completed_first_cohort']['binding_sha256']==module.sha(binding)
    with pytest.raises(AssertionError):module.run_measurements(snapshot,results,manifest,'python',{'vi':authority},'2099-01-01T00:00:00+00:00')
    assert len(events)==2


@pytest.mark.parametrize('fault',('outside','relative','collision','wrong-phase'))
def test_failure_output_containment_and_exclusive_paths(tmp_path,fault):
    module=load();binding={'output_root':str(tmp_path)};output=tmp_path/'vi.value-failure-v1.json';phase='measurement'
    if fault=='outside':output=tmp_path.parent/'outside.json'
    elif fault=='relative':binding['output_root']='relative'
    elif fault=='collision':output.write_text('retained')
    else:phase='unknown'
    with pytest.raises(AssertionError):module.failure_path(binding,phase,output)


def workflow_blocks(text):
    blocks=[];current=None
    for line in text.splitlines():
        if " <<'PY'"in line:current=[]
        elif current is not None and line=='          PY':blocks.append('\n'.join(current));current=None
        elif current is not None:assert line.startswith('          ');current.append(line[10:])
    assert current is None
    return blocks


def test_workflow_preserves_capacity_auth_offline_F1_and_all_case_launches():
    text=(ROOT/'.github/workflows/vi-initial-failure-capture.yml').read_text();old=(ROOT/'.github/workflows/vi-initial.yml').read_text()
    assert text.index('ROOT_PREPARATION_CAPACITY_BEGIN')<text.index('actions/checkout')<text.index('actions/artifacts/')
    blocks=workflow_blocks(text);before=workflow_blocks(old);assert blocks[0]==before[0]
    for block in blocks:compile(block,'<inert-new-workflow>','exec')
    for literal in("assert len(manifest['wheels']) == 14","manifest['source'] == frozen['prepared_source']","runtime']['sdk_sha256']","qualification['passed'] is True","probe['run_cohorts']","probe['run_measurements']","probe['validate_reports']"):
        assert literal in text
    assert "revision != '16a5b50193125199ad26cf5a89acec5babf5290c'"in text
    assert 'archive.unlink()'in text and 'if: always()'in text and 'path: results/'in text
    assert 'Retain capture source and text companions'in text and "['capture']['dependencies']"in text
    assert 'NPY_DISABLE_CPU_FEATURES='not in text
    # Scientific setup blocks differ only in the explicit new context/launcher.
    for index in(3,4,5):
        normalized=blocks[index].replace('tools/vi4_failure_capture.py','tools/vi_first_cohort_probe.py').replace(CONTEXT,'coverage/contracts/vi-measurement-v5.json')
        assert normalized==before[index]


@pytest.mark.parametrize('free',(1,30*1024**3))
def test_actual_capacity_block_refuses_low_space_before_transport(tmp_path,monkeypatch,free):
    import shutil
    monkeypatch.chdir(tmp_path);(tmp_path/'results').mkdir();monkeypatch.setenv('RUNNER_TEMP',str(tmp_path));monkeypatch.setattr(shutil,'disk_usage',lambda p:SimpleNamespace(free=free))
    text=(ROOT/'.github/workflows/vi-initial-failure-capture.yml').read_text();block=workflow_blocks(text)[0]
    if free<30*1024**3:
        with pytest.raises(AssertionError):exec(compile(block,'<actual-capacity>','exec'),{})
    else:exec(compile(block,'<actual-capacity>','exec'),{})
    assert json.loads((tmp_path/'results/preparation-capacity.json').read_bytes())['passed']==(free>=30*1024**3)
