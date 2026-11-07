"""Actual wrapper boundaries with inert arrays/targets only; scientific imports refused."""
import argparse
import ast
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('lbp_fp_diagnostic',ROOT/'tools/lbp_cohort_fp_diagnostic.py')
D=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(D)

class Dtype:
    hasobject=False
    metadata=None
    str='<f8'
    descr=[('',str)]

class ndarray:
    __module__='numpy'  # Inert controlled stand-in, no real NumPy/factory.
    def __init__(self,output=True,byte=b'A'):
        self.dtype=Dtype();self.shape=(256,257,3)if output else(256,257)
        self.strides=(6168,24,8)if output else(257,1)
        self.raw=byte*(1579008 if output else 65792);self.nbytes=len(self.raw)
    def tobytes(self):return self.raw

class FP:
    def __init__(self):self.raw=b'C'*32
    def save(self):return self.raw
    def restore(self,saved):self.raw=bytes(saved)

def fake_probe(tmp_path,fp,mismatch='post',codec=False,tail=None):
    path=ROOT/'benchmarks/mark_boundaries_execution.py'
    source=path.read_text();tree=ast.parse(source)
    assertion=next(n for n in ast.walk(tree)if isinstance(n,ast.Assert)and ast.get_source_segment(source,n)=='assert reference_before==native_before and reference_FP==native_FP')
    exact=ast.get_source_segment(source,assertion)
    counts=dict(original=0,native=0);errors=[];terminal=tmp_path/'terminal.json'
    expected,actual=ndarray(),ndarray(byte=b'B')
    if codec:actual.dtype.hasobject=True
    def original():counts['original']+=1;fp.raw=b'R'*32;return expected
    def native():counts['native']+=1;fp.raw=b'N'*32;return actual
    body='''def execute():
    args=(INPUT,32,33,16,17);options={'lbp_code':0,'alpha':.5};spec={'id':'gray-u8-C'}
    before=[{'payload':'inert-saved-input'}];caller=FP.save()
    reference_before={'control':895,'status':32,'mxcsr':8096}
    native_before=dict(reference_before)
    expected=ORIGINAL();reference_FP=dict(reference_before)
    actual=NATIVE();native_FP=dict(reference_before)
    if MISMATCH=='pre':native_before['status']=0
    if MISMATCH=='post':native_FP['status']=0
    returns=['_lbp_prepare']+['_lbp_binary']*24+['_lbp_assign']*8
    requirements={'get_num_threads':0,'_lbp_prepare':1,'_lbp_binary':24,'_lbp_assign':8};accepted=[True]
    try:
        ASSERTION
    finally:FP.restore(caller)
    PROBE.validate_native_entry(locals())
    raise STOP()
'''.replace('ASSERTION',exact)
    class Stop(BaseException):pass
    def validate(c):return c
    probe=SimpleNamespace(validate_native_entry=validate,__file__=str(ROOT/'tools/mark_lbp_first_cohort_probe.py'))
    ns=dict(INPUT=ndarray(False),FP=fp,ORIGINAL=original,NATIVE=native,MISMATCH=mismatch,PROBE=probe,STOP=Stop)
    exec(compile(body,str(path),'exec'),ns)
    main='''def main():
    counts=COUNTS;restored=False;outer=FP.save()
    try:
        try:EXECUTE()
        except STOP:pass
    except BaseException as error:ERRORS.append(error);raise
    finally:FP.restore(outer);restored=True
    if TAIL is not None:TAIL()
    TERMINAL.write_text(JSON.dumps(dict(status='PASS_REAL_CLI_STOP_BEFORE_BODY',counts=counts,
        sampler_body_calls=0,timed_batches=0,outer_state_restored=True,poststop_source_authority_unchanged=True)))
'''
    pns=dict(COUNTS=counts,FP=fp,EXECUTE=ns['execute'],STOP=Stop,ERRORS=errors,TERMINAL=terminal,JSON=json,TAIL=tail)
    exec(compile(main,probe.__file__,'exec'),pns);probe.main=pns['main']
    return probe,path,terminal,counts,errors,expected,actual

@pytest.mark.parametrize('mismatch',('pre','post'))
def test_actual_assertion_failure_keeps_immediate_FP_and_original_exception(tmp_path,mismatch):
    fp=FP();p,path,t,counts,errors,e,a=fake_probe(tmp_path,fp,mismatch)
    validate=p.validate_native_entry;oldargv=sys.argv;oldpath=list(sys.path);trace,profile=sys.gettrace(),sys.getprofile()
    with pytest.raises(AssertionError)as caught:D.run_diagnostic(p,['inert'],fp.save,fp.restore,tmp_path/'out.json',path,t)
    r=json.loads((tmp_path/'out.json').read_bytes())
    assert caught.value is errors[0]and counts==dict(original=1,native=1)
    assert r['probe_counts']==counts and r['probe_restored']is True
    assert r['strict_trace']['reference_before']==dict(control=895,status=32,mxcsr=8096)
    assert r['strict_trace']['native_before']['status']==(0 if mismatch=='pre'else 32)
    assert r['strict_trace']['native_FP']['status']==(0 if mismatch=='post'else 32)
    assert r['expected']['payload_sha256']==hashlib.sha256(e.raw).hexdigest()
    assert r['actual']['payload_sha256']==hashlib.sha256(a.raw).hexdigest()
    assert r['input_after']['nbytes']==65792 and r['driver_caller_raw32']==(b'C'*32).hex()
    assert r['own_before_raw32']==r['after_old_probe_raw32']==r['own_restored_raw32']==(b'C'*32).hex()
    assert r['own_full32_restored']is True and fp.raw==b'C'*32
    assert p.validate_native_entry is validate and sys.argv is oldargv and sys.path==oldpath
    assert sys.gettrace()is trace and sys.getprofile()is profile


def test_success_retains_actual_validated_values_instead_of_losing_nonreproduction(tmp_path):
    fp=FP();p,path,t,counts,_,e,a=fake_probe(tmp_path,fp,'none');validate=p.validate_native_entry
    r=D.run_diagnostic(p,['inert'],fp.save,fp.restore,tmp_path/'out.json',path,t)
    assert r['status']=='PASS_OLD_PROBE_SAMPLER_STOP_NO_TIMING'and r['value_frame_found']is True
    assert r['strict_trace']['reference_FP']==r['strict_trace']['native_FP']
    assert r['expected']['payload_sha256']==hashlib.sha256(e.raw).hexdigest()
    assert r['actual']['payload_sha256']==hashlib.sha256(a.raw).hexdigest()
    assert counts==dict(original=1,native=1)and r['terminal']['sampler_body_calls']==0
    assert p.validate_native_entry is validate and fp.raw==b'C'*32

@pytest.mark.parametrize('fault',('codec','writer','final','restore','fallback'))
def test_secondary_failure_never_replaces_primary_or_repeats_targets(tmp_path,monkeypatch,fault):
    fp=FP();p,path,t,counts,errors,_,_=fake_probe(tmp_path,fp,codec=fault=='codec')
    output=tmp_path/'out.json';writer=D.write_record
    if fault in('writer','fallback'):
        def broken(path,value):
            if Path(path)==output:output.write_text('{partial');raise OSError('primary writer')
            if fault=='fallback':raise OSError('fallback writer')
            return writer(path,value)
        monkeypatch.setattr(D,'write_record',broken)
    def final():
        if fault=='final':raise ValueError('authority mutated')
    calls=[]
    def restore(saved):
        calls.append(1)
        if fault=='restore'and len(calls)==1:raise ValueError('restore failure')
        fp.restore(saved)
    with pytest.raises(AssertionError)as caught:D.run_diagnostic(p,['inert'],fp.save,restore,output,path,t,final)
    assert caught.value is errors[0]and counts==dict(original=1,native=1)and fp.raw==b'C'*32
    if fault=='fallback':assert caught.value.__notes__ and output.read_text()=='{partial'
    else:
        r=json.loads((output.with_suffix('.failure.json')if fault=='writer'else output).read_bytes())
        assert r['probe_counts']==counts and r['strict_trace']['native_FP']['status']==0
        assert r.get('writer_error')or r['secondary_errors']

@pytest.mark.parametrize('fault',('shape','size','object','type'))
def test_bounded_codec_refuses_unsupported_objects(fault):
    v=ndarray()
    if fault=='shape':v.shape=(1,)
    if fault=='size':v.nbytes=1579009
    if fault=='object':v.dtype.hasobject=True
    if fault=='type':v=object()
    with pytest.raises(AssertionError):D.array_words(v,output=True)

@pytest.mark.parametrize('fault',('none','unbound','wrapper','binding','cohort','context','source','case','budget','deadline','existing'))
def test_external_GO_before_import_or_target_and_no_reciprocal_cycle(tmp_path,fault):
    root=tmp_path/'snapshot';root.mkdir();controller=tmp_path/'controller';(controller/'coverage/contracts').mkdir(parents=True)
    source=root/'source.txt';source.write_text('inert-source')
    context=dict(controller_dependencies={},prepared_dependencies={'source.txt':D.digest(source)},qualified_source='a'*40)
    cp=controller/D.CONTEXT;cp.write_text(json.dumps(context))
    binding=tmp_path/'binding.json';binding.write_text(json.dumps(dict(source_root=str(root),source_revision='a'*40)))
    oldgo=tmp_path/'oldgo.json';oldgo.write_text('{}');newgo=tmp_path/'newgo.json';output=tmp_path/'out.json'
    now=datetime.now(timezone.utc)
    go=dict(status='ROOT_AUTHORIZED_ONE_LBP_FP_DIAGNOSTIC',wrapper_sha256=D.digest(D.__file__),api='feature.draw_multiblock_lbp',case_id='gray-u8-C',binding_sha256=D.digest(binding),cohort_GO_sha256=D.digest(oldgo),context_sha256=D.digest(cp),maximum_original_topcalls=1,maximum_native_topcalls=1,sampler_body_calls=0,timed_batches=0,created_utc=(now-timedelta(minutes=1)).isoformat(),deadline_utc=(now+timedelta(minutes=30)).isoformat(),output_path=str(output),source_revision='a'*40,diagnostic_source='b'*40)
    if fault=='unbound':go['status']='UNBOUND'
    if fault in('wrapper','binding','context'):go[fault+'_sha256']='0'*64
    if fault=='cohort':go['cohort_GO_sha256']='0'*64
    if fault=='source':go['source_revision']='b'*40
    if fault=='case':go['case_id']='RGB-f32-F'
    if fault=='budget':go['maximum_native_topcalls']=2
    if fault=='deadline':go['deadline_utc']=(now-timedelta(seconds=1)).isoformat()
    if fault=='existing':output.write_text('old evidence')
    newgo.write_text(json.dumps(go));args=argparse.Namespace(binding=binding,root_go=oldgo,diagnostic_go=newgo,diagnostic_go_sha256=D.digest(newgo),output=output)
    if fault=='none':assert D.validate_authority(args,controller)[1]==root
    else:
        with pytest.raises(AssertionError):D.validate_authority(args,controller)
    assert 'diagnostic_go_sha256'not in json.loads(binding.read_bytes())


def run_blocks():
    blocks=[];current=None
    for line in (ROOT/'.github/workflows/lbp-cohort-fp-diagnostic.yml').read_text().splitlines():
        if line=='        run: |':current=[]
        elif current is not None:
            if line.startswith('          '):current.append(line[10:])
            else:blocks.append('\n'.join(current)+'\n');current=None
    if current is not None:blocks.append('\n'.join(current)+'\n')
    return blocks


def test_all_actual_inline_python_compiles_and_no_measurement_stage():
    blocks=run_blocks();assert len(blocks)==5
    for i,block in enumerate(blocks):
        lines=block.splitlines()
        if any(" - <<'PY'"in line for line in lines):
            start=next(j for j,line in enumerate(lines)if" - <<'PY'"in line);end=lines.index('PY',start+1)
            compile('\n'.join(lines[start+1:end])+'\n','actual-inline-'+str(i),'exec')
    workflow=(ROOT/'.github/workflows/lbp-cohort-fp-diagnostic.yml').read_text()
    assert 'default: UNBOUND'in workflow and 'ROOT_AUTHORIZED_ONE_LBP_FP_DIAGNOSTIC'in workflow
    assert "GITHUB_RUN_ATTEMPT'] == '1'"in workflow
    assert 'run_measurement'not in workflow and 'if: always()'in workflow and 'path: results/'in workflow
    assert workflow.index('UNBOUND diagnostic')<workflow.index('gh api "repos/')
    assert "raise SystemExit(done.returncode or checked.returncode)"in workflow
    assert "results/'postbatch-identity-exit.json'"in workflow


def test_frozen_dependency_closure_and_actual_source_only_single_call():
    context=json.loads((ROOT/D.CONTEXT).read_bytes())
    for group in ('controller_dependencies','prepared_dependencies'):
        for name,pin in context[group].items():assert D.digest(ROOT/name)==pin,name
    tree=ast.parse((ROOT/'tools/lbp_cohort_fp_diagnostic.py').read_bytes())
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run_diagnostic')
    assert sum(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='main'for n in ast.walk(run))==1
    assert context['actual_authority']is None and context['original_budget_now']==context['native_budget_now']==0
    assert context['future_budget']==dict(original=1,native=1,sampler_body=0,timing=0)
    assert 'sys.setprofile'not in ast.get_source_segment((ROOT/'tools/lbp_cohort_fp_diagnostic.py').read_text(),next(n for n in ast.walk(run)if isinstance(n,ast.FunctionDef)and n.name=='capture_validated_entry'))


@pytest.mark.parametrize('mismatch',('none','post'))
@pytest.mark.parametrize('cached',(False,True))
def test_actual_main_loads_existing_probe_once_and_restores_entry(tmp_path,monkeypatch,mismatch,cached):
    if cached:monkeypatch.setitem(sys.modules,'numpy',SimpleNamespace(INERT=True))
    fp=FP();probe,path,terminal,counts,errors,_,_=fake_probe(tmp_path,fp,mismatch)
    binding=dict(output_root=str(tmp_path))
    context=dict(driver_context='coverage/contracts/draw-multiblock-lbp-benchmark-execution-v5.json',probe='tools/mark_lbp_first_cohort_probe.py')
    checks=[]
    probe.read_manifest=lambda controller:(None,{})
    probe.verify_source_state=lambda b,m:None
    probe.verify_helper_state=lambda b,g:None
    (tmp_path/'go.json').write_text('{}')
    def authority(args,controller):checks.append(1);return binding,ROOT,context
    monkeypatch.setattr(D,'validate_authority',authority)
    common=SimpleNamespace(fp_controller=lambda root,c:(fp.save,fp.restore,None))
    old_spec=D.importlib.util.spec_from_file_location;old_module=D.importlib.util.module_from_spec
    def spec(name,path):
        value=common if name=='_lbp_diagnostic_common'else probe
        return SimpleNamespace(value=value,loader=SimpleNamespace(exec_module=lambda module:None))
    monkeypatch.setattr(D.importlib.util,'spec_from_file_location',spec)
    monkeypatch.setattr(D.importlib.util,'module_from_spec',lambda spec:spec.value)
    # Actual main expects the unchanged probe terminal path.
    folder=tmp_path/'lbp';folder.mkdir()
    original_main=probe.main
    def invoke():
        assert sys.argv[-2:]==['--route','lbp']
        original_main()
        (folder/'terminal.json').write_bytes(terminal.read_bytes())
    probe.main=invoke
    output=tmp_path/'output.json'
    monkeypatch.setattr(sys,'argv',['diagnostic','--binding',str(tmp_path/'binding.json'),'--root-go',str(tmp_path/'go.json'),'--diagnostic-go',str(tmp_path/'diagnostic-go.json'),'--diagnostic-go-sha256','0'*64,'--output',str(output)])
    if mismatch=='post':
        with pytest.raises(AssertionError):D.main()
    else:assert D.main()['value_frame_found']is True
    assert counts==dict(original=1,native=1)and len(checks)==2
    r=json.loads(output.read_bytes());assert r['strict_trace']['reference_FP']['status']==32
    assert fp.raw==b'C'*32


def inline_python(block):
    lines=block.splitlines();start=next(i for i,s in enumerate(lines)if" - <<'PY'"in s)
    return '\n'.join(lines[start+1:lines.index('PY',start+1)])+'\n'


def test_actual_workflow_UNBOUND_stops_before_fetch_or_helper(monkeypatch):
    monkeypatch.setenv('DIAGNOSTIC_AUTHORITY','UNBOUND')
    with pytest.raises(AssertionError,match='UNBOUND diagnostic'):
        exec(compile(inline_python(run_blocks()[0]),'actual-first-workflow-step','exec'),{})


@pytest.mark.parametrize('primary_exit',(0,17))
def test_actual_workflow_terminal_path_keeps_primary_exit_and_offline_verifier_once(tmp_path,monkeypatch,primary_exit):
    import runpy,subprocess
    controller=tmp_path/'controller';snapshot=tmp_path/'snapshot';results=tmp_path/'results'
    for folder in (controller/'tools',controller/'coverage/contracts',snapshot,results):folder.mkdir(parents=True,exist_ok=True)
    for name in ('lbp_only_comparison.py','lbp_cohort_fp_diagnostic.py'):(controller/'tools'/name).write_text('inert')
    (controller/'coverage/contracts/lbp-cohort-fp-diagnostic-v2.json').write_text('{}')
    (controller/'coverage/contracts/mark-lbp-measurement-v5.json').write_text(json.dumps(dict(producer=dict(revision='a'*40),runtime_sha256=hashlib.sha256(b'inert-runtime').hexdigest())))
    (results/'execution-clock.json').write_text(json.dumps(dict(created_utc='INERT_CREATED',deadline_utc='INERT_DEADLINE')))
    (results/'lbp-only-root-GO.json').write_text('{}');(results/'lbp-only-root-GO.sha256').write_text('0'*64)
    stage=tmp_path/'stage';stage.mkdir();(stage/'runtime.json').write_bytes(b'inert-runtime')
    binding=results/'binding.json';binding.write_text('{}');cohortgo=results/'cohortgo.json';cohortgo.write_text('{}')
    calls=[];prepared=[]
    def prepare(*args):prepared.append(args);return dict(binding=str(binding),go=str(cohortgo))
    monkeypatch.setattr(runpy,'run_path',lambda path:dict(prepare_cohort=prepare))
    def run(argv,**kwargs):calls.append(argv);return SimpleNamespace(returncode=primary_exit if len(calls)==1 else 0)
    monkeypatch.setattr(subprocess,'run',run)
    monkeypatch.setenv('GITHUB_WORKSPACE',str(tmp_path));monkeypatch.setenv('GITHUB_SHA','b'*40)
    with pytest.raises(SystemExit)as caught:exec(compile(inline_python(run_blocks()[-1]),'actual-final-workflow-step','exec'),{})
    assert caught.value.code==primary_exit and len(prepared)==1 and len(calls)==2
    assert calls[0][1]==str(controller/'tools/lbp_cohort_fp_diagnostic.py')
    assert 'verify'in calls[1]and json.loads((results/'diagnostic-exit.json').read_bytes())==dict(exit=primary_exit)
    assert json.loads((results/'postbatch-identity-exit.json').read_bytes())==dict(exit=0)
    go=json.loads((results/'diagnostic-root-GO.json').read_bytes())
    assert go['maximum_original_topcalls']==go['maximum_native_topcalls']==1 and go['sampler_body_calls']==go['timed_batches']==0
    assert '--measurement-go'not in calls[0]

@pytest.mark.parametrize('fault',('C1','terminal'))
@pytest.mark.parametrize('codec',(False,True))
def test_actual_old_poststop_finish_failure_uses_validated_refs_and_keeps_primary(tmp_path,monkeypatch,fault,codec):
    # Exercise frozen old finish_probe itself after execute has unwound.
    oldspec=importlib.util.spec_from_file_location('inert_old_probe_C1',ROOT/'tools/mark_lbp_first_cohort_probe.py')
    old=importlib.util.module_from_spec(oldspec);oldspec.loader.exec_module(old)
    folder=tmp_path/'old-finish';folder.mkdir()
    binding_path=tmp_path/'binding.json';go_path=tmp_path/'cohort-go.json'
    source=tmp_path/'source';source.mkdir();(source/'contract.json').write_text('{}')
    binding=dict(source_root=str(source),routes={'lbp':{'contract':'contract.json'}},base_contract_sha256=D.digest(source/'contract.json'))
    binding_path.write_text(json.dumps(binding))
    go=dict(probe_sha256=D.digest(old.__file__),deadline_utc='2999-01-01T00:00:00+00:00')
    go_path.write_text(json.dumps(go))
    primary=AssertionError('INERT_ACTUAL_C1_SOURCE_CHANGED')if fault=='C1'else OSError('INERT_ACTUAL_TERMINAL_WRITER')
    def source_check(b,m):
        if fault=='C1':raise primary
    monkeypatch.setattr(old,'verify_source_state',source_check)
    monkeypatch.setattr(old,'verify_helper_state',lambda b,g:None)
    writer=old.write_json
    def write(path,value):
        if fault=='terminal'and Path(path).name=='terminal.json':raise primary
        writer(path,value)
    monkeypatch.setattr(old,'write_json',write)
    def tail():
        old.finish_probe(folder,dict(route='lbp',counts=dict(original=1,native=1)),binding,binding_path,go_path,go,D.digest(binding_path),D.digest(go_path),True,dict(first_cohort=dict(routes=binding['routes'])))
    fp=FP();probe,path,terminal,counts,_,expected,actual=fake_probe(tmp_path,fp,'none',codec=codec,tail=tail)
    original_validate=probe.validate_native_entry;output=tmp_path/'diagnostic.json'
    with pytest.raises(type(primary))as caught:D.run_diagnostic(probe,['inert'],fp.save,fp.restore,output,path,terminal)
    assert caught.value is primary and counts==dict(original=1,native=1)
    record=json.loads(output.read_bytes())
    assert record['capture_origin']=='validated-entry-before-old-tail-failure'
    assert record['value_frame_found']is True and record['probe_restored']is True and record['probe_counts']==counts
    assert record['strict_trace']['reference_before']==record['strict_trace']['native_before']
    assert record['strict_trace']['reference_FP']==record['strict_trace']['native_FP']
    assert record['driver_caller_raw32']==(b'C'*32).hex()
    assert record['expected']['payload_sha256']==hashlib.sha256(expected.raw).hexdigest()
    if codec:
        assert record['secondary_errors'][0]['phase']=='value-codec'
        assert record['secondary_errors'][0]['type']=='AssertionError'
    else:
        assert record['actual']['payload_sha256']==hashlib.sha256(actual.raw).hexdigest()
        assert record['input_after']['nbytes']==65792 and record['secondary_errors']==[]
    assert record['primary_type']==type(primary).__name__ and record['primary_error']==str(primary)
    assert record['own_before_raw32']==record['after_old_probe_raw32']==record['own_restored_raw32']==(b'C'*32).hex()
    assert fp.raw==b'C'*32 and probe.validate_native_entry is original_validate
    assert record['target_calls_added_by_wrapper']==record['sampler_body_calls_authorized']==0
    if fault=='C1':assert json.loads((folder/'failure.json').read_bytes())['phase']=='poststop-source-authority'


def test_versioned_context_preserves_all_old_authority_and_dependency_fields():
    current=json.loads((ROOT/D.CONTEXT).read_bytes());previous=current['predecessor_context']
    raw=(ROOT/previous['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==previous['sha256']=='47432de249ee20eca1ad315ca64d1f1dbdc020c2e7854dcd38426f8400238b5e'
    old=json.loads(raw)
    assert {k:v for k,v in current.items()if k not in('schema','predecessor_context')}=={k:v for k,v in old.items()if k!='schema'}

    assert repr(previous['path']) in (ROOT/'.github/workflows/lbp-cohort-fp-diagnostic.yml').read_text(), 'Collector retains frozen predecessor context'
