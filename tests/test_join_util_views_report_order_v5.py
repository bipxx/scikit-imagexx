"""Synthetic saved reports/journals only; no real-data replay or target body."""
import ast,copy,hashlib,importlib.util,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
CONTEXT=json.loads((ROOT/'coverage/contracts/join-util-views-report-order-v5.json').read_bytes())
spec=importlib.util.spec_from_file_location('offline_report_v5',ROOT/'tools/validate_join_util_views_reports_v5.py')
TOOL=importlib.util.module_from_spec(spec);spec.loader.exec_module(TOOL)
H=lambda b:hashlib.sha256(b).hexdigest()


def test_exact_two_line_inverse_and_frozen_actual_sampler_callers():
    text=(ROOT/'tools/validate_join_util_views_reports_v5.py').read_text()
    fn=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='validate_reports');current=ast.get_source_segment(text,fn)
    assert H(current.encode())==CONTEXT['corrected_function_source_sha256']
    x,y=CONTEXT['inverse']['current'],CONTEXT['inverse']['predecessor'];assert current.count(x)==1
    original=current.replace(x,y,1);assert H(original.encode())==CONTEXT['original_function_source_sha256']
    old=(ROOT/CONTEXT['original_validator_path']).read_text();old_fn=next(n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef)and n.name=='validate_reports')
    assert original==ast.get_source_segment(old,old_fn)
    for name,pin in CONTEXT['source_pins_canonical'].items():assert H((ROOT/name).read_bytes().replace(b'\r\n',b'\n'))==pin
    sampler=(ROOT/'tools/montage_sustained.py').read_text();method=next(n for n in ast.parse(sampler).body if isinstance(n,ast.FunctionDef)and n.name=='measure')
    assert H((ast.get_source_segment(sampler,method)+'\n').encode())==CONTEXT['measurement_method_sha256']
    assert 'round_id + case_index + pass_index'in ast.unparse(method)
    for name,offset in (('next_util_execution',0),('label_pair_metrics',1),('find_boundaries',1),('ellipse_perimeter',1)):
        calls=[n for n in ast.walk(ast.parse((ROOT/f'benchmarks/{name}.py').read_bytes()))if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='measure']
        assert len(calls)==1
        keywords={k.arg:k.value for k in calls[0].keywords}
        assert isinstance(keywords['pass_index'],ast.Constant)and keywords['pass_index'].value==offset
        assert isinstance(keywords['case_index'],ast.Name)and keywords['case_index'].id=='index'
    imports=[n for n in ast.walk(ast.parse(text))if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert not any((getattr(n,'module','')or'').split('.')[0]in('numpy','scipy','skimage','skimagexx')for n in imports)
    assert not any(isinstance(n,ast.FunctionDef)and n.name in('measure','run_measurements','prepare_host_helper','run_cohorts')for n in ast.walk(ast.parse(text)))


def synthetic(tmp_path,kind,observed_offset=None):
    """Two cases exercise both case-index phases; times are invented metadata."""
    snapshot=tmp_path/'snapshot';results=tmp_path/'results';snapshot.mkdir();results.mkdir()
    def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    save(snapshot/'coverage/contracts/next-util-benchmark-execution-v2.json',{'method':{'method_sha256':'synthetic-method'}})
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
def test_correct_source_order_and_both_case_indices_pass(tmp_path,monkeypatch,kind):
    args=synthetic(tmp_path,kind);writes=[];monkeypatch.setattr(TOOL,'write_json',lambda path,value:writes.append((path,value)))
    TOOL.validate_reports(*args)
    assert len(writes)==1 and writes[0][0]==args[1]/'summary.json'
    assert writes[0][1]['families']['synthetic']==dict(raw_samples=28,wins=[],losses=['synthetic-0','synthetic-1'],ties=[])
    assert not(args[1]/'summary.json').exists()


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


def test_authority_locations_relocate_in_memory_without_byte_change(tmp_path):
    values={};before={}
    for role,name in (('binding','binding.json'),('go','root-go.json'),('measurement','measurement.json')):
        p=tmp_path/'authority-route'/name;p.parent.mkdir(exist_ok=True);p.write_bytes(b'{"host":"/home/runner/original","unchanged":true}\r\n');before[p]=p.read_bytes();values[role]='/home/runner/results/authority-route/'+name
    (tmp_path/'cohort-authorities.json').write_text(json.dumps({'route':values}))
    mapped=TOOL.local_authorities(tmp_path,{'first_cohort':{'routes':{'route':{}}}})
    assert mapped['route']['binding']==str((tmp_path/'authority-route/binding.json').resolve())
    assert all(p.read_bytes()==b and TOOL.sha(p)==H(b)for p,b in before.items())


def test_summary_sink_only_new_output_and_root_admission_required(tmp_path,monkeypatch):
    inputs=tmp_path/'failed';outputs=tmp_path/'successor';inputs.mkdir();outputs.mkdir();monkeypatch.setattr(TOOL,'_READ_ONLY',inputs);monkeypatch.setattr(TOOL,'_OUTPUT',outputs)
    TOOL.write_json(inputs/'summary.json',{'synthetic':True});assert not(inputs/'summary.json').exists()and json.loads((outputs/'summary.json').read_bytes())=={'synthetic':True}
    with pytest.raises(FileExistsError):TOOL.write_json(inputs/'summary.json',{})
    with pytest.raises(AssertionError):TOOL.write_json(inputs/'other.json',{})
    monkeypatch.setattr(sys,'argv',['offline','--snapshot',str(tmp_path),'--results',str(inputs),'--output',str(tmp_path/'never')])
    with pytest.raises(SystemExit):TOOL.main()
    assert not(tmp_path/'never').exists()
