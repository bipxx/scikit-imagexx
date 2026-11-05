"""Inert controller/schema/selector checks; no algorithm or timer execution."""
import ast
import builtins
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import types
import pytest

ROOT=Path(__file__).parents[1]

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

C=module('tools/region_stain_compare.py','region_stain_controller')
P=json.loads((ROOT/'coverage/contracts/region-stain-same-host-v1.json').read_bytes())


def test_imports_refuse_scientific_algorithms(monkeypatch):
    old=builtins.__import__
    def refused(name,*args,**kwargs):
        assert name.split('.')[0] not in ('numpy','scipy','skimage','skimagexx'),name
        return old(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',refused)
    module('tools/region_stain_compare.py','inert_controller')
    module('benchmarks/stain_f32_holdouts.py','inert_holdouts')


def test_fixed_contract_workflow_sources_and_pool_scope():
    assert P['passes']==['baseline','candidate','candidate','baseline'] and P['pairs']==[[0,1],[3,2]]
    assert [len(P[k]) for k in ('region_cases','half_cases','holdout_cases')]==[12,4,4]
    assert len({s['id'] for k in ('region_cases','half_cases','holdout_cases') for s in P[k]})==20
    assert P['sampling']==dict(threads=1,repeat=7,target_ms=100,warmup_ms=250,cohorts=20,samples_per_pass=280,samples_total=1120,deadline='required immutable UTC deadline; checked before every actualbatch')
    g=P['fixed_gates'];assert g['all20_candidate_over_prior_max']==1.05 and g['region_targets_candidate_over_prior_max']==.5 and g['stain_targets_candidate_over_prior_max']==.8
    context=json.loads((ROOT/'coverage/contracts/region-stain-controller-v1-context.json').read_bytes())['entries']
    for name,pin in P['controller_source_sha256'].items():
        raw=context[name]['utf8'].encode()if name in context else(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
        assert hashlib.sha256(raw).hexdigest()==pin
    workflow=context['.github/workflows/region-stain-same-host.yml']['utf8']
    assert "python-version: '3.11.17'" in workflow and 'if: always()' in workflow
    assert 'eea321b2a334ba51c475d7733ad7547fa24c553b' in workflow
    assert 'candidate_run:' in workflow and 'candidate_source:' in workflow and 'candidate_artifact:' in workflow and 'deadline_utc:' in workflow
    for name in C.POOLS:assert name+": '1'" in workflow
    tree=ast.parse((ROOT/'tools/region_stain_compare.py').read_text());execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
    source=ast.get_source_segment((ROOT/'tools/region_stain_compare.py').read_text(),execute)
    assert source.index('manifests_equal')<source.index('all-runtimes-preflight-complete.json')<source.index('enumerate(protocol[\'passes\']')
    assert 'ci_measurement_artifacts.py' in source and 'verify' in source and "'--require-hashes'" in source and "'--no-index'" in source
    assert 'CONTRACT=' not in source and 'write_text' not in source.split('for index,role in enumerate')[1].split('result=acceptance')[0]


def selection():
    b=P['baseline'];r=dict(id=int(b['run']),head_sha=b['source'],status='completed',conclusion='success',head_repository={'full_name':'bipxx/qupathxx'},path='.github/workflows/ci.yml',event='workflow_dispatch')
    a=dict(id=b['artifact'],name='ci-performance-minimum',expired=False,size_in_bytes=b['bytes'],digest='sha256:'+b['sha256'],workflow_run={'id':int(b['run']),'head_sha':b['source']})
    return r,{'total_count':1,'artifacts':[a]}


def test_exact_baseline_selection():
    r,a=selection();b=P['baseline'];assert C.validate_selection(r,a,'baseline',b['source'],b['run'],b['artifact'],'bipxx/qupathxx',P)==a['artifacts'][0]


@pytest.mark.parametrize('fault',('failed','wrong-kind','wrong-digest','wrong-source','expired','wrong-repo','pagination'))
def test_artifact_authentication_negative(fault):
    r,a=selection();b=P['baseline']
    if fault=='failed':r['conclusion']='failure'
    if fault=='wrong-kind':a['artifacts'][0]['name']='numpy-minimum-ciede2000'
    if fault=='wrong-digest':a['artifacts'][0]['digest']='sha256:'+'0'*64
    if fault=='wrong-source':a['artifacts'][0]['workflow_run']['head_sha']='0'*40
    if fault=='expired':a['artifacts'][0]['expired']=True
    if fault=='wrong-repo':r['head_repository']['full_name']='other/repo'
    if fault=='pagination':a['total_count']=2
    with pytest.raises(AssertionError):C.validate_selection(r,a,'baseline',b['source'],b['run'],b['artifact'],'bipxx/qupathxx',P)


@pytest.mark.parametrize('timed', (False,True,'ciede-no-repeat'))
def test_prepare_only_actual_step_admission(timed):
    names=['Install exact offline hash-locked runtime','Verify source, wheel, runtime, package and SDK before measurements','Verify measurement inputs remained immutable']
    steps=[{'name':n,'conclusion':'success'} for n in names]
    steps += [{'name':f'Run python benchmarks/closed_{index}.py --repeat 7','conclusion':'success' if timed is True and index==0 else 'skipped'} for index in range(34)]
    steps += [{'name':'Run python benchmarks/ciede2000.py --threads 1 --output build/ci-ciede2000.json','conclusion':'success' if timed=='ciede-no-repeat' else 'skipped'}]
    assert len([step for step in steps if step['name'].startswith('Run python benchmarks/')])==35
    jobs={'total_count':1,'jobs':[{'conclusion':'success','steps':steps}]}
    if timed:
        with pytest.raises(AssertionError):C.validate_prepare_jobs(jobs)
    else:C.validate_prepare_jobs(jobs)


def passes():
    ids=[s['id'] for k in ('region_cases','half_cases','holdout_cases') for s in P[k]]
    old={cid:{'original':10.,'cpp':10.} for cid in ids};new=copy.deepcopy(old)
    for cid in P['fixed_gates']['region_targets']:new[cid]['cpp']=4.
    for cid in P['fixed_gates']['stain_targets']:new[cid]['cpp']=7.
    return [copy.deepcopy(old),copy.deepcopy(new),copy.deepcopy(new),copy.deepcopy(old)]


def test_both_pairs_and_all40_fixed_gates():
    result=C.acceptance(passes(),P);assert result['decision']=='KEEP' and len(result['checks'])==40


@pytest.mark.parametrize('case',[s['id'] for k in ('region_cases','half_cases','holdout_cases') for s in P[k]])
@pytest.mark.parametrize('pair',(1,2))
def test_every_cohort_each_pair_rejects_regression(case,pair):
    data=passes();data[pair][case]['cpp']=11.
    assert C.acceptance(data,P)['decision']=='DISCARD'


def test_original_normalized_gate_cannot_be_hidden_by_direct_gain():
    data=passes();data[1][P['holdout_cases'][0]['id']]['original']=1.
    assert C.acceptance(data,P)['decision']=='DISCARD'


def sample_report(tmp_path,pass_index=1):
    events=[];samples=[[],[]];counts=[[],[]];records=[]
    def emit(f,phase,r,o,ns):
        common=dict(case_index=0,case_id='case',function=f,round=r,ordinal=o,loops=1)
        before=dict(common,event='batch_before',requested_phase=phase)
        after=dict(common,event='batch_after',phase=phase,elapsed_ns=ns,complete=True,accepted=phase=='sample',failure=None)
        events.extend([before,after]);records.append(after)
        if phase=='sample':samples[f].append(ns/1e6);counts[f].append(1)
    for f in (0,1):emit(f,'warmup',-1,f,250_000_000)
    orders=[]
    for r in range(7):
        order=[0,1] if (r+pass_index)%2==0 else [1,0];orders.append(order)
        for o,f in enumerate(order):emit(f,'sample',r,o,100_000_000)
    row=dict(case_id='case',threads=1,native_accepted=True,actual_C_returns=['entry'],samples_ms=samples,loops=counts,batch_records=records,warmup_elapsed_ns=[250_000_000]*2,sample_order=orders,reference_ms=100.,cpp_ms=100.,speedup=1.)
    events.append(dict(event='measure_returned',case_index=0,case_id='case',timing=copy.deepcopy(row)))
    report={'metadata':dict(threads=1,repeat=7,target_ms=100,warmup_calibration_min_ms=250,reference_pool_limits={k:'1' for k in C.POOLS},fallback_enabled=False,raw_samples=14),'results':[row]}
    path=tmp_path/'journal.jsonl';path.write_text('\n'.join(map(json.dumps,events))+'\n')
    return report,path,events


def test_raw_journal_medians_warmup_sample_order(tmp_path):
    report,path,events=sample_report(tmp_path);assert C.validate_report(report,path,['case'])=={'case':{'original':100.,'cpp':100.}}


@pytest.mark.parametrize('fault',('short-sample','warm','median','failure','extra','order','loops','missing-witness'))
def test_raw_journal_negative(tmp_path,fault):
    report,path,events=sample_report(tmp_path)
    if fault=='short-sample':events[5]['elapsed_ns']=99_000_000
    if fault=='warm':events[1]['elapsed_ns']=1
    if fault=='median':report['results'][0]['cpp_ms']=90.
    if fault=='failure':events[5]['complete']=False
    if fault=='extra':events+=events[:2]
    if fault=='order':events[5]['ordinal']=9
    if fault=='loops':events[5]['loops']=0
    if fault=='missing-witness':report['results'][0]['actual_C_returns']=[]
    path.write_text('\n'.join(map(json.dumps,events))+'\n')
    with pytest.raises(AssertionError):C.validate_report(report,path,['case'])


@pytest.mark.parametrize('spec',P['holdout_cases'],ids=[s['id'] for s in P['holdout_cases']])
def test_source_safe_holdout_factory_original_refused(spec):
    import numpy as np
    h=module('benchmarks/stain_f32_holdouts.py','factory_intake')
    reference=types.SimpleNamespace(rgb_from_hed=np.array([[.65,.7,.29],[.07,.99,.11],[.27,.57,.78]],dtype=np.float64))
    def denied(*a,**k):pytest.fail('no original or native algorithm authorized during fixture intake')
    reference.combine_stains=reference.separate_stains=denied
    image,matrix=h.fixture(spec,np,reference)
    assert image.dtype==np.dtype('float32') and image.shape==(512,512,3)
    assert image.flags.owndata and image.flags.c_contiguous and 0<=image.min()<=image.max()<=1
    assert matrix.shape==(3,spec['k']) and np.isfinite(matrix).all() and np.abs(matrix).max()<=1


@pytest.mark.parametrize('deadline',('2026-11-04T12:00:00','2026-11-04T12:00:00+01:00'))
def test_controller_rejects_non_UTC_before_any_execution(deadline,monkeypatch):
    monkeypatch.setattr(C,'execute',lambda args:pytest.fail('invaliddeadline reached controller execution'))
    with pytest.raises(AssertionError):
        C.main(['execute','--repository','bipxx/qupathxx','--candidate-source','0'*40,'--candidate-run','1','--candidate-artifact','1','--snapshots','unused','--results','unused','--deadline-utc',deadline])


COORDS=ROOT/'coverage/contracts/regionprops-coords-scaled-compare-v1.json'


def coords_passes():
    protocol=json.loads(COORDS.read_bytes())
    old={s['id']:dict(original=1.,cpp=1.)for s in protocol['coords_cases']}
    new=copy.deepcopy(old)
    for cid in protocol['fixed_gates']['target_ids']:new[cid]['cpp']=.8
    return [copy.deepcopy(old),copy.deepcopy(new),copy.deepcopy(new),copy.deepcopy(old)],protocol


def test_coords_six_case_protocol_and32_fixed_inequalities():
    data,p=coords_passes();result=C.acceptance(data,p)
    assert result['decision']=='KEEP' and len(result['checks'])==12
    assert sum(len(row['limits'])for row in result['checks'])==32
    benchmark=json.loads((ROOT/'coverage/contracts/regionprops-coords-scaled-benchmark-v1.json').read_bytes())
    assert p['optimization_version']==2 and p['mode']=='coords-scaled' and p['families']==['coords-scaled']
    assert p['coords_cases']==benchmark['cases'] and p['sampling']['samples_total']==336
    assert p['sampling']['warmup_groups']==48
    assert p['state'].startswith('PROSPECTIVE_')
    assert hashlib.sha256((ROOT/'coverage/contracts/regionprops-coords-scaled-benchmark-v1.json').read_bytes()).hexdigest()==p['benchmark_contract_sha256']
    for name,pin in p['controller_source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin


@pytest.mark.parametrize('case',range(6))
@pytest.mark.parametrize('pair',(1,2))
@pytest.mark.parametrize('fault',('direct','normalized'))
def test_coords_every_control_each_pair_is_strict(case,pair,fault):
    data,p=coords_passes();cid=p['coords_cases'][case]['id']
    if fault=='direct':data[pair][cid]['cpp']=1.06
    else:data[pair][cid]['original']=.5
    assert C.acceptance(data,p)['decision']=='DISCARD'


@pytest.mark.parametrize('target',(0,1))
@pytest.mark.parametrize('pair',(1,2))
@pytest.mark.parametrize('fault',('target-direct','target-original'))
def test_coords_targets_have_separate_direct_and_original_caps(target,pair,fault):
    data,p=coords_passes();cid=p['fixed_gates']['target_ids'][target]
    data[pair][cid]=dict(cpp=.91,original=2.)if fault=='target-direct'else dict(cpp=.8,original=.8)
    result=C.acceptance(data,p);assert result['decision']=='DISCARD'
    check=next(c for c in result['checks']if c['case_id']==cid and c['candidate_pass']==pair+1)
    assert sum(l['value']>l['limit']for l in check['limits'])==1


def test_coords_target_normalized_control_is105_not90():
    data,p=coords_passes()
    for cid in p['fixed_gates']['target_ids']:
        for i in (0,3):data[i][cid]=dict(cpp=.85,original=1.)
        for i in (1,2):data[i][cid]=dict(cpp=.75,original=.85)
    assert C.acceptance(data,p)['decision']=='KEEP'


@pytest.mark.parametrize('role,partial',(('baseline',False),('candidate',False),('candidate',True)))
def test_coords_report_preserves_provider_partition_and_actual_pass(tmp_path,role,partial):
    report,path,events=sample_report(tmp_path,2);row=report['results'][0]
    report['metadata'].update(backend_role='prior-cpp'if role=='baseline'else'native',pass_index=2)
    candidate=role=='candidate';row['native_accepted']=candidate and not partial
    row['actual_C_returns']=(['local','origin_into','multiply_f64']+([]if partial else['add_f64_into']))*2 if candidate else[]
    row['worker_snapshots']=[1,1]if candidate else[]
    row['phases']=[dict(index_owned=True,origin_owned=True,multiply_owned=not partial,add_owned=None if partial else True,native_accepted=not partial)]*2 if candidate else[]
    path.write_text('\n'.join(map(json.dumps,events))+'\n')
    policy=dict(role=role,pass_index=2,cases=[dict(native_partition='integer-provider'if partial else'f64')])
    assert C.validate_report(report,path,['case'],policy)=={'case':dict(original=100.,cpp=100.)}
    fault=copy.deepcopy(report);fault['results'][0]['native_accepted']=not row['native_accepted']
    with pytest.raises(AssertionError):C.validate_report(fault,path,['case'],policy)
    fault=copy.deepcopy(report);fault['metadata']['pass_index']=1
    with pytest.raises(AssertionError):C.validate_report(fault,path,['case'],policy)


def test_coords_driver_uses_common_controller_and_exact_forwarded_manifest(tmp_path,monkeypatch):
    root=tmp_path/'prepared-source';manifest=root/'build/prepared/ci-artifacts/measurement.json'
    manifest.parent.mkdir(parents=True);manifest.write_text('{}\n')
    seen=[]
    def inert_runpath(path,run_name):seen.append((path,list(C.sys.argv)))
    monkeypatch.setattr(C.runpy,'run_path',inert_runpath)
    args=types.SimpleNamespace(source_root=root,family='coords-scaled',source_revision='f'*40,role='baseline',pass_index=4,deadline_utc='2099-01-01T00:00:00+00:00',output=tmp_path/'none.json',preflight=False)
    C.driver(args)
    script,argv=seen[0]
    assert script==str(ROOT/'benchmarks/regionprops_coords_scaled.py')
    assert argv[argv.index('--backend-role')+1]=='prior-cpp' and argv[argv.index('--pass-index')+1]=='4'
    assert argv[argv.index('--source-root')+1]==str(root)
    assert argv[argv.index('--input-manifest')+1]==str(manifest)
    assert argv[argv.index('--input-manifest-sha256')+1]==hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_coords_prospective_execute_refuses_before_api_or_output(tmp_path,monkeypatch):
    monkeypatch.setattr(C,'CONTRACT',COORDS)
    monkeypatch.setattr(C.subprocess,'check_output',lambda *a,**kw:pytest.fail('prospective execute reached API/Git'))
    args=types.SimpleNamespace(candidate_source='f'*40,candidate_run='1',candidate_artifact=1,repository='bipxx/scikit-imagexx',results=tmp_path/'must-not-exist')
    with pytest.raises(AssertionError,match='prospective comparison'):C.execute(args)
    assert not args.results.exists()


@pytest.mark.parametrize('name',('tools/region_stain_compare.py','.github/workflows/region-stain-same-host.yml'))
def test_coords_legacy_exact_inverse_and_unrelated_mutation_rejected(name):
    raw=(ROOT/name).read_bytes().replace(b'\r\n',b'\n');projected=C.legacy_source(name,raw)
    legacy=json.loads((ROOT/'coverage/contracts/regionprops-performance-compare-v2.json').read_bytes())
    assert hashlib.sha256(projected).hexdigest()==legacy['controller_source_sha256'][name]
    with pytest.raises(AssertionError):C.legacy_source(name,raw+b'\n#unrelated mutation\n')
