"""Synthetic metadata/guard tests only: no native/image calls or clocks."""
import ast,copy,hashlib,importlib.util,json,os,statistics,sys,types
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('tv_compare_under_test',ROOT/'benchmarks/tv_chambolle_compare.py');t=importlib.util.module_from_spec(spec);sys.modules[spec.name]=t;spec.loader.exec_module(t);a=t.a

@pytest.fixture
def experiment(monkeypatch):
    # Inert metadata, never a real certificate/native acceptance or CLI bypass.
    bridge={'SYNTHETIC':'source witness'};monkeypatch.setattr(a,'BEFORE_BRIDGE_DIGEST',a.digest(bridge))
    scope=dict(image_paths=192,metadata_paths=3,sdk_families=18,sdk_members=63,sdk_family_names=['family'+str(i) for i in range(18)])
    products={}
    for role,source in [('before',a.BEFORE_PRODUCT),('after',a.AFTER_PRODUCT)]:
        products[role]={}
        for profile in a.PROFILES:
            pin=dict(revision=source,qualification_sha256='1'*64,product_sha256=('2' if role=='before' else '3')*64,wheel_sha256='4'*64,source_sha256='5'*64,scope=scope,artifact_run='111',artifact_name='rank-statistics-product-'+profile)
            products[role][profile]=dict(pin=pin,artifact=dict(id='123',digest='6'*64,run='111',name=pin['artifact_name']),core_sha256=('7' if role=='before' else '8')*64)
    return dict(schema='tv-chambolle-cursor-experiment-v1',status='root-reviewed',controller_revision='c'*40,before_product_source=a.BEFORE_PRODUCT,intended_before_source=a.BEFORE_ANCHOR,after_product_source=a.AFTER_PRODUCT,original_version='0.26.0',rounds=7,workers=1,case_count=400,timed_count=70,factory_sha256=a.FROZEN_FACTORY_SHA256,manifest_sha256=a.sha(ROOT/'coverage/contracts/tv-chambolle-benchmark.json'),root_after_admission=dict(source=a.AFTER_PRODUCT,passed=True,sha256='9'*64),products=products,approved_after_source_sha256={n:'a'*64 for n in a.ALLOWED_DELTA},before_source_equivalence=bridge)

def synthetic_report(e,measure=False):
    M=a.read_json(ROOT/'coverage/contracts/tv-chambolle-benchmark.json');expected=[M['reference_profiles']['modern']['observations'][str(i)] for i in range(400)]
    report=dict(schema='tv-cursor-controlled-v1',status='complete-controlled-measurement' if measure else 'complete-parity-check',controller_revision=e['controller_revision'],before_product_source=a.BEFORE_PRODUCT,intended_before_source=a.BEFORE_ANCHOR,after_product_source=a.AFTER_PRODUCT,profile='modern',check_only=not measure,failures=[],checks={},worker_identities={},results=[])
    for mode in (('normal',) if measure else ('normal','scalar')):
        report['checks'][mode]={r:dict(checks=[dict(workers=w,digests=expected[:]) for w in (1,4)],mismatches=[]) for r in a.ROLES};identities={}
        for index,role in enumerate(a.ROLES):
            frame=e['products']['after' if role=='after' else 'before']['modern'];identities[role]=dict(role=role,profile='modern',pid=101+index,cpu={'SYNTHETIC':'not actualCPU'},default_workers=1,scalar=mode=='scalar',has_avx2=mode=='normal',fallback_enabled=False,source=frame['pin']['revision'],product_sha256=frame['pin']['product_sha256'],core_sha256=frame['core_sha256'],core_path='/SYNTHETIC/'+('after' if role=='after' else 'before')+'/core')
        report['worker_identities'][mode]=identities
    if measure:
        for i in range(330,400):
            index=i-330;after=(1.,2.,4.)[index%3];durations={'original':2.,'before':3.,'after':after};samples={r:[durations[r]]*7 for r in a.ROLES};orders=t.rotating_triads(i);raw=[]
            for round_index,order in enumerate(orders):
                for role in order:raw.append(dict(round=round_index,role=role,ms=durations[role],elapsed_ns=int(durations[role]*1e6),loops=1,identity=report['worker_identities']['normal'][role]))
            report['results'].append(dict(case_index=i,inventory=M['case_inventory'][i],triad_order=orders,calibration={r:dict(ms=durations[r],elapsed_ns=int(durations[r]*1e6),loops=1,identity=report['worker_identities']['normal'][r]) for r in a.ROLES},samples_ms=samples,raw_samples=raw,medians_ms=durations,before_speedup=2./3.,after_speedup=2./after,improvement=3./after,slower_than_before=after>3.,slower_than_original=after>2.))
        report['summary']=t.summarize(report['results'])
    return report


def test_unaccepted_after_blocks_before_any_fixture_import(tmp_path,monkeypatch,experiment):
    assert a.validate_experiment(experiment,'c'*40)==experiment
    experiment['root_after_admission']=None;experiment['controller_revision']=a.revision();p=tmp_path/'experiment.json';p.write_text(json.dumps(experiment))
    monkeypatch.setattr(t,'factory',lambda *a,**k:pytest.fail('fixtures reached before root admission'))
    with pytest.raises(RuntimeError,match='AFTER'):a.load_experiment(p,a.sha(p),experiment['controller_revision'])

@pytest.mark.parametrize('field,value',[('workers',True),('workers',4),('rounds',6),('rounds',True),('case_count',True),('case_count',399),('timed_count',69),('status','proposal'),('before_product_source',a.BEFORE_ANCHOR),('intended_before_source',a.BEFORE_PRODUCT),('after_product_source','a'*40),('factory_sha256','0'*64),('manifest_sha256','0'*64),('controller_revision','d'*40)])
def test_experiment_semantic_guards_are_reached(experiment,field,value):
    a.validate_experiment(experiment,'c'*40);experiment[field]=value
    with pytest.raises(RuntimeError):a.validate_experiment(experiment,'c'*40)

@pytest.mark.parametrize('mutation',['afterpending','missingprofile','boolscope','extraTU','bridge','artifact','unknownfield'])
def test_product_scope_source_equivalence_and_plumbing_fail_closed(experiment,mutation):
    a.validate_experiment(experiment,'c'*40)
    if mutation=='afterpending':experiment['products']['after']['modern']=None
    elif mutation=='missingprofile':experiment['products']['before'].pop('windows')
    elif mutation=='boolscope':experiment['products']['before']['modern']['pin']['scope']['image_paths']=True
    elif mutation=='extraTU':experiment['approved_after_source_sha256']['src/rank.cpp']='a'*64
    elif mutation=='bridge':experiment['before_source_equivalence']['changed']=True
    elif mutation=='artifact':experiment['products']['before']['modern']['artifact']['run']='112'
    else:experiment['ignored_case_ids']=[330]
    with pytest.raises(RuntimeError):a.validate_experiment(experiment,'c'*40)


def test_seven_rotations_balanced_and_complete():
    for index in range(330,400):
        orders=t.rotating_triads(index);assert len(orders)==7 and all(set(r)==set(a.ROLES) for r in orders)
        for role in a.ROLES:
            positions=[sum(order[p]==role for order in orders) for p in range(3)];assert max(positions)-min(positions)<=1

@pytest.mark.parametrize('bad',[(False,True,'true',1,False,330),(True,False,'true',1,False,330),(True,True,'false',1,False,330),(True,True,'true',4,False,330),(True,True,'true',True,False,330),(True,True,'true',1,True,330),(True,True,'true',1,False,329),(True,True,'true',1,False,400),(True,True,'true',1,False,True)])
def test_sample_gate_blocks_before_clocks(bad,monkeypatch):
    monkeypatch.setattr(t.time,'perf_counter_ns',lambda:pytest.fail('timer reached'))
    with pytest.raises(RuntimeError):t.sample_admission(*bad)


def test_actual_full_report_summary_serializes_all70_inert_rows(experiment):
    report=synthetic_report(experiment,True);t.validate_report(report,experiment)
    assert len(report['results'])==70 and sum(len(r['raw_samples']) for r in report['results'])==1470
    assert report['summary']['overall']['slower_than_before'] and report['summary']['overall']['slower_than_original']
    assert {r['case_index'] for r in report['results']}==set(range(330,400));json.dumps(report,allow_nan=False)

@pytest.mark.parametrize('mutation',['missingrow','index','spec','sample','rawtime','raworder','identity','default','CPU','samecore','fallback','scope','summary','policy','scalar','duplicatepid','digest','missingmode'])
def test_report_guards_do_not_hide_after_an_earlier_failure(experiment,mutation):
    report=synthetic_report(experiment,mutation!='missingmode');t.validate_report(report,experiment)
    if mutation=='missingrow':report['results'].pop()
    elif mutation=='index':report['results'][0]['case_index']=0
    elif mutation=='spec':report['results'][0]['inventory']=report['results'][1]['inventory']
    elif mutation=='sample':report['results'][0]['samples_ms']['after'].pop()
    elif mutation=='rawtime':report['results'][0]['raw_samples'][0]['elapsed_ns']+=1
    elif mutation=='raworder':report['results'][0]['raw_samples'].reverse()
    elif mutation=='summary':report['summary']['overall']['case_count']=69
    elif mutation=='digest':report['checks']['normal']['after']['checks'][0]['digests'][0]='0'*64
    elif mutation=='missingmode':report['checks'].pop('scalar')
    elif mutation=='scope':report['before_product_source']=a.BEFORE_ANCHOR
    else:
        key=report['worker_identities']['normal']['after']
        if mutation=='identity':key['core_sha256']='0'*64
        elif mutation=='default':key['default_workers']=True
        elif mutation=='CPU':key['cpu']={'DIFFERENT':'CPU'}
        elif mutation=='samecore':key['core_path']=report['worker_identities']['normal']['before']['core_path']
        elif mutation=='fallback':key['fallback_enabled']=True
        elif mutation=='scalar':key['scalar']=True
        elif mutation=='duplicatepid':key['pid']=report['worker_identities']['normal']['before']['pid']
        else:key['has_avx2']=False
    with pytest.raises(RuntimeError):t.validate_report(report,experiment)


def build_identity(tmp_path):
    spec=importlib.util.spec_from_file_location('_historical_metadata_fixture',ROOT/'tests/test_rank_statistics_compare.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    path=tmp_path/'product';facts=h.synthetic_build_facts(path);rank=facts['translation_units'][0]
    for filename in ('tv_chambolle.cpp','tv_denoise.cpp'):
        unit=copy.deepcopy(rank);old=unit['source'];unit['source']=str(Path(facts['source_root'])/'src'/filename);Path(unit['source']).write_bytes(b'INERT TV SOURCE')
        oldout=unit['outputs'][0];newout=str(Path(oldout).with_name(filename+'.obj'));Path(newout).write_bytes(b'INERT TV OBJECT');unit['outputs']=[newout];unit['argv']=[v.replace(old,unit['source']).replace(oldout,newout) for v in unit['argv']]
        facts['translation_units'].append(unit);facts['link_commands'][0]['inputs'].append(newout);facts['link_commands'][0]['argv'].append(newout);facts['link_inputs'].append(dict(path=newout,generated=True))
    identity,_=h.capture_synthetic_build(path,facts);return h.c,identity


def test_real_common_model_fork_masks_only_two_TUs_and_preserves_rank_guard(tmp_path):
    c,identity=build_identity(tmp_path);original=inspect_source=__import__('inspect').getsource(c._common_build_identity);first=a.tv_common(identity,c)
    changed=copy.deepcopy(identity)
    for path,record in changed['file_records']['sources'].items():
        if path.replace('\\','/').endswith(tuple('/'+n for n in a.TV_TUS)):record['canonical_lf_sha256']='b'*64
    assert a.tv_common(changed,c)==first
    rank=next(p for p in changed['file_records']['sources'] if p.replace('\\','/').endswith('/src/rank.cpp'));changed['file_records']['sources'][rank]['canonical_lf_sha256']='c'*64
    assert a.tv_common(changed,c)!=first
    assert __import__('inspect').getsource(c._common_build_identity)==original
    assert "=='src/rank.cpp'" in original

@pytest.mark.parametrize('mutation',['tool','flag','header','external','missingTV','headerclosure'])
def test_common_model_preserves_all_nonTV_inputs(tmp_path,mutation):
    c,identity=build_identity(tmp_path);old=a.tv_common(identity,c);new=copy.deepcopy(identity)
    if mutation=='tool':next(iter(new['file_records']['binaries'].values()))['sha256']='c'*64
    elif mutation=='flag':new['raw']['translation_units'][0]['argv'].append('-ffast-math')
    elif mutation=='header':next(iter(new['file_records']['headers'].values()))['sha256']='c'*64
    elif mutation=='external':next(r for p,r in new['file_records']['link_inputs'].items() if not next(x for x in new['raw']['link_inputs'] if x['path']==p)['generated'])['sha256']='c'*64
    elif mutation=='missingTV':new['raw']['translation_units']=[u for u in new['raw']['translation_units'] if not u['source'].endswith('tv_denoise.cpp')]
    else:new['file_records']['headers'].pop(next(iter(new['file_records']['headers'])))
    if mutation in ('missingTV','headerclosure'):
        with pytest.raises((RuntimeError,KeyError)):a.tv_common(new,c)
    else:assert a.tv_common(new,c)!=old


def test_workflow_barrier_and_measurement_exclusion():
    source=(ROOT/'.github/workflows/tv-chambolle-compare.yml').read_text();measure=source.split('  measure:',1)[1]
    assert 'needs: [prepare, barrier]' in measure and '--receipts' in measure
    assert 'pytest' not in measure.replace('no builds or pytest','') and 'pip wheel' not in measure and 'cmake' not in measure
    assert '--no-index' in measure and '--no-compile' in measure
    assert a.ACCEPTED_PRODUCTS is None

def test_actual_inherited_stdio_authentication_and_exclusive_receipt(tmp_path,monkeypatch):
    # Real transport/authenticator, inert metadata only; no subprocess/native call.
    import io
    nonces,auth=t.startup_plan();monkeypatch.setattr(t.os,'getpid',lambda:321);monkeypatch.setattr(t.os,'getppid',lambda:auth['controller_pid'])
    claim={'authentication':auth};permit=dict(auth);path=tmp_path/'claim.json';a.save_new(path,claim)
    frame=t.startup_frame(auth,'original','modern','run','1',321,nonces['original'])
    result=t.authenticate_startup(permit,claim,path,'original','modern','run','1',io.StringIO(json.dumps(frame)+'\n'))
    assert result['worker_pid']==321 and result['startup_receipt_sha256']==a.sha(tmp_path/'claim-original-stdio-startup.json')
    with pytest.raises(FileExistsError):t.authenticate_startup(permit,claim,path,'original','modern','run','1',io.StringIO(json.dumps(frame)+'\n'))


def test_new_writer_returns_exact_receipt_path(tmp_path):
    path=tmp_path/'receipt.json';assert a.save_new(path,{'metadata':True})==path
    with pytest.raises(FileExistsError):a.save_new(path,{'metadata':False})


@pytest.mark.parametrize('mutation',['receipt_checks','receipt_common','report_bytes','controller','profile','stale'])
def test_real_barrier_binds_all_report_observations(tmp_path,experiment,mutation):
    paths=[]
    for profile in a.PROFILES:
        report=synthetic_report(experiment,False);report['profile']=profile
        M=a.read_json(ROOT/'coverage/contracts/tv-chambolle-benchmark.json')
        expected=[M['reference_profiles'][profile]['observations'][str(i)] for i in range(400)]
        for mode in report['checks']:
            for role in t.ROLES:
                report['worker_identities'][mode][role]['profile']=profile
                for row in report['checks'][mode][role]['checks']:row['digests']=expected
        report['source_comparison']={'common_sha256':'f'*64};report['experiment_sha256']='e'*64
        output=tmp_path/profile;output.mkdir();a.save_new(output/'report.json',report)
        r=dict(schema='tv-cursor-preparation-v1',passed=True,profile=profile,controller_revision=experiment['controller_revision'],generation='run',attempt='1',experiment_sha256='e'*64,report_sha256=a.sha(output/'report.json'),modes=['normal','scalar'],workers=[1,4],case_count=400,timed_count=70,common_build_sha256='f'*64,reports={mode:a.digest(report['checks'][mode]) for mode in ('normal','scalar')})
        a.save_new(output/'preparation.json',r);paths.append(output/'preparation.json')
    assert set(t.verify_barrier(paths,experiment,'e'*64,'run','1'))==set(a.PROFILES)
    r=a.read_json(paths[0])
    if mutation=='receipt_checks':r['reports']['normal']='a'*64
    elif mutation=='receipt_common':r['common_build_sha256']='a'*64
    elif mutation=='report_bytes':r['report_sha256']='a'*64
    elif mutation=='controller':r['controller_revision']='b'*40
    elif mutation=='profile':r['profile']='modern'
    else:r['generation']='old'
    paths[0].write_text(json.dumps(r))
    with pytest.raises(RuntimeError):t.verify_barrier(paths,experiment,'e'*64,'run','1')

@pytest.mark.parametrize('changed',[False,True])
def test_controller_checkout_source_guard(monkeypatch,changed):
    monkeypatch.setattr(a,'revision',lambda:'c'*40)
    calls=[]
    def metadata_only_git(argv,**kwargs):
        calls.append(argv);return types.SimpleNamespace(returncode=1 if changed else 0)
    monkeypatch.setattr(a.subprocess,'run',metadata_only_git)
    if changed:
        with pytest.raises(RuntimeError,match='checkout changed'):a.verify_controller_checkout('c'*40)
    else:assert a.verify_controller_checkout('c'*40)=='c'*40
    assert calls==[['git','-C',str(a.ROOT),'diff','--quiet','HEAD','--']]
    with pytest.raises(RuntimeError,match='revision differs'):a.verify_controller_checkout('d'*40)


@pytest.mark.parametrize('mutation',['missing','elapsed','identity'])
def test_actual_report_calibration_closure(experiment,mutation):
    report=synthetic_report(experiment,True);t.validate_report(report,experiment)
    value=report['results'][0]['calibration']
    if mutation=='missing':value.pop('before')
    elif mutation=='elapsed':value['before']['elapsed_ns']+=1
    else:value['before']['identity']=dict(value['before']['identity'],pid=999)
    with pytest.raises(RuntimeError):t.validate_report(report,experiment)
