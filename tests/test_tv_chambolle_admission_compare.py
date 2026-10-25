"""Synthetic metadata/guard tests only: no native/image calls or clocks."""
import ast,copy,hashlib,importlib.util,json,os,statistics,sys,types
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('tv_compare_under_test',ROOT/'benchmarks/tv_chambolle_admission_compare.py');t=importlib.util.module_from_spec(spec);sys.modules[spec.name]=t;spec.loader.exec_module(t);a=t.a

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
    return dict(schema='tv-chambolle-admission-experiment-v1',status='root-reviewed',controller_revision='c'*40,before_product_source=a.BEFORE_PRODUCT,intended_before_source=a.BEFORE_ANCHOR,after_product_source=a.AFTER_PRODUCT,original_version='0.26.0',rounds=7,workers=1,case_count=400,timed_count=70,factory_sha256=a.FROZEN_FACTORY_SHA256,manifest_sha256=a.sha(ROOT/'coverage/contracts/tv-chambolle-benchmark.json'),root_after_admission=dict(source=a.AFTER_PRODUCT,passed=True,sha256='9'*64),products=products,approved_after_source_sha256={n:(a.PRIVATE_HEADER_SHA256 if n==a.PRIVATE_HEADER else 'a'*64) for n in a.ALLOWED_DELTA},before_source_equivalence=bridge)

def synthetic_defaults(report,profile=None):
    # Inert wire metadata, never actual image/native evidence.
    profile=profile or report['profile'];report['default_validation_schema']=a.DEFAULT_SCHEMA
    report['default_checks']={};report['default_observation_assets']={}
    M=a.read_json(ROOT/'coverage/contracts/tv-chambolle-benchmark.json')
    obs=dict(result={'SYNTHETIC':'not image data'},error=None,warnings=[],events=[],before={},after={},pointer_stable=True,bases=dict(before={},after={},result=[]))
    for mode in report['checks']:
        report['default_checks'][mode]={}
        for role in a.ROLES:
            value=dict(schema=a.DEFAULT_SCHEMA,role=role,profile=profile,mode=mode,mismatches=[],checks=[])
            for workers in (1,4):
                records=[]
                for i in range(330,400):
                    rule=a.default_rule(i,M['case_inventory'][i]);count=0 if role=='original' else rule['finite_calls']
                    stages={n:dict(c_call=0,c_return=0,c_exception=0) for n in ('divergence','gradient','norm','dual')}
                    if role!='original' and rule['tagged']:
                        for n in ('norm','dual'):stages[n]=dict(c_call=1,c_return=1,c_exception=0)
                    returns=[dict(dtype=rule['dtype'],shape=rule['spatial_shape'],weight=rule['weight'],eps=rule['eps'],max_iterations=rule['max_iterations'],profile='1.24.4' if profile=='minimum' else '2.5.4',buffer_elements=8192,iterations=rule['max_iterations'],input_alias=rule['max_iterations']==1) for _ in range(count)]
                    dispatch=dict(finite=dict(helper_calls=count,c_call=count,c_return=count,c_exception=0,accepted=count,rejected=0,returns=returns),stages=stages,original=dict(outer=int(role=='original'),inner=rule['channels'] if role=='original' else 0,denied=0),helper_marker='owned-_finite_call-real-builtin-v1')
                    state=dict(policy=dict(a.DEFAULT_NP_POLICY),handler_is_none=True,providers_pristine=True,buffer_elements=8192,workers=workers)
                    records.append(dict(case_index=i,observation_digest=a.digest(obs),observation_payload=a.pack_default_observation(obs),state_before=copy.deepcopy(state),state_after=copy.deepcopy(state),dispatch=dispatch))
                value['checks'].append(dict(workers=workers,records=records,digests=[r['observation_digest'] for r in records]))
            report['default_checks'][mode][role]=a.ingest_default_reply(report,value)
    return report


def synthetic_report(e,measure=False):
    M=a.read_json(ROOT/'coverage/contracts/tv-chambolle-benchmark.json');expected=[M['reference_profiles']['modern']['observations'][str(i)] for i in range(400)]
    report=dict(schema='tv-admission-controlled-v2',status='complete-controlled-measurement' if measure else 'complete-parity-check',controller_revision=e['controller_revision'],before_product_source=a.BEFORE_PRODUCT,intended_before_source=a.BEFORE_ANCHOR,after_product_source=a.AFTER_PRODUCT,profile='modern',check_only=not measure,failures=[],checks={},worker_identities={},results=[])
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
        auth={'schema':'tv-admission-original-auth-artifacts-v1','original_raw_upload':True,
              'claim':dict(path='authentication/claim.json',sha256='1'*64,bytes=100),
              'permit':dict(path='worker-permit.json',sha256='2'*64,bytes=100),'startup_receipts':{}}
        for i,role in enumerate(a.ROLES):
            digest=str(i+3)*64;report['worker_identities']['normal'][role]['startup_authentication']={'startup_receipt_sha256':digest}
            auth['startup_receipts'][role]=dict(path='authentication/claim-'+role+'-stdio-startup.json',sha256=digest,bytes=100)
        report['authentication_artifacts']=auth
    return synthetic_defaults(report)


def test_unaccepted_after_blocks_before_any_fixture_import(tmp_path,monkeypatch,experiment):
    assert a.validate_experiment(experiment,'c'*40)==experiment
    experiment['root_after_admission']=None;experiment['controller_revision']=a.revision();p=tmp_path/'experiment.json';p.write_text(json.dumps(experiment))
    monkeypatch.setattr(t,'factory',lambda *a,**k:pytest.fail('fixtures reached before root admission'))
    with pytest.raises(RuntimeError,match='AFTER'):a.load_experiment(p,a.sha(p),experiment['controller_revision'])

@pytest.mark.parametrize('field,value',[('workers',True),('workers',4),('rounds',6),('rounds',True),('case_count',True),('case_count',399),('timed_count',69),('status','proposal'),('before_product_source',a.BEFORE_ANCHOR),('intended_before_source',a.BEFORE_PRODUCT),('after_product_source','a'*40),('factory_sha256','0'*64),('manifest_sha256','0'*64),('controller_revision','d'*40)])
def test_experiment_semantic_guards_are_reached(experiment,field,value):
    a.validate_experiment(experiment,'c'*40);experiment[field]=value
    with pytest.raises(RuntimeError):a.validate_experiment(experiment,'c'*40)

@pytest.mark.parametrize('mutation',['afterpending','missingprofile','boolscope','extraTU','bridge','artifact','headerpin','unknownfield'])
def test_product_scope_source_equivalence_and_plumbing_fail_closed(experiment,mutation):
    a.validate_experiment(experiment,'c'*40)
    if mutation=='afterpending':experiment['products']['after']['modern']=None
    elif mutation=='missingprofile':experiment['products']['before'].pop('windows')
    elif mutation=='boolscope':experiment['products']['before']['modern']['pin']['scope']['image_paths']=True
    elif mutation=='extraTU':experiment['approved_after_source_sha256']['src/rank.cpp']='a'*64
    elif mutation=='bridge':experiment['before_source_equivalence']['changed']=True
    elif mutation=='artifact':experiment['products']['before']['modern']['artifact']['run']='112'
    elif mutation=='headerpin':experiment['approved_after_source_sha256'][a.PRIVATE_HEADER]='0'*64
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
    with pytest.raises(RuntimeError):t.sample_admission(*bad,default_validated=True)


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
    for filename in ('tv_denoise.cpp','bind_tv_chambolle.cpp','tv_chambolle.cpp'):
        unit=copy.deepcopy(rank);old=unit['source'];unit['source']=str(Path(facts['source_root'])/'src'/filename);Path(unit['source']).write_bytes(b'INERT TV SOURCE')
        oldout=unit['outputs'][0];newout=str(Path(oldout).with_name(filename+'.obj'));Path(newout).write_bytes(b'INERT TV OBJECT');unit['outputs']=[newout];unit['argv']=[v.replace(old,unit['source']).replace(oldout,newout) for v in unit['argv']]
        facts['translation_units'].append(unit);facts['link_commands'][0]['inputs'].append(newout);facts['link_commands'][0]['argv'].append(newout);facts['link_inputs'].append(dict(path=newout,generated=True))
    identity,_=h.capture_synthetic_build(path,facts);return h.c,identity



def refresh_identity(c,identity):
    # Consistent synthetic metadata, not an artifact/qualification substitute.
    identity['common']=c._common_build_identity(identity);identity['common_sha256']=c.digest(identity['common'])

def test_real_common_model_fork_masks_only_two_TUs_and_preserves_rank_guard(tmp_path):
    c,identity=build_identity(tmp_path);original=inspect_source=__import__('inspect').getsource(c._common_build_identity);first=a.tv_common(identity,c)
    changed=copy.deepcopy(identity)
    for path,record in changed['file_records']['sources'].items():
        if path.replace('\\','/').endswith(tuple('/'+n for n in a.TV_TUS)):record['canonical_lf_sha256']='b'*64
    refresh_identity(c,changed)
    assert a.tv_common(changed,c)==first
    rank=next(p for p in changed['file_records']['sources'] if p.replace('\\','/').endswith('/src/rank.cpp'));changed['file_records']['sources'][rank]['canonical_lf_sha256']='c'*64
    refresh_identity(c,changed)
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
    else:
        refresh_identity(c,new)
        assert a.tv_common(new,c)!=old


def test_workflow_barrier_and_measurement_exclusion():
    source=(ROOT/'.github/workflows/tv-chambolle-admission-compare.yml').read_text();measure=source.split('  measure:',1)[1]
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
        report=synthetic_report(experiment,False);report['profile']=profile;synthetic_defaults(report,profile)
        M=a.read_json(ROOT/'coverage/contracts/tv-chambolle-benchmark.json')
        expected=[M['reference_profiles'][profile]['observations'][str(i)] for i in range(400)]
        for mode in report['checks']:
            for role in t.ROLES:
                report['worker_identities'][mode][role]['profile']=profile
                for row in report['checks'][mode][role]['checks']:row['digests']=expected
        report['source_comparison']={'common_sha256':'f'*64};report['experiment_sha256']='e'*64
        output=tmp_path/profile;output.mkdir();a.save_new(output/'report.json',report)
        r=dict(schema='tv-admission-preparation-v2',passed=True,profile=profile,controller_revision=experiment['controller_revision'],generation='run',attempt='1',experiment_sha256='e'*64,report_sha256=a.sha(output/'report.json'),modes=['normal','scalar'],workers=[1,4],case_count=400,timed_count=70,common_build_sha256='f'*64,reports={mode:a.digest(report['checks'][mode]) for mode in ('normal','scalar')},default_validation_schema=a.DEFAULT_SCHEMA,default_timed_count=70,default_reports={mode:a.digest(dict(checks=report['default_checks'][mode],assets=report['default_observation_assets'])) for mode in ('normal','scalar')})
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
    assert calls==[['git','-c','core.autocrlf=true','-C',str(a.ROOT),'diff','--quiet','HEAD','--']]
    with pytest.raises(RuntimeError,match='revision differs'):a.verify_controller_checkout('d'*40)


@pytest.mark.parametrize('mutation',['missing','elapsed','identity'])
def test_actual_report_calibration_closure(experiment,mutation):
    report=synthetic_report(experiment,True);t.validate_report(report,experiment)
    value=report['results'][0]['calibration']
    if mutation=='missing':value.pop('before')
    elif mutation=='elapsed':value['before']['elapsed_ns']+=1
    else:value['before']['identity']=dict(value['before']['identity'],pid=999)
    with pytest.raises(RuntimeError):t.validate_report(report,experiment)

def projection_pair(tmp_path,crlf=False):
    c,before=build_identity(tmp_path);after=copy.deepcopy(before);facts=after['raw']
    path=Path(facts['source_root'])/a.PRIVATE_HEADER
    raw=(ROOT/a.PRIVATE_HEADER).read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(raw).hexdigest()==a.PRIVATE_HEADER_SHA256
    if crlf:raw=raw.replace(b'\n',b'\r\n')
    path.write_bytes(raw);key=str(path.resolve());after['file_records']['headers'][key]=c._build_file_record(key,True)
    for unit in facts['translation_units']:
        if c._below_build_root(unit['source'],facts['source_root']) in a.TV_TUS:unit['dependencies'].append(key)
    refresh_identity(c,after)
    return c,before,after,Path(facts['source_root'])


@pytest.mark.parametrize('crlf',[False,True])
def test_exact_private_header_two_edges_project_after_full_raw_authentication(tmp_path,crlf):
    c,before,after,source=projection_pair(tmp_path,crlf);saved=copy.deepcopy((before,after))
    proof=a.header_projection(before,after,a.tv_common(before,c),a.tv_common(after,c),source,c)
    assert proof['removed_edges']==2 and proof['removed_global_entries']==1
    assert proof['actual_raw_bytes']==(853 if crlf else 837)
    assert proof['canonical_lf_sha256']==a.PRIVATE_HEADER_SHA256
    assert proof['approved_added_used_header_edges']==sorted(a.TV_TUS)
    assert (before,after)==saved
    assert a.TV_TUS=={'src/tv_denoise.cpp','src/bind_tv_chambolle.cpp'}


@pytest.mark.parametrize('change',['bytes','rawrecord','canonical','missingedge','duplicateedge','thirdedge',
                                   'beforeedge','extraheader','tool','flag','approvedflag','external','link','configuration',
                                   'unchanged_chambolle','extraTU','stalecommon','borrowedmodel'])
def test_header_projection_rejects_every_extra_input_change(tmp_path,change):
    c,before,after,source=projection_pair(tmp_path)
    assert a.header_projection(before,after,a.tv_common(before,c),a.tv_common(after,c),source,c)
    facts=after['raw'];header=next(p for p in after['file_records']['headers'] if p.replace('\\','/').endswith('/'+a.PRIVATE_HEADER))
    if change=='bytes':(source/a.PRIVATE_HEADER).write_bytes(b'not the reviewed header')
    elif change=='rawrecord':after['file_records']['headers'][header]['sha256']='b'*64
    elif change=='canonical':after['file_records']['headers'][header]['canonical_lf_sha256']='b'*64
    elif change=='missingedge':next(u for u in facts['translation_units'] if header in u['dependencies'])['dependencies'].remove(header)
    elif change=='duplicateedge':next(u for u in facts['translation_units'] if header in u['dependencies'])['dependencies'].append(header)
    elif change=='thirdedge':next(u for u in facts['translation_units'] if Path(u['source']).name=='tv_chambolle.cpp')['dependencies'].append(header)
    elif change=='beforeedge':
        before['file_records']['headers'][header]=copy.deepcopy(after['file_records']['headers'][header])
        next(u for u in before['raw']['translation_units'] if u['source'].endswith('tv_denoise.cpp'))['dependencies'].append(header)
        refresh_identity(c,before)
    elif change=='extraheader':
        p=source/'src/unreviewed.hpp';p.write_bytes(b'unreviewed');key=str(p.resolve());after['file_records']['headers'][key]=c._build_file_record(key,True);facts['translation_units'][0]['dependencies'].append(key)
    elif change=='tool':next(iter(after['file_records']['binaries'].values()))['sha256']='b'*64
    elif change=='flag':facts['translation_units'][-1]['argv'].append('-ffast-math')
    elif change=='approvedflag':next(u for u in facts['translation_units'] if Path(u['source']).name=='bind_tv_chambolle.cpp')['argv'].append('-ffast-math')
    elif change=='external':next(r for p,r in after['file_records']['link_inputs'].items() if not next(x for x in facts['link_inputs'] if x['path']==p)['generated'])['sha256']='b'*64
    elif change=='link':facts['link_commands'][-1]['argv'].append('UNAPPROVED-LINK-FLAG')
    elif change=='configuration':facts['configuration']['compiler_version']='UNAPPROVED-TOOLCHAIN'
    elif change=='unchanged_chambolle':next(r for p,r in after['file_records']['sources'].items() if Path(p).name=='tv_chambolle.cpp')['canonical_lf_sha256']='b'*64
    elif change=='extraTU':facts['translation_units'].append(copy.deepcopy(facts['translation_units'][-1]))
    elif change=='stalecommon':after['common']['configuration']['compiler_version']='UNAUTHENTICATED'
    if change not in ('stalecommon','borrowedmodel'):
        try:refresh_identity(c,after)
        except RuntimeError:return # malformed raw closure is rejected before projection
    with pytest.raises(RuntimeError):
        bm=a.tv_common(before,c);am=a.tv_common(after,c)
        if change=='borrowedmodel':am['binaries']=copy.deepcopy(bm['binaries']);am['configuration']['compiler_version']='BORROWED-MODEL'
        a.header_projection(before,after,bm,am,source,c)


@pytest.mark.parametrize('change',['missingclaim','missingstartup','claimbytes','claimoutside','startupid','startuphash','indexpath','indexrole','indexoriginal','indexbytes'])
def test_actual_original_auth_files_are_uploaded_and_fail_closed(tmp_path,monkeypatch,experiment,change):
    import io
    report=synthetic_report(experiment,True);report['experiment_sha256']='e'*64
    nonces,auth=t.startup_plan();output=tmp_path/'results';claim_path=output/'authentication/claim.json';receipts=[]
    for p in a.PROFILES:
        q=tmp_path/(p+'-preparation.json');a.save_new(q,{'SYNTHETIC':'not numerical/native evidence'});receipts.append(str(q))
    claim=dict(profile='modern',generation='run',attempt='1',receipt_hashes=[a.sha(p) for p in receipts],authentication=auth);a.save_new(claim_path,claim)
    permit=dict(profile='modern',experiment_sha256='e'*64,**auth,claim=str(claim_path),claim_sha256=a.sha(claim_path),receipts=receipts);a.save_new(output/'worker-permit.json',permit)
    for i,role in enumerate(a.ROLES):
        pid=101+i;monkeypatch.setattr(t.os,'getpid',lambda pid=pid:pid);monkeypatch.setattr(t.os,'getppid',lambda:auth['controller_pid'])
        frame=t.startup_frame(auth,role,'modern','run','1',pid,nonces[role]);report['worker_identities']['normal'][role]['startup_authentication']=t.authenticate_startup(permit,claim,claim_path,role,'modern','run','1',io.StringIO(json.dumps(frame)+'\n'))
    records=t.verify_authentication_artifacts(output,report);report['authentication_artifacts']=records;t.validate_report(report,experiment)
    assert records['original_raw_upload'] is True and len(list((output/'authentication').glob('*.json')))==4
    startup=output/'authentication/claim-original-stdio-startup.json'
    if change=='missingclaim':claim_path.unlink()
    elif change=='missingstartup':startup.unlink()
    elif change=='claimbytes':claim_path.write_text('{}')
    elif change=='claimoutside':permit['claim']=str(tmp_path/'outside.json');(output/'worker-permit.json').write_text(json.dumps(permit))
    elif change=='startupid':
        r=a.read_json(startup);r['worker_pid']=999;startup.write_text(json.dumps(r))
    elif change=='startuphash':report['worker_identities']['normal']['original']['startup_authentication']['startup_receipt_sha256']='0'*64
    elif change=='indexpath':report['authentication_artifacts']['claim']['path']='../claim.json'
    elif change=='indexrole':report['authentication_artifacts']['startup_receipts'].pop('original')
    elif change=='indexoriginal':report['authentication_artifacts']['original_raw_upload']=False
    else:report['authentication_artifacts']['claim']['bytes']=True
    if change.startswith('index'):
        with pytest.raises(RuntimeError):t.validate_report(report,experiment)
    else:
        with pytest.raises((RuntimeError,FileNotFoundError)):t.verify_authentication_artifacts(output,report)


def test_historical_science_and_controller_timing_bodies_are_unchanged():
    old=ast.parse((ROOT/'benchmarks/tv_chambolle_compare.py').read_bytes());new=ast.parse((ROOT/'benchmarks/tv_chambolle_admission_compare.py').read_bytes())
    def node(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    def same(left,right):assert ast.dump(left,include_attributes=False)==ast.dump(right,include_attributes=False)
    for name in ('factory','rotating_triads','summarize'):same(node(old,name),node(new,name))
    gate=copy.deepcopy(node(new,'sample_admission'));gate.args.kwonlyargs=[];gate.args.kw_defaults=[];gate.body.pop()
    same(node(old,'sample_admission'),gate)
    left=node(old,'worker_main');right=node(new,'worker_main')
    def branch(fn):return next(n for n in ast.walk(fn) if isinstance(n,ast.If) and ast.unparse(n.test)=="operation == 'validate'")
    same(ast.Module(body=branch(left).body,type_ignores=[]),ast.Module(body=branch(right).body,type_ignores=[]))
    def timer(fn):return next(n for n in ast.walk(fn) if isinstance(n,ast.With) and ast.unparse(n.items[0].context_expr)=='warnings.catch_warnings()')
    same(timer(left),timer(right))
    assert a.text_sha(ROOT/'benchmarks/tv_chambolle.py')==a.FROZEN_FACTORY_SHA256
    assert a.text_sha(ROOT/'benchmarks/rank_statistics_compare.py')==a.RANK_VALIDATOR_SHA256
    assert "claim=output/'authentication'/'claim.json'" in (ROOT/'benchmarks/tv_chambolle_admission_compare.py').read_text()


@pytest.mark.parametrize('fault',['missing','partial','policy','handler','buffer','provider','worker','return','exception','accepted','upstream','profile','shape','parameter','alias','dispatch','asset','old'])
def test_default70_rejects_wrong_timed_path_before_clocks(experiment,monkeypatch,fault):
    report=synthetic_report(experiment,True);t.validate_report(report,experiment)
    monkeypatch.setattr(t.time,'perf_counter_ns',lambda:pytest.fail('clock reached'))
    group=report['default_checks']['normal'];row=group['after']['checks'][0]['records'][0];finite=row['dispatch']['finite']
    if fault=='missing':report.pop('default_checks')
    elif fault=='partial':group['after']['checks'][0]['records'].pop()
    elif fault in ('policy','handler','buffer','provider','worker'):
        key={'policy':'policy','handler':'handler_is_none','buffer':'buffer_elements','provider':'providers_pristine','worker':'workers'}[fault]
        row['state_before'][key]={'divide':'warn','over':'warn','under':'warn','invalid':'warn'} if fault=='policy' else (4096 if fault=='buffer' else 3 if fault=='worker' else False)
    elif fault=='return':finite['c_return']-=1
    elif fault=='exception':finite['c_return']-=1;finite['c_exception']+=1
    elif fault=='accepted':finite['accepted']=0
    elif fault=='upstream':row['dispatch']['original']['denied']=1
    elif fault=='profile':finite['returns'][0]['profile']='1.24.4'
    elif fault=='shape':finite['returns'][0]['shape']=[1]
    elif fault=='parameter':finite['returns'][0]['weight']*=2
    elif fault=='alias':finite['returns'][0]['input_alias']=not finite['returns'][0]['input_alias']
    elif fault=='dispatch':group['before']['checks'][0]['records'][0]['dispatch']['helper_marker']='boolean-label'
    elif fault=='asset':next(iter(report['default_observation_assets'].values()))['sha256']='0'*64
    else:report['schema']='tv-admission-controlled-v1'
    with pytest.raises(RuntimeError):t.validate_report(report,experiment)


@pytest.mark.parametrize('fault',['base','mutation','pointer','hash','size','trailing'])
def test_default_raw_observation_integrity(fault):
    value=dict(result=None,error=None,warnings=[],events=[],before={},after={},pointer_stable=True,bases=dict(before={},after={},result=[]))
    if fault=='base':value['bases']['after']={'changed':True}
    elif fault=='mutation':value['after']={'changed':True}
    elif fault=='pointer':value['pointer_stable']=False
    packed=a.pack_default_observation(value)
    if fault=='hash':packed['sha256']='0'*64
    elif fault=='size':packed['bytes']+=1
    elif fault=='trailing':packed['payload']+='AAAA'
    with pytest.raises(RuntimeError):a.unpack_default_observation(packed)


def test_default_gate_cannot_be_replaced_by_legacy400_or_policy_interposition():
    with pytest.raises(RuntimeError):t.sample_admission(True,True,'true',1,False,330)
    t.sample_admission(True,True,'true',1,False,330,default_validated=True)
    tree=ast.parse((ROOT/'benchmarks/tv_chambolle_admission_compare.py').read_bytes())
    for name in ('observe_default','default_state','check_default_worker'):
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert not any(isinstance(n,ast.Attribute) and n.attr in ('seterr','seterrcall') for n in ast.walk(fn))
    witness=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DefaultWitness');source=ast.unparse(witness)
    assert "event == 'return'" in source and "'c_return'" in source and 'type(arg) is tuple' in source
    assert 'self.frame' not in source and 'self.image' not in source and 'setattr(' not in source
