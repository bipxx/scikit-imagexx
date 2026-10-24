"""HARNESS-ONLY metadata controls. No product, image call or timing acceptance."""
import ast,copy,hashlib,importlib.util,io,json,os,subprocess,sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('max_tree_comparison_controls',ROOT/'benchmarks/max_tree_compare.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);a=c.a


def synthetic_certificate():
    profiles={}
    for i,p in enumerate(a.PROFILES):
        pin=dict(revision='8'*40,qualification_sha256='1'*64,product_sha256='2'*64,wheel_sha256='3'*64,source_sha256='4'*64,scope=copy.deepcopy(a.SCOPE),artifact_run='123',artifact_name='rank-statistics-product-'+p)
        profiles[p]=dict(pin=pin,artifact=dict(id=str(100+i),run='123',name=pin['artifact_name'],digest='5'*64),core_sha256='6'*64,python_version='3.11.14' if p=='minimum' else '3.13.14')
    return dict(schema='max-tree-public-product-v1',status='root-accepted-qualified-public199',source='8'*40,run='123',controller_revision='7'*40,max_tree_runtime_source_sha256=copy.deepcopy(a.RUNTIME_SOURCE_PINS),baseline={n:a.PINS[n] for n in ('benchmarks/max_tree.py','coverage/contracts/max-tree-benchmark.json','coverage/contracts/max-tree.json')},profiles=profiles)


def write(path,value):path.write_text(json.dumps(value),encoding='utf8');return path


def test_immutable_baseline_and_historical_modules():
    a.frozen_sources();b,M=c.baseline()
    assert len(M['cases'])==len(b.CASES)==252 and M['counts']['reference_numeric']==233 and M['counts']['reference_metadata']==118
    assert a.ACCEPTED_PRODUCT_PINS is None and b.ACCEPTED_PRODUCT_PINS is None
    assert a.SCOPE['image_paths']==199 and a.SCOPE['sdk_families']==20 and a.SCOPE['sdk_members']==65
    assert len(a.FAMILIES)==20 and len(set(a.FAMILIES))==20


@pytest.mark.parametrize('fault',['absent','status','source','run','controller','missing-profile','duplicate-artifact','scope','float-scope','families','core','artifact-digest','artifact-run','artifact-name','python','baseline','runtime-source','extra'])
def test_certificate_fails_closed_before_any_fixture(fault,tmp_path):
    value=synthetic_certificate()
    if fault=='absent':value=None
    elif fault=='status':value['status']='reference-only'
    elif fault=='source':value['source']='0'*40
    elif fault=='run':value['run']=123
    elif fault=='controller':value['controller_revision']='0'*40
    elif fault=='missing-profile':value['profiles'].pop('minimum')
    elif fault=='duplicate-artifact':value['profiles']['modern']['artifact']['id']=value['profiles']['windows']['artifact']['id']
    elif fault=='scope':value['profiles']['windows']['pin']['scope']['sdk_members']=59
    elif fault=='float-scope':value['profiles']['windows']['pin']['scope']['image_paths']=199.
    elif fault=='families':value['profiles']['windows']['pin']['scope']['sdk_family_names'].reverse()
    elif fault=='core':value['profiles']['windows']['core_sha256']=None
    elif fault=='artifact-digest':value['profiles']['windows']['artifact']['digest']='bad'
    elif fault=='artifact-run':value['profiles']['windows']['artifact']['run']='124'
    elif fault=='artifact-name':value['profiles']['windows']['artifact']['name']='private-core'
    elif fault=='python':value['profiles']['minimum']['python_version']='3.13.14'
    elif fault=='baseline':value['baseline']['benchmarks/max_tree.py']='0'*64
    elif fault=='runtime-source':value['max_tree_runtime_source_sha256']['src/max_tree.cpp']='0'*64
    else:value['accepted']=True
    path=write(tmp_path/'certificate.json',value)
    with pytest.raises((RuntimeError,TypeError)):a.certificate(path,a.sha(path),'7'*40)


def test_exact_certificate_bytes_and_metadata_only_positive(tmp_path):
    path=tmp_path/'certificate.json';raw=(json.dumps(synthetic_certificate(),indent=2)+'\r\n').encode();path.write_bytes(raw)
    assert a.certificate(path,hashlib.sha256(raw).hexdigest(),'7'*40)==synthetic_certificate()
    with pytest.raises(RuntimeError):a.certificate(path,hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest(),'7'*40)


@pytest.mark.parametrize('measure,validated,workers,scalar,index',[
    (False,True,1,False,0),(True,False,1,False,0),(True,True,4,False,0),
    (True,True,True,False,0),(True,True,1,True,0),(True,True,1,False,-1),
    (True,True,1,False,252),(True,True,1,False,True)])
def test_timer_admission_rejects_partial_or_check_only(measure,validated,workers,scalar,index,monkeypatch):
    monkeypatch.setenv('GITHUB_ACTIONS','true')
    with pytest.raises(RuntimeError):c.sample_admission(validated,measure,workers,scalar,index)


def test_local_timer_guard_precedes_products_or_fixtures(monkeypatch):
    monkeypatch.delenv('GITHUB_ACTIONS',raising=False)
    with pytest.raises(RuntimeError):c.controller(type('Args',(),{})())
    with pytest.raises(RuntimeError):c.sample_admission(True,True,1,False,0)


@pytest.mark.parametrize('fault',['version','float-types','bool-types','operations','parallel','stages','float-stages','borrowed','class-spoof'])
def test_bad_actual_core_admission_precedes_fixture_import(fault,monkeypatch):
    # Every fake is intentionally REJECTED. No fake core is admitted or run.
    from types import SimpleNamespace as NS
    package=Path(sys.prefix)/'lib/site-packages/skimagexx'
    core=NS(__name__='skimagexx._core',__file__=str(package/'_core.fake'),max_tree_interface='max-tree-prepared-v1',
        max_tree_type_capabilities=1023,max_tree_operation_capabilities=63,max_tree_parallel_traversal=False,
        max_tree_stage_interface='max-tree-stages-v1',max_tree_stage_capabilities=31,max_tree_run=len)
    wrapper=NS(__file__=str(package/'_max_tree.py'),_core=core,_require_core=lambda:core,_STAGE_ENTRIES=())
    if fault=='version':core.max_tree_interface='stale'
    elif fault=='float-types':core.max_tree_type_capabilities=1023.
    elif fault=='bool-types':core.max_tree_type_capabilities=True
    elif fault=='operations':core.max_tree_operation_capabilities=1
    elif fault=='parallel':core.max_tree_parallel_traversal=True
    elif fault=='stages':core.max_tree_stage_interface='stale'
    elif fault=='float-stages':core.max_tree_stage_capabilities=31.
    elif fault=='class-spoof':
        fake=type('builtin_function_or_method',(),{'__module__':'skimagexx._core','__name__':'max_tree_run'})
        core.max_tree_run=fake()
    xx=NS(__file__=str(package/'__init__.py'),_core=core)
    monkeypatch.setitem(sys.modules,'skimagexx',xx);monkeypatch.setitem(sys.modules,'skimagexx._core',core)
    real=c.importlib.import_module
    monkeypatch.setattr(c.importlib,'import_module',lambda name:wrapper if name=='skimagexx._max_tree' else NS() if name=='skimagexx.coverage' else real(name))
    realsha=a.sha;monkeypatch.setattr(a,'sha',lambda p:'6'*64 if str(p)==core.__file__ else realsha(p))
    with pytest.raises(RuntimeError):c.public_admission(Path('unreadable-product'),'windows',synthetic_certificate())


def test_clock_contains_only_whole_public_call_and_destruction():
    tree=ast.parse(Path(c.__file__).read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='callback_pair')
    call=next(n for n in ast.walk(fn) if isinstance(n,ast.FunctionDef) and n.name=='call')
    assert len(call.body)==1 and isinstance(call.body[0],ast.Return)
    assert ast.unparse(call.body[0])=='return function(image, **options)'
    worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker_main')
    loop=next(n for n in ast.walk(worker) if isinstance(n,ast.For) and ast.unparse(n.iter)=='range(loops)')
    assert [ast.unparse(n) for n in loop.body]==['result = call()', 'del result']
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('observe_case','certify_entry','array_record','digest') for n in ast.walk(loop))


def test_all252_randomized_pairs_and_separate_provided_summary():
    b,_=c.baseline();rows=[]
    for i,case in enumerate(b.CASES):
        assert len(c.orders(i))==7 and all(sorted(x)==['native','original'] for x in c.orders(i))
        rows.append(dict(index=i,case=case,speedup=1.,slower=False))
    summary=c.summarize(rows)
    assert summary['overall']['cases']==252 and summary['rebuild']['cases']==222 and summary['provided_tree']['cases']==30
    with pytest.raises(RuntimeError):c.summarize(rows[:-1])


def counters(**completed):return {name:dict(attempted=completed.get(name,0),completed=completed.get(name,0)) for name in ('aggregate',*c.STAGES.values())}


@pytest.mark.parametrize('fault',['none','bool','partial','attempted','aggregate','stage'])
def test_real_completed_counter_floor(fault):
    row=counters(aggregate=1)
    if fault=='none':row=counters()
    elif fault=='bool':row['aggregate']['completed']=True
    elif fault=='partial':row.pop('area')
    elif fault=='attempted':row['aggregate']['attempted']=0
    elif fault=='aggregate':row['aggregate']['completed']=2;row['aggregate']['attempted']=2
    else:row['area']=dict(attempted=1,completed=1)
    with pytest.raises(RuntimeError):c.counter_check(row,aggregate=1)


@pytest.fixture(scope='module')
def synthetic_preparation():
    # Construct ONLY a JSON validation model from frozen observations. These
    # are explicitly not executed native results or qualification evidence.
    _,M=c.baseline();matrix=a.read(ROOT/'coverage/contracts/max-tree.json');profile='windows'
    def stages(entries):
        values={stage:sum(e['name']==source and 'after' in e for e in entries) for source,stage in c.STAGES.items()}
        return counters(**values)
    old=[];pristine=[]
    for spec in matrix['cases']:
        key=matrix['profiles'][profile]['snapshots'][spec['id']];expected=matrix['observations'][key]
        old.append(dict(id=spec['id'],mode=spec['mode'],digest=key,counts=stages(expected['entries'])))
        if spec['mode']=='numeric':pristine.append(dict(id=spec['id'],digest=a.digest(expected['result']),counts=counters(aggregate=1)))
    candidates=[dict(id=row['id'],digest=row['digest'],counts=stages(row['observation']['entries'])) for row in M['profiles'][profile]['cases']]
    plain=[dict(id=row['id'],digest=a.digest(row['observation']['result']),counts=counters(aggregate=1)) for row in M['profiles'][profile]['cases']]
    checks=[dict(workers=n,source351=old,pristine233=pristine,source252=candidates,pristine252=plain) for n in (1,4)]
    identity=dict(pid=42,source='8'*40,product_sha256='2'*64,core_sha256='6'*64,default_workers=1,fallback_enabled=False,scalar=False,has_avx2=True,cpu={'cpu_model':'HARNESS_ONLY_NOT_MEASURED'},authentication=dict(worker_pid=42,worker_parent=41,launcher_pid=42,receipt_sha256='a'*64))
    scalar=copy.deepcopy(identity);scalar.update(scalar=True,has_avx2=False)
    row=dict(original351=[r['digest'] for r in old],original252=[r['digest'] for r in candidates],checks=checks)
    return dict(schema='max-tree-public-comparison-v1',passed=True,failures=[],profile=profile,source='8'*40,product_sha256='2'*64,controller_sha256=a.sha(c.__file__),counts=copy.deepcopy(c.COUNTS),check_only=True,results=[],identities={'normal':identity,'scalar':scalar},checks={'normal':row,'scalar':copy.deepcopy(row)})


def test_full_metadata_model_is_valid_without_executing_products(synthetic_preparation):
    assert c.validate_report(synthetic_preparation,synthetic_certificate()) is synthetic_preparation


@pytest.mark.parametrize('fault',['normal-only','worker1-only','bool-worker','source-id','metadata-mode','partial351','partial233','partial252','pristine252','wrong-digest','zero-cpp','partial-stages','worker-core','source','cpu','scalar','startup','timing-in-prepare'])
def test_report_rejects_missing_or_relabelled_parity(synthetic_preparation,fault):
    r=copy.deepcopy(synthetic_preparation);workers=r['checks']['normal']['checks'][0]
    if fault=='normal-only':r['checks'].pop('scalar')
    elif fault=='worker1-only':r['checks']['normal']['checks'].pop()
    elif fault=='bool-worker':workers['workers']=True
    elif fault=='source-id':workers['source351'][0]['id']='renamed'
    elif fault=='metadata-mode':workers['source351'][0]['mode']='metadata'
    elif fault=='partial351':workers['source351'].pop()
    elif fault=='partial233':workers['pristine233'].pop()
    elif fault=='partial252':workers['source252'].pop()
    elif fault=='pristine252':workers['pristine252'].pop()
    elif fault=='wrong-digest':workers['source252'][0]['digest']='0'*64
    elif fault=='zero-cpp':workers['pristine233'][0]['counts']=counters()
    elif fault=='partial-stages':workers['source252'][0]['counts']=counters()
    elif fault=='worker-core':r['identities']['normal']['core_sha256']='0'*64
    elif fault=='source':r['source']='0'*40
    elif fault=='cpu':r['identities']['normal']['cpu']={}
    elif fault=='scalar':r['identities']['scalar']['has_avx2']=True
    elif fault=='startup':r['identities']['normal']['authentication']['worker_pid']=999
    else:r['results']=[{'fake':'timing'}]
    with pytest.raises(RuntimeError):c.validate_report(r,synthetic_certificate())


@pytest.fixture(scope='module')
def synthetic_measurement(synthetic_preparation):
    # Fake numbers exercise REPORT ARITHMETIC ONLY; never exported as samples.
    import inspect
    _,M=c.baseline();r=copy.deepcopy(synthetic_preparation);r.update(check_only=False,checks={},identities={'normal':r['identities']['normal']})
    source=inspect.getsource(c.callback_pair);node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='call');text=ast.get_source_segment(source,node);sha=hashlib.sha256(text.encode()).hexdigest()
    rows=[]
    for index,case in enumerate(M['cases']):
        def sample(backend):return dict(elapsed_ns=1000000 if backend=='original' else 2000000,loops=1,ms=1. if backend=='original' else 2.,identity=r['identities']['normal'],callback_source_sha256=sha,counts=counters(aggregate=int(backend=='native')))
        samples=[dict(round=i,backend=backend,**sample(backend)) for i,order in enumerate(c.orders(index)) for backend in order]
        expected=M['profiles']['windows']['cases'][index]['observation']
        callback=dict(case=case,input_state=dict(**expected['before'],provided=expected['provided_before']),callback_source=text,callback_source_sha256=sha,callback_AST=ast.dump(node,include_attributes=False),actual_targets={k:dict(module='skimage.morphology.max_tree' if k=='original' else 'skimagexx._max_tree',name=case['function']) for k in ('original','native')})
        rows.append(dict(index=index,case=case,callback=callback,orders=c.orders(index),raw_samples=samples,calibration={k:sample(k) for k in ('original','native')},medians_ms={'original':1.,'native':2.},speedup=.5,slower=True))
    r['results']=rows;r['summary']=c.summarize(rows);return r


def test_synthetic_statistics_model_retains_all_regressions(synthetic_measurement):
    c.validate_report(synthetic_measurement,synthetic_certificate())
    assert len(synthetic_measurement['summary']['overall']['slower'])==252


@pytest.mark.parametrize('fault',['partial','round','backend','loops','elapsed','nan','cpu','callback','target','input','provided','cpp','calibration','ratio','regression','summary'])
def test_corrupt_raw_paired_report_is_rejected(synthetic_measurement,fault):
    r=copy.deepcopy(synthetic_measurement);row=r['results'][0];value=row['raw_samples'][0]
    if fault=='partial':r['results'].pop()
    elif fault=='round':value['round']=9
    elif fault=='backend':value['backend']='foreign'
    elif fault=='loops':value['loops']=True
    elif fault=='elapsed':value['elapsed_ns']=0
    elif fault=='nan':value['ms']=float('nan')
    elif fault=='cpu':value['identity']=dict(value['identity'],cpu={'cpu_model':'foreign'})
    elif fault=='callback':row['callback']['callback_source']='forbidden observer()'
    elif fault=='target':row['callback']['actual_targets']['native']['module']='skimage.morphology.max_tree'
    elif fault=='input':row['callback']['input_state']['image']['sha256']='0'*64
    elif fault=='provided':row['callback']['input_state']['provided']={'foreign':{}}
    elif fault=='cpp':row['calibration']['native']['counts']=counters()
    elif fault=='calibration':row['calibration'].pop('original')
    elif fault=='ratio':row['speedup']=2.
    elif fault=='regression':row['slower']=False
    else:r['summary']['overall']['slower']=[]
    with pytest.raises(RuntimeError):c.validate_report(r,synthetic_certificate())


def synthetic_receipt(profile,cert_sha='a'*64):
    return dict(schema='max-tree-preparation-v1',passed=True,profile=profile,certificate_sha256=cert_sha,generation='99',attempt='1',modes=['normal','scalar'],workers=[1,4],counts=copy.deepcopy(c.COUNTS),report_sha256='b'*64,product_sha256='2'*64,controller_sha256='c'*64)


@pytest.mark.parametrize('fault',['profile','run','attempt','scalar','workers','count','status','reference','certificate','report','product'])
def test_preparation_receipts_cannot_be_partial_or_reused(fault):
    row=synthetic_receipt('windows')
    if fault=='profile':row['profile']='debug'
    elif fault=='run':row['generation']='98'
    elif fault=='attempt':row['attempt']='0'
    elif fault=='scalar':row['modes']=['normal']
    elif fault=='workers':row['workers']=[1]
    elif fault=='count':row['counts']['candidates']=251
    elif fault=='status':row['passed']=False
    elif fault=='reference':row['schema']='max-tree-benchmark-reference-v1'
    elif fault=='certificate':row['certificate_sha256']='0'*64
    elif fault=='report':row['report_sha256']='wrong'
    else:row['product_sha256']=None
    with pytest.raises(RuntimeError):a.validate_receipt(row,'a'*64,'99','1')


@pytest.mark.parametrize('n',[0,1,2,4])
def test_three_profile_barrier_precedes_any_report_or_fixture(n):
    with pytest.raises(RuntimeError,match='ALL3'):a.barrier([Path('unreadable')]*n,'a'*64,'99','1',synthetic_certificate())


def test_claim_is_exclusive_and_tied_to_receipts(tmp_path,monkeypatch):
    monkeypatch.setenv('RUNNER_TEMP',str(tmp_path))
    monkeypatch.setattr(a,'barrier',lambda *args:{'HARNESS_ONLY':'metadata no product'})
    receipts=[]
    for i in range(3):p=tmp_path/str(i);p.write_text(str(i));receipts.append(p)
    claim,_=a.claim(receipts,'a'*64,'99','1',synthetic_certificate(),'windows','n'*64)
    assert a.read(claim)['receipt_sha256']==[a.sha(p) for p in receipts]
    with pytest.raises(FileExistsError):a.claim(receipts,'a'*64,'99','1',synthetic_certificate(),'windows','z'*64)


def test_real_existing_interpreter_stdio_startup_and_replay(tmp_path,monkeypatch):
    # Only startup() runs in this REAL subprocess. No product/core/image import.
    nonce='a'*64;permit=tmp_path/'permit.json'
    value=dict(controller_pid=os.getpid(),controller_sha256=a.sha(c.__file__),generation='99',attempt='1',profile='metadata',certificate_sha256='b'*64,nonce_sha256=hashlib.sha256(nonce.encode()).hexdigest())
    write(permit,value)
    child=tmp_path/'startup.py';child.write_text('import importlib.util,json,sys\ns=importlib.util.spec_from_file_location("startup_metadata",'+repr(str(Path(c.__file__).resolve()))+');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\np,r=m.startup(sys.argv[1],sys.stdin)\nprint(json.dumps(r))\n',encoding='utf8')
    env=dict(os.environ,GITHUB_ACTIONS='true',GITHUB_RUN_ID='99',GITHUB_RUN_ATTEMPT='1',PYTHONDONTWRITEBYTECODE='1')
    def run():
        proc=subprocess.Popen([sys.executable,str(child),str(permit)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
        out,err=proc.communicate(json.dumps(dict(value,nonce=nonce,launcher_pid=proc.pid))+'\n');return proc.returncode,out,err
    code,out,err=run();assert code==0,err;assert json.loads(out)['worker_pid']>0
    code,out,err=run();assert code!=0 and 'FileExistsError' in err


@pytest.mark.parametrize('fault',['secret','source','stale','parent','interactive'])
def test_bad_stdio_startup_denied(fault,tmp_path,monkeypatch):
    nonce='a'*64;permit=tmp_path/'permit.json';value=dict(controller_pid=os.getppid(),controller_sha256=a.sha(c.__file__),generation='99',attempt='1',profile='metadata',certificate_sha256='b'*64,nonce_sha256=hashlib.sha256(nonce.encode()).hexdigest())
    monkeypatch.setenv('GITHUB_ACTIONS','true');monkeypatch.setenv('GITHUB_RUN_ID','99');monkeypatch.setenv('GITHUB_RUN_ATTEMPT','1')
    frame=dict(value,nonce=nonce,launcher_pid=os.getpid())
    if fault=='secret':frame['nonce']='z'*64
    elif fault=='source':value['controller_sha256']='0'*64;frame['controller_sha256']='0'*64
    elif fault=='stale':value['attempt']='0';frame['attempt']='0'
    elif fault=='parent':frame['launcher_pid']=99999999;value['controller_pid']=99999998;frame['controller_pid']=99999998
    write(permit,value)
    stream=io.StringIO(json.dumps(frame)+'\n')
    if fault=='interactive':stream.isatty=lambda:True
    with pytest.raises(RuntimeError):c.startup(permit,stream)


def test_workflow_fresh_hosts_no_build_or_test_in_measurement():
    text=(ROOT/'.github/workflows/max-tree-compare.yml').read_text();prepare,measure=text.split('\n  measure:',1)
    assert 'needs: prepare' in measure and 'max-tree-receipt-*' in measure
    for forbidden in ('pytest','cmake','ninja','pip wheel','build-whole','prepare --profile'):
        assert forbidden not in measure
    assert '--no-index' in measure and '--require-hashes' in measure and '--only-binary=:all:' in measure
    assert 'write_bytes(raw)' in prepare and 'write_bytes(raw)' in measure
    assert 'max_tree_compare.py fetch' in prepare and 'max_tree_compare.py measure' in measure
    assert 'build/barrier/max-tree-receipt-*/preparation/preparation.json' in measure
    assert prepare.count('profile: ')==3 and measure.count('profile: ')==3


@pytest.mark.parametrize('initial',[True,False])
def test_authentic_stdlib_compat_disables_initial_policy(initial):
    # Real repository _compat, not a fake core or native admission fixture.
    spec=importlib.util.spec_from_file_location('authentic_policy_control',ROOT/'python/skimagexx/_compat.py')
    policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
    assert policy._fallback_enabled is True
    policy.set_fallback_enabled(initial)
    c.disable_fallback(policy)
    assert policy._fallback_enabled is False
    assert policy.set_fallback_enabled(False) is False
    with pytest.raises(NotImplementedError,match='fallback is disabled'):
        policy.fallback('not_imported','not_called')
    node=next(n for n in ast.parse(Path(c.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='public_admission')
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='disable_fallback' for n in ast.walk(node))


@pytest.mark.parametrize('raw',[b'{"claim":1}\n',b'{"claim":1}\r\n',b' {"claim":1} ',b'{"label":"\xc3\xa9"}\r\n\r\n'])
def test_claim_retention_is_byte_exact_and_never_replaces_anchor(raw,tmp_path):
    anchor=tmp_path/'global-claim.json';anchor.write_bytes(raw)
    output=tmp_path/'uploaded';output.mkdir()
    row=a.retain_claim(anchor,output)
    assert anchor.read_bytes()==(output/row['path']).read_bytes()==raw
    assert row==dict(anchor=str(anchor.resolve()),path='measurement-claim.json',sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    with pytest.raises(FileExistsError):a.retain_claim(anchor,output)
    assert anchor.read_bytes()==raw


def test_retained_claim_does_not_weaken_global_once_guard(tmp_path,monkeypatch):
    monkeypatch.setenv('RUNNER_TEMP',str(tmp_path))
    monkeypatch.setattr(a,'barrier',lambda *args:[])
    receipts=[]
    for i in range(3):
        p=tmp_path/str(i);p.write_bytes(str(i).encode());receipts.append(p)
    anchor,_=a.claim(receipts,'a'*64,'99','1',synthetic_certificate(),'windows','n'*64)
    raw=anchor.read_bytes();output=tmp_path/'upload-one';output.mkdir();a.retain_claim(anchor,output)
    other=tmp_path/'upload-two';other.mkdir()
    with pytest.raises(FileExistsError):a.claim(receipts,'a'*64,'99','1',synthetic_certificate(),'windows','z'*64)
    assert anchor.read_bytes()==raw and (output/'measurement-claim.json').read_bytes()==raw
    # Controller must retain bytes after creating its fresh output, before workers.
    fn=next(n for n in ast.parse(Path(c.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='controller')
    text=ast.unparse(fn)
    assert text.index('output.mkdir(')<text.index('a.retain_claim(')<text.index('worker = Worker(')


def test_claim_retention_rejects_changed_anchor_hash(tmp_path,monkeypatch):
    anchor=tmp_path/'claim';anchor.write_bytes(b'original\r\n');output=tmp_path/'upload';output.mkdir()
    original=a.sha
    monkeypatch.setattr(a,'sha',lambda path:'0'*64 if Path(path)==anchor else original(path))
    with pytest.raises(RuntimeError,match='claim changed'):a.retain_claim(anchor,output)
    assert anchor.read_bytes()==(output/'measurement-claim.json').read_bytes()==b'original\r\n'


def test_prior_closing_allocation_source_is_not_admitted(tmp_path):
    value=synthetic_certificate();value['max_tree_runtime_source_sha256']['python/skimagexx/_max_tree.py']='faf537524b273dcdbb25a702590d139fdf94f8823b480bf7cef0f2cd3aa8eb53'
    path=write(tmp_path/'old-source.json',value)
    with pytest.raises(RuntimeError,match='runtime source differs'):a.certificate(path,a.sha(path),'7'*40)
