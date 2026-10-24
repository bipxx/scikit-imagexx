"""Metadata/harness-only tests. Fake artifacts never prove native admission."""
import ast
import base64
import builtins
import csv
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import zipfile
import pytest
ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('r4_controlled_tooling',ROOT/'benchmarks/rank_statistics_compare.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def wheel(path,name,version,entries=None):
    entries=dict(entries or {name.replace('-','_')+'/__init__.py':b'# SYNTHETIC\n'})
    prefix=name.replace('-','_')+'-'+version+'.dist-info/'
    entries[prefix+'METADATA']=('Name: '+name+'\nVersion: '+version+'\n').encode()
    rows=[]
    for n,b in entries.items():rows.append([n,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(b).digest()).rstrip(b'=').decode(),str(len(b))])
    rows.append([prefix+'RECORD','','']);stream=io.StringIO();csv.writer(stream).writerows(rows);entries[prefix+'RECORD']=stream.getvalue().encode()
    with zipfile.ZipFile(path,'w') as z:
        for n,b in entries.items():z.writestr(n,b)
    return entries


def synthetic_build_facts(path,revision='a'*40,compiler_id='MSVC',lto=False):
    """Small files stand in for tools/objects; none is executable or native proof."""
    source=path/'source';build=path.parent/(path.name+'-SYNTHETIC-BUILD');build.mkdir(parents=True,exist_ok=True)
    def put(relative,body):
        p=build/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(body);return str(p.resolve())
    for relative,body in {'src/rank.cpp':b'SYNTHETIC counts source\n','src/unchanged.cpp':b'SYNTHETIC unchanged source\n','include/skimagexx/h0.hpp':b'// fake\n'}.items():
        p=source/relative;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():p.write_bytes(body)
    roles={'GNU':['compiler','backend','assembler','linker','link_driver','archiver','ranlib'],
           'MSVC':['compiler','frontend','backend','linker','archiver']}[compiler_id]
    if compiler_id=='GNU' and lto:roles+=['lto_backend','lto_wrapper','lto_plugin']
    binaries={r:put('bin/'+r,b'SYNTHETIC NONEXECUTABLE TOOL '+r.encode()) for r in roles}
    roots={'project':[str(source.resolve())]};dependencies=[str((source/'include/skimagexx/h0.hpp').resolve())]
    for family in ('python','numpy','pybind11','toolchain'):
        header=put('headers/'+family+'/header.h',('SYNTHETIC '+family).encode());roots[family]=[str(Path(header).parent)];dependencies.append(header)
    rank=put('objects/rank.obj',b'SYNTHETIC rank object');other=put('objects/other.obj',b'SYNTHETIC other object');archive=put('lib/native.lib',b'SYNTHETIC generated archive');external=put('build-env/external.lib',b'SYNTHETIC external library')
    core=put('_core.synthetic.so',b'NOT A NATIVE MODULE')
    native=path/'wheels/native.whl'
    if not native.exists():native.parent.mkdir(parents=True,exist_ok=True);wheel(native,'scikit-imagexx','0.1.0',{'skimagexx/_core.synthetic.so':Path(core).read_bytes()})
    units=[]
    for filename,output in [('rank.cpp',rank),('unchanged.cpp',other)]:
        cpp=str((source/'src'/filename).resolve());units.append(dict(source=cpp,target='static',argv=[binaries['compiler'],'-O3','-ffp-contract=off',*(['-flto=auto'] if lto else []),'-I'+str(source.resolve()),'-o',output,'-c',cpp],dependencies=list(dependencies),outputs=[output]))
    links=[dict(target='archive',tool='archiver',argv=[binaries['archiver'],'qc',archive,rank,other],inputs=[rank,other],outputs=[archive]),dict(target='module',tool='compiler',argv=[binaries['compiler'],'-shared','-o',core,archive,external],inputs=[archive,external],outputs=[core])]
    evidence={r:[put('evidence/'+r+'.txt',('SYNTHETIC RAW '+r).encode())] for r in ('configuration','compiler','commands','dependencies')}
    return dict(schema=c.BUILD_INPUT_SCHEMA,revision=revision,profile='windows',source_root=str(source.resolve()),build_root=str(build.resolve()),configuration=dict(compiler_id=compiler_id,compiler_version='SYNTHETIC-1',generator='SYNTHETIC',generator_version='1',target_arch='x64',toolset='SYNTHETIC',crt='SYNTHETIC',build_type='Release',openmp=True,avx2=True,lto=lto),binaries=binaries,header_roots=roots,translation_units=units,link_commands=links,link_inputs=[dict(path=p,generated=p!=external) for p in (rank,other,archive,external)],evidence=evidence,product_wheel=str(native.resolve()),core=core)


def capture_synthetic_build(path,facts):
    facts_path=path.parent/(path.name+'-facts.json');facts_path.write_text(json.dumps(facts))
    c.capture_build_identity(facts_path,path/'build-identity.json')
    identity=c.load_json(path/'build-identity.json')
    records={'build-identity.json':dict(bytes=(path/'build-identity.json').stat().st_size,sha256=c.sha(path/'build-identity.json'))}
    records.update({name:dict(bytes=r['bytes'],sha256=r['sha256']) for name,r in identity['evidence_records'].items()})
    qualification=dict(schema='public-qualification-v2',passed=True,profile='windows',source=dict(revision=facts['revision']),records=records)
    return identity,qualification


@pytest.fixture
def fake_product(tmp_path):
    path=tmp_path/'SYNTHETIC-NONNATIVE';(path/'wheels').mkdir(parents=True);source=path/'source';source.mkdir()
    revision='a'*40;headers={f'include/skimagexx/h{i}.hpp':b'// fake\n' for i in range(54)}
    sdk=dict(headers,**{'lib/skimagexx.lib':b'NOT A LIBRARY','lib/cmake/skimagexx/a.cmake':b'fake','lib/cmake/skimagexx/b.cmake':b'fake','lib/cmake/skimagexx/c.cmake':b'fake','share/licenses/scikit-imagexx/LICENSE':b'fake'})
    entries=dict(sdk,**{'skimagexx/__init__.py':b'# fake\n','skimagexx/_core.synthetic.so':b'NOT A NATIVE MODULE'})
    wheel(path/'wheels/native.whl','scikit-imagexx','0.1.0',entries)
    for name,version in (('scikit-image','0.26.0'),('numpy','2.5.4'),('scipy','1.18.1')):wheel(path/'wheels'/str(name+'.whl'),name,version)
    contents=dict(headers,**{'python/skimagexx/__init__.py':b'# fake\n','readme.txt':b'SYNTHETIC SOURCE\n','src/rank.cpp':b'SYNTHETIC counts source\n','src/unchanged.cpp':b'SYNTHETIC unchanged source\n','coverage/contracts/rank-statistics-benchmark.json':json.dumps(dict(case_contract_sha256='fixed',case_inventory=[{}]*1379)).encode()})
    for n,b in contents.items():p=source/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    with zipfile.ZipFile(path/'source.zip','w') as z:
        for n,b in contents.items():z.writestr(n,b)
    scope=c.config()['planned_scope']
    sdk_receipt=dict(families=scope['sdk_family_names'],checks=2*scope['sdk_families'],SDK_sha256={n:hashlib.sha256(b).hexdigest() for n,b in sdk.items()})
    (path/'sdk.json').write_text(json.dumps(sdk_receipt))
    qualification=dict(schema='public-qualification-v2',SDK_checks=26,passed=True,no_timing=True,failures=[],profile='windows',source=dict(revision=revision),expected_registry=dict(image_count=scope['image_paths'],metadata_count=scope['metadata_paths'],registered_count=scope['image_paths']+scope['metadata_paths'],functions=list(map(str,range(scope['image_paths']+scope['metadata_paths'])))),records={'sdk.json':dict(bytes=(path/'sdk.json').stat().st_size,sha256=c.sha(path/'sdk.json'))})
    (path/'qualification.json').write_text(json.dumps(qualification));(path/'runtime-lock.txt').write_text('SYNTHETIC')
    wheels={p.name:{k:v for k,v in c.verify_wheel(p).items() if k!='members'} for p in (path/'wheels').glob('*.whl')}
    runtime=dict(schema='ci-offline-measurement-v1',source=dict(revision=revision),wheels=wheels,lock_sha256=c.sha(path/'runtime-lock.txt'),runtime=dict(sdk_sha256={n:hashlib.sha256(b).hexdigest() for n,b in sdk.items()}))
    (path/'runtime.json').write_text(json.dumps(runtime))
    _,build_qualification=capture_synthetic_build(path,synthetic_build_facts(path,revision))
    qualification['records'].update(build_qualification['records']);(path/'qualification.json').write_text(json.dumps(qualification))
    cfg=dict(accepted_products={'before':{'windows':dict(revision=revision,qualification_sha256=c.sha(path/'qualification.json'),wheel_sha256=c.sha(path/'wheels/native.whl'),source_sha256=c.sha(path/'source.zip'),product_sha256='pending',scope=scope,artifact_run='1',artifact_name='synthetic-only')}},source_text_sha256={'readme.txt':c.text_sha(source/'readme.txt')},case_contract_sha256='fixed',profiles={'windows':dict(numpy='2.5.4',scipy='1.18.1')})
    def reseal():
        p=path/'product.json';p.write_text(json.dumps(dict(schema='rank-statistics-comparison-product-v1',revision=revision,profile='windows',files_sha256=c.tree(path,('product.json',)))))
        cfg['accepted_products']['before']['windows']['product_sha256']=c.sha(p)
    reseal();return path,cfg,reseal


def test_synthetic_complete_capsule_metadata_only(fake_product):
    path,cfg,_=fake_product
    assert c.verify_product(path,'before','windows',cfg)['revision']=='a'*40


@pytest.mark.parametrize('failure',['file-extra','file-bytes','runtime-closure','qualification','sdk58','header-extra','python-extra','numpy-runtime'])
def test_complete_product_tamper_or_incomplete_metadata_rejected(fake_product,failure):
    path,cfg,reseal=fake_product
    if failure=='file-extra':(path/'unrecorded').write_text('tamper')
    elif failure=='file-bytes':(path/'source/readme.txt').write_text('tamper')
    elif failure=='runtime-closure':
        p=path/'runtime.json';j=c.load_json(p);j['wheels'].pop('scipy.whl');p.write_text(json.dumps(j));reseal()
    elif failure=='qualification':
        p=path/'qualification.json';j=c.load_json(p);j['passed']=False;p.write_text(json.dumps(j));cfg['accepted_products']['before']['windows']['qualification_sha256']=c.sha(p);reseal()
    elif failure=='sdk58':
        p=path/'runtime.json';j=c.load_json(p);j['runtime']['sdk_sha256'].pop('include/skimagexx/h0.hpp');p.write_text(json.dumps(j));reseal()
    elif failure=='numpy-runtime':cfg['profiles']['windows']['numpy']='1.24.4'
    else:
        name='include/skimagexx/omitted.hpp' if failure=='header-extra' else 'python/skimagexx/omitted.py'
        p=path/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'new')
        with zipfile.ZipFile(path/'source.zip','w') as z:
            for p in (path/'source').rglob('*'):
                if p.is_file():z.writestr(p.relative_to(path/'source').as_posix(),p.read_bytes())
        cfg['accepted_products']['before']['windows']['source_sha256']=c.sha(path/'source.zip');reseal()
    with pytest.raises(RuntimeError):c.verify_product(path,'before','windows',cfg)


@pytest.mark.parametrize('role',['before','after'])
@pytest.mark.parametrize('profile',c.PROFILES)
def test_phase1_products_are_pending_before_fixture_import(role,profile,monkeypatch,tmp_path):
    monkeypatch.setattr(c,'harness',lambda *_:pytest.fail('fixture import reached'))
    with pytest.raises(RuntimeError,match='pending'):c.verify_product(tmp_path,role,profile,c.config())


def test_failed65_source_is_not_a_baseline_even_with_hashes():
    cfg=dict(accepted_products={'before':{'windows':dict(revision=c.BLOCKED_SOURCE,qualification_sha256='a'*64,product_sha256='b'*64,wheel_sha256='c'*64,source_sha256='d'*64,scope=c.config()['planned_scope'],artifact_run='1',artifact_name='synthetic-only')}})
    with pytest.raises(RuntimeError,match='failed65c'):c.accepted_pin(cfg,'before','windows')


@pytest.mark.parametrize('member',['../escape','/absolute','C:/escape','dir\\escape'])
def test_artifact_path_containment(member):
    with pytest.raises(RuntimeError):c.safe_name(member)


def test_wheel_record_verification_checks_exact_file_bytes(tmp_path):
    p=tmp_path/'fake.whl';wheel(p,'numpy','2.5.4');c.verify_wheel(p)
    with zipfile.ZipFile(p) as z:entries={n:z.read(n) for n in z.namelist()}
    entries['numpy/__init__.py']+=b'tamper'
    with zipfile.ZipFile(p,'w') as z:
        for n,b in entries.items():z.writestr(n,b)
    with pytest.raises(RuntimeError,match='RECORD bytes'):c.verify_wheel(p)


def receipts(tmp_path,monkeypatch):
    cfg=c.config();cfg['accepted_products']={r:{p:dict(revision=('a' if r=='before' else 'b')*40,qualification_sha256='1'*64,product_sha256=('2' if r=='before' else '3')*64,wheel_sha256='4'*64,source_sha256='5'*64,scope=cfg['planned_scope'],artifact_run='1',artifact_name='synthetic-only') for p in c.PROFILES} for r in ('before','after')}
    paths=[]
    for profile in c.PROFILES:
        build=dict(schema=c.BUILD_IDENTITY_SCHEMA,common_sha256='8'*64,products=dict(before='6'*64,after='7'*64))
        row=dict(schema='rank-statistics-comparison-preparation-v1',receipt_id=profile,generation='run',attempt='1',contract_sha256=c.text_sha(c.CONTRACT),profile=profile,native_admission=True,reference_only=False,modes=['normal','scalar'],workers=[1,4],canonical_cases=1379,alias_checks=4,timed_ids=cfg['timed_ids'],products={r:dict(product_sha256=cfg['accepted_products'][r][profile]['product_sha256'],build_identity_sha256=build['products'][r],build_common_sha256=build['common_sha256']) for r in ('before','after')})
        directory=tmp_path/profile;directory.mkdir();p=directory/'preparation.json'
        frozen=json.loads((ROOT/'coverage/contracts/rank-statistics-benchmark.json').read_text())['reference_profiles'][profile]['observations']
        digests=[frozen[str(i)] for i in range(1379)]
        value=dict(checks=[dict(workers=w,digests=digests,errors=[None]*1379,warnings=[[]]*1379) for w in (1,4)],aliases=cfg['algorithm_names'],mismatches=[])
        identities={mode:{role:dict(role=role,pid=index+1,source=cfg['accepted_products']['before' if role=='original' else role][profile]['revision'],product_sha256=cfg['accepted_products']['before' if role=='original' else role][profile]['product_sha256'],profile=profile,default_workers=1,scalar=mode=='scalar',has_avx2=mode=='normal',core_sha256='a'*64 if role!='after' else 'b'*64,core_path='/fake/'+role+'/core') for index,role in enumerate(c.ROLES)} for mode in ('normal','scalar')}
        report=dict(source_comparison=dict(changed=['src/rank.cpp'],patch_sha256=cfg['counts_cpp_patch_sha256'],scope=cfg['planned_scope'],full_source_index_sha256=['a'*64,'b'*64],build_identity=build),schema='rank-statistics-controlled-v1',status='complete-parity-check',check_only=True,reference_only=False,profile=profile,results=[],failures=[],contract_sha256=c.text_sha(c.CONTRACT),worker_identities=identities,checks={mode:{role:value for role in c.ROLES} for mode in ('normal','scalar')})
        (directory/'report.json').write_text(json.dumps(report));row['report_sha256']=c.sha(directory/'report.json')
        p.write_text(json.dumps(row));paths.append(p)
    return cfg,paths


def test_complete_barrier_is_only_metadata_not_native_admission(tmp_path,monkeypatch):
    cfg,paths=receipts(tmp_path,monkeypatch);assert len(c.receipt_barrier(paths,cfg,'run','1'))==3


@pytest.mark.parametrize('failure',['partial','duplicate-profile','duplicate-id','reference','no-native','stale-run','stale-attempt','mode','worker','case','alias','timed-id','product'])
def test_barrier_rejects_incomplete_replayed_or_reference_receipts(tmp_path,monkeypatch,failure):
    cfg,paths=receipts(tmp_path,monkeypatch);j=c.load_json(paths[0])
    if failure=='partial':paths.pop()
    elif failure=='duplicate-profile':j['profile']='modern'
    elif failure=='duplicate-id':j['receipt_id']='modern'
    elif failure=='reference':j['reference_only']=True
    elif failure=='no-native':j['native_admission']=False
    elif failure=='stale-run':j['generation']='old'
    elif failure=='stale-attempt':j['attempt']='0'
    elif failure=='mode':j['modes']=['normal']
    elif failure=='worker':j['workers']=[1]
    elif failure=='case':j['canonical_cases']=1378
    elif failure=='alias':j['alias_checks']=3
    elif failure=='timed-id':j['timed_ids']=j['timed_ids'][:-1]
    else:j['products']['before']['product_sha256']='bad'
    paths[0].write_text(json.dumps(j))
    with pytest.raises(RuntimeError):c.receipt_barrier(paths,cfg,'run','1')


def test_claim_outputs_cannot_overwrite_or_reuse(tmp_path):
    p=tmp_path/'claim.json';c.write_new(p,dict(claimed=True))
    with pytest.raises(FileExistsError):c.write_new(p,{})


def test_seeded_triads_contain_exact_serial_backends_and_are_reproducible():
    a=c.seeded_triads(959981);assert a==c.seeded_triads(959981) and len(a)==7
    assert all(sorted(order)==sorted(c.ROLES) for order in a)
    assert len({tuple(order) for order in a})>1
    with pytest.raises(RuntimeError):c.seeded_triads(1,6)


def test_scheduler_rejects_reentrant_operation_and_recovers():
    scheduler=c.SerialScheduler()
    class Reentrant:
        def request(self,command):return scheduler.request(self,command)
    with pytest.raises(RuntimeError,match='overlapping'):scheduler.request(Reentrant(),{})
    assert scheduler.active is False
    class Plain:
        def request(self,command):return command
    assert scheduler.request(Plain(),dict(serial=True))==dict(serial=True)


@pytest.mark.parametrize('kind',['python','cython','swallowed'])
def test_untimed_antiforwarding_restores_profile_and_cython_identities(monkeypatch,kind):
    def original():return 1
    cy=types.SimpleNamespace(**{name:original for name in ('_geometric_mean','_noise_filter','_subtract_mean','_windowed_hist')})
    generic=types.SimpleNamespace(generic_cy=cy);h=types.SimpleNamespace(NAMES=('x',),resolve=lambda *_:original)
    importer=builtins.__import__
    monkeypatch.setattr(builtins,'__import__',lambda name,*a,**k:generic if name=='skimage.filters.rank.generic' else importer(name,*a,**k))
    previous=sys.getprofile();entries=dict(vars(cy))
    with pytest.raises((AssertionError,RuntimeError),match='forwarding'):
        with c.no_forwarding(h):
            if kind=='cython':cy._noise_filter()
            elif kind=='swallowed':
                try:original()
                except AssertionError:pass
            else:original()
    assert sys.getprofile() is previous and vars(cy)==entries


def test_reference_only_cannot_enter_native_or_barrier_modes(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'harness',lambda *_:pytest.fail('fixtures reached'))
    with pytest.raises(RuntimeError,match='reference receipt'):c.main(['barrier','--reference-only'])
    with pytest.raises(RuntimeError,match='requires'):c.main(['reference','--reference-only','--output',str(tmp_path/'x.json')])


def test_fixed_corpus_and_all376_ids_preserved():
    old=json.loads((ROOT/'coverage/contracts/rank-statistics-benchmark.json').read_text());cfg=c.config()
    assert old['case_inventory']==cfg['case_inventory'] and len(cfg['case_inventory'])==1379
    assert cfg['case_contract_sha256']==old['case_contract_sha256'] and len(cfg['timed_ids'])==len(set(cfg['timed_ids']))==376
    for name,value in cfg['source_text_sha256'].items():assert c.text_sha(ROOT/name)==value
    assert not any(cfg['accepted_products'][r][p] for r in ('before','after') for p in c.PROFILES)


def test_timer_is_confined_to_validated_measurement_worker_sample_branch():
    tree=ast.parse((ROOT/'benchmarks/rank_statistics_compare.py').read_text())
    owners={}
    for function in [n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]:
        owners[function.name]=[n.func.id for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='perf_counter_ns']
    assert owners['worker_main']==['perf_counter_ns','perf_counter_ns']
    assert all(not v for k,v in owners.items() if k!='worker_main')
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker_main')
    text=ast.unparse(function)
    assert text.index('early_native_admission()')<text.index('harness(')<text.index('cases = list(')
    assert text.index('sample_admission(')<text.index('from time import perf_counter_ns')


def test_workflow_barrier_and_measurement_host_have_no_build_or_pytest():
    text=(ROOT/'.github/workflows/rank-statistics-comparison.yml').read_text()
    measure=text.split('  measure:',1)[1]
    assert 'needs: [prepare, barrier]' in measure
    assert not any(token in measure for token in ('pip wheel','cmake --build','pytest','python -m build'))
    assert '--no-index --no-deps --require-hashes' in measure and '--only-binary=:all:' in measure
    assert 'rank-statistics-compare-prepared-${{ matrix.profile }}' in measure
    assert 'receipt_barrier' in text and '--check-only' in text and "SKIMAGEXX_DISABLE_SIMD" not in measure


@pytest.mark.parametrize('failure',['missing-mode','missing-backend','missing-workers','digest','alias','same-core','same-pid','private-source','report-timed','report-bytes'])
def test_complete_receipts_bind_actual_full_untimed_reports(tmp_path,monkeypatch,failure):
    cfg,paths=receipts(tmp_path,monkeypatch);p=paths[0].parent/'report.json';report=c.load_json(p)
    if failure=='missing-mode':report['checks'].pop('scalar')
    elif failure=='missing-backend':report['checks']['normal'].pop('before')
    elif failure=='missing-workers':report['checks']['normal']['before']['checks'].pop()
    elif failure=='digest':report['checks']['normal']['before']['checks'][0]['digests'][0]='tamper'
    elif failure=='alias':report['checks']['normal']['after']['aliases'].pop()
    elif failure=='same-core':report['worker_identities']['normal']['after']['core_sha256']='a'*64
    elif failure=='same-pid':report['worker_identities']['normal']['after']['pid']=1
    elif failure=='private-source':report['worker_identities']['normal']['before']['source']=c.BLOCKED_SOURCE
    elif failure=='report-timed':report['results']=[{}]
    else:report['schema']='tamper'
    p.write_text(json.dumps(report))
    if failure!='report-bytes':
        receipt=c.load_json(paths[0]);receipt['report_sha256']=c.sha(p);paths[0].write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError):c.receipt_barrier(paths,cfg,'run','1')


def test_measurement_receipts_cannot_be_replayed_with_a_different_output(tmp_path):
    receipt=tmp_path/'receipt.json';receipt.write_text('{}')
    root=tmp_path/'runner-temp'
    first=c.claim_measurement([receipt],'modern','run','1',root)
    assert first.is_file()
    with pytest.raises(FileExistsError):c.claim_measurement([receipt],'modern','run','1',root)
    # Distinct profile is the independently scheduled third host, not reuse.
    assert c.claim_measurement([receipt],'minimum','run','1',root)!=first


@pytest.mark.parametrize('failure',['unvalidated','check-only','local','worker4','scalar','untimed'])
def test_sample_guard_rejects_before_any_operation_timer(failure):
    state=dict(validated=True,measure=True,actions='true',workers=1,scalar=False,index=3,timed_ids=[3])
    if failure=='unvalidated':state['validated']=False
    elif failure=='check-only':state['measure']=False
    elif failure=='local':state['actions']=None
    elif failure=='worker4':state['workers']=4
    elif failure=='scalar':state['scalar']=True
    else:state['index']=4
    with pytest.raises(RuntimeError):c.sample_admission(**state)


def early_metadata(monkeypatch):
    # Only metadata objects; these must never be interpreted as public proof.
    state={'workers':1};core=types.SimpleNamespace(__name__='metadata_core',get_num_threads=lambda:state['workers'],set_num_threads=lambda n:state.update(workers=n),rank_statistics_interface='rank-statistics-v1',rank_statistics_capabilities=0xe000,rank_vector_interface='rank-vector-v1',rank_vector_capabilities=0x10000)
    entries=[]
    for name in ('rank_statistics_run','rank_windowed_histogram_run','rank_noise_footprint'):
        def entry():raise AssertionError('metadata stub cannot execute algorithms')
        entry.__name__=name;entry.__module__=core.__name__;setattr(core,name,entry);entries.append(entry)
    predicate=c.inspect.isbuiltin;monkeypatch.setattr(c.inspect,'isbuiltin',lambda obj:any(obj is e for e in entries) or predicate(obj))
    xx=types.ModuleType('skimagexx');xx._core=core;xx.get_num_threads=core.get_num_threads;xx.set_num_threads=core.set_num_threads;xx.set_fallback_enabled=lambda _:None
    rank=types.ModuleType('skimagexx.filters.rank');generic=types.ModuleType('skimagexx.filters.rank.generic')
    names=('geometric_mean','noise_filter','subtract_mean','windowed_histogram');paths={prefix+n for prefix in ('filters.rank.','filters.rank.generic.') for n in names}
    for name in names:
        namespace={'_core':core,'__name__':'skimagexx._rank'};exec('def '+name+'(*a,**k): raise AssertionError("metadata only")',namespace)
        setattr(rank,name,namespace[name]);setattr(generic,name,namespace[name])
    coverage=types.ModuleType('skimagexx.coverage');coverage.native_functions=lambda:sorted(paths)
    for name,module in (('skimagexx',xx),('skimagexx.coverage',coverage),('skimagexx.filters.rank',rank),('skimagexx.filters.rank.generic',generic)):monkeypatch.setitem(sys.modules,name,module)
    return xx,rank,generic,core,state,paths


def test_early_public_guard_accepts_only_explicit_harness_metadata(monkeypatch):
    xx,*_=early_metadata(monkeypatch)
    assert c.early_native_admission() is xx


@pytest.mark.parametrize('failure',['missing-path','float-capability','negative-capability','wrong-interface','wrong-entry-name','spoof-entry','upstream','alternate-core','alias','worker4','setter'])
def test_early_native_guard_fails_before_fixture_import(monkeypatch,failure):
    xx,rank,generic,core,state,paths=early_metadata(monkeypatch)
    if failure=='missing-path':paths.pop()
    elif failure=='float-capability':core.rank_statistics_capabilities=float(0xe000)
    elif failure=='negative-capability':core.rank_vector_capabilities=-1
    elif failure=='wrong-interface':core.rank_vector_interface='private'
    elif failure=='wrong-entry-name':core.rank_statistics_run.__name__='rank_run'
    elif failure=='spoof-entry':core.rank_statistics_run=type('builtin_function_or_method',(),{'__name__':'rank_statistics_run','__module__':core.__name__})()
    elif failure=='upstream':rank.geometric_mean.__module__='skimage.filters.rank.generic'
    elif failure=='alternate-core':rank.geometric_mean.__globals__['_core']=object()
    elif failure=='alias':generic.geometric_mean=rank.noise_filter
    elif failure=='worker4':state['workers']=4
    else:xx.set_num_threads=lambda n:None
    monkeypatch.setattr(c,'harness',lambda *_:pytest.fail('fixture imported'))
    with pytest.raises(RuntimeError):c.early_native_admission()


def test_direct_measurement_worker_needs_authenticated_parent_permit(monkeypatch):
    monkeypatch.setenv('GITHUB_ACTIONS','true')
    with pytest.raises(RuntimeError,match='permit'):c.worker_permit(None,'modern',c.config())


def permit_metadata(tmp_path,monkeypatch):
    cfg,paths=receipts(tmp_path,monkeypatch)
    monkeypatch.setenv('GITHUB_ACTIONS','true');monkeypatch.setenv('GITHUB_RUN_ID','run');monkeypatch.setenv('GITHUB_RUN_ATTEMPT','1')
    nonces,authentication=c.startup_plan();authentication['controller_pid']=c.os.getppid()
    claim=c.claim_measurement(paths,'windows','run','1',tmp_path/'r',authentication)
    permit=dict(profile='windows',**authentication,receipts=[str(p) for p in paths],claim=str(claim),claim_sha256=c.sha(claim))
    path=tmp_path/'permit.json';path.write_text(json.dumps(permit))
    frame=c.startup_frame(authentication,'before','windows','run','1',c.os.getpid(),nonces['before'])
    return cfg,path,permit,frame


@pytest.mark.parametrize('failure',['parent','tool','claim','profile'])
def test_worker_permit_rejects_wrong_parent_source_or_claim(tmp_path,monkeypatch,failure):
    cfg,path,permit,frame=permit_metadata(tmp_path,monkeypatch)
    c.worker_permit(path,'windows',cfg,'before',io.StringIO(json.dumps(frame)+'\n'))
    if failure=='parent':permit['controller_pid']=-1
    elif failure=='tool':permit['tool_sha256']='tampered'
    elif failure=='profile':permit['profile']='modern'
    else:permit['claim_sha256']='tampered'
    path.write_text(json.dumps(permit))
    with pytest.raises(RuntimeError):c.worker_permit(path,'windows',cfg,'before',io.StringIO(json.dumps(frame)+'\n'))


@pytest.mark.parametrize('failure',['nonce','source','stale-run','stale-attempt','wrong-role','wrong-launcher','missing-frame','unknown-parent'])
def test_stdio_startup_rejects_unbound_or_stale_controller_frame(tmp_path,monkeypatch,failure):
    cfg,path,permit,frame=permit_metadata(tmp_path,monkeypatch)
    if failure=='nonce':frame['nonce']='0'*64
    elif failure=='source':frame['tool_sha256']='bad'
    elif failure=='stale-run':frame['generation']='old'
    elif failure=='stale-attempt':frame['attempt']='0'
    elif failure=='wrong-role':frame['role']='after'
    elif failure=='wrong-launcher':frame['launcher_pid']=-1
    elif failure=='unknown-parent':
        claim=c.load_json(permit['claim']);claim['authentication']['controller_pid']=permit['controller_pid']+100000
        permit['controller_pid']=claim['authentication']['controller_pid'];frame['controller_pid']=permit['controller_pid'];frame['launcher_pid']=c.os.getpid()+100000
        Path(permit['claim']).write_text(json.dumps(claim));permit['claim_sha256']=c.sha(permit['claim']);path.write_text(json.dumps(permit))
    stream=io.StringIO('' if failure=='missing-frame' else json.dumps(frame)+'\n')
    with pytest.raises(RuntimeError):c.worker_permit(path,'windows',cfg,'before',stream)


def test_consumed_stdio_permit_cannot_start_the_same_role_again(tmp_path,monkeypatch):
    cfg,path,permit,frame=permit_metadata(tmp_path,monkeypatch)
    result=c.worker_permit(path,'windows',cfg,'before',io.StringIO(json.dumps(frame)+'\n'))
    assert result['mechanism']=='one-use-inherited-stdio-v1'
    with pytest.raises(FileExistsError):c.worker_permit(path,'windows',cfg,'before',io.StringIO(json.dumps(frame)+'\n'))


def test_real_existing_venv_subprocess_authenticates_exact_stdio_chain(tmp_path):
    profile='windows' if c.os.name=='nt' else 'modern'
    result=c.startup_probe(sys.executable,tmp_path/'real-probe',profile)
    assert result['probe_only'] and not result['native_admission'] and not result['barrier_eligible'] and result['operation_timer_calls']==0
    reply=result['reply'];assert reply['probe_only'] and reply['worker_pid']!=result['controller_pid']
    assert reply['controller_pid']==result['controller_pid'] and reply['launcher_pid']==result['popen_pid']
    assert reply['worker_parent_pid'] in (result['controller_pid'],result['popen_pid'])
    assert result['returncode']==0


def test_validation_finishes_before_any_one_worker_warm_or_sample():
    tree=ast.parse((ROOT/'benchmarks/rank_statistics_compare.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='controller')
    text=ast.unparse(function)
    assert text.index('validate_workers(workers, scheduler)')<text.index("operation='warm'")<text.index("operation='sample'")
    assert text.index('receipt_barrier(')<text.index('Worker(')


@pytest.mark.parametrize('failure',['missing-scope','cross-coverage','missing-profile','template','hash','run-binding'])
def test_external_experiment_is_immutable_complete_and_common_coverage(tmp_path,monkeypatch,failure):
    cfg,_=receipts(tmp_path,monkeypatch)
    external=dict(schema='rank-statistics-experiment-v1',template_sha256=c.text_sha(c.CONTRACT),accepted_products=cfg['accepted_products'])
    if failure=='missing-scope':external['accepted_products']['before']['windows'].pop('scope')
    elif failure=='cross-coverage':
        scope=dict(external['accepted_products']['after']['windows']['scope']);scope['image_paths']+=1;external['accepted_products']['after']['windows']['scope']=scope
    elif failure=='missing-profile':external['accepted_products']['after'].pop('minimum')
    elif failure=='template':external['template_sha256']='bad'
    elif failure=='run-binding':external['accepted_products']['before']['windows']['artifact_run']=''
    p=tmp_path/'external.json';p.write_text(json.dumps(external))
    with pytest.raises(RuntimeError):c.config(p,'wrong' if failure=='hash' else c.sha(p))


def test_external_config_does_not_rewrite_shared_template(tmp_path,monkeypatch):
    cfg,_=receipts(tmp_path,monkeypatch);before=c.CONTRACT.read_bytes()
    p=tmp_path/'external.json';p.write_text(json.dumps(dict(schema='rank-statistics-experiment-v1',template_sha256=c.text_sha(c.CONTRACT),accepted_products=cfg['accepted_products'])))
    loaded=c.config(p,c.sha(p))
    assert loaded['experiment_sha256']==c.sha(p) and c.CONTRACT.read_bytes()==before
    assert all(c.config()['accepted_products'][r][p] is None for r in ('before','after') for p in c.PROFILES)


def test_counts_patch_allows_line_offsets_but_no_changed_context():
    patch='@@ -1 +1 @@\n-old_bookkeeping\n+counts_population_only\n'
    assert c.apply_counts_patch('prefix\nold_bookkeeping\nsuffix\n',patch)=='prefix\ncounts_population_only\nsuffix\n'
    with pytest.raises(RuntimeError):c.apply_counts_patch('old_bookkeeping\nold_bookkeeping\n',patch)
    with pytest.raises(RuntimeError):c.apply_counts_patch('different\n',patch)


@pytest.mark.parametrize('failure',['other-source','other-rank-edit','missing-counts','scope','registry'])
def test_full_source_pair_only_allows_the_approved_counts_delta(tmp_path,monkeypatch,failure):
    # This existing unit isolates the source-patch guard. Separate tests below
    # exercise the real common-build gate with byte-authenticated fake inputs.
    monkeypatch.setattr(c,'verify_common_build',lambda *a:dict(schema=c.BUILD_IDENTITY_SCHEMA,common_sha256='8'*64,products=dict(before='6'*64,after='7'*64)))
    cfg=c.config();scope=cfg['planned_scope'];patch='@@ -1 +1 @@\n-old_bookkeeping\n+counts_population_only\n'
    cfg['counts_cpp_patch']=patch;cfg['counts_cpp_patch_sha256']=hashlib.sha256(patch.encode()).hexdigest()
    pin=lambda role:dict(revision=('a' if role=='before' else 'b')*40,qualification_sha256='1'*64,product_sha256='2'*64,wheel_sha256='3'*64,source_sha256='4'*64,scope=scope,artifact_run='1',artifact_name='metadata-only')
    cfg['accepted_products']={r:{'windows':pin(r)} for r in ('before','after')}
    for role,text in (('before','old_bookkeeping\n'),('after','counts_population_only\n')):
        root=tmp_path/role;(root/'source/src').mkdir(parents=True);(root/'source/src/rank.cpp').write_text(text)
        (root/'qualification.json').write_text(json.dumps(dict(expected_registry=dict(functions=['same'],image_count=186))))
    c.verify_pair_sources(tmp_path/'before',tmp_path/'after','windows',cfg)
    if failure=='other-source':(tmp_path/'after/source/src/other.cpp').write_text('changed')
    elif failure=='other-rank-edit':(tmp_path/'after/source/src/rank.cpp').write_text('counts_population_only\nextra_edit\n')
    elif failure=='missing-counts':(tmp_path/'after/source/src/rank.cpp').write_text('old_bookkeeping\n')
    elif failure=='scope':cfg['accepted_products']['after']['windows']['scope']=dict(scope,image_paths=190)
    else:(tmp_path/'after/qualification.json').write_text(json.dumps(dict(expected_registry=dict(functions=['different'],image_count=186))))
    with pytest.raises(RuntimeError):c.verify_pair_sources(tmp_path/'before',tmp_path/'after','windows',cfg)


@pytest.mark.parametrize('field',['canonical_cases','alias_checks','workers','timed_ids'])
def test_preparation_scope_rejects_float_or_boolean_integer_lookalikes(tmp_path,monkeypatch,field):
    cfg,paths=receipts(tmp_path,monkeypatch);row=c.load_json(paths[0])
    if field=='canonical_cases':row[field]=1379.0
    elif field=='alias_checks':row[field]=4.0
    elif field=='workers':row[field]=[True,4]
    else:row[field]=[float(v) for v in row[field]]
    paths[0].write_text(json.dumps(row))
    with pytest.raises(RuntimeError):c.receipt_barrier(paths,cfg,'run','1')


def test_scope_totals_are_explicit_and_not_hardwired_for_future_common_coverage(tmp_path,monkeypatch):
    cfg,_=receipts(tmp_path,monkeypatch)
    for role in ('before','after'):
        for profile in c.PROFILES:
            scope=dict(cfg['accepted_products'][role][profile]['scope']);scope['image_paths']=190
            scope['sdk_families']=14;scope['sdk_family_names']=[*scope['sdk_family_names'],'rank_decisions']
            cfg['accepted_products'][role][profile]['scope']=scope
    path=tmp_path/'external.json';path.write_text(json.dumps(dict(schema='rank-statistics-experiment-v1',template_sha256=c.text_sha(c.CONTRACT),accepted_products=cfg['accepted_products'])))
    loaded=c.config(path,c.sha(path))
    assert c.accepted_pin(loaded,'before','windows')['scope']['image_paths']==190
    assert c.accepted_pin(loaded,'after','windows')['scope']['sdk_families']==14


def synthetic_build_pair(tmp_path,change=None,compiler_id='MSVC',lto=False):
    cfg=c.config();cfg['accepted_products']={r:{} for r in ('before','after')};roots=[]
    for role,revision in (('before','a'*40),('after','b'*40)):
        path=tmp_path/role;path.mkdir();facts=synthetic_build_facts(path,revision,compiler_id,lto)
        if role=='after' and change:change(facts)
        identity,q=capture_synthetic_build(path,facts);q['expected_registry']={'functions':['same']};(path/'qualification.json').write_text(json.dumps(q))
        cfg['accepted_products'][role]['windows']=dict(revision=revision,qualification_sha256=c.sha(path/'qualification.json'),product_sha256='1'*64,wheel_sha256=c.sha(path/'wheels/native.whl'),source_sha256='2'*64,scope=cfg['planned_scope'],artifact_run='1',artifact_name='SYNTHETIC-NONNATIVE')
        roots.append(path)
    return roots,cfg


def test_actual_metadata_capture_does_not_execute_tools_or_import_numerical_code(tmp_path,monkeypatch):
    path=tmp_path/'SYNTHETIC';path.mkdir();facts=synthetic_build_facts(path)
    monkeypatch.setattr(c.subprocess,'Popen',lambda *a,**k:pytest.fail('tool execution reached'))
    original=builtins.__import__
    def checked(name,*args,**kwargs):
        assert name.split('.')[0] not in ('numpy','scipy','skimage','skimagexx')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',checked)
    identity,q=capture_synthetic_build(path,facts)
    result=c.verify_build_identity(path,'windows',q,facts['revision'])
    assert result['common_sha256']==identity['common_sha256'] and identity['raw']==facts
    assert identity['file_records']['binaries']['backend']['sha256']==c.sha(facts['binaries']['backend'])


@pytest.mark.parametrize('compiler_id,lto',[('MSVC',False),('GNU',False),('GNU',True)])
def test_common_build_is_root_normalized_and_generated_outputs_are_observations(tmp_path,compiler_id,lto):
    def variation(facts):
        Path(facts['translation_units'][0]['source']).write_text('SYNTHETIC approved counts change\n')
        # Real LTO outputs can contain nonreproducible stamps. Every output raw
        # hash remains in its role receipt; no input library/header is masked.
        for command in [*facts['translation_units'],*facts['link_commands']]:
            for p in command['outputs']:
                if p!=facts['core']:Path(p).write_bytes(Path(p).read_bytes()+b' generated stamp')
    roots,cfg=synthetic_build_pair(tmp_path,variation,compiler_id,lto)
    result=c.verify_common_build(*roots,'windows',cfg)
    assert result['schema']==c.BUILD_IDENTITY_SCHEMA and len(result['common_sha256'])==64
    before=c.load_json(roots[0]/'build-identity.json');after=c.load_json(roots[1]/'build-identity.json')
    assert before['file_records']['outputs']!=after['file_records']['outputs']
    assert before['common']==after['common']


@pytest.mark.parametrize('change',['compiler','backend','linker','toolset','flags','FP','OpenMP','AVX2','LTO','header','numpy-header','external-build-env-library','other-source'])
def test_common_build_rejects_any_different_actual_common_input(tmp_path,change):
    def mutate(facts):
        if change in ('compiler','backend','linker'):Path(facts['binaries'][change]).write_bytes(b'DIFFERENT TOOL')
        elif change=='toolset':facts['configuration']['toolset']='different-toolset'
        elif change=='flags':facts['translation_units'][0]['argv'].append('-funroll-loops')
        elif change=='FP':facts['translation_units'][0]['argv'].append('-ffast-math')
        elif change in ('OpenMP','AVX2','LTO'):facts['configuration'][change.lower()]=not facts['configuration'][change.lower()]
        elif change in ('header','numpy-header'):
            family='toolchain' if change=='header' else 'numpy';Path(facts['header_roots'][family][0],'header.h').write_bytes(b'DIFFERENT HEADER')
        elif change=='external-build-env-library':Path(next(r['path'] for r in facts['link_inputs'] if not r['generated'])).write_bytes(b'DIFFERENT BUILD-ENV LIBRARY')
        else:Path(facts['translation_units'][1]['source']).write_text('UNAPPROVED SOURCE\n')
    roots,cfg=synthetic_build_pair(tmp_path,mutate)
    with pytest.raises(RuntimeError,match='compiler, headers, used inputs or build flags differ'):c.verify_common_build(*roots,'windows',cfg)


@pytest.mark.parametrize('role',['lto_backend','lto_wrapper','lto_plugin'])
def test_executed_gnu_lto_binaries_are_mandatory(tmp_path,role):
    path=tmp_path/'SYNTHETIC';path.mkdir();facts=synthetic_build_facts(path,compiler_id='GNU',lto=True);facts['binaries'].pop(role)
    with pytest.raises(RuntimeError,match='LTO tool'):capture_synthetic_build(path,facts)


@pytest.mark.parametrize('role',['lto_backend','lto_wrapper','lto_plugin'])
def test_actual_lto_binary_byte_difference_rejects_common_build(tmp_path,role):
    def mutate(facts):Path(facts['binaries'][role]).write_bytes(b'DIFFERENT EXECUTED LTO TOOL')
    roots,cfg=synthetic_build_pair(tmp_path,mutate,'GNU',True)
    with pytest.raises(RuntimeError,match='compiler, headers, used inputs or build flags differ'):c.verify_common_build(*roots,'windows',cfg)


@pytest.mark.parametrize('flag',['-B/unrecorded/backend','/B1unrecorded','-fplugin=/unrecorded/plugin.so','--specs=/unrecorded/specs'])
def test_unrecorded_compiler_overrides_or_plugins_fail_closed(tmp_path,flag):
    path=tmp_path/'SYNTHETIC';path.mkdir();facts=synthetic_build_facts(path);facts['translation_units'][0]['argv'].append(flag)
    with pytest.raises(RuntimeError):capture_synthetic_build(path,facts)


@pytest.mark.parametrize('failure',['backend','flags','response','link-response','used-headers','header-family','link-edges','generated-external','global-inputs','source-root','LTO-hidden'])
def test_build_capture_fails_closed_on_missing_or_untraceable_facts(tmp_path,failure):
    path=tmp_path/'SYNTHETIC';path.mkdir();facts=synthetic_build_facts(path)
    if failure=='backend':facts['binaries'].pop('backend')
    elif failure=='flags':facts['translation_units'][0]['argv']=[]
    elif failure=='response':facts['translation_units'][0]['argv'].append('@unexpanded.rsp')
    elif failure=='link-response':facts['link_commands'][1]['argv'].append('-Wl,@unexpanded.rsp')
    elif failure=='used-headers':facts['translation_units'][0]['dependencies']=[]
    elif failure=='header-family':facts['header_roots'].pop('python')
    elif failure=='link-edges':facts['link_commands'][0].pop('inputs')
    elif failure=='generated-external':next(r for r in facts['link_inputs'] if not r['generated'])['generated']=True
    elif failure=='global-inputs':facts['link_inputs'].pop()
    elif failure=='source-root':facts['source_root']=str(Path(facts['source_root']).anchor)
    else:facts['translation_units'][0]['argv'].append('-flto=auto')
    with pytest.raises(RuntimeError):capture_synthetic_build(path,facts)


def test_debug_capture_is_honest_but_never_release_comparison_admission(tmp_path):
    path=tmp_path/'SYNTHETIC';path.mkdir();facts=synthetic_build_facts(path);facts['profile']='debug';facts['configuration']['build_type']='Debug'
    identity,q=capture_synthetic_build(path,facts);q['profile']='debug'
    with pytest.raises(RuntimeError,match='Release'):c.verify_build_identity(path,'debug',q,facts['revision'])
    assert identity['raw']['configuration']['build_type']=='Debug'


@pytest.mark.parametrize('failure',['missing','unqualified','evidence-unqualified','evidence-tamper','common-tamper','source-tamper'])
def test_identity_and_raw_evidence_require_original_qualification_binding(tmp_path,failure):
    path=tmp_path/'SYNTHETIC';path.mkdir();facts=synthetic_build_facts(path);identity,q=capture_synthetic_build(path,facts)
    if failure=='missing':(path/'build-identity.json').unlink()
    elif failure=='unqualified':q['records'].pop('build-identity.json')
    elif failure=='evidence-unqualified':q['records'].pop(next(iter(identity['evidence_records'])))
    elif failure=='evidence-tamper':(path/next(iter(identity['evidence_records']))).write_bytes(b'changed raw facts')
    elif failure=='source-tamper':Path(facts['translation_units'][0]['source']).write_text('changed source\n')
    else:
        identity['common']['configuration']['toolset']='tamper';(path/'build-identity.json').write_text(json.dumps(identity));q['records']['build-identity.json']=dict(bytes=(path/'build-identity.json').stat().st_size,sha256=c.sha(path/'build-identity.json'))
    with pytest.raises(RuntimeError):c.verify_build_identity(path,'windows',q,facts['revision'])


def test_old_qualified_product_without_build_capture_fails_before_fixtures(fake_product,monkeypatch):
    path,cfg,reseal=fake_product;q=c.load_json(path/'qualification.json');q['records'].pop('build-identity.json');(path/'qualification.json').write_text(json.dumps(q));cfg['accepted_products']['before']['windows']['qualification_sha256']=c.sha(path/'qualification.json');reseal()
    monkeypatch.setattr(c,'harness',lambda *_:pytest.fail('fixture import reached'))
    with pytest.raises(RuntimeError,match='build identity missing'):c.verify_product(path,'before','windows',cfg)


def test_declared_roots_do_not_normalize_unrelated_flags_or_paths():
    roots=[('/owned/source','$SOURCE'),('/owned/build','$BUILD')]
    assert c._normalize_build_argument('-I/owned/source/include',roots)=='-I$SOURCE/include'
    assert c._normalize_build_argument('/Fo/owned/build/a.obj',roots)=='/Fo$BUILD/a.obj'
    for arg in ('-I/owned/source-other/include','-I/system/include','-O3','-fplugin=/system/compiler/plugin.so'):
        assert c._normalize_build_argument(arg,roots)==arg


@pytest.mark.parametrize('field',['common_sha256','before_identity','missing'])
def test_barrier_binds_original_preparation_common_build_identity(tmp_path,monkeypatch,field):
    cfg,paths=receipts(tmp_path,monkeypatch);row=c.load_json(paths[0]);report_path=paths[0].parent/'report.json';report=c.load_json(report_path)
    if field=='common_sha256':row['products']['after']['build_common_sha256']='9'*64
    elif field=='before_identity':row['products']['before']['build_identity_sha256']='9'*64
    else:report['source_comparison'].pop('build_identity');report_path.write_text(json.dumps(report));row['report_sha256']=c.sha(report_path)
    paths[0].write_text(json.dumps(row))
    with pytest.raises(RuntimeError):c.receipt_barrier(paths,cfg,'run','1')
