"""TV harness metadata guards; synthetic modules never qualify native code."""
import ast
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import skimagexx as xx
from benchmarks import tv_chambolle as h


def manifest():return json.loads(h.MANIFEST.read_text())


def test_fixed400_ids_and330_source_observations_are_not_trimmed():
    m=manifest();cases=h.corpus();source=json.loads(h.SOURCE_CONTRACT.read_text())
    assert len(cases)==400 and sum(c.timed for c in cases)==70
    assert [c.fixture for c in cases if c.source_id]==source['cases']
    assert all(not c.timed for c in cases if c.source_id)
    assert [h.inventory(c) for c in cases]==m['case_inventory']
    assert h.digest(m['case_inventory'])==m['case_contract_sha256']
    assert h.text_sha(h.SOURCE_CONTRACT)==m['source_contract_sha256']
    assert all(h.text_sha(h.ROOT/p)==sha for p,sha in m['fixture_source_sha256'].items())
    assert m['accepted_public'] is None and all(v is None for v in m['accepted_products'].values())


@pytest.mark.parametrize('case',[c for c in h.corpus() if c.timed],ids=lambda c:c.id)
def test_timing_fixture_storage_data_hash_and_workspace_certificate(case):
    passed,image,owner,options,events=h.fixture(case)
    other=h.fixture(case)[1]
    assert h.contracts().array(image)==h.contracts().array(other)
    assert not events and image.size<=257*259
    spatial=image.ndim-(1 if 'channel_axis' in options else 0)
    assert image.size*8*(4*spatial+7)<=h.LIMIT
    assert np.isfinite(image).all()
    assert case.source_id is None and case.fixture['cohort'] in ('tiny','small','raster','volume','channels','layout')


@pytest.mark.parametrize('mode',['unregistered','stage-only','missing-products'])
def test_native_pending_rejects_before_fixtures_capsules_or_timers(monkeypatch,mode):
    m=manifest()
    monkeypatch.setattr(h,'contracts',lambda:pytest.fail('fixture import reached before admission'))
    monkeypatch.setattr(h,'measure',lambda *_:pytest.fail('timer reached'))
    coverage=__import__('skimagexx.coverage',fromlist=['native_functions'])
    monkeypatch.setattr(coverage,'native_functions',lambda:() if mode=='unregistered' else (h.API,))
    if mode=='missing-products':m['accepted_public']={'stage':'not sufficient'}
    with pytest.raises(RuntimeError,match='unregistered|public TV admission pending'):h.public_admission(m)


@pytest.mark.parametrize('argv',[
    ['--reference-only'],['--reference-only','--check-only','--capsule','missing'],
    ['--check-only','--threads','4'],['--check-only','--repeat','2'],
    ['--check-only','--threads','1','1'],['--prepare-capsule','--check-only','--threads','1'],
    [],['--threads','1','4','--output','ignored','--capsule','missing'],
])
def test_bad_cli_modes_reject_without_numerical_calls(monkeypatch,argv):
    monkeypatch.delenv('GITHUB_ACTIONS',raising=False)
    monkeypatch.setattr(h,'contracts',lambda:pytest.fail('fixture creation reached'))
    monkeypatch.setattr(h,'measure',lambda *_:pytest.fail('timer reached'))
    with pytest.raises(SystemExit):h.main(argv)


def test_reference_check_never_times_and_restores_fallback_workers(monkeypatch,tmp_path):
    case=h.corpus()[0];m=manifest();m['case_inventory']=[h.inventory(case)];m['case_contract_sha256']=h.digest(m['case_inventory']);m['reference_profiles']={}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(m));monkeypatch.setattr(h,'MANIFEST',path);monkeypatch.setattr(h,'corpus',lambda:(case,))
    monkeypatch.setattr(h,'measure',lambda *_:pytest.fail('check-only timer reached'))
    monkeypatch.setattr(h,'seal',lambda *_:pytest.fail('reference capsule reached'));monkeypatch.setattr(h,'verify',lambda *_:pytest.fail('reference capsule reached'))
    workers=xx.get_num_threads();fallback=xx.set_fallback_enabled(False);xx.set_fallback_enabled(fallback)
    output=tmp_path/'report.json'
    assert h.main(['--reference-only','--check-only','--output',str(output)])==0
    report=json.loads(output.read_text());assert report['status']=='complete-reference-check' and report['metadata']['native_checks']==0 and not report['results']
    assert xx.get_num_threads()==workers
    assert xx.set_fallback_enabled(fallback)==fallback


@pytest.mark.parametrize('forward',['public','helper'])
def test_exact_original_forwarding_is_caught_and_prior_profile_restored(forward):
    case=h.corpus()[0];module=__import__('skimage.restoration._denoise',fromlist=['_denoise_tv_chambolle_nd'])
    def existing(frame,event,arg):pass
    before=sys.getprofile();sys.setprofile(existing)
    try:
        fn=h.resolve(True) if forward=='public' else lambda image,**options:module._denoise_tv_chambolle_nd(image)
        with pytest.raises(AssertionError,match='forwarding detected'):h.native_observe(fn,case)
        assert sys.getprofile() is existing
    finally:sys.setprofile(before)


def test_whole_call_callback_excludes_fixture_but_keeps_output_allocation():
    case=next(c for c in h.corpus() if c.timed);calls=[]
    def function(image,**options):calls.append((image,options));return image.copy()
    callback,unchanged=h.timing_callback(function,case)
    a=callback();b=callback();assert a is not b and calls[0][0] is calls[1][0] and unchanged()
    assert a.shape==b.shape and a.tobytes()==b.tobytes()


def test_missing_capsule_and_source_tamper_fail_closed(tmp_path):
    with pytest.raises(FileNotFoundError):h.verify(tmp_path)
    (tmp_path/'tv-chambolle-artifacts.json').write_text(json.dumps(dict(schema='tv-chambolle-exact-offline-v1',files_sha256={},corpus_manifest_sha256='0'*64,product_source={})))
    with pytest.raises(AssertionError):h.verify(tmp_path)


def test_checks_complete_before_only_measurement_entry_and_seven_rounds():
    tree=ast.parse(Path(h.__file__).read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    timers=[n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='measure'];assert len(timers)==1
    text=ast.unparse(main);assert text.index('if args.check_only:')<text.index('timing = measure(')
    assert "if report['failures']:" in text and text.index("if report['failures']:")<text.index('timing = measure(')
    assert ast.literal_eval(timers[0].args[2])==7
    assert "args.threads != [1]" in text and 'slower_than_original' in text


def test_source330_fixture_fullfile_pin_remains_strict():
    m=manifest();assert set(m['fixture_source_sha256'])=={'tests/test_tv_chambolle_contracts.py'}
    assert m['counts']==dict(source_untimed=330,additional_timed_candidates=70,total=400,algorithms=1,public_paths=1,aliases=0)


def test_workflow_has_all_three_preparation_barrier_and_fresh_artifact_measurement():
    text=(h.ROOT/'.github/workflows/tv-chambolle-benchmark.yml').read_text()
    prepare,measure=text.split('  measure:\n',1)
    assert 'needs: prepare' in measure and 'matrix:' in prepare and 'matrix:' in measure
    assert prepare.count('profile: windows')==1 and prepare.count('profile: modern')==1 and prepare.count('profile: minimum')==1
    assert 'steps.certificate.outputs.python' in prepare and 'steps.certificate.outputs.python' in measure
    assert '--prepare-capsule --check-only --threads 1 4' in prepare
    assert '--check-only --threads 1 4' in prepare and "SKIMAGEXX_DISABLE_SIMD: '1'" in prepare
    assert 'python -m pytest' in prepare and 'pip wheel' not in prepare and 'pip build' not in prepare
    assert 'rank-statistics-product-${{ matrix.profile }}' in prepare and 'qualified_run:' in text and 'qualified_source:' in text
    assert 'certificate_json:' in text and 'artifact_digest' in prepare and 'head_sha' in prepare
    assert 'python -m pip install --force-reinstall --no-index --no-deps --require-hashes' in prepare
    assert '--seal-preparation --check-only' in prepare and 'tv-chambolle-receipt-*' in measure
    assert 'pytest' not in measure and 'pip wheel' not in measure and 'cmake' not in measure
    assert '--no-index --no-deps --require-hashes' in measure and '--threads 1 --repeat 7' in measure
    assert 'actions/download-artifact@v4' in measure and 'if: always()' in measure
    assert 'admit_native' not in text and 'public_admission' in prepare


def test_timing_profiles_have_no_protocol_nonfinite_or_error_reclassification():
    rows=h.timing_profiles()
    assert {p['dtype'] for p in rows}=={'u1','f4','f8'}
    assert {len(p['shape']) for p in rows}=={2,3}
    assert {p.get('axis') for p in rows}=={None,0,-1}
    assert {p['iterations'] for p in rows}=={1,2,8,200}
    assert all(p['pattern'] in ('ramp','constant') for p in rows)
    assert not any(any(k in p for k in ('intake','parameter','warning_hook','fp_action','allocation','policy')) for p in rows)


@pytest.mark.parametrize('fault',('version','finite-version','float-cap','bool-cap','entry-name','spoof-entry','other-core','default4','signature','source-pin','binary-pin','profile','private-env','reference-pool4'))
def test_public_guard_metadata_stubs_reject_before_any_fixture(tmp_path,monkeypatch,fault):
    """Inert metadata-only entry stubs; never actual native/public proof."""
    binary=tmp_path/'tooling-core-not-a-binary';binary.write_bytes(b'harness-only')
    class Entry:
        def __init__(self,name):self.__name__=name;self.__module__='tooling-core'
        def __call__(self,*args):pytest.fail('metadata entry executed')
    core=types.SimpleNamespace(__name__='tooling-core',__file__=str(binary),tv_chambolle_interface='tv-chambolle-stages-v1',tv_chambolle_capabilities=63,tv_chambolle_finite_interface='tv-chambolle-finite-v1',tv_chambolle_finite_capabilities=3,tv_chambolle_finite_profiles=('1.24.4','2.5.4'))
    entries=('tv_chambolle_divergence_into','tv_chambolle_gradient_into','tv_chambolle_norm_into','tv_chambolle_dual_into','tv_chambolle_finite_run')
    for name in entries:setattr(core,name,Entry(name))
    wrapper=types.ModuleType('skimagexx._tv_chambolle');wrapper._core=core
    exec('def denoise_tv_chambolle(image, weight=0.1, eps=0.0002, max_num_iter=200, *, channel_axis=None):\n    raise AssertionError("harness stub must never execute")',wrapper.__dict__)
    function=wrapper.denoise_tv_chambolle;monkeypatch.setitem(sys.modules,wrapper.__name__,wrapper)
    workers=[1]
    def get():return workers[0]
    def set_workers(n):workers[0]=n
    core.get_num_threads=get;core.set_num_threads=set_workers
    monkeypatch.setattr(xx,'_core',core);monkeypatch.setattr(xx,'get_num_threads',get);monkeypatch.setattr(xx,'set_num_threads',set_workers);monkeypatch.setattr(xx,'set_fallback_enabled',lambda _:False)
    coverage=__import__('skimagexx.coverage',fromlist=['native_functions']);monkeypatch.setattr(coverage,'native_functions',lambda:(h.API,))
    importer=h.importlib.import_module
    monkeypatch.setattr(h.importlib,'import_module',lambda name:wrapper if name==wrapper.__name__ else importer(name))
    monkeypatch.setattr(h.inspect,'isbuiltin',lambda value:isinstance(value,Entry))
    original=h.resolve(True);monkeypatch.setattr(h,'resolve',lambda original_flag=False:original if original_flag else function)
    monkeypatch.setattr(h,'source_fingerprint',lambda _:dict(source_sha256='c'*64))
    monkeypatch.setattr(h,'contracts',lambda:pytest.fail('fixture import reached'));monkeypatch.setattr(h,'measure',lambda *_:pytest.fail('timer reached'))
    pin=dict(core_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),product_source_sha256='c'*64)
    if fault=='version':core.tv_chambolle_interface='wrong'
    elif fault=='finite-version':core.tv_chambolle_finite_interface='wrong'
    elif fault=='float-cap':core.tv_chambolle_finite_capabilities=3.
    elif fault=='bool-cap':core.tv_chambolle_finite_capabilities=True
    elif fault=='entry-name':core.tv_chambolle_finite_run=Entry('wrong')
    elif fault=='spoof-entry':core.tv_chambolle_finite_run=object()
    elif fault=='other-core':wrapper._core=object()
    elif fault=='default4':workers[0]=4
    elif fault=='signature':function.__signature__=__import__('inspect').Signature()
    elif fault=='source-pin':pin['product_source_sha256']='f'*64
    elif fault=='binary-pin':pin['core_sha256']='e'*64
    elif fault=='profile':core.tv_chambolle_finite_profiles=('2.5.4',)
    elif fault=='private-env':monkeypatch.setenv('SKIMAGEXX_TV_CHAMBOLLE_STAGE_CORE','forbidden-private-path')
    else:monkeypatch.setenv('OPENBLAS_NUM_THREADS','4')
    with pytest.raises(AssertionError):h.public_admission(manifest(),pin)

def tooling_certificate():
    """Synthetic pins for metadata guards only, never accepted product bytes."""
    families=['tooling'+str(i) for i in range(14)]+['tv_chambolle','tv_reductions','tv_denoise']
    pins={}
    for i,name in enumerate(h.PROFILES):
        pins[name]=dict(revision='a'*40,qualification_sha256='b'*64,product_sha256=str(i+1)*64,wheel_sha256='c'*64,source_sha256='d'*64,core_sha256='e'*64,product_source_sha256='f'*64,scope=dict(h.TV_SCOPE,sdk_family_names=families),artifact_run='123',artifact_name='rank-statistics-product-'+name,artifact_id=str(i+10),artifact_digest='9'*64,python_version='3.11.14' if name=='minimum' else '3.13.14')
    return dict(schema='tv-qualified-product-certificate-v1',status='accepted-qualified-tv',qualified_run='123',qualified_source='a'*40,profiles=pins)


@pytest.mark.parametrize('fault',('missing-profile','null','reference-only','bad-source','wrong-run','wrong-artifact','bad-digest','SDK59','family14','image190','duplicate-artifact','wrong-python'))
def test_certificate_rejects_unqualified_or_cross_scope_metadata(tmp_path,fault):
    c=tooling_certificate();p=c['profiles']['modern']
    if fault=='missing-profile':del c['profiles']['minimum']
    elif fault=='null':c['profiles']['minimum']=None
    elif fault=='reference-only':c['status']='reference-only'
    elif fault=='bad-source':c['qualified_source']='not-a-revision'
    elif fault=='wrong-run':p['artifact_run']='456'
    elif fault=='wrong-artifact':p['artifact_name']='public-qualification-modern'
    elif fault=='bad-digest':p['artifact_digest']='missing'
    elif fault=='SDK59':p['scope']['sdk_members']=59
    elif fault=='family14':p['scope']['sdk_families']=14
    elif fault=='image190':p['scope']['image_paths']=190
    elif fault=='duplicate-artifact':p['artifact_id']=c['profiles']['minimum']['artifact_id']
    else:p['python_version']='3.11.14'
    path=tmp_path/'certificate.json';path.write_text(json.dumps(c))
    with pytest.raises((AssertionError,TypeError)):h.certificate(path,'123','a'*40)


def tooling_barrier(tmp_path,monkeypatch):
    """No native code: exercise the pre-fixture receipt boundary only."""
    monkeypatch.setenv('GITHUB_ACTIONS','true');monkeypatch.setenv('GITHUB_RUN_ID','999');monkeypatch.setenv('GITHUB_RUN_ATTEMPT','2')
    capsule=tmp_path/'capsule';capsule.mkdir();c=tooling_certificate();cert=capsule/'tv-certificate.json';cert.write_text(json.dumps(c))
    seal=capsule/'tv-chambolle-artifacts.json';seal.write_bytes(b'synthetic metadata seal')
    barrier=tmp_path/'barrier';barrier.mkdir()
    for name,pin in c['profiles'].items():
        row=dict(schema='tv-preparation-receipt-v1',status='ready',profile=name,workflow_run='999',workflow_attempt='2',qualified_run='123',qualified_source='a'*40,certificate_sha256=h.artifacts.sha(cert),product_sha256=pin['product_sha256'],capsule_sha256=h.artifacts.sha(seal),modes=['normal','scalar'],workers=[1,4],case_count=400,timed_count=70,native_checks_per_mode=800,pipeline_default_finite_required=True)
        (barrier/('tv-receipt-'+name+'.json')).write_text(json.dumps(row))
    return barrier,capsule


@pytest.mark.parametrize('fault',('missing','duplicate','stale-run','stale-attempt','reference','one-worker','normal-only','partial-rows','wrong-product','wrong-source','wrong-cert','wrong-capsule','no-default-loop'))
def test_preparation_barrier_rejects_before_any_fixture_or_timer(tmp_path,monkeypatch,fault):
    barrier,capsule=tooling_barrier(tmp_path,monkeypatch)
    assert len(h.preparation_barrier(barrier,capsule))==3  # Inert metadata control only.
    path=barrier/('tv-receipt-'+h.profile()+'.json');row=json.loads(path.read_text())
    if fault=='missing':path.unlink()
    elif fault=='duplicate':(barrier/'tv-receipt-extra.json').write_text(path.read_text())
    else:
        key,value={'stale-run':('workflow_run','998'),'stale-attempt':('workflow_attempt','1'),'reference':('status','reference-only'),'one-worker':('workers',[1]),'normal-only':('modes',['normal']),'partial-rows':('case_count',399),'wrong-product':('product_sha256','0'*64),'wrong-source':('qualified_source','0'*40),'wrong-cert':('certificate_sha256','0'*64),'wrong-capsule':('capsule_sha256','0'*64),'no-default-loop':('pipeline_default_finite_required',False)}[fault]
        row[key]=value;path.write_text(json.dumps(row))
    monkeypatch.setattr(h,'contracts',lambda:pytest.fail('fixture import reached'));monkeypatch.setattr(h,'measure',lambda *_:pytest.fail('timer reached'))
    with pytest.raises(AssertionError):h.preparation_barrier(barrier,capsule)


def tooling_reports(path):
    """Inert report/XML inputs solely for fail-closed completeness guards."""
    import xml.etree.ElementTree as ET
    m=manifest();inventory=m['case_inventory']
    checks=[dict(index=i,**r,status='passed',actual_by_workers={'1':{},'4':{}}) for i,r in enumerate(inventory)]
    for mode in ('normal','scalar'):
        report=dict(status='complete-parity-check',failures=[],results=[],metadata=dict(check_only=True,reference_only=False,workers=[1,4],case_count=400,timing_eligible=70,native_checks=800,simd_mode=mode,case_contract_sha256=m['case_contract_sha256'],case_spec_sha256='a'*64,artifacts={'synthetic':'metadata only'}),checks=checks)
        (path/(mode+'.json')).write_text(json.dumps(report))
        suites=ET.Element('testsuites');suite=ET.SubElement(suites,'testsuite',tests='2148',failures='0',errors='0',skipped='0')
        for i in range(632):
            tagged=i>=628;name=('layout/tagged/'+str(i) if tagged else 'tooling/'+str(i))
            case=ET.SubElement(suite,'testcase',name='test_actual_default_finite_loop_once_per_spatial_channel['+name+']');props=ET.SubElement(case,'properties')
            for key,value in [('actual_default','True'),('actual_compiled_finite_loops','[]' if tagged else '[{"synthetic":true}]'),('actual_compiled_stage_calls','{"tv_chambolle_norm_into":1,"tv_chambolle_dual_into":1}' if tagged else '{}')]:ET.SubElement(props,'property',name=key,value=value)
        for i in range(2148-632):ET.SubElement(suite,'testcase',name='tooling-control-'+str(i))
        ET.ElementTree(suites).write(path/('pipeline-'+mode+'.xml'))


@pytest.mark.parametrize('fault',('missing-scalar','partial-rows','changed-case','wrong-workers','reference-only','timed','worker-failure','skip','missing-loop','forced-loop'))
def test_preparation_reports_require_every_fixed_row_and_actual_default_counter_shape(tmp_path,fault):
    import xml.etree.ElementTree as ET
    tooling_reports(tmp_path);assert set(h.preparation_reports(tmp_path))=={'normal','scalar'}
    path=tmp_path/'scalar.json';row=json.loads(path.read_text())
    if fault=='missing-scalar':path.unlink()
    elif fault in ('skip','missing-loop','forced-loop'):
        xml=tmp_path/'pipeline-scalar.xml';tree=ET.parse(xml);case=tree.getroot().find('.//testcase')
        if fault=='skip':tree.getroot().find('testsuite').set('skipped','1')
        else:
            key='actual_compiled_finite_loops' if fault=='missing-loop' else 'actual_default'
            next(p for p in case.findall('properties/property') if p.get('name')==key).set('value','[]' if fault=='missing-loop' else 'False')
        tree.write(xml)
    else:
        if fault=='partial-rows':row['checks'].pop()
        elif fault=='changed-case':row['checks'][0]['fixture']={'changed':'fixture'}
        elif fault=='wrong-workers':row['metadata']['workers']=[1]
        elif fault=='reference-only':row['metadata']['reference_only']=True
        elif fault=='timed':row['results']=[{'forbidden':'timing'}]
        else:row['checks'][0]['actual_by_workers'].pop('4')
        path.write_text(json.dumps(row))
    with pytest.raises((AssertionError,FileNotFoundError)):h.preparation_reports(tmp_path)


def test_qualified_artifact_uses_complete_generic_verifier_plus_tv_specific_scopes():
    source=Path(h.__file__).read_text();tree=ast.parse(source)
    qualified=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='qualified_product')
    text=ast.unparse(qualified)
    assert 'controller.verify_product(' in text and 'artifacts.verify(' in text
    assert "q['source'] == artifacts.source_state(ROOT)" in text
    assert "API in q['expected_registry']['functions']" in text
    assert "mode + '-public-TV'" in text and "len(package['members_sha256']) == 108" in text
    assert 'independent-verification.json' in text and 'core_sha256' in text
    main=ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main'))
    assert main.index('preparation_barrier(args.barrier')<main.index('qualified = contracts()')<main.index('timing = measure(')
    assert 'pip wheel' not in source and 'subprocess.run' not in source


def test_caught_upstream_attempt_cannot_hide_behind_owned_return():
    case=h.corpus()[0];original=h.resolve(True);before=sys.getprofile()
    def catches(image,**options):
        try:original(image,**options)
        except AssertionError:pass
        return image
    with pytest.raises(AssertionError,match='caught attempts'):h.native_observe(catches,case)
    assert sys.getprofile() is before

def test_native_check_only_artifact_failure_cannot_import_fixtures_or_enter_timer(tmp_path,monkeypatch):
    monkeypatch.setattr(h,'contracts',lambda:pytest.fail('fixture creation reached before product seal'))
    monkeypatch.setattr(h,'measure',lambda *_:pytest.fail('check-only entered timer'))
    output=tmp_path/'report.json'
    assert h.main(['--check-only','--capsule',str(tmp_path/'missing-product'),'--output',str(output)])==1
    report=json.loads(output.read_text());assert not report['checks'] and not report['results'] and report['status']=='failed'


def test_scientific_timing_fixture_literal_and_matrix_are_frozen():
    m=manifest();assert m['source_contract_sha256']=='e9ea5775b97f5bce2d5926a6eb411449908fdc219e88fb663b7fb1424edcb539'
    assert h.digest(h.timing_profiles())=='c16fa4e2af58a5d385cef6c6b6c2c28d9f3b9e502f0d6094be4420cda17b8e79'
    assert m['qualified_artifact_protocol']['product_members']==108
    assert m['qualified_artifact_protocol']['sdk_members']==62 and m['qualified_artifact_protocol']['sdk_families']==17


@pytest.mark.parametrize('payload',('{"certificate":"ascii"}\n','{\n  "certificate": "pretty"\n}\n','{\r\n  "certificate": "CRLF"\r\n}\r\n','{"certificate":"\u00e9"}\n'))
def test_workflow_certificate_utf8_bytes_are_identical_across_newline_platforms(payload,tmp_path):
    """Execute the actual workflow write expression with modeled text newlines."""
    text=(h.ROOT/'.github/workflows/tv-chambolle-benchmark.yml').read_text()
    lines=[line.strip() for line in text.splitlines() if "pathlib.Path('build/tv-certificate.json')" in line]
    assert len(lines)==1
    expression=ast.parse(lines[0]);call=expression.body[0].value
    assert isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='write_bytes'
    assert ast.dump(call.args[0])==ast.dump(ast.parse("os.environ['CERTIFICATE_JSON'].encode('utf-8')",mode='eval').body)
    hashes=[];legacy_hashes=[]
    for platform,newline in (('posix','\n'),('windows','\r\n')):
        target=tmp_path/(platform+'.json')
        class Destination:
            def write_bytes(self,data):return target.write_bytes(data)
            def write_text(self,data):
                with target.open('w',encoding='utf-8',newline=newline) as stream:return stream.write(data)
        namespace=dict(pathlib=types.SimpleNamespace(Path=lambda _:Destination()),os=types.SimpleNamespace(environ={'CERTIFICATE_JSON':payload}))
        exec(compile(expression,'<actual-workflow-certificate-write>','exec'),namespace)
        assert target.read_bytes()==payload.encode('utf-8')
        hashes.append(hashlib.sha256(target.read_bytes()).hexdigest())
        legacy=tmp_path/(platform+'-legacy-text.json')
        with legacy.open('w',encoding='utf-8',newline=newline) as stream:stream.write(payload)
        legacy_hashes.append(hashlib.sha256(legacy.read_bytes()).hexdigest())
    assert hashes[0]==hashes[1]
    assert legacy_hashes[0]!=legacy_hashes[1]  # The old text writer reproduces the actual barrier bug.
