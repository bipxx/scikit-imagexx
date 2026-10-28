"""Source-only H203 guards; no array, algorithm, candidate or native calls."""
import ast
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import types
import zipfile

import pytest

ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


q=load('h203_q_metadata','tools/public_qualification.py')
p=load('h203_stager_metadata','tools/rank_statistics_products.py')
gate=load('h203_installed_gate_source','tests/test_h_extrema_public.py')


def test_exact_closed_migration_and_old_science_remain_separate():
    result=q.h_extrema_source_metadata(lambda name:(ROOT/name).read_bytes())
    assert set(result)==set(q.H_EXTREMA_HISTORICAL_RAW_PINS)
    assert all(r['historical_raw_sha256']!=r['effective_raw_sha256'] for r in result.values())
    assert q.H_EXTREMA_HISTORICAL_RAW_PINS==p.H_EXTREMA_HISTORICAL_RAW_PINS
    assert q.H_EXTREMA_FROZEN_RAW_PINS==p.H_EXTREMA_FROZEN_RAW_PINS


@pytest.mark.parametrize('path',tuple(q.H_EXTREMA_RUNTIME_SOURCE_PINS)+tuple(q.H_EXTREMA_FROZEN_RAW_PINS))
def test_every_frozen_h_runtime_reference_reader_pin_rejects_tamper(path):
    def read(name):
        raw=(ROOT/name).read_bytes()
        return raw+b'\nchanged' if name==path else raw
    with pytest.raises(q.QualificationError,match='H .*differ'):
        q.h_extrema_source_metadata(read)


def test_public_scientific_bodies_have_only_closed_unsafe_return_delta():
    old={n.name:n for n in ast.parse((ROOT/gate.PRIVATE_PATH).read_bytes()).body if isinstance(n,ast.FunctionDef)}
    actual=gate.public_assertion_nodes()
    assert len(actual)==9 and tuple(n.name for n in actual)==gate.FUNCTIONS
    for function in actual:
        if function.name=='test_exact_owned_pipeline':
            inverse=copy.deepcopy(function)
            old_branch=next(n for n in ast.walk(old[function.name]) if isinstance(n,ast.If) and ast.unparse(n.test)=='name in UNSAFE_IDS')
            branch=next(n for n in ast.walk(inverse) if isinstance(n,ast.If) and ast.unparse(n.test)=='name in UNSAFE_IDS')
            assert isinstance(branch.body[-1],ast.Return)
            branch.body[-1]=copy.deepcopy(old_branch.body[-1])
            assert ast.dump(inverse,include_attributes=False)==ast.dump(old[function.name],include_attributes=False)
        else:
            assert ast.dump(function,include_attributes=False)==ast.dump(old[function.name],include_attributes=False)
    compile(ast.Module(body=actual,type_ignores=[]),'<source-only>','exec')


def test_runtime_authentication_AST_keeps_every_original_metadata_check():
    current=ast.parse((ROOT/'tests/h_extrema_public_authentication.py').read_bytes())
    for path,name,want_reads in (('tests/h_extrema_reference.py','authenticate_legacy',1),
                                 ('tests/h_extrema_floating_reference.py','authenticate_floating',2)):
        old=copy.deepcopy(next(n for n in ast.parse((ROOT/path).read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='authenticate'))
        counters=dict(reads=0,legacy_call=0,legacy_load=0)
        class ExactMetadataAdapter(ast.NodeTransformer):
            def visit_Assign(self,node):
                if ast.unparse(node)=="old = load('h_floating_unchanged_authentication', 'h_extrema_reference.py')":
                    counters['legacy_load']+=1;return None
                return self.generic_visit(node)
            def visit_Call(self,node):
                text=ast.unparse(node)
                if text=='(ROOT / name).read_bytes()':
                    counters['reads']+=1;return ast.copy_location(ast.parse('source_bytes(name, expected)',mode='eval').body,node)
                if text=='old.authenticate(profile)':
                    counters['legacy_call']+=1;return ast.copy_location(ast.parse('authenticate_legacy(profile)',mode='eval').body,node)
                return self.generic_visit(node)
        expected=ExactMetadataAdapter().visit(old);expected.name=name
        actual=next(n for n in current.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert counters==dict(reads=want_reads,legacy_call=int(name=='authenticate_floating'),legacy_load=int(name=='authenticate_floating'))
        assert ast.dump(expected,include_attributes=False)==ast.dump(actual,include_attributes=False)
        # Every actual upstream version, distribution-path/hash and warning
        # helper check is inherited verbatim; only closed source reads differ.
        assert any(isinstance(n,ast.Assert) for n in ast.walk(actual))


def test_cross_family_reconstruction_raw_inverse_is_distinct_from_original600():
    auth=load('h203_cross_family_metadata','tests/h_extrema_public_authentication.py')
    old=auth.validate_reconstruction(lambda n:(ROOT/n).read_bytes())
    assert hashlib.sha256(old).hexdigest()==auth.OLD
    assert hashlib.sha256(old.replace(b'\r\n',b'\n')).hexdigest()==auth.OLD_LF
    assert auth.OLD not in ('9a506a465f56e295edeb21a83b7e5988a7cbf8a25288513b935e59f1e85e05de',auth.CURRENT)


@pytest.mark.parametrize('fault',('wrong-current','wrong-old','extra-science','extra-body','scope','mapping','old-pin'))
def test_cross_family_adapter_rejects_unlisted_successors(fault):
    auth=load('h203_cross_family_negative','tests/h_extrema_public_authentication.py')
    integration=json.loads((ROOT/auth.RECORD).read_bytes())
    record=json.loads(integration['historical_integration_UTF8'])
    current=json.loads((ROOT/auth.POLICY_RECORD).read_bytes())['previous_source_UTF8'].encode()
    migration=(ROOT/'coverage/evidence/reconstruction-public-assertion-migration.json').read_bytes()
    if fault=='wrong-current':record['current_raw_sha256']='0'*64
    elif fault=='wrong-old':record['historical_raw_sha256']='0'*64
    elif fault=='extra-science':current=current.replace(b'def value(a):',b'def altered_value(a):',1)
    elif fault=='extra-body':record['allowed_body']='test_native_pending_is_not_upstream_forwarding'
    elif fault=='scope':record['schema']='general-source-mismatch-escape'
    elif fault=='mapping':migration+=b'\nchanged'
    else:
        with pytest.raises(AssertionError,match='unknown historical'):auth.source_bytes(auth.PATH,'0'*64)
        return
    with pytest.raises(AssertionError):auth.verify_reconstruction(record,current,migration)


def test_closed_chain_retains_the_exact_e39_verifier_and_runtime_authentication():
    auth=load('h203_policy_chain_metadata','tests/h_extrema_public_authentication.py')
    record=json.loads((ROOT/auth.RECORD).read_bytes())
    old=record['historical_adapter_UTF8'].encode()
    assert hashlib.sha256(old).hexdigest()==record['historical_adapter_raw_sha256']=='ab065c5a2a7b5db7a27f3d00b42818e3ed5c1bcd5be90ef7f7935699e6c75330'
    previous={n.name:n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef)}
    current={n.name:n for n in ast.parse((ROOT/'tests/h_extrema_public_authentication.py').read_bytes()).body if isinstance(n,ast.FunctionDef)}
    for name in ('sha','body','verify_reconstruction','source_bytes','load','authenticate_legacy','authenticate_floating'):
        assert ast.dump(current[name],include_attributes=False)==ast.dump(previous[name],include_attributes=False)
    text=ast.unparse(current['validate_reconstruction'])
    assert text.index('verify_current_to_f82(')<text.index('verify_reconstruction(')
    policy_raw=(ROOT/auth.POLICY_RECORD).read_bytes()
    predecessor=auth.verify_current_to_f82(record,(ROOT/auth.PATH).read_bytes(),policy_raw,(ROOT/auth.POLICY_VALIDATOR).read_bytes())
    assert hashlib.sha256(predecessor).hexdigest()==auth.CURRENT
    assert hashlib.sha256(predecessor.replace(b'\r\n',b'\n')).hexdigest()==auth.CURRENT_LF
    assert predecessor==json.loads(policy_raw)['previous_source_UTF8'].encode()
    old_result=auth.verify_reconstruction(json.loads(record['historical_integration_UTF8']),predecessor,
        (ROOT/'coverage/evidence/reconstruction-public-assertion-migration.json').read_bytes())
    assert auth.validate_reconstruction(lambda n:(ROOT/n).read_bytes())==old_result


@pytest.mark.parametrize('fault',('current-pin','current-LF','current-bytes','policy-pin','policy-bytes',
    'validator-pin','validator-bytes','extra-body','missing-body','body-order','policy-source',
    'public-source','scope','historical-record','historical-record-pin','old-migration'))
def test_two_step_policy_chain_rejects_unreviewed_sources_or_scope(fault):
    auth=load('h203_policy_chain_negative','tests/h_extrema_public_authentication.py')
    record=json.loads((ROOT/auth.RECORD).read_bytes())
    updates={}
    if fault=='current-pin':record['current_raw_sha256']='0'*64
    elif fault=='current-LF':record['current_canonical_lf_sha256']=record['current_raw_sha256']
    elif fault=='current-bytes':updates[auth.PATH]=(ROOT/auth.PATH).read_bytes().replace(b'def observe(',b'def changed_observe(',1)
    elif fault=='policy-pin':record['reviewed_policy_migration_raw_sha256']='0'*64
    elif fault=='policy-bytes':updates[auth.POLICY_RECORD]=(ROOT/auth.POLICY_RECORD).read_bytes()+b'\n'
    elif fault=='validator-pin':record['reviewed_validator_raw_sha256']='0'*64
    elif fault=='validator-bytes':updates[auth.POLICY_VALIDATOR]=(ROOT/auth.POLICY_VALIDATOR).read_bytes()+b'\n'
    elif fault=='extra-body':record['allowed_assertion_bodies'].append('observe')
    elif fault=='missing-body':record['allowed_assertion_bodies'].pop()
    elif fault=='body-order':record['allowed_assertion_bodies'].reverse()
    elif fault=='policy-source':record['reviewed_policy_source']='0'*40
    elif fault=='public-source':record['previous_public_H_source']='0'*40
    elif fault=='scope':record['schema']='general-historical-read-override'
    elif fault=='historical-record':record['historical_integration_UTF8']+='\n'
    elif fault=='historical-record-pin':record['historical_integration_raw_sha256']='0'*64
    else:
        name='coverage/evidence/reconstruction-public-assertion-migration.json'
        updates[name]=(ROOT/name).read_bytes()+b'\n'
    updates[auth.RECORD]=json.dumps(record).encode()
    def read(name):return updates[name] if name in updates else (ROOT/name).read_bytes()
    with pytest.raises(AssertionError):auth.validate_reconstruction(read)


def test_original_authentication_precedes_sole_installed_import_and_fixture_calls():
    source=(ROOT/'tests/test_h_extrema_public.py').read_text()
    assert source.index('reference.authenticate_floating(profile)')<source.index('import skimagexx as xx')<source.index("namespace['test_exact_owned_pipeline'](spec, workers)")
    assert 'types.ModuleType' not in source and "load('_h_extrema_core'" not in source
    assert "for fallback in (False, True):" in source
    assert "q.h_extrema_admission(core, morphology, owned, native_functions())" in source
    assert "q.reconstruction_admission(core, morphology, reconstruction, native_functions())" in source
    assert "sys.flags.optimize != 0" in source
    assert '3544' in source and "len(blocked) == 28" in source


def test_actual_child_diagnostic_projection_uses_closed_v2_record():
    # Execute only the actual metadata assignment from the child, never the
    # installed audit or its image loop. Authentication already validates this
    # record before the projection; no second source-auth layer is introduced.
    tree=ast.parse((ROOT/'tests/test_h_extrema_public.py').read_bytes())
    audit=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='audit_installed')
    projections=[n for n in audit.body if isinstance(n,ast.Assign) and
                 any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and
                     t.value.id=='authentication' and isinstance(t.slice,ast.Constant) and
                     t.slice.value=='reconstruction_reference_metadata' for t in n.targets)]
    assert len(projections)==1
    auth=load('h203_projection_authentication','tests/h_extrema_public_authentication.py')
    cross=json.loads((ROOT/auth.RECORD).read_bytes())
    code=compile(ast.Module(body=projections,type_ignores=[]),'<actual-child-projection>','exec')
    namespace={'cross':cross,'authentication':{}}
    exec(code,namespace)
    projected=namespace['authentication']['reconstruction_reference_metadata']
    assert projected=={key:cross[key] for key in (
        'schema','previous_public_H_source','reviewed_policy_source','current_raw_sha256',
        'current_canonical_lf_sha256','reviewed_policy_migration_raw_sha256',
        'reviewed_validator_raw_sha256','allowed_assertion_bodies')}
    assert projected['schema']=='h-reconstruction-closed-trace-policy-integration-v2'
    assert projected['reviewed_policy_source']=='67325657613921fffad5964ddd31a5cba5f32613'
    assert len(projected['allowed_assertion_bodies'])==2
    # The historical v1 remains preserved inside the record. It is not the
    # current projection schema and must fail here rather than masquerade as v2.
    historical=json.loads(cross['historical_integration_UTF8'])
    with pytest.raises(KeyError):exec(code,{'cross':historical,'authentication':{}})


def h_stub(monkeypatch):
    core=types.SimpleNamespace(__name__='unit_core',get_num_threads=lambda:1,has_openmp=True,
        h_extrema_interface='h-extrema-integer-v1',h_extrema_real_interface='h-extrema-real-v1',
        h_extrema_full_API=False,h_extrema_integer_widths=(8,16,32,64))
    for name in ('h_extrema_validate_metadata','h_extrema_integer_range','h_extrema_integer_seed','h_extrema_integer_mask',
                 'h_extrema_counts','h_extrema_real_mask','h_extrema_real_counts'):
        def forbidden(*a,**kw):raise AssertionError('metadata stub executed pixels')
        forbidden.__name__=name;forbidden.__module__=core.__name__;forbidden.unit_builtin=True
        setattr(core,name,forbidden)
    def h_maxima(image,h,footprint=None):raise AssertionError('wrapper executed')
    def h_minima(image,h,footprint=None):raise AssertionError('wrapper executed')
    for f in (h_maxima,h_minima):f.__module__='skimagexx._h_extrema'
    wrapper=types.SimpleNamespace(_core=core,h_maxima=h_maxima,h_minima=h_minima)
    morphology=types.SimpleNamespace(h_maxima=h_maxima,h_minima=h_minima,__all__=['h_maxima','h_minima'])
    original=q.inspect.isbuiltin
    monkeypatch.setattr(q.inspect,'isbuiltin',lambda f:original(f) or getattr(f,'unit_builtin',False))
    return core,morphology,wrapper,list(q.H_EXTREMA_PATHS)


def test_metadata_admission_performs_no_pixel_calls(monkeypatch):
    q.h_extrema_admission(*h_stub(monkeypatch))


@pytest.mark.parametrize('fault',('missing','duplicate','workers','OpenMP','real-interface','fullAPI','core','entry','owner','alias','export','signature','width-bool'))
def test_metadata_admission_rejects_missing_real_or_ordinary_ownership(monkeypatch,fault):
    core,morphology,wrapper,registered=h_stub(monkeypatch)
    if fault=='missing':registered.pop()
    elif fault=='duplicate':registered*=2
    elif fault=='workers':core.get_num_threads=lambda:4
    elif fault=='OpenMP':core.has_openmp=False
    elif fault=='real-interface':core.h_extrema_real_interface='integer-only'
    elif fault=='fullAPI':core.h_extrema_full_API=True
    elif fault=='core':wrapper._core=object()
    elif fault=='entry':core.h_extrema_real_mask.unit_builtin=False
    elif fault=='owner':core.h_extrema_real_mask.__module__='upstream'
    elif fault=='alias':morphology.h_maxima=wrapper.h_minima
    elif fault=='export':morphology.__all__.append('h_maxima')
    elif fault=='signature':wrapper.h_minima.__signature__=q.inspect.Signature()
    else:core.h_extrema_integer_widths=(True,16,32,64)
    with pytest.raises(q.QualificationError):q.h_extrema_admission(core,morphology,wrapper,registered)


def test_combined_scope_and_physical_source_census_preserve_nmi_and_old_h():
    text=(ROOT/'tools/public_qualification.py').read_text()
    # Project the closed historical literals explicitly; current staging also
    # requires its separate source-derived install layout.
    declarations={'FAMILIES','PRODUCT_SCOPE','PRODUCT_MEMBERS','SDK_MEMBERS'}
    legacy='\n'.join(ast.get_source_segment(text,n) for n in ast.parse(text).body
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in declarations for t in n.targets))
    scope=p.qualification_scope_text(legacy)
    assert (scope['name'],scope['members'],scope['sdk'],len(scope['families']))==('h_extrema23',120,68,23)
    assert q.product_census('nmi22')==q.product_census('h_extrema22')==(118,67)
    assert q.FAMILIES[:-1]==q.NMI_FAMILIES and q.FAMILIES[-2:]==('normalized_mutual_information','h_extrema')
    current=p.qualification_scope(ROOT)
    if current['name']=='h_extrema23':
        assert q.registry_expectation(ROOT)['image_count']==203
        assert len(tuple((ROOT/'python/skimagexx').rglob('*.py')))==41
        assert len(tuple((ROOT/'include/skimagexx').glob('*.hpp')))==56
    else:
        contract=q.current_contract(lambda n:(ROOT/n).read_bytes())
        assert len(contract['baseline_image_paths'])==203
        assert q.registry_expectation(ROOT)['image_count']==len(contract['image_paths'])
        assert current['name']==q.CURRENT_PRODUCT_SCOPE
        assert len(tuple((ROOT/'python/skimagexx').rglob('*.py')))==sum(n.startswith('skimagexx/') and n.endswith('.py') for n in current['install_members'])
        assert len(tuple((ROOT/'include/skimagexx').glob('*.hpp')))==sum(n.startswith('include/skimagexx/') and n.endswith('.hpp') for n in current['SDK_members'])
    assert len(q.sdk_family_declarations((ROOT/'tests/cmake_consumer/CMakeLists.txt').read_text()))>=23
    for name,want in q.NMI_IMPORTED_SOURCE_PINS.items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==want


def h_archive(records):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,raw in records.items():archive.writestr(name,raw)
    return zipfile.ZipFile(io.BytesIO(stream.getvalue()))


@pytest.fixture(scope='module')
def h_records():
    names={*q.H_EXTREMA_RUNTIME_SOURCE_PINS,*q.H_EXTREMA_FROZEN_RAW_PINS,'tools/public_qualification.py'}
    return {name:(ROOT/name).read_bytes() for name in names}


def test_stager_authenticates_actual_829_and_closed448_inverse(h_records):
    snapshot={'files_sha256':{n:hashlib.sha256(r.replace(b'\r\n',b'\n')).hexdigest() for n,r in h_records.items()}}
    with h_archive(h_records) as archive:p.validate_h_extrema_source(archive,snapshot,{'name':'h_extrema23'})


@pytest.mark.parametrize('fault',('runtime','historical','reader','migration','snapshot','missing'))
def test_stager_no_unlisted_mismatch_escape(h_records,fault):
    records=dict(h_records)
    snapshot={'files_sha256':{n:hashlib.sha256(r.replace(b'\r\n',b'\n')).hexdigest() for n,r in records.items()}}
    name={'runtime':'src/h_extrema.cpp','historical':'coverage/contracts/h-extrema-floating.json',
          'reader':'tests/h_extrema_source_provenance.py','migration':'coverage/evidence/h-extrema-reference-assertion-migration.json',
          'snapshot':'tests/h_extrema_reference.py','missing':'src/bind_h_extrema.cpp'}[fault]
    if fault=='missing':records.pop(name)
    elif fault=='snapshot':snapshot['files_sha256'][name]='0'*64
    else:
        records[name]+=b'\nchanged'
        snapshot['files_sha256'][name]=hashlib.sha256(records[name].replace(b'\r\n',b'\n')).hexdigest()
    with h_archive(records) as archive:
        with pytest.raises(RuntimeError,match='H '):p.validate_h_extrema_source(archive,snapshot,{'name':'h_extrema23'})
