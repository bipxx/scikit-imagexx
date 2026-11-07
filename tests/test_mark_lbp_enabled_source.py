"""Source-only enabled MARK/LBP wiring, platform boundaries and exact transports."""
import ast,base64,hashlib,json,runpy,shlex,zlib
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
H=lambda b:hashlib.sha256(b).hexdigest()

def test_exact_admitted_components_without_family_rewrites():
    d=json.loads((ROOT/'coverage/contracts/mark-lbp-enabled-components-v1.json').read_bytes())
    assert d['source_budget']==0
    context=json.loads((ROOT/'coverage/contracts/mark-boundaries-benchmark-execution-v8.json').read_bytes())
    lbp=json.loads((ROOT/'coverage/contracts/draw-multiblock-lbp-benchmark-execution-v7.json').read_bytes())
    adapter=runpy.run_path(str(ROOT/'benchmarks/mark_boundaries_execution.py'))
    for name,pin in d['source_files_raw_sha256'].items():
        c,path=(context,'coverage/contracts/mark-boundaries-benchmark-execution-v8.json')if name in context['integrated_source_predecessors']else(lbp,'coverage/contracts/draw-multiblock-lbp-benchmark-execution-v7.json')
        raw=adapter['historical_bytes'](ROOT,name,context_path=path,section='integrated_source_predecessors')if name in c['integrated_source_predecessors']else(ROOT/name).read_bytes()
        assert H(raw)==pin,name

def test_ordinary_single_core_build_and_public_routes():
    c=(ROOT/'CMakeLists.txt').read_text();b=(ROOT/'src/bindings.cpp').read_text()
    assert 'SKIMAGEXX_PRIVATE_LBP'not in c and 'PATTERN "draw_multiblock_lbp.hpp" EXCLUDE'not in c
    for leaf in ('mark_boundaries','draw_multiblock_lbp'):
        assert 'src/'+leaf+'.cpp'in c and 'src/bind_'+leaf+'.cpp'in c
        assert b.count('bind_'+leaf+'(m);')==1
    assert 'from ._mark_boundaries import mark_boundaries'in(ROOT/'python/skimagexx/segmentation.py').read_text()
    assert 'from ._draw_multiblock_lbp import draw_multiblock_lbp'in(ROOT/'python/skimagexx/feature.py').read_text()
    assert 'install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})'in c

def test_source_registry_default_binary_arity_dtypes_and_strict_checker():
    n=runpy.run_path(str(ROOT/'python/skimagexx/coverage.py'))
    for api in ('segmentation.mark_boundaries','feature.draw_multiblock_lbp'):
        assert api in n['native_functions']()and len(n['native_dtypes'](api))==12
    s=(ROOT/'python/skimagexx/coverage.py').read_text()
    assert "return (image,labels),{'mode':'thick'}"in s and "return (image,3,3,2,2),{'lbp_code':170,'alpha':0.5}"in s
    assert "actual.tobytes(order='C')==expected.tobytes(order='C')"in s
    assert 'actual.strides==expected.strides'in s and 'not np.shares_memory(actual,image)'in s

def test_standard_install_source_layout_and_real_profile_selectors():
    q=runpy.run_path(str(ROOT/'tools/public_qualification.py'));names=[p.relative_to(ROOT).as_posix()for prefix in ('python/skimagexx','include')for p in(ROOT/prefix).rglob('*')if p.is_file()]
    layout=q['current_install_layout'](lambda n:(ROOT/n).read_bytes(),names,'modern')
    assert layout['product_count']>0 and layout['SDK_count']>0
    assert 'include/skimagexx/mark_boundaries.hpp'in layout['SDK_members']and 'include/skimagexx/draw_multiblock_lbp.hpp'in layout['SDK_members']
    assert q['CURRENT_FAMILIES'][-7:]==('mark_boundaries','draw_multiblock_lbp','finite_radon','variation_of_information','flood_fill','rolling_ball','inverse_gaussian_gradient')
    base=q['groups_for_source'](ROOT)
    for profile in ('minimum','modern','debug','windows'):
        groups=q['applicable_public_groups'](base,profile)
        assert all(groups[n]==v for n,v in base.items())
        if profile=='windows':assert 'public-mark-saved-linux'not in groups and 'public-lbp-saved-linux'not in groups
        else:assert groups['public-mark-saved-linux']==('test_mark_boundaries_public.py',)and groups['public-lbp-saved-linux']==('test_draw_multiblock_lbp_public.py',)
    assert 'test_mark_lbp_enabled_public.py'in base['public-post203']

def test_actual_prebuild_command_selects_enabled_and_admitted_benchmark_guards():
    lines=(ROOT/'.github/workflows/public-qualification.yml').read_text().splitlines()
    line=next(x for x in lines if 'python -m pytest'in x and 'qualification-tool-guards.log'in x)
    a,b=line.split('2>&1 |');assert shlex.split(b)==['tee','build/qualification-tool-guards.log']
    for name in ('test_mark_lbp_enabled_source.py','test_draw_multiblock_lbp_native_source.py','test_mark_boundaries_native_contract.py','test_mark_boundaries_benchmark_execution.py','test_draw_multiblock_lbp_benchmark_execution.py'):
        assert shlex.split(a).count('tests/'+name)==1

def test_exact_original188_source_transport_and_phase_scope():
    t=json.loads((ROOT/'tests/fixtures/draw_multiblock_lbp_originals_v1.json').read_bytes())
    raw=zlib.decompress(base64.b64decode(t['admission_raw_zlib_base64']));assert H(raw)==t['admission_raw_sha256']=='1bcfff2be1b241b989a6142a756131474e89e756dd30047b0e653dde3d822726';ad=json.loads(raw)
    for profile,p in t['profiles'].items():
        body=zlib.decompress(base64.b64decode(p['raw_zlib_base64']));rows=json.loads(body)
        assert H(body)==p['raw_sha256']==ad['raw_files'][f'build/draw-multiblock-lbp-codec-capture-v1/capture-v1/{profile}/observations.json']['sha256']
        assert len(rows)==94 and[r['id']for r in rows]==ad['case_ids']
        assert sum(r['error']is None for r in rows)==71
    phase=(ROOT/'coverage/contracts/draw-multiblock-lbp-observer-phase-scope-v1.json').read_bytes();assert H(phase)=='4c21baa192a904a957435cb2178f361cffb36d74a84399912daa98e5ef5a26e7'
    text=(ROOT/'tests/test_draw_multiblock_lbp_public.py').read_text();assert "p=='/after_observer_fenv/mxcsr'"in text and 'bytes(env())==bytes(process)'in text
    assert 'source-built-v1'in text and 'private-fixed-v1'not in text

def test_complete_named_sdk_consumers_and_exact71_word_transports():
    c=json.loads((ROOT/'coverage/contracts/mark-lbp-enabled-transports-v1.json').read_bytes())
    for name,row in c['source_transports'].items():
        raw=(ROOT/name).read_bytes();assert H(raw)==row['published_raw_sha256']
        assert raw.startswith(row['inverse_prefix'].encode());raw=raw[len(row['inverse_prefix']):]
        assert raw.count(row['inverse_current'].encode())==1;raw=raw.replace(row['inverse_current'].encode(),row['inverse_predecessor'].encode())
        assert H(raw)==row['retained_raw_sha256']
    source=(ROOT/'tests/cmake_consumer/draw_multiblock_lbp_enabled_consumer.cpp').read_text()
    assert 'lbp_saved_minimum();lbp_saved_modern();draw_multiblock_lbp_consumer();'in source
    assert 'mark_boundaries_consumer(workers)'in(ROOT/'tests/cmake_consumer/mark_boundaries_consumer.cpp').read_text()

def test_windows_neutral_consumers_have_no_linux_envelope_or_skips():
    source=(ROOT/'tests/test_mark_lbp_enabled_public.py').read_text()
    assert 'pytest.skip'not in source
    assert 'full32'not in source and 'fegetenv'not in source
    assert "('rank5','swapped-labels','nonexact-wide-labels')"in source
    assert 'np.zeros((2,3),dtype=dtype)'in source

def test_enabled_shared_inverse_and_old_metrics_polygon_guards():
    h=runpy.run_path(str(ROOT/'tools/join_source_context.py'))
    c=json.loads((ROOT/'coverage/contracts/mark-lbp-enabled-additive-context-v1.json').read_bytes())
    for name,row in c['sources'].items():
        raw=(ROOT/name).read_bytes();assert H(h['enabled_predecessor_bytes'](ROOT,name,raw))==row['predecessor_canonical_lf_sha256']
        with pytest.raises(AssertionError):h['enabled_predecessor_bytes'](ROOT,name,raw+b'\n#unadmitted')


def test_real_qualification_parser_and_exact_saved_group_count_guards(monkeypatch):
    import importlib.util
    spec=importlib.util.spec_from_file_location('enabled_qualifier_cli',ROOT/'tools/public_qualification.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
    calls=[];monkeypatch.setattr(q,'child',lambda out,group:calls.append(group))
    tree=ast.parse((ROOT/'tools/public_qualification.py').read_bytes())
    seal=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='seal')
    for group,count in (('public-mark-saved-linux',267),('public-lbp-saved-linux',188),('public-radon-saved-linux',224),('public-lbp-metadata-linux',24)):
        q.main(['_child','--output',str(ROOT/'build/parser-control-never-created'),'--group',group])
        node=next(n for n in ast.walk(seal)if isinstance(n,ast.If)and ast.unparse(n.test)==f"group == '{group}'")
        code=compile(ast.Module(body=[node],type_ignores=[]),'<actual seal group guard>','exec')
        exec(code,dict(group=group,row={'tests':count},require=q.require))
        for wrong in (0,count-1,count+1):
            with pytest.raises(q.QualificationError):exec(code,dict(group=group,row={'tests':wrong},require=q.require))
    assert calls==['public-mark-saved-linux','public-lbp-saved-linux','public-radon-saved-linux','public-lbp-metadata-linux']


def test_active_benchmark_context_successors_preserve_exact_old_authority_and_source_bodies():
    adapter=runpy.run_path(str(ROOT/'benchmarks/mark_boundaries_execution.py'))
    for name in ('mark-boundaries-benchmark-execution-v8.json','draw-multiblock-lbp-benchmark-execution-v7.json'):
        current=json.loads((ROOT/'coverage/contracts'/name).read_bytes());p=current['integration_predecessor_context'];raw=(ROOT/p['path']).read_bytes();assert H(raw)==p['raw_sha256'];old=json.loads(raw)
        expected_previous={'mark-boundaries-benchmark-execution-v8.json':'mark-boundaries-benchmark-execution-v7.json','draw-multiblock-lbp-benchmark-execution-v7.json':'draw-multiblock-lbp-benchmark-execution-v6.json'}
        assert p['path']=='coverage/contracts/'+expected_previous[name]
        mutable={'integration_predecessor_context','schema','successor_source_predecessors','source_files_canonical_sha256','historical_sources','historical_446_sources','route','predecessor_context','integration_scope','integrated_source_predecessors','reused_executor_canonical_sha256','common_boundary_canonical_sha256','metadata_fixture_canonical_sha256'}
        for key in old:
            if key not in mutable:assert current[key]==old[key],key
        route=dict(current['route']);route['contract']=old['route']['contract'];assert route==old['route']
        for source in current['successor_source_predecessors']:
            before=adapter['historical_bytes'](ROOT,source,context_path='coverage/contracts/'+name,section='successor_source_predecessors')
            assert H(before)==old['source_files_canonical_sha256'][source]
        for source,pin in old['source_files_canonical_sha256'].items():
            if source not in current['successor_source_predecessors']:assert current['source_files_canonical_sha256'][source]==pin
