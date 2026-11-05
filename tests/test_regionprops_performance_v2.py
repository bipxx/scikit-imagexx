"""Prospective optimization-v2 source/guard checks, no scientific calls."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
ROOT=Path(__file__).parents[1]
CONTRACT=ROOT/'coverage/contracts/regionprops-performance-compare-v2.json'


def controller():
    spec=importlib.util.spec_from_file_location('region_v2_controller',ROOT/'tools/region_stain_compare.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def test_v2_fixed_baseline_cases_method_and_each_pair_limits():
    p=json.loads(CONTRACT.read_bytes())
    assert p['baseline']==dict(source='6a2b33a2f78606b9315dd0fcc50777b6e85ac38f',run='41985437089',artifact=13396062741,bytes=85524974,sha256='106f30693ea990776cfeb4d12afe97a55dfe86bae9818b5a40f17271e46126e0')
    assert p['families']==['region','scalars','half','holdouts'] and p['passes']==['baseline','candidate','candidate','baseline'] and p['pairs']==[[0,1],[3,2]]
    keys=['region_cases','scalar_cases','half_cases','holdout_cases'];assert [len(p[k])for k in keys]==[12,8,4,4]
    ids=[s['id']for k in keys for s in p[k]];assert len(set(ids))==28
    assert p['sampling']==dict(threads=1,repeat=7,target_ms=100,warmup_ms=250,cohorts=28,samples_per_pass=392,samples_total=1568,deadline='required immutable UTC deadline; checked before every actualbatch')
    limits=p['fixed_gates']['per_case_limits'];assert set(limits)==set(ids)
    for cid in ids:
        expected=.5 if cid in ('regionprops-scalars/medium-getters-f-off','regionprops-state/large2d-f-coords','regionprops-state/medium3d-f-cache-off')else 2/3 if cid in ('regionprops-scalars/tiny-warm-c-on','regionprops-scalars/3d-warm-f-on')else 1.05
        assert limits[cid]==expected
    c=controller();passes=[{cid:dict(original=1.,cpp=1.)for cid in ids}for i in range(4)]
    for i in (1,2):
        for cid in ids:passes[i][cid]['cpp']=limits[cid]*.99
    assert c.acceptance(passes,p)['decision']=='KEEP'
    for i in (1,2):
        for cid in ids:
            fault=copy.deepcopy(passes);fault[i][cid]['cpp']=limits[cid]*1.001
            assert c.acceptance(fault,p)['decision']=='DISCARD'
            fault=copy.deepcopy(passes);fault[i][cid]['original']=.5
            assert c.acceptance(fault,p)['decision']=='DISCARD'


def test_v2_active_closures_and_immutable_timed_bodies():
    for filename,oldfile,driver in [('regionprops-performance-state-v2.json','regionprops-scalar-runtime-v1.json','regionprops_state.py'),('regionprops-performance-scalars-v2.json','regionprops-scalars-benchmark-v1.json','regionprops_scalars.py')]:
        previous=json.loads((ROOT/'coverage/contracts'/filename).read_bytes())
        filename=filename.replace('-v2.json','-v3.json')
        active=json.loads((ROOT/'coverage/contracts'/filename).read_bytes());old=json.loads((ROOT/'coverage/contracts'/oldfile).read_bytes())
        assert {k:v for k,v in active.items()if k not in ('source_closure','scope')}=={k:v for k,v in previous.items()if k not in ('source_closure','scope')}
        assert active['cases']==old['cases']
        assert {k:v for k,v in active.items()if k not in ('source_closure','scope','native_entries')}=={k:v for k,v in old.items()if k not in ('source_closure','scope','native_entries')}
        for name,pin in active['source_closure'].items():assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin,name
        current=(ROOT/'benchmarks'/driver).read_bytes().replace(b'\r\n',b'\n')
        predecessor=current.replace(filename.encode(),oldfile.encode())
        if driver=='regionprops_scalars.py':assert hashlib.sha256(predecessor).hexdigest()==old['source_closure']['benchmarks/regionprops_scalars.py']
        else:
            predecessor=predecessor.replace(oldfile.encode(),b'regionprops-state-benchmark.json')
            assert hashlib.sha256(predecessor).hexdigest()=='12c237328a25b462362b19551af70b52dc0868a66bc72c1a4ecfbe5755a39587'
    native=active['native_entries']
    assert 'regionprops_state_extent_f64'not in native and {'regionprops_state_area_bbox_get','regionprops_state_extent_get'}<=set(native)


def test_v2_controller_source_workflow_and_no_science_import(monkeypatch):
    import builtins
    p=json.loads(CONTRACT.read_bytes())
    for name,pin in p['controller_source_sha256'].items():
        predecessor=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
        if name=='tools/region_stain_compare.py':
            predecessor=predecessor.replace(b"CONTRACT=ROOT/'coverage/contracts/regionprops-performance-compare-v3.json'",b"CONTRACT=ROOT/'coverage/contracts/region-stain-same-host-v1.json'")
            predecessor=predecessor.replace(b"active=load(ROOT/'coverage/contracts/regionprops-performance-compare-v3.json')",b"active=load(ROOT/'coverage/contracts/regionprops-performance-compare-v2.json')")
        elif name=='.github/workflows/region-stain-same-host.yml':
            predecessor=predecessor.replace(b'cf0abd3f8dba9acbc988a59a53ae32cbcccfab4b',p['baseline']['source'].encode())
            predecessor=predecessor.replace(b'regionprops-performance-compare-v3.json',b'regionprops-performance-compare-v2.json')
        assert hashlib.sha256(predecessor).hexdigest()==pin,name
    previous=builtins.__import__
    def checked(name,*a,**kw):
        assert name.split('.')[0]not in {'numpy','scipy','skimage','skimagexx'}
        return previous(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',checked);assert callable(controller().acceptance)
    workflow=(ROOT/'.github/workflows/region-stain-same-host.yml').read_text()
    assert '--contract controller/coverage/contracts/regionprops-performance-compare-v3.json' in workflow
    active=json.loads((ROOT/'coverage/contracts/regionprops-performance-compare-v3.json').read_bytes())
    assert 'ref: '+active['baseline']['source']in workflow and 'if: always()'in workflow
    source=(ROOT/'tools/region_stain_compare.py').read_text();ast.parse(source)
    assert "choices=('region','scalars','half','holdouts')"in source
    assert "protocol.get('families',['region','half','holdouts'])"in source
    assert "startswith('Run python benchmarks/')"in source


def test_v2_installed_gate_and_source_selector_closure():
    ci=(ROOT/'.github/workflows/ci.yml').read_text();q=(ROOT/'.github/workflows/public-qualification.yml').read_text()
    assert ci.count('tests/test_regionprops_scalar_boundary_v2.py')==4
    for name in ['tests/test_regionprops_performance_v2.py','tests/test_region_stain_compare.py::test_fixed_contract_workflow_sources_and_pool_scope','tests/test_regionprops_strided_sdk_scope.py']:assert name in q
    c=ROOT/'CMakeLists.txt';text=c.read_text();assert 'target_compile_options(test_regionprops_state_native PRIVATE -UNDEBUG -ffp-contract=off -fno-fast-math -frounding-math)'in text
