"""Inert fixed-gate tests; synthetic latencies only, no scientific imports."""
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import textwrap
import pytest
ROOT=Path(__file__).resolve().parents[1]
WORKFLOW=(ROOT/'.github/workflows/label-points-zero-provider-same-host.yml').read_text()
SCRIPTS=[textwrap.dedent(s)for s in re.findall(r"python - <<'PY'\n(.*?)          PY\n",WORKFLOW,re.S)]
PREP=ast.parse(SCRIPTS[1])
PAIR=next(n for n in PREP.body if isinstance(n,ast.If)and ast.unparse(n.test)=="role == 'candidate'")
CONTRACT='coverage/contracts/label-points-zero-provider-candidate.json'
CONTRACT_SHA='b70415e455b317b44b8ce803d2f60544b022ae6bedaad4b02ef03c1eed923fc6'
PINS=ast.literal_eval(next(n.value for n in PAIR.body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='candidate_pins'))
TARGETS={'sparse-64x64-i64','sparse-1024x1024-i64','sparse-64x64x64-f32'}
CORRECTION="    scope=scope.replace('owned zero-fill/scatter','NumPy zero-allocation provider and owned ordered scatter')\n"


def test_fixed_source_transport_and_no_timing_host_build_or_test():
    assert len(SCRIPTS)==3
    for s in SCRIPTS:compile(s,'workflow-inline','exec')
    assert '40724150045' in SCRIPTS[0] and '40730844227' in SCRIPTS[0]
    assert '910c31a609c157c28cbbe56933b8448b710af95e' in SCRIPTS[0]
    assert '168e6b736221a488aad74b821dcfe74dbc87470c' in SCRIPTS[0]
    assert '12833680393' in SCRIPTS[0] and '38797cf494532e16da41f751062c57815de2d566d8ebf11f3b59be26ba4be536' in SCRIPTS[0]
    assert '12835986984' in SCRIPTS[0] and '6f7d6fcbd1506c2e61031d925575526ea8abf4c30c7770bf61a06aa04965cc08' in SCRIPTS[0]
    assert 'PENDING' not in WORKFLOW
    timed=WORKFLOW.split('      - name: Measure',1)[1].split('      - name:',1)[0]
    assert 'for pass in 1-baseline 2-candidate 3-candidate 4-baseline' in timed
    assert timed.count('ci_measurement_artifacts.py verify')==2
    assert 'benchmarks/label_points.py --threads 1 --repeat 7' in timed
    assert not re.search(r'pytest|cmake|ctest|pip install|pip wheel|__module__',timed)
    assert '--no-index --no-deps --only-binary=:all: --require-hashes' in WORKFLOW
    assert 'no normalization authorized' in SCRIPTS[1]
    assert '== 79' in SCRIPTS[1] and '== 168' in SCRIPTS[1]
    assert "{'passed': 1435}" in SCRIPTS[1] and '== 245' in SCRIPTS[1]
    assert 'if: always()' in WORKFLOW and 'name: label-points-zero-provider-same-host-minimum' in WORKFLOW
    assert '2026-10-28T18:42:47.988490+00:00' in SCRIPTS[2]
    loop=next(n for n in PREP.body if isinstance(n,ast.For)and isinstance(n.target,ast.Tuple))
    for role,expected in [('baseline',7134),('candidate',7146)]:
        pairs=eval(compile(ast.Expression(loop.iter),'actual-counts','eval'),dict(role=role))
        assert len(pairs)==33 and sum(c*2 for n,c in pairs if n!='color-transforms-regression')==expected
    assert len(PINS)==8 and 'src/simd_label_points_avx2.cpp' not in PINS


@pytest.mark.parametrize('bad',[None,'foreign-file','simd','case-contract','dependency','pytest','runtime','science','allowed-but-unpinned','contract'])
def test_actual_complete_source_and_dependency_closure(bad):
    runtime=dict(python='fixed',implementation='fixed',cache_tag='fixed',machine='fixed',system='fixed',dependency_libraries_sha256={'a':'b'},versions={'scikit-imagexx':'0.1','pytest':'same'},packages_sha256={k:{'file':'same'}for k in ('numpy','scipy','skimage')})
    wheels={f'dep{i}':dict(name=f'dep{i}',sha256='same')for i in range(13)};wheels['product']=dict(name='scikit-imagexx',sha256='old')
    fixed={'src/simd_label_points_avx2.cpp':'fixed','CMakeLists.txt':'fixed','coverage/contracts/label-points-benchmark.json':'fixed'}
    old=dict(runtime=runtime,wheels=wheels,source=dict(files_sha256={**fixed,**dict.fromkeys(PINS,'old')}))
    new=deepcopy(old);new['source']['files_sha256'].update(PINS);new['source']['files_sha256'][CONTRACT]=CONTRACT_SHA;new['wheels']['product']['sha256']='new'
    if bad=='foreign-file':new['source']['files_sha256']['foreign']='bad'
    elif bad=='simd':new['source']['files_sha256']['src/simd_label_points_avx2.cpp']='bad'
    elif bad=='case-contract':new['source']['files_sha256']['coverage/contracts/label-points-benchmark.json']='bad'
    elif bad=='dependency':new['wheels']['dep0']['sha256']='bad'
    elif bad=='pytest':new['runtime']['versions']['pytest']='bad'
    elif bad=='runtime':new['runtime']['python']='bad'
    elif bad=='science':new['runtime']['packages_sha256']['numpy']['file']='bad'
    elif bad=='allowed-but-unpinned':new['source']['files_sha256']['src/bind_label_points.cpp']='bad'
    elif bad=='contract':new['source']['files_sha256'][CONTRACT]='bad'
    body=[n for n in PAIR.body[1:]if not(isinstance(n,ast.Assert)and 'Path('in ast.unparse(n))and not isinstance(n,ast.Expr)]
    code=compile(ast.Module(body=body,type_ignores=[]),'actual-pair-closure','exec')
    if bad:
        with pytest.raises(AssertionError):exec(code,dict(old=old,new=new))
    else:exec(code,dict(old=old,new=new))


@pytest.mark.parametrize('bad',[None,'struct','into-abi','missing-assign','numerical-harness-edit','extra-metadata-edit'])
def test_actual_into_abi_and_harness_inverse_AST(tmp_path,monkeypatch,bad):
    monkeypatch.chdir(tmp_path)
    header='struct LabelPointsInput {const void* axes;int ndim;};\nbool label_points_into(const LabelPointsInput&,unsigned long*,int);\n'
    old="def main():\n    result=measure(1,2,7,19)\n    scope='owned zero-fill/scatter'\n"
    new=old+CORRECTION
    candidate_header=header+'bool label_points_assign(const LabelPointsInput&,unsigned long*,int);\n'
    if bad=='struct':candidate_header=candidate_header.replace('int ndim','long ndim')
    elif bad=='into-abi':candidate_header=candidate_header.replace('bool label_points_into','void label_points_into')
    elif bad=='missing-assign':candidate_header=header
    elif bad=='numerical-harness-edit':new=new.replace('measure(1,2,7,19)','measure(1,2,3,19)')
    elif bad=='extra-metadata-edit':new+='    another=1\n'
    for role,harness,h in [('baseline',old,header),('candidate',new,candidate_header)]:
        p=tmp_path/'snapshots'/role;(p/'include/skimagexx').mkdir(parents=True);(p/'benchmarks').mkdir()
        (p/'include/skimagexx/label_points.hpp').write_text(h);(p/'benchmarks/label_points.py').write_text(harness)
    node=next(n for n in PREP.body if isinstance(n,ast.FunctionDef));ns=dict(Path=Path,re=re,ast=ast);exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-source-predicate','exec'),ns)
    if bad:
        with pytest.raises(AssertionError):ns['assert_existing_sources']()
    else:ns['assert_existing_sources']()


def run_decision(tmp_path,monkeypatch,scenario):
    import datetime as dt
    real=dt.datetime
    class Fixed(real):
        @classmethod
        def now(cls,tz=None):return real(2026,10,28,19 if scenario=='expired'else 17,tzinfo=dt.timezone.utc)
    monkeypatch.setattr(dt,'datetime',Fixed);monkeypatch.chdir(tmp_path)
    contract=json.loads((ROOT/'coverage/contracts/label-points-benchmark.json').read_bytes())
    fixed_contract=tmp_path/'snapshots/baseline/coverage/contracts/label-points-benchmark.json';fixed_contract.parent.mkdir(parents=True);fixed_contract.write_text(json.dumps(contract))
    identities={};fingerprints={}
    for role in ('baseline','candidate'):
        source=tmp_path/'snapshots'/role;prepared=source/'build/prepared/ci-artifacts';prepared.mkdir(parents=True);(source/'benchmarks').mkdir()
        manifest=dict(source=dict(revision=role*5),runtime=dict(python='frozen',packages_sha256={'skimagexx':{'_core.so':role*8}}));raw=json.dumps(manifest).encode();(prepared/'measurement.json').write_bytes(raw)
        fingerprints[role]=dict(source_sha256=role,source_hash_schema='fixture',source_hash_scope='fixture');(source/'benchmarks/provenance.py').write_text('def source_fingerprint(root):return '+repr(fingerprints[role])+'\n')
        identities[role]=dict(status='verify',revision=manifest['source']['revision'],wheel_count=14,artifact_sha256=hashlib.sha256(raw).hexdigest())
    target=next(iter(sorted(TARGETS)))
    for name in ('1-baseline','2-candidate','3-candidate','4-baseline'):
        role=name.split('-')[1];candidate=role=='candidate';folder=tmp_path/'results'/name;folder.mkdir(parents=True);rows=[]
        for spec in contract['cases']:
            key=spec['case_id'];cpp=5. if candidate else 10.;ref=20.
            if scenario=='global-direct'and candidate:cpp=10/(1.06 if key in TARGETS else 1.)
            if scenario=='global-normalized'and candidate:ref=cpp*2*(1.06 if key in TARGETS else 1.)
            if scenario.startswith('target-repeat:')and name=='3-candidate'and key==scenario.split(':')[1]:cpp=10/1.04
            if scenario=='target-original'and key==target:ref=3.75 if candidate else 5.
            if scenario=='preserved-original'and key=='dense-64x64-f64':ref=4. if candidate else 5.
            if scenario=='other-direct-floor'and candidate and key=='dense-64x64-f64':cpp=10/.96
            if scenario=='other-normalized-floor'and candidate and key=='dense-64x64-f64':ref=9.6
            if scenario=='inclusive-floor'and candidate and key=='dense-64x64-f64':cpp=10/.97
            rows.append(dict(**spec,api='util.label_points',backend='native',threads=1,native_accepted=True,actual_C_returns=['label_points_apply'],max_abs_error=0,inputs=[dict(sha256='a'*64)],output_metadata={'dtype':'<u8'},output_sha256='b'*64,reference_sha256='b'*64,oracle_sha256='b'*64,sample_labels=['scikit-image','current'],samples_ms=[[ref]*7,[cpp]*7],loops=[1,1],reference_ms=ref,cpp_ms=cpp,speedup=ref/cpp))
        scope=contract['scope']+' '+contract['timed_boundary']+' '+contract['frontier']
        if candidate:scope=scope.replace('owned zero-fill/scatter','NumPy zero-allocation provider and owned ordered scatter')
        meta=dict(**fingerprints[role],source_before=fingerprints[role],repeat=7,threads=1,raw_samples=126,fallback_enabled=False,avx2=True,reference='0.26.0',numpy='1.24.4',scipy='1.15.3',python='frozen',core_sha256=role*8,numpy_error_policy=dict(divide='warn',over='warn',under='ignore',invalid='warn'),numpy_buffer_elements=8192,selected_apis=['util.label_points'],harness_sha256=PINS['benchmarks/label_points.py']if candidate else '903c46306ed0a3e7a57d21a32e794fb9b48fe6af7ea7b685d7cf3ea66b1d0850',reference_pool_limits={n:'1'for n in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')},contract_sha256='3fceb3b66c254f1330f30f779a34a042162c08cf59921bf1343b06376f450221',original_source_sha256='8c409baaf3527b3466c8d8e973e4482182bdf172e8486c33d91eb469403742b2',target_ms=19,seed=2026,provider_frontier=scope)
        if name=='3-candidate':
            if scenario=='input-drift':rows[0]['inputs'][0]['sha256']='f'*64
            elif scenario=='wrong-C':rows[0]['actual_C_returns']=[]
            elif scenario=='wrong-core':meta['core_sha256']='bad'
            elif scenario=='false-owned-zeros':meta['provider_frontier']=contract['scope']+' '+contract['timed_boundary']+' '+contract['frontier']
        (folder/'label-points.json').write_text(json.dumps(dict(metadata=meta,results=rows)))
        for when in ('before','after'):(folder/('identity-'+when+'.log')).write_text(json.dumps(identities[role]))
    exec(compile(SCRIPTS[2],'actual-summary','exec'),{})
    return json.loads((tmp_path/'results/comparison.json').read_bytes())


@pytest.mark.parametrize('scenario,decision',[('keep','KEEP'),('inclusive-floor','KEEP'),('global-direct','DISCARD'),('global-normalized','DISCARD'),('target-original','DISCARD'),('preserved-original','DISCARD'),('other-direct-floor','DISCARD'),('other-normalized-floor','DISCARD'),('expired','DISCARD')]+[('target-repeat:'+key,'DISCARD')for key in sorted(TARGETS)])
def test_exact_frozen_decision(tmp_path,monkeypatch,scenario,decision):
    out=run_decision(tmp_path,monkeypatch,scenario)
    assert out['status']==decision and out['raw_samples']==504 and len(out['rows'])==9
    if scenario=='target-original':assert out['target_repeat_wins']==3 and out['target_original_repeat_wins']==2
    if scenario=='keep':assert out['target_repeat_wins']==3 and out['preserved_original_repeat_wins']==6


@pytest.mark.parametrize('scenario',['input-drift','wrong-C','wrong-core','false-owned-zeros'])
def test_actual_metadata_and_C_guards(tmp_path,monkeypatch,scenario):
    with pytest.raises(AssertionError):run_decision(tmp_path,monkeypatch,scenario)


@pytest.mark.parametrize('mutate',[False,True])
def test_actual_raw_contract_file_gate(tmp_path,monkeypatch,mutate):
    import base64,zlib
    raw=zlib.decompress(base64.b64decode(CONTRACT_COMPRESSED))
    assert raw.count(b'\r\n')==0 and hashlib.sha256(raw).hexdigest()==CONTRACT_SHA
    assert hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==CONTRACT_SHA
    monkeypatch.chdir(tmp_path);path=tmp_path/'snapshots/candidate'/CONTRACT;path.parent.mkdir(parents=True);path.write_bytes(raw+(b'changed'if mutate else b''))
    node=next(n for n in PAIR.body if isinstance(n,ast.Assert)and 'Path('in ast.unparse(n))
    code=compile(ast.Module(body=[node],type_ignores=[]),'actual-contract-file-gate','exec')
    if mutate:
        with pytest.raises(AssertionError):exec(code,dict(hashlib=hashlib,Path=Path,contract=CONTRACT))
    else:exec(code,dict(hashlib=hashlib,Path=Path,contract=CONTRACT))

CONTRACT_COMPRESSED='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'
