"""Inert fixed-gate tests; synthetic latencies only, no scientific imports."""
import ast
import base64,zlib
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import textwrap
import pytest
ROOT=Path(__file__).resolve().parents[1]
WORKFLOW=(ROOT/'.github/workflows/label-points-metadata-same-host.yml').read_text()
SCRIPTS=[textwrap.dedent(s)for s in re.findall(r"python - <<'PY'\n(.*?)          PY\n",WORKFLOW,re.S)]
PREP=ast.parse(SCRIPTS[1])
PAIR=next(n for n in PREP.body if isinstance(n,ast.If)and ast.unparse(n.test)=="role == 'candidate'")
CONTRACT='coverage/contracts/label-points-metadata-candidate.json'
CONTRACT_SHA='2276bdf9fc048b923613f9964c6a096169f7483d1babd9b95c802347cc75b3f9'
DELTA=ast.literal_eval(next(n.value for n in PAIR.body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='preexisting_metadata_delta'))
BINDER_SHA='66fc484b8676e0fffd12c81748e21d38217d3becadac59f294d0ee78a49f689e'
TARGETS={'sparse-64x64-i64'}



def test_fixed_source_transport_and_no_timing_host_build_or_test():
    assert len(SCRIPTS)==3
    for s in SCRIPTS:compile(s,'workflow-inline','exec')
    assert '40724150045' in SCRIPTS[0] and '40738793434' in SCRIPTS[0]
    assert '910c31a609c157c28cbbe56933b8448b710af95e' in SCRIPTS[0]
    assert '69e390dd5e48d5199c9a249fccafd83daa027d3c' in SCRIPTS[0]
    assert '12833680393' in SCRIPTS[0] and '38797cf494532e16da41f751062c57815de2d566d8ebf11f3b59be26ba4be536' in SCRIPTS[0]
    assert '12839223780' in SCRIPTS[0] and '700cfabeae84a880753995117cd3fe2df6ec5e552c72c8c20309128d103f4503' in SCRIPTS[0]
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
    assert 'if: always()' in WORKFLOW and 'name: label-points-metadata-same-host-minimum' in WORKFLOW
    assert '2026-10-28T19:39:29.511203+00:00' in SCRIPTS[2]
    loop=next(n for n in PREP.body if isinstance(n,ast.For)and isinstance(n.target,ast.Tuple))
    for role,expected in [('baseline',7134),('candidate',7134)]:
        pairs=eval(compile(ast.Expression(loop.iter),'actual-counts','eval'),dict(role=role))
        assert len(pairs)==33 and sum(c*2 for n,c in pairs if n!='color-transforms-regression')==expected
    assert len(DELTA)==21
    assert not any(n.startswith(('src/','include/','python/'))for n in DELTA)
    assert 'keep = (direct > 1.01 and normalized > 1.01' in SCRIPTS[2]
    assert 'len(targets) == 1 and len(preserved) == 7 and len(others) == 8' in SCRIPTS[2]


@pytest.mark.parametrize('bad',[None,'foreign-file','simd','python','harness','case-contract','dependency','pytest','runtime','science','binder','contract','old-metadata','new-metadata','SDK','package-python','Montage'])
def test_actual_closed_source_and_dependency_closure(bad):
    runtime=dict(python='fixed',implementation='fixed',cache_tag='fixed',machine='fixed',system='fixed',dependency_libraries_sha256={'a':'b'},versions={'scikit-imagexx':'0.1','pytest':'same'},packages_sha256={k:{'file':'same'}for k in ('numpy','scipy','skimage')},sdk_sha256={'lib/libskimagexx.a':'same'})
    runtime['packages_sha256']['skimagexx']={'_core.so':'old','_label_points.py':'same'}
    wheels={f'dep{i}':dict(name=f'dep{i}',sha256='same')for i in range(13)};wheels['product']=dict(name='scikit-imagexx',sha256='old')
    fixed={n:'same'for n in ['src/bind_label_points.cpp','src/simd_label_points_avx2.cpp','python/skimagexx/_label_points.py','benchmarks/label_points.py','coverage/contracts/label-points-benchmark.json']}
    fixed.update({n:v['old']for n,v in DELTA.items()if v['old']is not None})
    old=dict(runtime=runtime,wheels=wheels,source=dict(files_sha256=fixed));new=deepcopy(old)
    for name,v in DELTA.items():
        if v['new']is None:new['source']['files_sha256'].pop(name)
        else:new['source']['files_sha256'][name]=v['new']
    new['source']['files_sha256'].update({'src/bind_label_points.cpp':BINDER_SHA,CONTRACT:CONTRACT_SHA});new['wheels']['product']['sha256']='new';new['runtime']['packages_sha256']['skimagexx']['_core.so']='new'
    mapping={'foreign-file':'foreign','simd':'src/simd_label_points_avx2.cpp','python':'python/skimagexx/_label_points.py','harness':'benchmarks/label_points.py','case-contract':'coverage/contracts/label-points-benchmark.json','binder':'src/bind_label_points.cpp','contract':CONTRACT,'Montage':'src/montage.cpp'}
    if bad in mapping:new['source']['files_sha256'][mapping[bad]]='bad'
    elif bad=='dependency':new['wheels']['dep0']['sha256']='bad'
    elif bad=='pytest':new['runtime']['versions']['pytest']='bad'
    elif bad=='runtime':new['runtime']['python']='bad'
    elif bad=='science':new['runtime']['packages_sha256']['numpy']['file']='bad'
    elif bad=='old-metadata':old['source']['files_sha256']['.gitattributes']='bad'
    elif bad=='new-metadata':new['source']['files_sha256']['.gitattributes']='bad'
    elif bad=='SDK':new['runtime']['sdk_sha256']['lib/libskimagexx.a']='bad'
    elif bad=='package-python':new['runtime']['packages_sha256']['skimagexx']['_label_points.py']='bad'
    body=[n for n in PAIR.body[1:]if not(isinstance(n,ast.Assert)and 'Path('in ast.unparse(n))and not isinstance(n,ast.Expr)]
    code=compile(ast.Module(body=body,type_ignores=[]),'actual-pair-closure','exec')
    if bad:
        with pytest.raises(AssertionError):exec(code,dict(old=old,new=new))
    else:exec(code,dict(old=old,new=new))


@pytest.mark.parametrize('bad',[None,'header','leaf','harness','changed-worker','fifth-reserve','owner-return'])
def test_actual_binder_only_inverse_and_immutable_production(tmp_path,monkeypatch,bad):
    monkeypatch.chdir(tmp_path)
    old=zlib.decompress(base64.b64decode(BINDER_BASE)).decode();new=zlib.decompress(base64.b64decode(BINDER_CANDIDATE)).decode()
    if bad=='changed-worker':new=new.replace('workers < 1','workers < 2')
    elif bad=='fifth-reserve':new=new.replace('sizes.reserve(shape.size());','sizes.reserve(shape.size());sizes.reserve(32);')
    elif bad=='owner-return':new=new.replace('return output;','return py::none();')
    fixed=['include/skimagexx/label_points.hpp','src/label_points.hpp','src/label_points.cpp','src/simd_label_points_avx2.cpp','python/skimagexx/_label_points.py','benchmarks/label_points.py']
    for role,binder in [('baseline',old),('candidate',new)]:
        p=tmp_path/'snapshots'/role
        for name in fixed+['src/bind_label_points.cpp']:
            f=p/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(binder if name=='src/bind_label_points.cpp'else 'fixed')
    mapping={'header':'include/skimagexx/label_points.hpp','leaf':'src/label_points.cpp','harness':'benchmarks/label_points.py'}
    if bad in mapping:(tmp_path/'snapshots/candidate'/mapping[bad]).write_text('changed')
    node=next(n for n in PREP.body if isinstance(n,ast.FunctionDef));ns=dict(Path=Path,re=re,ast=ast);exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-source-predicate','exec'),ns)
    if bad:
        with pytest.raises(AssertionError):ns['assert_existing_sources']()
    else:ns['assert_existing_sources']()


def run_decision(tmp_path,monkeypatch,scenario):
    import datetime as dt
    real=dt.datetime
    class Fixed(real):
        @classmethod
        def now(cls,tz=None):return real(2026,10,28,20 if scenario=='expired'else 18,tzinfo=dt.timezone.utc)
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
            if scenario=='preserved-sparse1024-original'and key=='sparse-1024x1024-i64':ref=4. if candidate else 5.
            if scenario=='mixed-sparse3D-loss'and key=='sparse-64x64x64-f32':ref=4.
            if scenario=='other-direct-floor'and candidate and key=='dense-64x64-f64':cpp=10/.96
            if scenario=='other-normalized-floor'and candidate and key=='dense-64x64-f64':ref=9.6
            if scenario=='inclusive-floor'and candidate and key=='dense-64x64-f64':cpp=10/.97
            rows.append(dict(**spec,api='util.label_points',backend='native',threads=1,native_accepted=True,actual_C_returns=['label_points_apply'],max_abs_error=0,inputs=[dict(sha256='a'*64)],output_metadata={'dtype':'<u8'},output_sha256='b'*64,reference_sha256='b'*64,oracle_sha256='b'*64,sample_labels=['scikit-image','current'],samples_ms=[[ref]*7,[cpp]*7],loops=[1,1],reference_ms=ref,cpp_ms=cpp,speedup=ref/cpp))
        scope=contract['scope']+' '+contract['timed_boundary']+' '+contract['frontier']
        meta=dict(**fingerprints[role],source_before=fingerprints[role],repeat=7,threads=1,raw_samples=126,fallback_enabled=False,avx2=True,reference='0.26.0',numpy='1.24.4',scipy='1.15.3',python='frozen',core_sha256=role*8,numpy_error_policy=dict(divide='warn',over='warn',under='ignore',invalid='warn'),numpy_buffer_elements=8192,selected_apis=['util.label_points'],harness_sha256='903c46306ed0a3e7a57d21a32e794fb9b48fe6af7ea7b685d7cf3ea66b1d0850',reference_pool_limits={n:'1'for n in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')},contract_sha256='3fceb3b66c254f1330f30f779a34a042162c08cf59921bf1343b06376f450221',original_source_sha256='8c409baaf3527b3466c8d8e973e4482182bdf172e8486c33d91eb469403742b2',target_ms=19,seed=2026,provider_frontier=scope)
        if name=='3-candidate':
            if scenario=='input-drift':rows[0]['inputs'][0]['sha256']='f'*64
            elif scenario=='wrong-C':rows[0]['actual_C_returns']=[]
            elif scenario=='wrong-core':meta['core_sha256']='bad'
            elif scenario=='wrong-frontier':meta['provider_frontier']='changed'
        (folder/'label-points.json').write_text(json.dumps(dict(metadata=meta,results=rows)))
        for when in ('before','after'):(folder/('identity-'+when+'.log')).write_text(json.dumps(identities[role]))
    exec(compile(SCRIPTS[2],'actual-summary','exec'),{})
    return json.loads((tmp_path/'results/comparison.json').read_bytes())


@pytest.mark.parametrize('scenario,decision',[('keep','KEEP'),('inclusive-floor','KEEP'),('global-direct','DISCARD'),('global-normalized','DISCARD'),('target-original','DISCARD'),('preserved-original','DISCARD'),('preserved-sparse1024-original','DISCARD'),('mixed-sparse3D-loss','KEEP'),('other-direct-floor','DISCARD'),('other-normalized-floor','DISCARD'),('expired','DISCARD')]+[('target-repeat:'+key,'DISCARD')for key in sorted(TARGETS)])
def test_exact_frozen_decision(tmp_path,monkeypatch,scenario,decision):
    out=run_decision(tmp_path,monkeypatch,scenario)
    assert out['status']==decision and out['raw_samples']==504 and len(out['rows'])==9
    if scenario=='target-original':assert out['target_repeat_wins']==1 and out['target_original_repeat_wins']==0
    if scenario=='mixed-sparse3D-loss':assert out['balanced_original_losses']==['sparse-64x64x64-f32']
    if scenario=='keep':assert out['target_repeat_wins']==1 and out['preserved_original_repeat_wins']==7


@pytest.mark.parametrize('scenario',['input-drift','wrong-C','wrong-core','wrong-frontier'])
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

CONTRACT_COMPRESSED='eJzlV0mP2zYUvudXEL52bGtfJrcmaFEULYImORUBQZGUxYxECiRle6bIf+8jtdhyZ5L2UPTQw4zl9/gWfW/h5z9eIbQxtOEd2dyjTUsq3m57JaQ1245bwoglW0okE/DEt8dwc+csFgmuiOFY86MwQknnImV1TIK6yvKSRVlSVmlQB2We8IxlSUqCgoRlmRZ0dOTMWyHXLsowoHFIsqCkYZrTqKBVxdOsjOOqSJKiysOA1GXKb10MzjoJ8igJ0yBI0rWaaCtqQi2cCaMijrMiiMt4feZJ9Ng0JEozl0dc5GVO66RM0jjiYcZIEtZ5GgZZRNO8CFPGI5ZmGSt4VYdhHVdpWfEoq0gC+cbZTX6Gqp6D3z9ACvJeKzaM6WTF3Sh7//Zn+J6X01fRkYOziJJoksw1AWE8iRjvuWRcUsGNcxaD+IuPXGv1xCUeLHVvEwVRtg2DbVR8CPP7uLyPyl0ahlEQfxcE90EwZss4YT7bZ6zKF62ORAsCPePiewFpD0oL23TOx/cC8tNozh0p2T7eI807deTwwQbJwBipk+TaoCOnVunXoDBcw4EHCQqkiXxAtdLwN2hnS4QU8jCdNq9RpbRWJ8TsY88hAuWoh5jkLMwO/apQR2xzh9Rg+8Ei0raKEgvtdod6zduBcXgYqlbQfa/FERobGXGQxA4aNPApIJEGoAGXkMP7n355i2hD5IHvRgBc7gzXovU1+H2sjNF0X8G7Yz9WeByrHe37zVQ6CgBoKPGeKmk19KbZf2MCd58NzAhYf3puDqc4Y2PB22F+BqfYKtxxYuBdGIzWOr+XUnM6I7p17pgcz9HtqZWH5qITkjpg9+bBt/H5/OLJ/tE2Sl4dXAPWP16H+yqgFYxB0xH9YPYvuPgW5IuHG6Q9vpVvZAyoKykoaa92BU3rNMoJjRIeVFEdx0kQRiFsqoLVDPZRwaM6TeCpykoSV0FNWEnqgNZZnNCsLMY2msrUcWkviwLSlKZX2ok2P3I5wHiiU6Navp1K7Zq4J9p3tLlDNYxOgwzp+LZRxiJ7UkjVtRtrBEvSio7DqXkv7ZcmunqalQts7vUFuzS3qwaENHybJecs2Yosmc76JSIXRX2tmEzCIErO7t9zZhflc6beqfcbR88G/GsyQw9zDS9lXnRttWCcXauvnc/qlXuv/DShYxsNu2HZfiCBenDiCpbPR4g+cIs7f2je75qcMJSpH7dGGiTL6j64fC8t4Ju2bTETGvYdPvhbYxfM0Wa1VLojrXiCVXR7ZIzvOuiFss0ZckIbfNkrPTHmNmz6bYtnMvkbVnBlHISEuZpt5rEmLYGNzjCMOvaDq9pVK/7fGm7GYERxgWepkyMNwa7Mv358VaQbk366ftmlKG4H/Ouo/8fAXl77n/TnTJpAaPXAF9S1K4fkxqwVkAMsYTvyNSdfxn5tMg3/SDTYfPt8+Oh00ZKwpysYKDx9MJ7/zlmNhOaiictsMVJrquKuHuBT3dBdwOoURJNreJxsZTW20BXCcCsSvTZaUDM9pz7Fme6qFtqrcoCPVxcGuDFcw+QB2Lw6GQwUDruYYLQk74xGRoaNOOM5pc1vfDAckcE2Dl3XJgwNcgIPGaCNlO9V9Zm7O/+kNNv/8A7Brw5HnflrJNXM8z58RFR1vWh9Uo7vyaHj2l34QDz7lixk4mo0Wr+QZuglP2Euj0Ir6e7ya5W/uDEQfVwNomXXKkdKWwzXMxBbL1/6AsLC/Q/i8UUtkF9HYkvkZ/IOQe2sIG2BWmW8AJRozm7/5t07SPwArW1GgjDOZ/wWdeIM2MwHHeUeWqDHkiHPHJH/yeL5s7FAmiDcYGBj2EeHCsTYjp729QDPNaDtEYM6g1faEtHt0BtICb49ca0cXXF4azT9Xph4vIFPVxY45kIvRdttXn159SdTvmXm'

BINDER_BASE='eJyVVltv2zYUfvevOHGBVEpVxW6LYZBtDcWwhwHF2vcgEGiJtjlLIkFSvizNf9/hxbYk2/EmIJHMc+V3zvnId6zOy6agMCzJnJaZ4KzWKl4JMRy8O8rmrC4yIiXZZ0pzSZa0pzEVe6MzHj/WTSX28Sq9JDt8dMW50gUGbS+VrGJatVc2NMfA6WBQk4oqQXIKYg8zOLicDAYbzgqwmba3Eoh9klS8aEqa3YP7COFlAPi4X3FBF0Fn+xkRotwPI3h6DnJeKw3GiW5ESe8BA7Ccqsh6aD1nimpFBI1QX8OWyzWVKoSPqdXg879xPz4L87AFBF4JpjCGnz8PNpDCePTpS9gJp1eSb62nDSkbmlEpufR7+Oj24O1B1USoFdfAG60YQvk0jozD52E46YS36caK/UODEGYzGJkkOospfP5kFj0Ch+W7WUctBEl1I2ubXs1rXDoFwlJjzraWUyNXxibTKZi3uqxnvw96BatorRhi3VN2CliFBvGewfgkXnAJvoyk0RwsZJC4pMNWEQ5I3P3Yf+P1Mvt9RfP1HzuS68DaxEJL3N+bG7Q+MAO+oXJRYpEQyK60lYnJGRV8uK/KvMzf17r47u3bkSO4P7jtR8SsjxFN2YzjKYxupmphj0WjVtmc5OugVZPAvMKe/gn+llEL/zOj16tlsDkmLY+hA996gJdjJUcTmEtK1pOWL6No5eF/cXteYauUQidx843kRSXLM8c/nQ5FEiE7092PNgRfOIsGi/3Ll0ybdZdRnxdulcDt82Fm3V4Crj8zlodT4Nsa6eHyyDg87IrP7yEFsntzwrDFCrZY2GHUEpniyoR5eun0tS1CW4nsmFNxX9MeZ0zgwwcjuDh9Zo9MMdwBqXPa2nLgnTwZ0+fbc2j7wZragyJJJMXUqRRol805cub2uvfzAbNqMSpVjvQsS7vFQu8Fho/X6MMJ37P3Z41gnqMJ07Q6EeivRnDXdUY0Dv2QIYFrtqHDMM6J0tM552WKffh/28xC6zA/VrDd/y62p/CuKS0VdWNzTd9swXm9zThrVuH9YbdLEnNWs3qpkiQ3TNu9YLgA/YruaJd4EJo8s7hcanmfZEE0MbzdS8Q1ecud35JbD0Z9Azdwfd6rkHd9rhdp7/ER/qLbcg+kLHlONC3MSSwaPKpIXeO5TEpGFGBPCiJRmHOOc4g1pzBvFgsT8ejLNrS3nnni+A2OLZzpaYeTUstzqtsqyU396Oop7eS+6BF0EPL4n6pbUE1YmSTfzKXkh22OP2uTOTP/X87r6soUHQtz+N06cy4sHZJx7Rc5VF4nHcweAP1pAyqaIHIOwrhqNJnjtdB5PZmYEQOS51RgtU7L3ZQNLktWZirnghaZpCUlOCb+3Wtb78uM3DlAnZsn/uOBRShqJx0dr5CXeszNtg/y5gR6kQPALb9GviOWSDaOApFrao4FxRuFttge5PbKdFXqUxyatngd/AuKu829'

BINDER_CANDIDATE='eJyVVltv2zYUfvevOHWB1EpVJe6KYZBtDUWxhwHF1oe9BYFAU7TNWRYFknLspvnvO7zIEn2JOwGJZJ4rz3fOR77lFS2bgsGwJHNW5rXglVbJqq6Hg7cH2ZxXRU6kJPtcaSHJkh1pTOu90RmP76pmU++TVXZO1n6EYqp0gUH7SyXfcK36K1tGMXA2GFRkw1RNKIN6DzNoXU4Gg63gBdhM+1sZ1fs03YiiKVl+A+4jgucB4ON+JQVbjILt56Suy/0whofHERWV0mCc6KYu2Q1gAE6Ziq2H3nOiqFakZjHqa3gScs2kiuBDZjXE/F/cj8/CPHwBI68EUxjDjx+tDWQwvv/4KQrC6ZUUT9bTlpQNy5mUQvo9fHB78PagKlKrldAgGq04lvJhHBuHj8NoEoS36SaKf2ejCGYzuDdJBIsZ/PLRLPoKtMtvZoFaBJLpRlY2vUpUuNQFQqgxZ4vl1MiVscl1BuatzuvZ71av4BtWKY617ikb20QyxeSWBfvoRe4Mr2n24iGoDcI3g3EnXggJvitIowVYBCB1NYh6mLaFffNt/1VUy/zLitH1HztC9cjaJLWWGPjVelkfmIHYMrkoEXPEJZT2MjE5o4IP91mZl/n7XBV/e/t+5BhuWrfHETHrQ0TTBcbxFO6vpuqQqBu1yueErkc9iEfmFR3p90DpjHr1PzF6uQiDzTHteYxc8a0HeD4geT+BuWRkPen5MopWHv2M21OErVIGQeLmG7mQSU5zR2dBwyMnkZ0ZljsbQiycRYNg//op12bdZXRMM9cgcPu8nVm35wrXHy23UbviA99mQHavTiL2TsEXCzu0WiKj9JSN6WG8QpoIBsya/YxiN4me1YL+v7uDf1YM+GbTaDIvWctMYBkYxFOlgBG6wry4upsTxSxzNssVvrmCL0BJWSZhT/WDGjvXM/ZrekR9E3j/3gjOTr0BiCuOBSYVZRZ8e4Jm7X4fjOnj9fm3fWhN7XmXppJhKZis0S6fC6T+p8veL5JFofe1YQtrldhfJ8SD23BidLlxTG+PJqudrDGSW3zH3520qXlQ1dlzzTbdafGbEbxxTohGKhpyPKU037JhlFCi9HQuRJlhK/zf5reFd4gc+qU/lS6bkz4zDyuxO+wwX9I3qTuv13lwzTd4Sdrt0tRcSHi1VGlKDf+HtygX4BjvHQvpEEtDc1uXc/PqkyyIJuY0OUrEj1rnzm/JrY/uz5MrDtZf7KncA46HoESzwlwf6gYPRFJVeJkgJScKsANrIlFIhUBSQAwZzJvFAq8t3VDZbvPWM09Pv8OhYXM9DZgvs2yqQujTq/rxxauFk3sQYwh27OvZoVUwTXiZpl/NTeqbBfvPymTOzf/nU5xc2eNDodvfvZPtzFKbjGun2FXlZRLU7BbQnzZFRROsnCth4qkud147EzMyQChlNaLVLYcpm7oseZkrKmpW5JKVzJCifx+1ofdlRui0QMF1Gf+Jka1Q3E86Ptx7z/WYm1Uf5NWJ8iJXALf8EvuOWCJ5OMJD7qgEAor3Fm1r28rtxeyi1Kc4NG3xMvgPw1AGRg=='
