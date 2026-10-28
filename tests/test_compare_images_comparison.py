"""Metadata-only tests of the fixed Compare ABBA contract; no scientific calls."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import textwrap
import pytest

ROOT=Path(__file__).resolve().parents[1]
WORKFLOW=(ROOT/'.github/workflows/compare-images-same-host.yml').read_text(encoding='utf-8')
SCRIPTS=[textwrap.dedent(s)for s in re.findall(r"python - <<'PY'\n(.*?)          PY\n",WORKFLOW,re.S)]
PREPARATION=ast.parse(SCRIPTS[1])
PAIR=next(n for n in PREPARATION.body if isinstance(n,ast.If)and any(isinstance(x,ast.Assign)and ast.unparse(x.targets[0])=='(old, new)'for x in n.body))
MUTABLE={'python/skimagexx/_compare_images.py','src/compare_images.cpp','src/compare_images.hpp','src/simd_compare_images_avx2.cpp','tests/test_compare_images_native.cpp'}


def test_frozen_products_command_normalization_and_isolated_timing():
    assert len(SCRIPTS)==4
    for source in SCRIPTS:compile(source,'inline-workflow','exec')
    assert "default: '40649386523'"in WORKFLOW and '317658444d7de04c751fe92d912c60853054c0e6'in SCRIPTS[0]
    assert "item['id'] == 12798648924"in SCRIPTS[0]and '0179be5c3b7b3232b5c005bbfc4ded321a11f34e64207913b0a6382ea684265e'in SCRIPTS[0]
    assert "re.fullmatch('[0-9a-f]{40}', revision)"in SCRIPTS[0]
    assert '4fae66de81eefb42a13a899c6a0a8e843f5aac8d5ab5f2e24818a1ba5679ecf8'in SCRIPTS[3]
    assert '2026-10-28T09:53:11.706388+00:00'in SCRIPTS[3]
    timed=WORKFLOW.split('      - name: Measure the fixed18',1)[1].split('      - name:',1)[0]
    assert 'for pass in 1-baseline 2-candidate 3-candidate 4-baseline; do'in timed
    assert 'benchmarks/compare_images.py --threads 1 --repeat 7'in timed
    assert timed.count('ci_measurement_artifacts.py verify')==2
    assert not re.search(r'pytest|cmake|ctest|pip install|pip wheel',timed)
    assert 'unset PYTHONPATH PYTHONHOME SKIMAGEXX_DISABLE_SIMD'in timed
    assert 'name: compare-images-same-host-minimum'in WORKFLOW and 'if: always()'in WORKFLOW
    normalization=WORKFLOW.split('      - name: Normalize only',1)[1].split('      - name: Measure',1)[0]
    assert '--no-index --no-deps --only-binary=:all: --require-hashes --force-reinstall'in normalization
    assert 'measurement-v2/pytest-lock.txt'in normalization
    families=next(ast.literal_eval(n.iter)for n in PREPARATION.body if isinstance(n,ast.For)and isinstance(n.target,ast.Tuple))
    assert ('compare-images',109)in families and sum(n*2 for family,n in families if family!='color-transforms-regression')==4710
    assert "{'passed': 1417}"in SCRIPTS[1]and '== 239'in SCRIPTS[1]


def manifest(candidate=False):
    common=ast.literal_eval(next(n.value for n in PAIR.body if isinstance(n,ast.Assign)and ast.unparse(n.targets[0])=='common'))
    common|={'benchmarks/compare_images.py','tests/compare_images_cases.py','tests/test_compare_images_public.py','python/skimagexx/_compare_images_checks.py','src/bind_compare_images.cpp','CMakeLists.txt'}
    runtime={k:'same'for k in('python','implementation','cache_tag','machine','system')}
    runtime.update(dependency_libraries_sha256={'numpy':{'blas.so':'a'}},versions={'numpy':'1.24.4','pytest':'9.3.0'if candidate else'9.2.0','scikit-imagexx':'0.1.0'},packages_sha256={n:{'source.py':'a'}for n in('numpy','scipy','skimage')})
    wheels={f'dep{i}.whl':dict(name=f'dep{i}',version='1',sha256='a'*64)for i in range(12)}
    version='9.3.0'if candidate else'9.2.0';digest='b'*64 if candidate else'07acdcf88ec174f907dc01af70bf52d49b014bbab03faeeade31c96b18c2e588'
    wheels['pytest-'+version+'-py3-none-any.whl']=dict(name='pytest',version=version,sha256=digest)
    wheels['native.whl']=dict(name='scikit-imagexx',version='0.1.0',sha256='c'*64 if candidate else'd'*64)
    return dict(runtime=runtime,wheels=wheels,source=dict(revision='c'*40 if candidate else'b'*40,files_sha256=dict.fromkeys(common|MUTABLE|{'src/resample.cpp','src/polygon.cpp','pyproject.toml'},'a')))


@pytest.mark.parametrize('bad',[None,'sampler','geometry','binding','build','harness','old-cases','new-file','science','pytest'])
def test_only_five_contract_files_and_test_runner_may_change(bad):
    old,new=manifest(),manifest(True)
    for name in MUTABLE:new['source']['files_sha256'][name]='changed'
    paths={'sampler':'src/resample.cpp','geometry':'src/polygon.cpp','binding':'src/bind_compare_images.cpp','build':'CMakeLists.txt','harness':'benchmarks/compare_images.py','old-cases':'tests/compare_images_cases.py','new-file':'new.cpp'}
    if bad in paths:new['source']['files_sha256'][paths[bad]]='changed'
    elif bad=='science':new['wheels']['dep0.whl']['sha256']='changed'
    elif bad=='pytest':new['runtime']['versions']['pytest']='mismatch'
    code=compile(ast.Module(body=PAIR.body[1:-1],type_ignores=[]),'pair-contract','exec')
    if bad:
        with pytest.raises(AssertionError):exec(code,dict(old=old,new=new,re=re))
    else:exec(code,dict(old=old,new=new,re=re))


@pytest.mark.parametrize('bad',[None,'missing-alignment','fallback','method-policy','shape-order','scalar-provider','grid-API'])
def test_source_gate_preserves_full_frontend_except_exact_copy_bypass(tmp_path,monkeypatch,bad):
    monkeypatch.chdir(tmp_path)
    old_python="""def compare_images(image0,image1):
    if image1.shape != image0.shape:raise ValueError('shape')
    img1=convert(image0)
    img2=convert(image1)
    first=np.require(img1,requirements=('C','A','O'))
    second=np.require(img2,requirements=('C','A','O'))
    if method=='diff':return same_math(first,second)
    return scalar_provider()
"""
    new_python=old_python
    for output,image in(('first','img1'),('second','img2')):
        new_python=new_python.replace(f'{output}=np.require',f'{output}={image} if ({image}.flags.c_contiguous and {image}.dtype.isnative and {image}.flags.aligned and {image}.flags.owndata)else np.require')
    header='enum class CompareMethod { diff,blend,checkerboard };\nstruct CompareInput {const void* data;int type;};\nstruct CompareGrid {int rows,columns;};\nbool compare_images_into(int first);\n'
    new_header=header+'bool fused_span(bool& finite);\n'
    if bad=='missing-alignment':new_python=new_python.replace(' and img1.flags.aligned','')
    elif bad=='fallback':new_python=new_python.replace("requirements=('C','A','O')","requirements=('C',)",1)
    elif bad=='method-policy':new_python=new_python.replace("method=='diff'","method=='blend'")
    elif bad=='shape-order':new_python=new_python.replace('image1.shape != image0.shape','image0.shape != image1.shape')
    elif bad=='scalar-provider':new_python=new_python.replace('scalar_provider()','other_provider()')
    elif bad=='grid-API':new_header=new_header.replace('rows,columns','columns,rows')
    for role,python,h in [('baseline',old_python,header),('candidate',new_python,new_header)]:
        for name,text in[('python/skimagexx/_compare_images.py',python),('src/compare_images.hpp',h)]:
            p=tmp_path/'snapshots'/role/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
    function=next(n for n in PREPARATION.body if isinstance(n,ast.FunctionDef)and n.name=='assert_existing_sources')
    namespace=dict(ast=ast,re=re,Path=Path);exec(compile(ast.Module(body=[function],type_ignores=[]),'copy-source-policy','exec'),namespace)
    if bad:
        with pytest.raises(AssertionError):namespace['assert_existing_sources']()
    else:namespace['assert_existing_sources']()


def decision_fixture(tmp_path,monkeypatch,scenario):
    # These are synthetic latency fields on real corpus metadata, never performance evidence.
    import datetime
    real=datetime.datetime
    class FixedDatetime(real):
        @classmethod
        def now(cls,tz=None):return real(2026,10,28,10 if scenario=='expired'else 9,tzinfo=datetime.timezone.utc)
    monkeypatch.setattr(datetime,'datetime',FixedDatetime);monkeypatch.chdir(tmp_path)
    original=json.loads((ROOT/'benchmarks/compare-images-minimum-initial.json').read_bytes())
    sources={};fingerprints={}
    for role in('baseline','candidate'):
        source=tmp_path/'snapshots'/role;prepared=source/'build/measurement-v2';prepared.mkdir(parents=True);(source/'benchmarks').mkdir()
        value=manifest(role=='candidate');value['runtime']['python']=original['metadata']['python'];value['runtime']['packages_sha256']['skimagexx']={'_core.so':('c'if role=='candidate'else'b')*64}
        raw=json.dumps(value).encode();(prepared/'measurement.json').write_bytes(raw)
        fingerprints[role]=dict(source_sha256=('c'if role=='candidate'else'b')*64,source_hash_schema='fixture',source_hash_scope='fixture')
        (source/'benchmarks/provenance.py').write_text('def source_fingerprint(root):return '+repr(fingerprints[role])+'\n')
        sources[role]=(value,hashlib.sha256(raw).hexdigest())
    for name in('1-baseline','2-candidate','3-candidate','4-baseline'):
        role=name.split('-')[1];candidate=role=='candidate';folder=tmp_path/'results'/name;folder.mkdir(parents=True)
        report=deepcopy(original);report['metadata'].update(fingerprints[role]);report['metadata']['source_before']=fingerprints[role]
        report['metadata']['core_sha256']=sources[role][0]['runtime']['packages_sha256']['skimagexx']['_core.so']
        for i,row in enumerate(report['results']):
            cpp=1. if candidate else 2.;reference=4.
            checker=row['method']=='checkerboard';first_target=row['case_id']=='side256:uint8:diff'
            if scenario=='target-direct'and first_target and name=='3-candidate':cpp=2.5
            if scenario=='target-normalized'and first_target and name=='2-candidate':reference=1.5
            if scenario=='diff-method'and row['method']=='diff':reference=.8 if candidate else.5
            if scenario=='blend-method'and row['method']=='blend':reference=.8 if candidate else.5
            if scenario=='checker-original'and checker and i==2 and name=='2-candidate':reference=.5
            if scenario=='checker-cpp-loss'and checker and candidate:cpp,reference=3.,6.
            if scenario=='target-original-loss'and first_target:reference=.8 if candidate else 1.
            if scenario=='direct-aggregate'and checker and candidate:cpp,reference=1000.,2000.
            if scenario=='normalized-aggregate'and checker:reference=2. if candidate else 2e12
            row.update(reference_ms=reference,cpp_ms=cpp,speedup=reference/cpp,samples_ms=[[reference]*7,[cpp]*7],loops=[1,1])
        identity=dict(status='verify',revision=sources[role][0]['source']['revision'],wheel_count=14,artifact_sha256=sources[role][1])
        for when in('before','after'):(folder/('identity-'+when+'.log')).write_text(json.dumps(identity))
        if name=='2-candidate':
            row=report['results'][0]
            if scenario=='missing':report['results'].pop()
            elif scenario=='duplicate':report['results'][-1]=deepcopy(row)
            elif scenario=='output':row['reference_sha256']='e'*64
            elif scenario=='metadata':row['output_metadata']['strides']=[1,1]
            elif scenario=='input':row['input_sha256'][0]='e'*64
            elif scenario=='not-accepted':row['native_accepted']=False
            elif scenario=='median':row['cpp_ms']+=.25
            elif scenario=='harness':report['metadata']['harness_sha256']='e'*64
            elif scenario=='core':report['metadata']['core_sha256']='e'*64
            elif scenario=='fingerprint':report['metadata']['source_sha256']='e'*64
            elif scenario=='identity':(folder/'identity-before.log').write_text('{}')
        (folder/'compare-images.json').write_text(json.dumps(report))


@pytest.mark.parametrize('bad',['missing','duplicate','output','metadata','input','not-accepted','median','harness','core','fingerprint','identity'])
def test_incomplete_or_changed_raw_reports_are_rejected(tmp_path,monkeypatch,bad):
    decision_fixture(tmp_path,monkeypatch,bad)
    with pytest.raises(AssertionError):exec(compile(SCRIPTS[3],'raw-contract','exec'),{})


@pytest.mark.parametrize('scenario',['normal','target-direct','target-normalized','diff-method','blend-method','checker-original','direct-aggregate','normalized-aggregate','expired','checker-cpp-loss','target-original-loss'])
def test_each_frozen_keep_requirement_and_allowed_retained_losses(tmp_path,monkeypatch,scenario):
    decision_fixture(tmp_path,monkeypatch,scenario);exec(compile(SCRIPTS[3],'fixed-decision','exec'),{})
    result=json.loads((tmp_path/'results/comparison.json').read_bytes());keep=scenario in('normal','checker-cpp-loss','target-original-loss')
    assert result['status']==('KEEP'if keep else'DISCARD')
    assert len(result['rows'])==18 and result['raw_samples']==1008
    assert result['groups']['targets_diff_blend']['cases']==12 and result['groups']['checker_math_controls']['cases']==6
    assert all(result['groups'][d]['cases']==6 for d in('uint8','float32','float64','diff','blend','checkerboard'))
    if scenario=='checker-cpp-loss':assert len(result['direct_losses'])==6 and result['checker_original_repeat_wins']==6
    if scenario=='target-original-loss':assert len(result['balanced_original_losses'])==1 and result['target_repeat_wins']==12
    if scenario in('target-direct','target-normalized'):assert result['target_repeat_wins']==11
    if scenario=='checker-original':assert result['checker_original_repeat_wins']==5
    if scenario=='diff-method':assert result['method_original']['diff']<1<result['candidate_original']
    if scenario=='blend-method':assert result['method_original']['blend']<1<result['candidate_original']
    assert 'not untouched code paths'in result['controls_scope']


def test_normalization_AST_is_identical_to_validated_coordinate_workflow():
    prior=(ROOT/'.github/workflows/profile-coordinates-same-host.yml').read_text(encoding='utf-8')
    source=[textwrap.dedent(v)for v in re.findall(r"python - <<'PY'\n(.*?)          PY\n",prior,re.S)][2]
    assert ast.dump(ast.parse(SCRIPTS[2]))==ast.dump(ast.parse(source))
    allowed=next(ast.literal_eval(n.value)for n in PAIR.body if isinstance(n,ast.Assign)and ast.unparse(n.targets[0])=='mutable')
    assert allowed==MUTABLE


@pytest.mark.parametrize('bad',[None,'missing','short','failure','smoke'])
def test_both_preparations_require_compare_pair_and_current_smoke(tmp_path,monkeypatch,bad):
    import os
    import xml.etree.ElementTree as ET
    monkeypatch.setenv('CANDIDATE_REVISION','c'*40)
    families=next(ast.literal_eval(n.iter)for n in PREPARATION.body if isinstance(n,ast.For)and isinstance(n.target,ast.Tuple))
    guards=next(ast.literal_eval(n.iter)for n in PREPARATION.body if isinstance(n,ast.For)and isinstance(n.target,ast.Tuple)and 'polygon-intake'in ast.unparse(n.iter))
    for family,count in families:
        for mode in('normal','scalar'):
            tree=ET.Element('testsuite')
            for i in range(count):ET.SubElement(tree,'testcase',name=str(i))
            if family=='compare-images'and mode=='scalar':
                if bad=='missing':continue
                if bad=='short':tree.remove(tree[-1])
                if bad=='failure':ET.SubElement(tree[0],'failure')
            (tmp_path/(family+'-'+mode+'.xml')).write_bytes(ET.tostring(tree))
    for family,count in guards:
        tree=ET.Element('testsuite')
        for i in range(count):ET.SubElement(tree,'testcase',name=str(i))
        (tmp_path/(family+'.xml')).write_bytes(ET.tostring(tree))
    smoke=dict(status='passed',exit_code=0,outcomes={'passed':1417},results=[{}]*1417,implementations={'native-partial':236,'metadata':3})
    if bad=='smoke':smoke['results'].pop()
    (tmp_path/'feature-smoke.json').write_text(json.dumps(smoke))
    value=manifest(True);value['schema']='ci-offline-measurement-v1';(tmp_path/'measurement.json').write_text(json.dumps(value))
    body=PREPARATION.body
    start=next(i for i,n in enumerate(body)if isinstance(n,ast.Assign)and ast.unparse(n.targets[0])=='manifest')
    stop=body.index(PAIR)
    code=compile(ast.Module(body=body[start:stop],type_ignores=[]),'both-preparations','exec')
    namespace=dict(prepared=tmp_path,role='candidate',json=json,ET=ET,os=os)
    if bad:
        with pytest.raises((AssertionError,FileNotFoundError)):exec(code,namespace)
    else:exec(code,namespace)
