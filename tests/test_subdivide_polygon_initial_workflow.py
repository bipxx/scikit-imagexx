"""Inert subdivision transport tests: stdlib values, no image libraries/calls."""
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import runpy
import statistics
import struct
import textwrap
import sys
import types
import xml.etree.ElementTree as ET
import pytest

ROOT=Path(__file__).resolve().parents[1]
WORKFLOW=(ROOT/'.github/workflows/subdivide-polygon-initial.yml').read_text()
SCRIPTS=[textwrap.dedent(s)for s in re.findall(r"python - <<'PY'\n(.*?)          PY\n",WORKFLOW,re.S)]

def pure_helpers():
    tree=ast.parse((ROOT/'benchmarks/subdivide_polygon.py').read_text())
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name in ('integer_masks','points','subdivision_oracle')]
    assert len(nodes)==3
    ns={'math':math}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'frozen-stdlib-helpers','exec'),ns)
    return ns

def validator():
    function=next(n for n in ast.parse(SCRIPTS[2]).body if isinstance(n,ast.FunctionDef))
    ns=dict(math=math,statistics=statistics,hashlib=hashlib,struct=struct)
    exec(compile(ast.Module(body=[function],type_ignores=[]),'actual-validator','exec'),ns)
    return ns['validate_report']

def test_fixed_transport_and_unchanged_admission_predicate():
    assert len(SCRIPTS)==3
    for script in SCRIPTS:compile(script,'workflow-inline','exec')
    rdp=(ROOT/'.github/workflows/approximate-polygon-initial.yml').read_text()
    reference=textwrap.dedent(re.findall(r"python - <<'PY'\n(.*?)          PY\n",rdp,re.S)[0])
    assert ast.dump(ast.parse(SCRIPTS[0]))==ast.dump(ast.parse(reference))
    for name in ('producer_run','producer_revision','artifact_id','artifact_sha256'):
        assert name+':\n        type: string\n        required: true'in WORKFLOW
    assert 'default:'not in WORKFLOW
    timed=WORKFLOW.split('      - name: Measure',1)[1].split('      - name:',1)[0]
    assert timed.count("str(harness),'--source-root',str(source),'--threads','1','--repeat','7'")==1
    assert timed.count('result=delegate(*args,**kwargs)')==1
    assert 'sched_setaffinity'not in timed
    assert timed.count('ci_measurement_artifacts.py verify')==2
    assert not re.search(r'pytest|cmake|ctest|pip install|pip wheel|__module__|setup.py',timed)
    assert '--no-index --no-deps --only-binary=:all: --require-hashes'in WORKFLOW
    assert "layout['product_count'] == 176 and layout['SDK_count'] == 83"in SCRIPTS[1]
    assert 'focused_count == 9346'in SCRIPTS[1]and '== 1447'in SCRIPTS[1]
    assert '== 247'in SCRIPTS[1]and '== 3'in SCRIPTS[1]
    assert '== 29'in SCRIPTS[0]and "step['conclusion'] == 'skipped'"in SCRIPTS[0]
    assert 'if: always()'in WORKFLOW and 'subdivide-polygon-initial-minimum'in WORKFLOW
    for name,pin in [('benchmarks/subdivide_polygon.py','101265e40bdd2c7026cdbb396f9670e2c07d5e32502be55de2411eea623faa38'),('coverage/contracts/subdivide-polygon-benchmark.json','2cdfe4191810ee7ae2de72a11fc921ced6edec72cb5ac282bdc4ae32780dd8da')]:
        assert pin in SCRIPTS[1]
        if name.startswith('coverage/'):
            assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin
    assert hashlib.sha256((ROOT/'benchmarks/subdivide_polygon.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest()=='0bce058ebf921b7f80f3a1dac74632de3e2ab5e7a7ee08369339077cfbb53010'
    assert 'import numpy'not in SCRIPTS[2]and 'import scipy'not in SCRIPTS[2]


def test_factory_and_pascal_scope_without_generating_benchmark_inputs():
    cases=runpy.run_path(str(ROOT/'tests/subdivide_polygon_contract.py'))['cases']()
    frozen=json.loads((ROOT/'coverage/contracts/subdivide-polygon.json').read_bytes())
    assert len(cases)==60 and [x['id']for x in cases]==[x['id']for x in frozen['cases']]
    c=json.loads((ROOT/'coverage/contracts/subdivide-polygon-benchmark.json').read_bytes())
    assert len(c['cases'])==14 and c['raw_samples']==196
    assert {(x['degree'],x['closed'])for x in c['cases']}=={(d,closed)for d in range(1,8)for closed in(False,True)}
    helpers=pure_helpers()
    for degree in range(1,8):assert list(helpers['integer_masks'](degree))==frozen['subdivision_masks'][str(degree)]
    node=next(x for x in ast.parse(SCRIPTS[1]).body if isinstance(x,ast.Assign)and isinstance(x.targets[0],ast.Name)and x.targets[0].id=='expected_files')
    ns={'factory_cases':cases};exec(compile(ast.Module(body=[node],type_ignores=[]),'counts','exec'),ns)
    assert ns['expected_files']=={'test_subdivide_polygon_integration':143,'test_subdivide_polygon_pipeline':136}


def fixture():
    # Tiny synthetic report metadata only; never instantiate the frozen cohort.
    source=dict(source_sha256='fixed',source_hash_schema='schema',source_hash_scope='scope')
    helpers=pure_helpers();masks=json.loads((ROOT/'coverage/contracts/subdivide-polygon.json').read_bytes())['subdivision_masks']
    apis=['measure.subdivide_polygon'];contract=dict(apis=apis,original_python_source_sha256='original',max_points=9,max_backing_elements=36,provider_frontier='owned',timed_boundary='wholecall',cases=[])
    meta=dict(**source,source_before=deepcopy(source),threads=1,repeat=7,target_ms=19,raw_samples=196,reference_pool_limits={k:'1'for k in('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')},fallback_enabled=False,selected_apis=apis,numpy_error_policy=dict(divide='warn',over='warn',under='ignore',invalid='warn'),numpy_error_handler_type='NoneType',numpy_buffer_elements=8192,harness_sha256='0bce058ebf921b7f80f3a1dac74632de3e2ab5e7a7ee08369339077cfbb53010',prepared_harness_sha256='101265e40bdd2c7026cdbb396f9670e2c07d5e32502be55de2411eea623faa38',original_source_sha256='original',core_sha256='core',python='3.11.17',numpy='1.24.4',scipy='1.15.3',reference='0.26.0',provider_frontier='owned wholecall')
    manifest=dict(source=dict(files_sha256={'benchmarks/subdivide_polygon.py':'101265e40bdd2c7026cdbb396f9670e2c07d5e32502be55de2411eea623faa38'}),runtime=dict(python='3.11.17',packages_sha256={'skimagexx':{'_core.cpython-311-x86_64-linux-gnu.so':'core'}}))
    rows=[]
    for degree in range(1,8):
        for closed in(False,True):
            index=len(rows);layout=('C','F','reverse-rows','step-rows','step-columns')[index%5]
            spec=dict(case_id=str(index),api=apis[0],degree=degree,closed=closed,preserve_ends=not closed and degree%2==0,base_points=8,points=8+int(closed),layout=layout)
            values=helpers['points'](spec);oracle=helpers['subdivision_oracle'](values,degree,spec['preserve_ends']);shape=[len(oracle),2];spec['expected_output_shape']=shape
            contract['cases'].append(deepcopy(spec));count=spec['points']
            strides,own,c,f,base={'C':([16,8],True,True,False,'NoneType'),'F':([8,count*8],True,False,True,'NoneType'),'reverse-rows':([-16,8],False,False,False,'ndarray'),'step-rows':([32,8],False,False,False,'ndarray'),'step-columns':([32,16],False,False,False,'ndarray')}[layout]
            input_metadata=dict(dtype='<f8',dtype_metadata=None,shape=[count,2],strides=strides,own=own,writeable=True,c=c,f=f,base_type=base)
            output_metadata=dict(dtype='<f8',dtype_metadata=None,shape=shape,strides=[16,8],own=True,writeable=True,c=True,f=False,base_type='NoneType')
            words=lambda pts:hashlib.sha256(b''.join(struct.pack('<dd',*pair)for pair in pts)).hexdigest()
            current=2. if index==0 else .5
            rows.append(dict(**spec,backend='native',threads=1,native_accepted=True,actual_C_returns=['subdivide_polygon_apply'],native_entry_module='skimagexx._core',max_abs_error=0.,output_sha256=words(oracle),reference_sha256=words(oracle),oracle_sha256=words(oracle),input_sha256=words(values),input_metadata=input_metadata,output_metadata=output_metadata,samples_ms=[[1.]*7,[current]*7],loops=[1,1],reference_ms=1.,cpp_ms=current,speedup=1/current))
    return dict(metadata=meta,results=rows),contract,manifest,source,helpers,masks


@pytest.mark.parametrize('fault',[None,'source','threads','pool-key','pool-value','python','numpy','frontier','handler','policy','buffer','harness','prepared-harness','core','sample-count','nan','loops','median','ratio','degree','closed','preserve','points','input-hash','oracle','output-shape','output-layout','output-own','input-stride','entry','module','accepted','backend','mask','missing-case'])
def test_actual_validator_rejects_mutations_and_retains_loss(fault):
    r,c,m,s,p,masks=fixture();row=r['results'][0]
    if fault=='source':r['metadata']['source_before']={}
    elif fault=='threads':row['threads']=4
    elif fault=='pool-key':r['metadata']['reference_pool_limits']['foreign']='1'
    elif fault=='pool-value':r['metadata']['reference_pool_limits']['OMP_NUM_THREADS']='4'
    elif fault=='python':r['metadata']['python']='changed'
    elif fault=='numpy':r['metadata']['numpy']='changed'
    elif fault=='frontier':r['metadata']['provider_frontier']='changed'
    elif fault=='handler':r['metadata']['numpy_error_handler_type']='callable'
    elif fault=='policy':r['metadata']['numpy_error_policy']['invalid']='changed'
    elif fault=='buffer':r['metadata']['numpy_buffer_elements']=0
    elif fault=='harness':r['metadata']['harness_sha256']='changed'
    elif fault=='prepared-harness':r['metadata']['prepared_harness_sha256']='changed'
    elif fault=='core':r['metadata']['core_sha256']='changed'
    elif fault=='sample-count':row['samples_ms'][0].pop()
    elif fault=='nan':row['samples_ms'][0][0]=float('nan')
    elif fault=='loops':row['loops'][0]=0
    elif fault=='median':row['cpp_ms']=3.
    elif fault=='ratio':row['speedup']=3.
    elif fault=='degree':row['degree']=8
    elif fault=='closed':row['closed']=True
    elif fault=='preserve':row['preserve_ends']=True
    elif fault=='points':row['points']=2
    elif fault=='input-hash':row['input_sha256']='changed'
    elif fault=='oracle':row['oracle_sha256']='changed'
    elif fault=='output-shape':row['output_metadata']['shape']=[1,2]
    elif fault=='output-layout':row['output_metadata']['f']=True
    elif fault=='output-own':row['output_metadata']['own']=False
    elif fault=='input-stride':row['input_metadata']['strides'][0]+=1
    elif fault=='entry':row['actual_C_returns']=[]
    elif fault=='module':row['native_entry_module']='fake'
    elif fault=='accepted':row['native_accepted']=False
    elif fault=='backend':row['backend']='fallback'
    elif fault=='mask':masks['1'][0][0]=2
    elif fault=='missing-case':r['results'].pop()
    if fault:
        with pytest.raises(AssertionError):validator()(r,c,m,s,p,masks)
    else:
        out=validator()(r,c,m,s,p,masks)
        assert out['raw_samples']==196 and out['losses']==['0']and len(out['wins'])==13


@pytest.mark.parametrize('fault',[None,'pipeline-missing','substituted','skipped','failed'])
def test_actual_279_node_gate_requires_both_modules(tmp_path,fault):
    expected={'test_subdivide_polygon_integration':143,'test_subdivide_polygon_pipeline':136}
    for mode in('normal','scalar'):
        tree=ET.Element('testsuite')
        for module,count in expected.items():
            for i in range(count):
                if fault=='pipeline-missing'and module.endswith('pipeline')and i==0:continue
                label='test_subdivide_polygon_integration'if fault=='substituted'and module.endswith('pipeline')and i==0 else module
                node=ET.SubElement(tree,'testcase',classname='tests.'+label,name=str(i))
                if i==0 and fault in('skipped','failed'):ET.SubElement(node,'skipped'if fault=='skipped'else'failure')
        ET.ElementTree(tree).write(tmp_path/('subdivide-polygon-'+mode+'.xml'))
    tree=ast.parse(SCRIPTS[1]);declaration=next(x for x in tree.body if isinstance(x,ast.Assign)and isinstance(x.targets[0],ast.Name)and x.targets[0].id=='expected_files')
    loop=next(x for x in tree.body if isinstance(x,ast.For)and isinstance(x.iter,ast.Tuple)and 'subdivide-polygon-normal.xml'in ast.unparse(x.iter))
    code=compile(ast.Module(body=[declaration,loop],type_ignores=[]),'actual-parity-xml','exec');ns=dict(factory_cases=[None]*60,ET=ET,Counter=Counter,prepared=tmp_path)
    if fault:
        with pytest.raises(AssertionError):exec(code,ns)
    else:exec(code,ns)


def launcher_source():
    start=WORKFLOW.index('          cat > "$output/launch.py" <<\'PY\'\n')
    lines=WORKFLOW[start:].splitlines()[1:]
    end=lines.index('          PY')
    return '\n'.join(line[10:]for line in lines[:end])+'\n'


@pytest.mark.parametrize('fail',[False,True])
def test_delegate_once_preserves_partial_returned_samples(tmp_path,monkeypatch,fail):
    source=launcher_source();ast.parse(source)
    product=tmp_path/'product';(product/'benchmarks').mkdir(parents=True)
    (product/'build/prepared/ci-artifacts').mkdir(parents=True)
    (product/'coverage/contracts').mkdir(parents=True)
    runfile=product/'benchmarks/run.py';runfile.write_bytes(b'# inert measure module\n')
    external=tmp_path/'external.py';external.write_bytes(b'# inert external harness\n')
    contract=dict(cases=[dict(case_id=str(i))for i in range(14)])
    (product/'coverage/contracts/subdivide-polygon-benchmark.json').write_text(json.dumps(contract))
    manifest=dict(source=dict(revision='a'*40,files_sha256={'benchmarks/run.py':hashlib.sha256(runfile.read_bytes()).hexdigest(),'benchmarks/subdivide_polygon.py':'prepared'}))
    (product/'build/prepared/ci-artifacts/measurement.json').write_text(json.dumps(manifest))
    output=tmp_path/'results';output.mkdir()
    module=types.ModuleType('run');module.__file__=str(runfile);calls=[]
    timing=dict(reference_ms=1.,cpp_ms=2.,speedup=.5,samples_ms=[[1.]*7,[2.]*7],loops=[3,4],sample_labels=['scikit-image','current'])
    reference,native=object(),object()
    def measure(*args,**kwargs):
        calls.append((args,kwargs))
        if fail and len(calls)==2:raise ValueError('forced second measure failure')
        return timing
    module.measure=measure
    def harness(path,run_name):
        assert run_name=='__main__'and (output/'process-before.json').is_file()
        for _ in range(14):assert module.measure(reference,native,7,19)is timing
        (output/'subdivide-polygon.json').write_text('{}')
    monkeypatch.setattr(sys,'argv',['launch.py',str(product),str(output),str(external)])
    monkeypatch.setattr(sys,'path',sys.path.copy())
    monkeypatch.setitem(sys.modules,'run',module)
    monkeypatch.setattr(runpy,'run_path',harness)
    if fail:
        with pytest.raises(ValueError,match='forced second measure failure'):exec(compile(source,'actual-launcher','exec'),{})
    else:exec(compile(source,'actual-launcher','exec'),{})
    final=json.loads((output/'process-after.json').read_bytes())
    journal=[json.loads(line)for line in(output/'returned-measures.jsonl').read_text().splitlines()]
    assert final['status']==('FAILED'if fail else'COMPLETE')
    assert final['completed_measures']==len(journal)==(1 if fail else 14)
    assert len(calls)==(2 if fail else 14)
    assert all(args==(reference,native,7,19)and not kwargs for args,kwargs in calls)
    assert all(row['timing']==timing and row['status']=='PENDING_FINAL_HARNESS_VALIDATION'for row in journal)
    assert [row['case_id']for row in journal]==[str(i)for i in range(len(journal))]
    if fail:assert final['failure']['type']=='ValueError'and 'forced second measure failure'in final['failure']['traceback']
    else:assert final['failure']is None and final['report_exists']is True
    assert "item['timing']"in SCRIPTS[2]and "len(journal) == len(report['results']) == 14"in SCRIPTS[2]
