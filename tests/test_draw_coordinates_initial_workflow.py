"""Inert checks of the actual artifact-only drawing workflow and raw validator."""
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
WORKFLOW=(ROOT/'.github/workflows/draw-coordinates-initial.yml').read_text()
SCRIPTS=re.findall(r"          python - <<'PY'\n(.*?)          PY",WORKFLOW,re.S)
SCRIPTS=['\n'.join(line[10:] for line in s.splitlines()) for s in SCRIPTS]


def test_recipe_is_admitted_offline_pinned_and_has_no_build_or_tests():
    assert len(SCRIPTS)==3
    for script in SCRIPTS:ast.parse(script)
    assert "run='40816710252'" in SCRIPTS[0] and "core_sha256='9d46467577182c0481a86f3996e1e7d218a3da4a8f6f21a39d0c862736c83706'" in SCRIPTS[0]
    assert "assert (run, revision, artifact, digest) ==" in SCRIPTS[0]
    assert 'execution_created_utc:' in WORKFLOW and 'deadline_utc:' in WORKFLOW
    assert '(deadline-created).total_seconds() <= 7200' in SCRIPTS[0]
    assert 'pip install --no-index --no-deps --only-binary=:all: --require-hashes' in WORKFLOW
    assert not any(text in WORKFLOW for text in ('python -m pytest','cmake --build','pip install numpy','pip install scipy'))
    launcher=re.search(r"          taskset -c 1 .*?<<'PY'.*?\n(.*?)          PY",WORKFLOW,re.S).group(1)
    ast.parse('\n'.join(line[10:] for line in launcher.splitlines()))
    assert 'taskset -c 1' in WORKFLOW and "--deadline-utc', os.environ['DEADLINE_UTC']" in WORKFLOW
    assert 'name: draw-coordinates-initial-minimum' in WORKFLOW and 'if: always()' in WORKFLOW
    assert "layout['product_count'] == 178 and layout['SDK_count'] == 84" in SCRIPTS[1]
    assert "['CURRENT_FAMILIES']) == 27" in SCRIPTS[1]
    assert "'test_draw_coordinates_public': 271" in SCRIPTS[1] and "'test_draw_coordinates_integration': 181" in SCRIPTS[1]
    assert '== 452' in SCRIPTS[1] and 'dict(actual_files) == expected_files' in SCRIPTS[1]
    assert 'focused_count == 10250' in SCRIPTS[1]
    assert 'd52f16ab3b6c67424de930597daa3c17eb9bca60a624aa3e61699afcd45a9a2d' in SCRIPTS[1]
    assert "smoke['outcomes']['passed'] == len(smoke['results']) == 1463" in SCRIPTS[1]
    assert 'draw-coordinates.batches.jsonl' in SCRIPTS[2]
    assert 'da27606ddd8e67a7631b172e5f98ff5ac2305afab5ec212c363acf081567b6fc' in SCRIPTS[1]


@pytest.mark.parametrize('include_path', (False, True), ids=('old-context-fails', 'fixed-context-imports'))
def test_real_isolated_runpy_context_stops_after_provenance_before_numpy(tmp_path, include_path):
    # These are the actual three source566 files, not fake scientific modules.
    pins = {
        'benchmarks/draw_coordinates.py': 'da27606ddd8e67a7631b172e5f98ff5ac2305afab5ec212c363acf081567b6fc',
        'benchmarks/provenance.py': 'e71b277a7d43d286d56ac442c900d9a5ae673ecf4e9a81a4e080df6b6e4b883d',
        'tools/montage_sustained.py': '29e5fec91dcc4b98f74b6de573ea943ef40f08c7f0187d133c06ad960ff3fd3d'}
    snapshot = tmp_path/'source566'
    for name, digest in pins.items():
        data = (ROOT/name).read_bytes().replace(b'\r\n', b'\n')
        assert hashlib.sha256(data).hexdigest() == digest
        target = snapshot/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    launcher = re.search(r"          taskset -c 1 .*?<<'PY'.*?\n(.*?)          PY", WORKFLOW, re.S).group(1)
    launcher = '\n'.join(line[10:] for line in launcher.splitlines())
    tree = ast.parse(launcher)
    path_node = next(n for n in ast.walk(tree) if isinstance(n, ast.Expr)
                     and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'sys.path.insert')
    run_node = next(n for n in ast.walk(tree) if isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'runpy.run_path')
    statements = [ast.get_source_segment(launcher, path_node)] if include_path else []
    statements.append(ast.get_source_segment(launcher, run_node))
    actual_snippet = '\n'.join(statements)
    child = tmp_path/'isolated_context.py'
    child.write_text('''import ast, builtins, hashlib, json, runpy, sys
from pathlib import Path
driver = Path('benchmarks/draw_coordinates.py').resolve()
tree = ast.parse(driver.read_bytes())
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
numpy_line = next(n.lineno for n in main.body if isinstance(n, ast.Import) and any(a.name == 'numpy' for a in n.names))
for name in ('numpy', 'scipy', 'skimage', 'skimagexx'): assert name not in sys.modules
scientific_imports = []
original_import = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] in ('numpy', 'scipy', 'skimage', 'skimagexx'):
        scientific_imports.append(name)
        raise AssertionError('Scientific import executed before intended stop')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guard
class ImportStop(Exception): pass
proof = {}
def trace(frame, event, arg):
    if event == 'line' and frame.f_code.co_name == 'main' and frame.f_lineno == numpy_line and Path(frame.f_code.co_filename).resolve() == driver:
        helper = sys.modules['provenance']
        helper_path = Path(helper.__file__).resolve()
        assert helper_path == Path('benchmarks/provenance.py').resolve()
        assert frame.f_locals['source_fingerprint'] is helper.source_fingerprint
        proof.update(helper_path=str(helper_path), helper_sha256=hashlib.sha256(helper_path.read_bytes()).hexdigest(), stopped_before_numpy=True)
        raise ImportStop('After real provenance import, before NumPy import')
    return trace
sys.argv = [str(driver), '--deadline-utc', '2099-01-01T00:00:00+00:00', '--output', 'never-created.json']
sys.settrace(trace)
try:
    exec(ACTUAL_SNIPPET)
except ImportStop:
    outcome = 'fixed-context-pass'
except ModuleNotFoundError as error:
    assert error.name == 'provenance'
    outcome = 'old-context-missing-provenance'
finally:
    sys.settrace(None)
assert not scientific_imports and not Path('never-created.json').exists() and not Path('never-created.batches.jsonl').exists()
print(json.dumps(dict(outcome=outcome, proof=proof, scientific_imports=scientific_imports, image_calls=0, timed_batches=0)))
'''.replace('ACTUAL_SNIPPET', repr(actual_snippet)), encoding='utf-8')
    result = subprocess.run([sys.executable, '-I', '-B', str(child)], cwd=snapshot,
                            capture_output=True, text=True, check=True)
    (tmp_path/'context-result.json').write_text(result.stdout, encoding='utf-8')
    (tmp_path/'context-stderr.txt').write_text(result.stderr, encoding='utf-8')
    proof = json.loads(result.stdout)
    assert proof['outcome'] == ('fixed-context-pass' if include_path else 'old-context-missing-provenance')
    assert proof['image_calls'] == proof['timed_batches'] == 0 and proof['scientific_imports'] == []
    if include_path:
        assert proof['proof']['helper_sha256'] == pins['benchmarks/provenance.py']
        assert proof['proof']['stopped_before_numpy'] is True


@pytest.mark.parametrize('fault',(None,'unsealed','foreign-source','foreign-artifact','timing-ran','missing-job'))
def test_actual_producer_and_clock_authentication(tmp_path,monkeypatch,fault):
    monkeypatch.chdir(tmp_path);Path('results').mkdir()
    env=dict(PRODUCER_RUN='123',PRODUCER_REVISION='a'*40,ARTIFACT_ID='456',ARTIFACT_SHA256='b'*64,GITHUB_REPOSITORY='bipxx/scikit-imagexx',EXECUTION_CREATED_UTC='2026-10-29T00:00:00+00:00',DEADLINE_UTC='2026-10-29T02:00:00+00:00')
    for k,v in env.items():monkeypatch.setenv(k,v)
    record=dict(id=123,head_sha='a'*40,head_repository=dict(full_name=env['GITHUB_REPOSITORY']),path='.github/workflows/ci.yml',status='completed',conclusion='success')
    item=dict(id=456,name='ci-performance-minimum',digest='sha256:'+'b'*64,expired=False,size_in_bytes=42,workflow_run=dict(id=123,head_sha='a'*40))
    jobs=[dict(name='numpy-minimum',conclusion='success',steps=[]),dict(name='measure-minimum',conclusion='success',steps=[dict(name='Run python benchmarks/'+str(i),conclusion='skipped')for i in range(29)])]
    if fault=='foreign-source':record['head_sha']='c'*40
    elif fault=='foreign-artifact':item['workflow_run']['id']=124
    elif fault=='timing-ran':jobs[1]['steps'][0]['conclusion']='success'
    elif fault=='missing-job':jobs.pop()
    replies=[record,dict(total_count=1,artifacts=[item]),dict(total_count=len(jobs),jobs=jobs)]
    monkeypatch.setattr(subprocess,'check_output',lambda args:json.dumps(replies.pop(0)).encode())
    script=SCRIPTS[0]
    for a,b in [('40816710252', '123'), ('566c1fcdc50dc6cdc0bb2c7a9de0a0e56449b6f8', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'), ('12873106882', '456'), ('b5c1b0d20ee9ed5558ef3ed2a9089f2899c4d6265a5f10553726e61a6e39dc91', 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'), ('9d46467577182c0481a86f3996e1e7d218a3da4a8f6f21a39d0c862736c83706', 'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')]:script=script.replace(a,b)
    if fault=='unsealed':script=script.replace("run='123'","run='TBD_RUN'")
    # Replace only the imported wall clock with a deterministic pre-deadline clock.
    script=script.replace('from datetime import datetime, timezone','from datetime import datetime as real_datetime, timezone\nclass datetime(real_datetime):\n    @classmethod\n    def now(cls, tz): return cls(2026,10,29,1,tzinfo=tz)')
    if fault:
        with pytest.raises(AssertionError):exec(script,{})
    else:
        exec(script,{})
        assert json.loads(Path('results/selection.json').read_bytes())==item


def validator():
    tree=ast.parse(SCRIPTS[2]);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
    from datetime import datetime
    ns=dict(datetime=datetime,math=math,statistics=statistics,re=re)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-draw-validator','exec'),ns)
    return ns['validate_report']


def fixture():
    source=dict(source_sha256='source',source_hash_schema='schema',source_hash_scope='scope')
    clock=dict(created_utc='2026-10-29T00:00:00+00:00',deadline_utc='2026-10-29T02:00:00+00:00')
    apis=['draw.line','draw.rectangle','draw.disk','draw.ellipse'];entries=dict(zip(apis,['draw_line','draw_rectangle','draw_ellipse','draw_ellipse']))
    contract=dict(cases=[],native_entries=entries,method_sha256='method',dtype_scope='geometry',max_output_coordinate_elements=1000000)
    reference=dict(apis=apis,source_canonical_lf_sha256='original',binary_sha256={'1.24.4':'binary'},benchmark_cases=contract['cases'])
    manifest=dict(source=dict(files_sha256={'benchmarks/draw_coordinates.py':'harness'}),runtime=dict(packages_sha256={'skimagexx':{'_core.fixture.so':'core'}},versions={'numpy':'1.24.4','scipy':'1.15.3','scikit-image':'0.26.0'}))
    meta=dict(**source,source_before=deepcopy(source),threads=1,repeat=7,target_ms=100,raw_samples=168,warmup_calibration_min_ms=250,interbatch_sleep_ms=0,pass_index=1,deadline_utc=clock['deadline_utc'],timestamp_utc='2026-10-29T01:00:00+00:00',reference_pool_limits={'OMP':'1'},fallback_enabled=False,selected_apis=apis,dtype_scope='geometry',numpy_error_policy=dict(divide='warn',over='warn',under='ignore',invalid='warn'),numpy_buffer_elements=8192,harness_sha256='harness',method_sha256='method',original_source_sha256='original',original_binary_sha256='binary',numpy='1.24.4',scipy='1.15.3',reference='0.26.0',core_sha256='core')
    rows=[];journal=[]
    for index in range(12):
        api=apis[index%4];rect=api=='draw.rectangle';line=api=='draw.line'
        args=([0,0,7,3]if line else[[0,0]]if rect else[[4,5],3]if api=='draw.disk'else[4,5,3,2])
        spec=dict(id=str(index),api=api,args=args,kwargs={'extent':[3,4]}if rect else{});contract['cases'].append(spec)
        shape=[4,3]if rect else[8]if line else[3]
        desc=dict(dtype='<i8',dtype_metadata=None,shape=shape,strides=[24,8]if rect else[8]if line else[16],own=rect,writeable=True,aligned=True,c=rect or line,f=line,base_type='NoneType'if rect else'memoryview'if line else'ndarray',base_shape=None if rect else shape if line else[3,2],base_strides=None if rect else[8]if line else[16,8],sha256='a'*64)
        output=dict(container='list'if rect else'tuple',arrays=[deepcopy(desc),deepcopy(desc)],shared_bases=[[a==b or not line for b in range(2)]for a in range(2)],output_aliases=[[a==b for b in range(2)]for a in range(2)],reference_sha256=['a'*64]*2,oracle_sha256=['a'*64]*2)
        afters=[];warm=[250000000]*2;orders=[[0,1]if(i+index+1)%2==0 else[1,0]for i in range(7)];samples=[[],[]];loops=[[],[]]
        def batch(function,round_id,ordinal,phase,elapsed,n):
            common=dict(case_index=index,case_id=str(index),function=function,round=round_id,ordinal=ordinal,loops=n)
            before=dict(**common,event='batch_before',requested_phase=phase)
            after=dict(**common,event='batch_after',phase=phase,elapsed_ns=elapsed,complete=True,accepted=phase=='sample',failure=None)
            journal.extend([before,after]);afters.append(after)
        for f in(0,1):batch(f,-1,f,'warmup',250000000,1)
        for i,order in enumerate(orders):
            for j,f in enumerate(order):
                elapsed=100000000 if f==0 else 200000000 if index==0 else 100000000
                n=100 if f==0 or index==0 else 200
                batch(f,i,j,'sample',elapsed,n);samples[f].append(elapsed/n/1e6);loops[f].append(n)
        a,b=map(statistics.median,samples)
        timing=dict(reference_ms=a,cpp_ms=b,speedup=a/b,samples_ms=samples,loops=loops,sample_labels=['scikit-image','current'],warmup_elapsed_ns=warm,sample_order=orders,batch_records=afters)
        journal.append(dict(event='measure_returned',case_index=index,case_id=str(index),status='PENDING_FINAL_HARNESS_VALIDATION',timing=deepcopy(timing)))
        rows.append(dict(case_id=str(index),api=api,parameters=dict(args=args,kwargs=spec['kwargs']),threads=1,native_accepted=True,backend='native',actual_C_returns=[entries[api]],native_entry_module='skimagexx._core',max_abs_error=0,warnings=[],output=output,**timing))
    return dict(metadata=meta,results=rows),contract,reference,manifest,source,journal,clock


@pytest.mark.parametrize('fault',(None,'source','core','entry','parameters','container','strides','base','owner','hash','samples','median','ratio','underlength','warmup','journal','order','deadline'))
def test_actual_validator_rejects_changed_evidence_and_keeps_losses(fault):
    args=fixture();r,c,ref,m,s,j,clock=args;row=r['results'][0]
    if fault=='source':r['metadata']['source_before']={}
    elif fault=='core':r['metadata']['core_sha256']='foreign'
    elif fault=='entry':row['actual_C_returns']=[]
    elif fault=='parameters':row['parameters']=dict(args=[],kwargs={})
    elif fault=='container':row['output']['container']='list'
    elif fault=='strides':row['output']['arrays'][0]['strides']=[16]
    elif fault=='base':row['output']['arrays'][0]['base_type']='ndarray'
    elif fault=='owner':row['native_entry_module']='fake'
    elif fault=='hash':row['output']['reference_sha256'][0]='b'*64
    elif fault=='samples':row['samples_ms'][0][0]=float('nan')
    elif fault=='median':row['cpp_ms']=1
    elif fault=='ratio':row['speedup']=9
    elif fault=='underlength':j[5]['elapsed_ns']=99999999
    elif fault=='warmup':row['warmup_elapsed_ns'][0]=249999999
    elif fault=='journal':j.pop()
    elif fault=='order':row['sample_order'][0].reverse()
    elif fault=='deadline':r['metadata']['deadline_utc']='2026-10-29T03:00:00+00:00'
    if fault:
        with pytest.raises(AssertionError):validator()(*args)
    else:
        summary=validator()(*args)
        assert summary['raw_samples']==168 and summary['losses']==['0'] and len(summary['wins'])==11
        assert summary['geometric_mean']>1
