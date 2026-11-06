"""LBP-specific source/authority controls; target and sampler bodies forbidden."""
import builtins,hashlib,json,math,runpy,subprocess,sys,types
from pathlib import Path
import pytest
from test_mark_boundaries_benchmark_execution import synthetic,FAULTS
ROOT=Path(__file__).resolve().parents[1]
CONTEXT=ROOT/'coverage/contracts/draw-multiblock-lbp-benchmark-execution-v2.json'
H=lambda b:hashlib.sha256(b).hexdigest()


def adapter():return runpy.run_path(str(ROOT/'benchmarks/draw_multiblock_lbp_execution.py'))


def test_exact_case_method_source_domains_and_historical_projection():
    a=adapter();c=json.loads(CONTEXT.read_bytes());old=json.loads((ROOT/'coverage/contracts/draw-multiblock-lbp-v1.json').read_bytes())
    m=runpy.run_path(str(ROOT/'benchmarks/draw_multiblock_lbp_native.py'))
    assert c['cases']==m['case_specs']()==old['benchmark_cases']and len(c['cases'])==4
    assert c['method']==json.loads((ROOT/'coverage/contracts/mark-boundaries-benchmark-execution-v3.json').read_bytes())['method']
    assert m['plan']()['root_GO']is None and m['plan']()['timed_batches']==0
    assert all(v is None for v in c['unbound'].values())
    for name,pin in c['source_files_canonical_sha256'].items():assert H((ROOT/name).read_bytes().replace(b'\r\n',b'\n'))==pin,name
    for name,row in c['historical_sources'].items():assert H(a['historical_bytes'](ROOT,name))==row['predecessor_canonical_sha256']
    mark=a['boundary']();mc=json.loads((ROOT/mark.CONTEXT).read_bytes())
    for name,row in mc['historical_446_sources'].items():assert H(mark.historical_bytes(ROOT,name,section='historical_446_sources'))==row['predecessor_canonical_sha256']


@pytest.mark.parametrize('cached',(False,True))
def test_real_cold_cached_UNBOUND_CLI_before_import(cached):
    code="""import builtins,runpy,sys,types
old=builtins.__import__
def refuse(name,*a,**k):
 if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):raise AssertionError('TARGET_IMPORT_REACHED')
 return old(name,*a,**k)
if sys.argv[2]=='True':sys.modules['numpy']=types.ModuleType('numpy')
builtins.__import__=refuse;sys.argv=[sys.argv[1]];runpy.run_path(sys.argv[0],run_name='__main__')
"""
    p=subprocess.run([sys.executable,'-I','-c',code,str(ROOT/'benchmarks/draw_multiblock_lbp_native.py'),str(cached)],capture_output=True,text=True)
    assert p.returncode!=0 and 'LBP_BENCHMARK_UNBOUND'in p.stderr and 'TARGET_IMPORT_REACHED'not in p.stderr and 'Traceback'not in p.stderr


@pytest.mark.parametrize('fault',FAULTS)
def test_actual_LBP_CLI_binding_refusal_zero_bodies(tmp_path,monkeypatch,fault):
    a=adapter();path,b,c=synthetic(tmp_path,monkeypatch,'none'if fault=='source'else fault,CONTEXT)
    if fault=='source':(Path(b['source_root'])/'python/skimagexx/_draw_multiblock_lbp.py').write_text('changed')
    old=builtins.__import__;attempts=[]
    def refuse(name,*args,**kw):
        if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):attempts.append(name);raise RuntimeError('LBP_REQUEST_REFUSED')
        return old(name,*args,**kw)
    if fault=='none-cached':monkeypatch.setitem(sys.modules,'numpy',types.ModuleType('numpy'))
    monkeypatch.setattr(builtins,'__import__',refuse);monkeypatch.setattr(sys,'path',list(sys.path))
    output=Path(b['output_root'])/'never.json'
    monkeypatch.setattr(sys,'argv',['lbp','--root-binding',str(path),'--first-cohort','--case',c['cases'][0]['id'],'--output',str(output)])
    module=runpy.run_path(str(ROOT/'benchmarks/draw_multiblock_lbp_native.py'))
    with pytest.raises(RuntimeError if fault.startswith('none')else AssertionError,match='LBP_REQUEST_REFUSED'if fault.startswith('none')else None):a['execute'](module)
    assert attempts==(['numpy']if fault.startswith('none')else[])and not output.exists()and not output.with_suffix('.batches.jsonl').exists()
    assert (Path(b['output_root'])/'lbp-first-cohort.claim.json').exists()==fault.startswith('none')
    if fault.startswith('none'):
        sys.argv[-1]=str(Path(b['output_root'])/'other.json')
        with pytest.raises(FileExistsError):a['execute'](module)
        assert attempts==['numpy']


@pytest.mark.parametrize('fault',('none','missing','wrong-family','wrong-route','cases','scope','core','cohort-binding','deadline'))
def test_LBP_separate_measurement_authority(tmp_path,monkeypatch,fault):
    a=adapter();path,b,c=synthetic(tmp_path,monkeypatch,context_path=CONTEXT);shared=a['boundary']();pin=H(path.read_bytes())
    done=dict(status='PASS_REAL_CLI_STOP_BEFORE_BODY',route='lbp',source=b['source_revision'],root_binding_sha256=pin,counts=dict(original=1,native=1),sampler_body_calls=0,timed_batches=0,outer_state_restored=True,poststop_source_authority_unchanged=True)
    if fault=='wrong-route':done['route']='mark'
    cohort=tmp_path/'synthetic-cohort.json';cohort.write_text(json.dumps(done))
    m=dict(schema='next-metrics-measurement-v1',status='ROOT_AUTHORIZED_ONE_INITIAL_COMPARISON',prepared_source=b['prepared_source'],producer=b['producer'],core_sha256=b['core_sha256'],runtime_sha256=b['runtime_sha256'],families={c['report_family']:dict(contract=adapter()['CONTEXT'],cases=4,case_ids=[s['id']for s in c['cases']],untimed_calls_per_case=1)},execution_scope=dict(workers=1,repeat=7,target_ms=100,warmup_min_ms=250,single_host_same_installed_runtime=True,no_automatic_retry=True,retain_all_losses=True),completed_first_cohort=dict(path=str(cohort),sha256=H(cohort.read_bytes()),binding_sha256=pin),deadline_utc='2099-01-01T00:00:00+00:00')
    if fault=='wrong-family':m['families']={'mark_boundaries_consumer':m['families'][c['report_family']]}
    elif fault=='cases':m['families'][c['report_family']]['cases']=3
    elif fault=='scope':m['execution_scope']['workers']=4
    elif fault=='core':m['core_sha256']='0'*64
    elif fault=='cohort-binding':m['completed_first_cohort']['binding_sha256']='0'*64
    elif fault=='deadline':m['deadline_utc']='2000-01-01T00:00:00+00:00'
    target=tmp_path/'measurement.json';target.write_text(json.dumps(m))
    if fault=='none':assert shared.validate_measurement(target,b,c,pin,context_path=a['CONTEXT'])[0]==H(target.read_bytes())
    else:
        with pytest.raises((AssertionError,KeyError)):shared.validate_measurement(None if fault=='missing'else target,b,c,pin,context_path=a['CONTEXT'])


def test_LBP_fixture_metadata_and_native_entry_counts_without_arrays():
    a=adapter();c=json.loads(CONTEXT.read_bytes())
    class Array:
        def __init__(self,n,dtype='int64',shape=None,modulus=None,layout='C'):self.n,self.dtype,self.shape,self.modulus,self.layout=n,dtype,shape,modulus,layout
        def __mod__(self,n):return Array(self.n,self.dtype,self.shape,n,self.layout)
        def reshape(self,shape):return Array(self.n,self.dtype,shape,self.modulus,self.layout)
        def astype(self,dtype):return Array(self.n,dtype,self.shape,self.modulus,self.layout)
        def __getitem__(self,key):assert key==(slice(None),slice(None,None,-1));return Array(self.n,self.dtype,self.shape,self.modulus,'reverse')
    fake=types.SimpleNamespace(int64='int64',arange=lambda n,dtype:Array(n),asfortranarray=lambda v:Array(v.n,v.dtype,v.shape,v.modulus,'F'))
    for spec in c['cases']:
        args,options=a['fixture'](spec,fake);image=args[0]
        assert image.n==math.prod(spec['shape'])and image.shape==spec['shape']and image.dtype==spec['dtype']and image.layout==spec['layout']and image.modulus==7
        assert args[1:]==(spec['r'],spec['c'],spec['width'],spec['height'])and options==dict(lbp_code=spec['code'],alpha=.5)
        assert a['native_requirements'](spec)==dict(get_num_threads=0,_lbp_prepare=1,_lbp_binary=24,_lbp_assign=8 if len(spec['shape'])==2 else 9)
    tree=__import__('ast').parse((ROOT/'benchmarks/draw_multiblock_lbp_execution.py').read_text())
    assert not any(isinstance(n,__import__('ast').FunctionDef)and n.name in('measure','validate_binding','sample_entry')for n in __import__('ast').walk(tree))
