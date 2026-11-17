"""Complete inert/source controls for separate legacy and fresh public VI large scopes."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
LEGACY='tests/test_variation_of_information_large_saved.py'
CURRENT='tests/test_vi_public_log2_provider_large.py'
BODY_SHA256='04254552494d6ba8d02c663595235b1477ea2a50954251deaba3b6112d19f4d4'
FRONTEND32E_SHA256='f42ca166318d77900e4df9ee00a72a5ddec7309a3098fb83287a7ecf621ca5fa'
GOLDS_SHA256='0c59120f511ef5b2368c01b3b6d010e45bc233deacd52f5c630d568141ec2f55'


def node(text,name):return next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name==name)


def test_every_old_body_assertion_and_fixed_oracle_remains_legacy_only():
    text=(ROOT/LEGACY).read_text();f=node(text,'test_admitted_large_legacy_scalar_words_and_state')
    body=''.join(text.splitlines(keepends=True)[f.body[0].lineno-1:f.end_lineno])
    assert hashlib.sha256(body.encode()).hexdigest()==BODY_SHA256
    assert 'Legacy scalar SDK'in ast.get_docstring(ast.parse(text))
    assert 'provider_predecessor_source'in text and FRONTEND32E_SHA256 in text
    assert "reference['target_immediate']"in body and "observation(result,np)==reference['result']"in body
    decorators=[ast.unparse(n)for n in f.decorator_list]
    assert "pytest.mark.parametrize('index', range(4))"in decorators and "pytest.mark.parametrize('workers', (1, 4))"in decorators
    assert "from test_variation_of_information_public import target as public_target"in text
    assert 'module.variation_of_information='not in text and 'pytest.skip'not in text
    assert hashlib.sha256((ROOT/'tests/data/variation_of_information_large_saved_v1.json').read_bytes()).hexdigest()==GOLDS_SHA256


def legacy_gateway(monkeypatch,refused=False,tampered=False):
    from variation_of_information_source_context import provider_predecessor_source
    text=(ROOT/LEGACY).read_text();function=node(text,'legacy_scalar_gateway');calls=[]
    class Array:
        flags=SimpleNamespace(aligned=True)
    class CSR:
        shape=(2,3);dtype=SimpleNamespace(kind='f',itemsize=8,isnative=True)
        data=Array();indices=Array();indptr=Array()
    result=object()
    class Core:
        def label_pair_check_storage(self,*a):pass
        def get_num_threads(self):return 1
        def variation_of_information_csr(self,*args,log2_provider=None):
            calls.append((args,log2_provider));return None if refused else result
    core=Core();sparse=SimpleNamespace(csr_array=CSR,csr_matrix=type('Matrix',(),{}))
    ns=dict(ast=ast,hashlib=hashlib,ROOT=ROOT,LEGACY_FRONTEND_SHA256=FRONTEND32E_SHA256,
        np=SimpleNamespace(ndarray=Array,float64='f64',int64='i64',ascontiguousarray=lambda v,dtype:v),
        importlib=SimpleNamespace(import_module=lambda name:sparse if name=='scipy.sparse'else None),SimpleNamespace=SimpleNamespace,
        provider_predecessor_source=lambda *a:provider_predecessor_source(*a)+(b' 'if tampered else b''))
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<actual legacy gateway>','exec'),ns)
    model=SimpleNamespace(contingency_table=lambda *a,**k:pytest.fail('Supplied CSR touched prefix provider'))
    gateway=ns['legacy_scalar_gateway'](model,core);image=SimpleNamespace(shape=(2,3))
    return gateway,image,CSR(),calls,result


def test_actual_hash_bound_gateway_selects_omitted_None_scalar_ABI(monkeypatch):
    gateway,image,table,calls,result=legacy_gateway(monkeypatch)
    assert gateway.variation_of_information(image,image,table=table)is result
    assert len(calls)==1 and len(calls[0][0])==6 and calls[0][1]is None


def test_legacy_gateway_refusal_propagates_without_fallback(monkeypatch):
    gateway,image,table,calls,_=legacy_gateway(monkeypatch,refused=True)
    with pytest.raises(NotImplementedError,match='no algorithm fallback'):gateway.variation_of_information(image,image,table=table)
    assert len(calls)==1


def test_legacy_gateway_refuses_unreviewed_predecessor_bytes(monkeypatch):
    with pytest.raises(AssertionError):legacy_gateway(monkeypatch,tampered=True)


def load_current():
    spec=importlib.util.spec_from_file_location('_vi_large_inert_current',ROOT/CURRENT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def assert_current_source(text):
    f=node(text,'test_current_public_large_original_pair')
    decorators=[ast.unparse(n)for n in f.decorator_list]
    assert "pytest.mark.parametrize('index', range(4))"in decorators and "pytest.mark.parametrize('workers', (1, 4))"in decorators
    assert text.count('restore(natural)')==2 and 'natural=env()'in text
    assert not any(s in text for s in('target_immediate','SAVED','allclose','isclose','pytest.skip','NPY_DISABLE_CPU_FEATURES','SKIMAGEXX_DISABLE_SIMD'))
    for required in("phase[0]='reference'","phase[0]='native'",'reference_runtime()',"pipeline(skimage)","pipeline(xx)","counts==dict(original=1,native=1)","len(returned)==11","sum(requirements.values())==11","len(batches)==4",'isinstance(np.log2,np.ufunc)','isinstance(np.matmul,np.ufunc)',"VI_PUBLIC_LOG2_LARGE_RECORD_DIR",'expected.tobytes()==actual.tobytes()',"observation(actual,np)==expected_metadata",'restore(caller)',"['numpy.log2','numpy.matmul','numpy.log2','numpy.matmul']",'np.matmul is true_matmul','benchmarks/vi_public_matmul_provider_execution.py'):
        assert required in text,required
    # Both checkpoints are required. A second occurrence of one short token
    # must not conceal removal of the post-target or final identity assertion.
    assert 'assert len(batches)==4 and np.log2 is true_provider and np.matmul is true_matmul'in text
    assert 'assert xx.metrics.variation_of_information is native and skimage.metrics.variation_of_information is reference and np.log2 is true_provider and np.matmul is true_matmul'in text
    assert text.index("record['expected']=encode(expected,output=True)")<text.index('expected_metadata=observation')
    assert text.index("record['actual']=encode(actual,output=True)")<text.index('assert observation(actual,np)')
    call=node(text,'call_once');body=ast.get_source_segment(text,call)
    assert len([n for n in ast.walk(call)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='call'])==1
    assert body.index('outputs[role]=value')<body.index('try:after=env()')
    assert 'if primary is None:raise'in body and 'primary=error;raise'in body and 'sys.setprofile(prior)'in body


def test_current_source_is_exact_eight_fresh_natural_public_pairs():
    assert_current_source((ROOT/CURRENT).read_text())


@pytest.mark.parametrize('old,new',(('restore(natural)','restore(caller)'),('len(batches)==4','len(batches)==0'),('len(returned)==11','len(returned)>=1'),("phase[0]='reference'","phase[0]='native'"),('expected.tobytes()==actual.tobytes()','True'),('np.matmul is true_matmul','True')))
def test_current_source_guard_refuses_weakened_or_seeded_protocol(old,new):
    text=(ROOT/CURRENT).read_text();assert old in text
    with pytest.raises(AssertionError):assert_current_source(text.replace(old,new,1))


def test_call_once_refusal_preserves_primary_and_profile_without_retry():
    module=load_current();primary=NotImplementedError('actual domain refusal');calls=[];prior=sys.getprofile();outputs={};notes=[]
    def refused():calls.append('target');raise primary
    with pytest.raises(NotImplementedError)as error:module.call_once(refused,(),{},lambda:b'caller',None,outputs,'actual',notes)
    assert error.value is primary and calls==['target']and sys.getprofile()is prior
    assert outputs=={}and notes==[]


def test_return_is_saved_before_post_target_snapshot_failure(tmp_path):
    module=load_current();value=object();outputs={};notes=[];events=[];primary=RuntimeError('snapshot-after-target');prior=sys.getprofile()
    def env():
        events.append('snapshot')
        if events.count('snapshot')==2:raise primary
        return b'natural'
    def target():events.append('target');return value
    with pytest.raises(RuntimeError)as error:module.call_once(target,(),{},env,None,outputs,'actual',notes)
    assert error.value is primary and outputs['actual']is value and events==['snapshot','target','snapshot']and sys.getprofile()is prior
    record={'diagnostics':notes};path=tmp_path/'record.json';module.retain_record(path,record,outputs,lambda *a,**k:{'raw_hex':'retained'})
    assert json.loads(path.read_bytes())['actual']['raw_hex']=='retained'and notes[0]['stage']=='post-target-FP'


def test_target_primary_survives_a_secondary_post_snapshot_failure():
    module=load_current();primary=NotImplementedError('target-primary');secondary=RuntimeError('snapshot-secondary');outputs={};notes=[];events=[];prior=sys.getprofile()
    def env():
        events.append('snapshot')
        if events.count('snapshot')==2:raise secondary
        return b'natural'
    def target():events.append('target');raise primary
    with pytest.raises(NotImplementedError)as error:module.call_once(target,(),{},env,None,outputs,'actual',notes)
    assert error.value is primary and outputs=={}and events==['snapshot','target','snapshot']and sys.getprofile()is prior
    assert notes==[dict(role='actual',stage='post-target-FP',type='RuntimeError',error='snapshot-secondary')]


def test_provider_context_exit_failure_keeps_both_already_returned_outputs(tmp_path):
    module=load_current();expected=object();actual=object();outputs={'expected':expected};notes=[];primary=AssertionError('provider-context-exit');events=[]
    class Providers:
        def __enter__(self):return []
        def __exit__(self,*args):raise primary
    def target():events.append('target');return actual
    def env():events.append('snapshot');return b'natural'
    with pytest.raises(AssertionError)as error:
        with Providers():module.call_once(target,(),{},env,None,outputs,'actual',notes)
    assert error.value is primary and outputs=={'expected':expected,'actual':actual}and events==['snapshot','target','snapshot']
    path=tmp_path/'record.json';module.retain_record(path,{},outputs,lambda value,**k:{'role':'expected'if value is expected else'actual'})
    saved=json.loads(path.read_bytes());assert saved['expected']['role']=='expected'and saved['actual']['role']=='actual'


def profile_fixture():
    module=load_current()
    def reference():pass
    def native():pass
    def prefix():pass
    def fallback():pass
    entries={name:object()for name in('get_num_threads','label_pair_check_storage','label_pair_counts_preweighted','variation_of_information_csr')}
    counts=dict(original=0,native=0);returned=[];accepted=[];suffix=[];phase=['reference']
    guard=module.ownership_profile(reference,native,prefix,fallback,entries,phase,counts,returned,accepted,suffix)
    def frame(fn,module='owned',**locals):return SimpleNamespace(f_code=fn.__code__,f_globals={'__name__':module},f_locals=locals)
    return guard,frame,reference,native,prefix,fallback,entries,phase,counts,returned,accepted,suffix


def test_actual_profile_counts_one_pair_all_eleven_returns_and_acceptance():
    guard,frame,reference,native,prefix,_,entries,phase,counts,returned,accepted,suffix=profile_fixture()
    guard(frame(reference),'call',None);phase[0]='native';guard(frame(native),'call',None)
    for key,count in dict(get_num_threads=2,label_pair_check_storage=7,label_pair_counts_preweighted=1,variation_of_information_csr=1).items():
        for _ in range(count):guard(frame(native),'c_return',entries[key])
    guard(frame(prefix,native_result=object()),'return',object());guard(frame(native),'return',object())
    assert counts==dict(original=1,native=1)and len(returned)==11 and accepted==suffix==[True]


@pytest.mark.parametrize('fault',('original','helper','fallback','compiled-original'))
def test_native_phase_denies_cached_original_helpers_and_fallback(fault):
    guard,frame,reference,native,_,fallback,_,phase,_,_,_,_=profile_fixture();phase[0]='native'
    if fault=='original':call=lambda:guard(frame(reference),'call',None)
    elif fault=='helper':call=lambda:guard(frame(native,'scipy.ndimage._filters'),'call',None)
    elif fault=='fallback':call=lambda:guard(frame(fallback),'call',None)
    else:call=lambda:guard(frame(native),'c_call',SimpleNamespace(__module__='skimage._compiled'))
    with pytest.raises(AssertionError):call()


def test_native_None_return_remains_a_refusal_not_accepted():
    guard,frame,_,native,prefix,_,_,phase,_,_,accepted,suffix=profile_fixture();phase[0]='native'
    guard(frame(native),'return',None);guard(frame(prefix,native_result=None),'return',None)
    assert accepted==suffix==[False]


def test_diagnostics_keep_other_words_when_one_output_codec_fails(tmp_path):
    module=load_current();record={'case':'current-case'};events=[];good=object();bad=object()
    def encode(value,**kwargs):
        events.append(value)
        if value is bad:raise TypeError('bad-POD')
        return {'raw_hex':'0100000000000000'}
    path=tmp_path/'record.json';module.retain_record(path,record,{'expected':bad,'actual':good},encode)
    saved=json.loads(path.read_bytes());assert events==[bad,good]and saved['expected']['unavailable']=='TypeError'and saved['actual']['raw_hex']=='0100000000000000'
    with pytest.raises(FileExistsError):module.retain_record(path,record,{},encode)
