"""Pure source/storage gates; no target or scientific import required."""
import ast,builtins,hashlib,json,subprocess,sys,types
from pathlib import Path
import pytest
from array_views_contract import APIS,cases,source_safe,seq,MAX_INPUT,MAX_OUTPUT
import array_views_observer as observer
ROOT=Path(__file__).resolve().parents[1]


def test_source_signatures_provider_and_manifest():
    raw=(ROOT/'tests/fixtures/array_views_upstream_v026.py.txt').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==observer.SOURCE_SHA256
    functions={n.name:n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)}
    blocks=functions['view_as_blocks'];windows=functions['view_as_windows']
    assert [a.arg for a in blocks.args.args]==['arr_in','block_shape']and not blocks.args.defaults
    assert [a.arg for a in windows.args.args]==['arr_in','window_shape','step']and windows.args.defaults[0].value==1
    assert ast.unparse(blocks.body[1].test)=='not isinstance(block_shape, tuple)'
    assert ast.unparse(windows.body[1].test)=='not isinstance(arr_in, np.ndarray)'
    for node in (blocks,windows):
        assert ast.unparse(node.body[-2].value)=='as_strided(arr_in, shape=new_shape, strides='+('new_strides)'if node is blocks else'strides)')
    data=json.loads((ROOT/'coverage/contracts/util-array-views-v1.json').read_bytes())
    assert data['cases']==cases()and data['apis']==list(APIS)
    assert data['budgets']==dict(original_calls=0,native_calls=0,builds=0,timing_batches=0)
    assert len({r['id']for r in cases()})==len(cases())
    assert data['standalone_CPP_proposal']['status']=='PROPOSAL_ONLY_NO_HEADER_LEAF_BINDER_AUTHORED'


@pytest.mark.parametrize('spec',cases(),ids=lambda s:s['id'])
def test_bounded_recipe_source_only(spec):
    bound=source_safe(spec)
    assert bound['input_elements']<=MAX_INPUT and bound['output_elements']<=MAX_OUTPUT
    assert bound['kind']in('pre_view_error','bounded_view')


@pytest.mark.parametrize('change',[
 dict(step=seq(-1,1)),dict(parameter=seq(float('inf'),2)),dict(shape=[10**9,6]),
 dict(parameter=seq(2.0,2.0),api=APIS[0]),dict(parameter=seq(True,True),api=APIS[0]),
 dict(parameter={'forged_protocol':True}),dict(input_kind='numpy_scalar',api=APIS[0]),
])
def test_quarantined_recipe_never_reaches_factory(change):
    row=next(s for s in cases()if s['id']=='view_as_windows/baseline').copy();row.update(change)
    with pytest.raises(AssertionError):source_safe(row)


def binding(tmp_path):
    row=dict(schema='array-views-intake-binding-v1',status='ROOT_BOUND_INTAKE_ONLY',mode='intake-only',source_revision=observer.source_revision(),source_files_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()for name in observer.SOURCE_FILES},apis=list(APIS),original_calls=0)
    path=tmp_path/'synthetic-intake-only-root.json';path.write_text(json.dumps(row));return path,row


@pytest.mark.parametrize('field,value',[('status','UNBOUND'),('original_calls',1),('mode','capture'),('source_revision','0'*40),('apis',[APIS[0]])])
def test_actual_CLI_rejects_envelope_before_science(tmp_path,monkeypatch,field,value):
    path,row=binding(tmp_path);row[field]=value;path.write_text(json.dumps(row))
    monkeypatch.setattr(observer,'reference_runtime',lambda:pytest.fail('invalid envelope reached runtime'))
    with pytest.raises(AssertionError):observer.main(['--root-binding',str(path),'--intake-only'])


def test_actual_CLI_rejects_changed_source_pin(tmp_path,monkeypatch):
    path,row=binding(tmp_path);row['source_files_sha256']['tests/array_views_contract.py']='0'*64;path.write_text(json.dumps(row))
    monkeypatch.setattr(observer,'reference_runtime',lambda:pytest.fail('changed source reached runtime'))
    with pytest.raises(AssertionError):observer.main(['--root-binding',str(path),'--intake-only'])


def test_actual_CLI_cached_numpy_import_request_is_refused(tmp_path,monkeypatch):
    path,_=binding(tmp_path);attempts=[];old_import=builtins.__import__;old_numpy=sys.modules.get('numpy')
    with monkeypatch.context()as scoped:
        if old_numpy is None:scoped.setitem(sys.modules,'numpy',types.ModuleType('numpy'))
        current=sys.modules['numpy']
        def refused(name,*args,**kwargs):
            if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):
                attempts.append(name);raise RuntimeError('ARRAY_VIEWS_INTAKE_IMPORT_REFUSED')
            return old_import(name,*args,**kwargs)
        scoped.setattr(builtins,'__import__',refused)
        with pytest.raises(RuntimeError,match='ARRAY_VIEWS_INTAKE_IMPORT_REFUSED'):
            observer.main(['--root-binding',str(path),'--intake-only'])
        assert sys.modules['numpy']is current
    assert attempts==['numpy']and builtins.__import__ is old_import and sys.modules.get('numpy')is old_numpy


def test_actual_CLI_cold_import_in_fresh_process(tmp_path):
    path,_=binding(tmp_path)
    script="""import builtins,sys
sys.path.insert(0,sys.argv[1])
import array_views_observer as module
assert 'numpy'not in sys.modules
old=builtins.__import__
def refused(name,*args,**kwargs):
 if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):raise RuntimeError('COLD_INTAKE_REFUSED_'+name)
 return old(name,*args,**kwargs)
builtins.__import__=refused
try:module.main(['--root-binding',sys.argv[2],'--intake-only'])
except RuntimeError as error:assert str(error)=='COLD_INTAKE_REFUSED_numpy'
else:raise AssertionError('cold original import escaped')
assert 'numpy'not in sys.modules
"""
    result=subprocess.run([sys.executable,'-c',script,str(ROOT/'tests'),str(path)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
