"""Prospective stdlib guards; no original/scientific/native image execution."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
if __package__:
    from .draw_polygon_contract import cases,source_safe,fixture,APIS,UNASSESSED
else:
    from draw_polygon_contract import cases,source_safe,fixture,APIS,UNASSESSED
ROOT=Path(__file__).parents[1]
MATRIX=ROOT/'coverage/contracts/draw-polygon-v1.json'


def test_draw_polygon_frozen_source_and_scope():
    data=json.loads(MATRIX.read_bytes());assert data['cases']==cases()and len(cases())==24
    assert len({s['id']for s in cases()})==24 and {a:sum(s['api']==a for s in cases())for a in APIS}==dict.fromkeys(APIS,12)
    assert data['bounds_by_case']=={s['id']:source_safe(s)for s in cases()}
    assert data['original_calls']==data['native_calls']==0 and data['timings']is False
    assert data['original_observations']=={'1.24.4':[],'2.5.4':[]}
    assert data['unassessed']==list(UNASSESSED)
    for path,pin in data['source_freeze'].items():assert hashlib.sha256((ROOT/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin
    text=(ROOT/data['borrowed_helper_methods']['module']).read_text();tree=ast.parse(text)
    for name,pin in data['borrowed_helper_methods']['canonicalLF_function_sha256'].items():
        n=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
        body=''.join(text.splitlines(keepends=True)[n.lineno-1:n.end_lineno]);assert hashlib.sha256(body.encode()).hexdigest()==pin


@pytest.mark.parametrize('spec',cases(),ids=lambda s:s['id'])
def test_draw_polygon_real_observer_call_expression_refuses_original(spec):
    # A tiny recording array adapter checks factory recipes without importing
    # NumPy. This is NOT actual NumPy layout/dtype execution or a golden result.
    class Flags:writeable=True
    class Array:
        def __init__(self,values,dtype,order):self.values=copy.deepcopy(values);self.dtype=dtype;self.order=order;self.flags=Flags();self.shape=None
        def __getitem__(self,key):return Array(self.values[key],self.dtype,self.order)
        def reshape(self,shape):self.shape=shape;return self
    class Recorder:
        @staticmethod
        def array(values,dtype,order):return Array(values,dtype,order)
    args,options,buffers,bounds=fixture(spec,Recorder)
    assert bounds==source_safe(spec)
    if spec['api']==APIS[0]:
        r=[v[0]for v in spec['vertices']];c=[v[1]for v in spec['vertices']]
        if spec['carrier']=='reverse':r.reverse();c.reverse()
        if spec['carrier']=='scalar':expected=(r[0],c[0])
        elif spec['carrier']=='rank2':expected=([r],[c])
        else:expected=(r,c)
        assert tuple(a.values if isinstance(a,Array)else a for a in args)==expected
        assert options==dict(shape=None if spec['shape']is None else tuple(spec['shape']))
    else:
        assert args[0]==tuple(spec['shape']) and not options
        actual=args[1].values if isinstance(args[1],Array)else args[1]
        assert actual==spec['vertices']
        if not spec['vertices']:assert args[1].shape==(0,2)
    for value in buffers.values():assert value.flags.writeable is not spec['readonly']
    tree=ast.parse((ROOT/'tests/draw_polygon_observer.py').read_bytes())
    call=next(n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='call')
    invoked=[]
    def refused(*a,**kw):invoked.append((a,kw));raise RuntimeError('original-entry-refused')
    with pytest.raises(RuntimeError,match='^original-entry-refused$'):
        eval(compile(ast.Expression(body=call),'<actual frozen observer call>','eval'),dict(call=refused,args=args,options=options))
    assert len(invoked)==1 and invoked[0][0]is not None


def test_draw_polygon_unfrozen_unsafe_inputs_refused_at_entry():
    for patch in [dict(vertices=[[float('nan'),0],[1,1]]),dict(shape=[5]),dict(shape=[]),dict(vertices=[[0,0]]*9),dict(carrier='protocol')]:
        spec=copy.deepcopy(cases()[0]);spec.update(patch)
        with pytest.raises(AssertionError):source_safe(spec)
    assert not any(s['shape']is not None and len(s['shape'])<2 for s in cases())


def test_draw_polygon_observer_import_is_science_free(monkeypatch):
    import builtins
    old=builtins.__import__
    def checked(name,*args,**kwargs):
        assert name.split('.')[0]not in{'numpy','scipy','skimage','skimagexx'}
        return old(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',checked)
    spec=importlib.util.spec_from_file_location('prospective_draw_polygon_observer',ROOT/'tests/draw_polygon_observer.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert set(module.SIGNATURES)==set(APIS)and callable(module.observe)and callable(module.reference_runtime)


def test_draw_polygon_predicate_reuse_and_safety_are_explicit():
    data=json.loads(MATRIX.read_bytes())
    assert 'VIEW(x=c,y=r)'in data['predicate_reuse']['match']
    assert 'QUERY(x=column,y=row)'in data['predicate_reuse']['match']
    assert 'canonical0/1'in data['predicate_reuse']['not_drop_in']
    assert 'Shortshape'in data['safety']['unsafe_exclusions'][0]
    observer=ast.parse((ROOT/'tests/draw_polygon_observer.py').read_bytes())
    assert len([n for n in ast.walk(observer)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='call'])==1
