"""Saved reference and prospective stdlib guards; no original/native execution."""
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
    assert data['original_calls']==48 and data['native_calls']==0 and data['timings']is False
    assert set(data['original_observations'])=={'1.24.4','2.5.4'}
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


@pytest.mark.parametrize('version',['1.24.4','2.5.4'])
def test_draw_polygon_saved_original_records(version):
    # Decode only retained words. No NumPy, original calls, or image algorithms.
    import base64
    import math
    import zlib
    data=json.loads(MATRIX.read_bytes());rows=data['original_observations'][version]
    assert [row['id']for row in rows]==[spec['id']for spec in cases()]
    assert len(rows)==24 and sum(row['error']is None for row in rows)==17
    assert sum(row['error']is not None for row in rows)==7
    def check_payload(value):
        if isinstance(value,dict):
            if value.get('type')=='ndarray':
                raw=zlib.decompress(base64.b64decode(value['raw_zlib_base64']))
                assert hashlib.sha256(raw).hexdigest()==value['raw_sha256']
                itemsize=1 if value['dtype']=='|b1'else int(value['dtype'][2:])
                assert len(raw)==math.prod(value['shape'])*itemsize
            for child in value.values():check_payload(child)
        elif isinstance(value,list):
            for child in value:check_payload(child)
    for spec,row in zip(cases(),rows):
        assert row['api']==spec['api'] and all(row['checks'].values())
        assert row['inputs_before']==row['inputs_after'] and row['warnings']==row['FP_events']==[]
        assert row['call_accounting']['top_level_api']==spec['api'] and row['call_accounting']['top_level_calls']==1
        check_payload(row)
        if row['error']is None:
            outputs=row['result']['items']if spec['api']==APIS[0]else[row['result']]
            assert all(a['type']=='ndarray'and a['own']and a['writeable']and a['aligned']and a['base_type']=='NoneType'for a in outputs)
            if spec['api']==APIS[0]:
                assert row['result']['container']=='tuple'and len(outputs)==2
                assert all(a['dtype']=='<i8'and len(a['shape'])==1 for a in outputs)
                assert outputs[0]['shape']==outputs[1]['shape'] and outputs[0]['shape'][0]<=data['bounds_by_case'][spec['id']]['max_output_pairs']
                assert row['output_pair_aliases'][1][0]is False
            else:
                assert outputs[0]['dtype']=='|b1'and outputs[0]['shape']==spec['shape']
                raw=zlib.decompress(base64.b64decode(outputs[0]['raw_zlib_base64']));assert set(raw)<={0,1}
        else:assert row['error'][0:2]==['builtins','ValueError']
    capture=data['original_capture'][version]
    assert capture['top_level_original_calls']==24 and capture['journal_events']==48 and capture['returns']==17 and capture['errors']==7
    assert capture['runtime']['numpy']==version and capture['dependent_polygon_calls']==10
    assert rows==data['original_observations']['1.24.4']
