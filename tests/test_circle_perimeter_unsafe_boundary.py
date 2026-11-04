"""Never execute unsafe originals; narrow quarantine and saved safe controls."""
import base64
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import zlib
import pytest
from circle_perimeter_contract import cases
from circle_perimeter_observer import observe
from draw_coordinates_contract import ScalarProvider
ROOT=Path(__file__).parents[1]
MESSAGE='draw.circle_perimeter: tuple shapes shorter than 2 entries are outside the assessed safe boundary'
NATIVE_IDS=('bresenham/clip','bresenham/shape-extra','andres/shape-list','andres/shape-negative-stride')
PROVIDER_IDS=('bresenham/shape-float','bytes-method','none-method','list-method')
CONTROL_IDS=NATIVE_IDS+PROVIDER_IDS+('invalid-method',)


@pytest.fixture(scope='module')
def owned():
    import skimagexx
    spec=importlib.util.spec_from_file_location('skimagexx._circle_perimeter_boundary_v1',ROOT/'python/skimagexx/_circle_perimeter.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.fixture(scope='module')
def expected():
    matrix=json.loads((ROOT/'coverage/contracts/circle-perimeter.json').read_bytes())
    return {p:{r['id']:r for r in rows}for p,rows in matrix['original_observations'].items()}


def test_circle_unsafe_boundary_prospective_scope():
    record=json.loads((ROOT/'coverage/contracts/circle-perimeter-unsafe-boundary-v1.json').read_bytes())
    assert record['quarantine']==dict(carrier='exact builtin tuple',lengths=[0,1])
    assert record['safe_control_ids']==list(CONTROL_IDS)and record['original_calls']==0
    assert record['exception']==['builtins','NotImplementedError',MESSAGE]
    assert record['source_freeze']['python/skimagexx/_circle_perimeter.py']==hashlib.sha256((ROOT/'python/skimagexx/_circle_perimeter.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest()


@pytest.mark.parametrize('short',((),(7,)),ids=('empty','one'))
@pytest.mark.parametrize('radius',(-3,0,4))
@pytest.mark.parametrize('method',('bresenham','andres','np-string'))
def test_exact_short_tuple_never_reaches_core_provider_or_shape_items(owned,monkeypatch,short,radius,method):
    import numpy as np
    events=[];calls=[]
    class ShapeElement:
        def __index__(self):raise AssertionError('shape element protocol touched')
        def __lt__(self,other):raise AssertionError('shape comparison touched')
    shape=()if not short else(ShapeElement(),)
    def denied(*a,**k):calls.append('forbidden');raise AssertionError('unsafe route reached core/provider')
    binary=importlib.import_module('skimage.draw._draw')
    monkeypatch.setattr(binary,'_circle_perimeter',denied)
    monkeypatch.setattr(owned._core,'draw_circle_perimeter',denied);monkeypatch.setattr(owned._core,'get_num_threads',denied)
    if method=='np-string':method=np.str_('andres')
    with pytest.raises(NotImplementedError,match='tuple shapes shorter than 2')as error:
        owned.circle_perimeter(ScalarProvider(4,events,'r'),ScalarProvider(5,events,'c'),ScalarProvider(radius,events,'radius'),method,shape)
    assert str(error.value)==MESSAGE and events==['r.__index__','c.__index__','radius.__index__']and calls==[]


@pytest.mark.parametrize('method',('unknown',b'bresenham',None,['bresenham']))
@pytest.mark.parametrize('shape',((),(7,)))
def test_known_method_errors_precede_quarantine(owned,monkeypatch,method,shape):
    events=[]
    def denied(*a,**k):raise AssertionError('method rejection forwarded')
    monkeypatch.setattr(importlib.import_module('skimage.draw._draw'),'_circle_perimeter',denied)
    monkeypatch.setattr(owned._core,'draw_circle_perimeter',denied);monkeypatch.setattr(owned._core,'get_num_threads',denied)
    with pytest.raises(ValueError,match='^Wrong method$'):
        owned.circle_perimeter(ScalarProvider(4,events,'r'),ScalarProvider(5,events,'c'),ScalarProvider(4,events,'radius'),method,shape)
    assert events==['r.__index__','c.__index__','radius.__index__']


@pytest.mark.parametrize('index',(0,1,2))
@pytest.mark.parametrize('kind',('overflow','float'))
def test_sequential_conversion_errors_precede_method_and_quarantine(owned,monkeypatch,index,kind):
    events=[];values=[ScalarProvider(v,events,n)for v,n in zip((4,5,4),('r','c','radius'))]
    values[index]=2**100 if kind=='overflow'else 4.0
    def denied(*a,**k):raise AssertionError('conversion error forwarded')
    monkeypatch.setattr(importlib.import_module('skimage.draw._draw'),'_circle_perimeter',denied)
    monkeypatch.setattr(owned._core,'draw_circle_perimeter',denied);monkeypatch.setattr(owned._core,'get_num_threads',denied)
    with pytest.raises(OverflowError if kind=='overflow'else TypeError):owned.circle_perimeter(*values,method='unknown',shape=(7,))
    assert events==[n+'.__index__'for n in ('r','c','radius')[:index]]


def test_unknown_method_callback_is_not_replayed_into_unsafe_provider(owned,monkeypatch):
    events=[]
    class Method:
        def __eq__(self,other):events.append('method=='+other);return other=='bresenham'
    def denied(*a,**k):raise AssertionError('unsafe callback route forwarded')
    monkeypatch.setattr(importlib.import_module('skimage.draw._draw'),'_circle_perimeter',denied)
    monkeypatch.setattr(owned._core,'draw_circle_perimeter',denied);monkeypatch.setattr(owned._core,'get_num_threads',denied)
    with pytest.raises(NotImplementedError):owned.circle_perimeter(ScalarProvider(4,events,'r'),5,4,Method(),())
    assert events==['r.__index__','method==bresenham']


@pytest.mark.parametrize('case_id',CONTROL_IDS)
@pytest.mark.parametrize('workers',(1,4))
def test_saved_safe_native_provider_and_error_routes_unchanged(owned,monkeypatch,expected,case_id,workers):
    import numpy as np
    import skimagexx as xx
    spec=next(s for s in cases()if s['id']==case_id);gold=expected[np.__version__][case_id]
    calls=[];native_returns=[];entry=owned._core.draw_circle_perimeter;previous=sys.getprofile();oldworkers=xx.get_num_threads()
    def provider(*a,**k):
        calls.append((a,k))
        if gold['error']is not None:raise ValueError(gold['error'][2])
        result=[]
        for value in gold['result']['items']:
            raw=zlib.decompress(base64.b64decode(value['raw_zlib_base64']))
            out=np.empty(value['shape'],dtype=np.dtype(value['dtype']));out[...] =np.frombuffer(raw,dtype=out.dtype).reshape(out.shape);result.append(out)
        return tuple(result)
    def profiler(frame,event,value):
        if event=='c_return'and value is entry:native_returns.append('circle')
    try:
        xx.set_num_threads(workers);monkeypatch.setattr(importlib.import_module('skimage.draw._draw'),'_circle_perimeter',provider)
        sys.setprofile(profiler)
        try:actual=observe(owned.circle_perimeter,spec,np)
        finally:sys.setprofile(previous)
        assert actual==gold
        assert len(calls)==int(case_id in PROVIDER_IDS)
        assert native_returns==(['circle']if case_id in NATIVE_IDS else[])
    finally:sys.setprofile(previous);xx.set_num_threads(oldworkers)
