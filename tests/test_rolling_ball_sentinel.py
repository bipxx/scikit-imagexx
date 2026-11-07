"""Rolling-ball scalar API metadata; inert dependencies, no image algorithms."""
import ast
import builtins
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import types
import warnings
import pytest
ROOT=Path(__file__).resolve().parents[1]
H=lambda raw:hashlib.sha256(raw).hexdigest()


def load_metadata(monkeypatch, available=True):
    """Execute real consumer/frontend modules; only array metadata is inert."""
    snapshot=json.loads((ROOT/'tests/fixtures/rolling_ball_sources_v1.json').read_bytes())['files']
    text=snapshot['_shared/utils.py']['text']
    assert H(text.encode())==snapshot['_shared/utils.py']['raw_sha256']
    nodes=[n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name in ('PatchClassRepr','DEPRECATED')]
    assert len(nodes)==2
    shared=types.ModuleType('skimage._shared.utils')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned upstream scalar marker>','exec'),shared.__dict__)
    numpy=types.ModuleType('numpy');numpy.__version__='1.24.4'
    numpy.lib=types.SimpleNamespace(NumpyVersion=lambda version:'1.24.4')
    numpy.dtype=lambda t:types.SimpleNamespace(num=t)
    for i,name in enumerate(('int8','uint8','int16','uint16','int32','uint32','int64','uint64','longlong','ulonglong')):setattr(numpy,name,i)
    fromnumeric=types.ModuleType('numpy.core.fromnumeric');fromnumeric.amax=lambda *a:(_ for _ in ()).throw(AssertionError('No numerical provider'))
    name='_rolling_sentinel_metadata';package=types.ModuleType(name);package.__path__=[str(ROOT/'python/skimagexx')]
    core=types.ModuleType(name+'._core')
    components=types.ModuleType(name+'._components');components.label=lambda *a:(_ for _ in ()).throw(AssertionError('No component algorithm'))
    kernels=types.ModuleType(name+'._rolling_ball_kernels');kernels._MAX_RANK=32;kernels._ordinary_scalar=lambda x:False
    for key,value in {'numpy':numpy,'numpy.core.fromnumeric':fromnumeric,name:package,name+'._core':core,name+'._components':components,name+'._rolling_ball_kernels':kernels,'skimage':types.ModuleType('skimage'),'skimage._shared':types.ModuleType('skimage._shared'),'skimage._shared.utils':shared}.items():monkeypatch.setitem(sys.modules,key,value)
    original_import=builtins.__import__;requests=[]
    def guarded(module,*args,**kwargs):
        if module.startswith('skimage'):
            requests.append(module)
            assert module=='skimage._shared.utils','No upstream algorithm imports'
            if not available:raise ImportError('Intentional upstream-unavailable metadata control')
        return original_import(module,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    def actual(module,filename):
        spec=importlib.util.spec_from_file_location(name+'.'+module,ROOT/'python/skimagexx'/filename)
        value=importlib.util.module_from_spec(spec);monkeypatch.setitem(sys.modules,spec.name,value);spec.loader.exec_module(value);return value
    consumer=actual('_component_consumers','_component_consumers.py')
    owned=actual('_rolling_ball','_rolling_ball.py')
    assert requests==['skimage._shared.utils']
    return owned,consumer,shared,snapshot


@pytest.mark.parametrize('available',(True,False))
def test_actual_default_is_shared_identity_and_strict_signature(monkeypatch,available):
    owned,consumer,shared,snapshot=load_metadata(monkeypatch,available)
    marker=consumer._DEPRECATED
    assert owned.DEPRECATED is marker
    assert inspect.signature(owned.rolling_ball).parameters['num_threads'].default is marker
    if available:
        assert marker is shared.DEPRECATED and isinstance(marker,type)
        # Actual upstream signature AST; replace only body/decorators, never invoke it.
        raw=snapshot['restoration/_rolling_ball.py']['text']
        assert H(raw.encode())==snapshot['restoration/_rolling_ball.py']['raw_sha256']
        node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='rolling_ball')
        node.decorator_list=[];node.body=[ast.Raise(exc=ast.Call(func=ast.Name(id='AssertionError',ctx=ast.Load()),args=[ast.Constant('No reference body')],keywords=[]),cause=None)]
        tree=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]));scope={'DEPRECATED':marker}
        exec(compile(tree,'<pinned actual upstream signature>','exec'),scope)
        assert inspect.signature(owned.rolling_ball)==inspect.signature(scope['rolling_ball'])
    else:
        assert marker is not shared.DEPRECATED and repr(marker)=='<DEPRECATED>'


@pytest.mark.parametrize('available',(True,False))
@pytest.mark.parametrize('case',('omitted','explicit','explicit-workers','deprecated','conflict','same-repr','same-repr-conflict'))
def test_actual_decorator_routes_scalar_metadata_without_algorithms(monkeypatch,available,case):
    owned,consumer,_,_=load_metadata(monkeypatch,available)
    calls=[]
    def inert(image,*,num_threads=consumer._DEPRECATED,workers=None):
        calls.append((image,num_threads,workers));return calls[-1]
    wrapped=owned._thread_deprecation(inert)
    class SameRepr:
        def __repr__(self):return '<DEPRECATED>'
    other=SameRepr();options={}
    if case in ('explicit','explicit-workers'):options['num_threads']=consumer._DEPRECATED
    elif case in ('deprecated','conflict'):options['num_threads']=3
    elif case.startswith('same-repr'):options['num_threads']=other
    if case in ('explicit-workers','conflict','same-repr-conflict'):options['workers']=4
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always')
        if case in ('conflict','same-repr-conflict'):
            with pytest.raises(ValueError,match='Both deprecated parameter'):wrapped('inert image',**options)
            assert not calls
        else:
            result=wrapped('inert image',**options)
            assert result[1] is consumer._DEPRECATED
            expected=4 if case=='explicit-workers' else 3 if case=='deprecated' else other if case=='same-repr' else None
            assert result[2] is expected or result[2]==expected
    assert len(observed)==(0 if case in ('omitted','explicit','explicit-workers') else 1)
    if observed:assert observed[0].category is FutureWarning


def test_exact_current_source_context_and_unchanged_operation():
    from tools.rolling_ball_sentinel_context import predecessor_bytes
    context=json.loads((ROOT/'coverage/contracts/rolling-ball-sentinel-v1.json').read_bytes())
    for name,row in context['sources'].items():
        raw=(ROOT/name).read_bytes();old=predecessor_bytes(ROOT,name,raw)
        assert H(old)==row['predecessor_canonical_sha256']
        with pytest.raises(AssertionError):predecessor_bytes(ROOT,name,raw+b'changed')
        with pytest.raises(AssertionError):predecessor_bytes(ROOT,name,old)
    name='python/skimagexx/_rolling_ball.py';raw=(ROOT/name).read_bytes();old=predecessor_bytes(ROOT,name,raw)
    def bodies(value):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(value).body if isinstance(n,ast.FunctionDef)}
    assert bodies(raw)==bodies(old)  # Includes complete scan, kernel, and decorator.
    assert '_Deprecated' not in raw.decode() and 'from ._component_consumers import _DEPRECATED as DEPRECATED' in raw.decode()
    for field in ('cases','method','route'):
        assert json.loads((ROOT/'coverage/contracts/rolling-ball-benchmark-execution-v3.json').read_bytes())[field]==json.loads((ROOT/'coverage/contracts/rolling-ball-benchmark-execution-v2.json').read_bytes())[field]


@pytest.mark.parametrize('transport',('LF','CRLF'))
def test_raw_canonical_and_common_entries_project_once(transport):
    from tools.rolling_ball_sentinel_context import predecessor_bytes
    from tools.join_source_context import rolling_predecessor_bytes,rolling_predecessor_raw_bytes
    c=json.loads((ROOT/'coverage/contracts/rolling-ball-sentinel-v1.json').read_bytes())
    for name in ('tools/join_source_context.py','tools/public_qualification.py','python/skimagexx/_rolling_ball.py'):
        raw=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
        if transport=='CRLF':raw=raw.replace(b'\n',b'\r\n')
        old=predecessor_bytes(ROOT,name,raw)
        for entry in (rolling_predecessor_bytes,rolling_predecessor_raw_bytes):
            actual=entry(ROOT,name,raw)
            assert actual  # Historical layers remain strict after this once-only boundary.
        assert H(old)==c['sources'][name]['predecessor_canonical_sha256']
    raw=b'unchanged\r\n';assert predecessor_bytes(ROOT,'unrelated',raw) is raw


def test_current_native_driver_and_measurement_source_pins_are_exact():
    native=json.loads((ROOT/'coverage/contracts/rolling-ball-native-v3.json').read_bytes())
    for group in ('source_files_raw_sha256','frozen_source_files_raw_sha256'):
        for name,pin in native[group].items():assert H((ROOT/name).read_bytes())==pin,name
    driver=json.loads((ROOT/'coverage/contracts/rolling-ball-benchmark-execution-v3.json').read_bytes())
    for name,pin in driver['source_files_canonical_sha256'].items():assert H((ROOT/name).read_bytes().replace(b'\r\n',b'\n'))==pin,name
    assert all(v is None for v in driver['unbound'].values())
    outer=json.loads((ROOT/'coverage/contracts/rolling-ball-measurement-v3.json').read_bytes())
    for name,pin in outer['benchmark_source_sha256'].items():assert H((ROOT/name).read_bytes().replace(b'\r\n',b'\n'))==pin,name
    for name,pin in outer['first_cohort']['source_files_raw_sha256'].items():assert H((ROOT/name).read_bytes())==pin,name
    assert outer['families']['rolling_ball_native']['contract']=='coverage/contracts/rolling-ball-benchmark-execution-v3.json'
    assert all(outer[k] is None for k in ('prepared_source','SDK_members','core_sha256','runtime_sha256','deadline_utc','benchmark_fingerprint'))
