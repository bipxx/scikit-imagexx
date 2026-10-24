"""Inert private-loader metadata; these checks never import image/prototype code."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).parents[1]
MIGRATION=ROOT/'coverage/evidence/private-rank-collection-migration.json'
DATA=json.loads(MIGRATION.read_text(encoding='utf-8'))
EXPECTED_FAMILIES=('bilateral','percentile','statistics')
SOURCES=DATA['sources']


def selected_loader(path,record):
    module=ast.parse(path.read_text(encoding='utf-8'))
    expected=ast.parse(record['allowed_top_level_guard_utf8']).body[0]
    guards=[n for n in module.body if isinstance(n,ast.If) and ast.dump(n)==ast.dump(expected)]
    assert len(guards)==1
    loads=[n for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='core' for t in n.targets)]
    assert len(loads)==1
    assert module.body.index(guards[0])<module.body.index(loads[0])
    return compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(guards[0]),copy.deepcopy(loads[0])],type_ignores=[])),str(path),'exec')


def inert_namespace(environ,find_spec,import_module,skips):
    def skip(reason,**options):
        skips.append(dict(reason=reason,**options))
        pytest.skip(reason,**options)
    return dict(os=SimpleNamespace(environ=environ),
                importlib=SimpleNamespace(util=SimpleNamespace(find_spec=find_spec),import_module=import_module),
                pytest=SimpleNamespace(skip=skip))


def test_exact_original_utf8_and_only_one_top_level_guard_change():
    assert DATA['schema']=='private-rank-load-guard-migration-v1'
    assert DATA['public_native_parity_promoted'] is False
    assert set(SOURCES)=={f'tests/test_rank_{family}_pipeline.py' for family in EXPECTED_FAMILIES}
    for filename,record in SOURCES.items():
        original=record['original_utf8']
        assert hashlib.sha256(original.encode()).hexdigest()==record['original_utf8_sha256']
        current=(ROOT/filename).read_bytes().replace(b'\r\n',b'\n')
        assert hashlib.sha256(current).hexdigest()==record['current_canonical_lf_sha256']
        before=ast.parse(original);after=ast.parse(current)
        expected=ast.parse(record['allowed_top_level_guard_utf8']).body[0]
        guards=[n for n in after.body if isinstance(n,ast.If) and ast.dump(n)==ast.dump(expected)]
        assert len(guards)==1 and len(after.body)==len(before.body)+1
        after.body.remove(guards[0])
        # Complete same-runtime AST comparison includes every numerical test,
        # callback, helper, class, import, global and original import call.
        assert ast.dump(after)==ast.dump(before)
        assert record['selector_environment']==f'SKIMAGEXX_RANK_{filename.split("test_rank_",1)[1].split("_pipeline",1)[0].upper()}_BACKEND'


@pytest.mark.parametrize('filename',SOURCES)
def test_missing_default_private_fixture_is_explicit_module_skip(filename):
    record=SOURCES[filename];calls=[];skips=[]
    def find(name):calls.append(('find',name));return None
    def load(name):pytest.fail('missing private fixture must not be imported')
    namespace=inert_namespace({},find,load,skips)
    with pytest.raises(pytest.skip.Exception,match='Private compiled fixture'):
        exec(selected_loader(ROOT/filename,record),namespace)
    assert calls==[('find',record['default_module'])]
    assert skips==[dict(reason=record['skip_reason'],allow_module_level=True)]
    assert record['default_module'] in skips[0]['reason'] and 'public parity' in skips[0]['reason']


@pytest.mark.parametrize('filename',SOURCES)
def test_discoverable_default_still_imports_normally(filename):
    record=SOURCES[filename];calls=[];skips=[];sentinel=object()
    def find(name):calls.append(('find',name));return object()
    def load(name):calls.append(('import',name));return sentinel
    namespace=inert_namespace({},find,load,skips);exec(selected_loader(ROOT/filename,record),namespace)
    assert namespace['core'] is sentinel and not skips
    assert calls==[('find',record['default_module']),('import',record['default_module'])]


@pytest.mark.parametrize('filename',SOURCES)
@pytest.mark.parametrize('selector',['chosen_private_fixture','_explicit_missing_private_fixture',''])
def test_explicit_selector_is_never_probed_or_skipped(filename,selector):
    record=SOURCES[filename];calls=[];skips=[];sentinel=object()
    def find(name):pytest.fail('explicit selector must not be silently probed/skipped')
    def load(name):calls.append(name);return sentinel
    namespace=inert_namespace({record['selector_environment']:selector},find,load,skips)
    exec(selected_loader(ROOT/filename,record),namespace)
    assert namespace['core'] is sentinel and calls==[selector] and not skips


@pytest.mark.parametrize('filename',SOURCES)
@pytest.mark.parametrize('explicit',[False,True])
@pytest.mark.parametrize('kind',['module-missing','dependency-missing','link-import','link-os','initialization','empty-selector'])
def test_explicit_or_found_private_load_errors_propagate(filename,explicit,kind):
    record=SOURCES[filename];skips=[];selector='' if kind=='empty-selector' else '_chosen_private_fixture'
    errors={'module-missing':ModuleNotFoundError('No module named chosen fixture',name=selector),
            'dependency-missing':ModuleNotFoundError('No module named unrelated_dependency',name='unrelated_dependency'),
            'link-import':ImportError('DLL load failed: linked library unavailable'),
            'link-os':OSError('undefined symbol: compiled_fixture_symbol'),
            'initialization':RuntimeError('generic_type: type is already registered'),
            'empty-selector':ValueError('Empty module name')}
    error=errors[kind];calls=[]
    def find(name):
        assert not explicit;calls.append(('find',name));return object()
    def load(name):calls.append(('import',name));raise error
    environ={record['selector_environment']:selector} if explicit else {}
    namespace=inert_namespace(environ,find,load,skips)
    with pytest.raises(type(error)) as caught:exec(selected_loader(ROOT/filename,record),namespace)
    assert caught.value is error and not skips
    assert calls[-1]==('import',selector if explicit else record['default_module'])


@pytest.mark.parametrize('filename',SOURCES)
@pytest.mark.parametrize('kind',['import-error','invalid-spec'])
def test_default_finder_fault_is_not_an_absent_fixture_skip(filename,kind):
    record=SOURCES[filename];skips=[];error=ImportError('finder dependency fault') if kind=='import-error' else ValueError('module.__spec__ is None')
    def find(name):raise error
    def load(name):pytest.fail('finder failure must propagate before import')
    namespace=inert_namespace({},find,load,skips)
    with pytest.raises(type(error)) as caught:exec(selected_loader(ROOT/filename,record),namespace)
    assert caught.value is error and not skips
