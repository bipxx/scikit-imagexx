"""Strict ABI record selection and immutable Linux guards; stdlib/inert only."""
import ast,base64,builtins,copy,hashlib,importlib.util,json,sys,zlib
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('level_set_reference_metadata',ROOT/'tests/level_set_reference.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
LINUX=ROOT/'coverage/contracts/level-set.json'
SUPPLEMENT=ROOT/'coverage/contracts/level-set-windows.json'


def test_supplement_exact_witness_provenance_and_complete_rows():
    data=json.loads(SUPPLEMENT.read_bytes());p=data['provenance']
    assert data['schema']=='level-set-windows-original-v1' and tuple(data['ids'])==r.WINDOWS_IDS
    assert tuple(row['id'] for row in data['observations'])==r.WINDOWS_IDS
    assert p['witness_run']==40695893979 and p['witness_source']=='6235188190b6541b972db65f4a1a664a094d0d3f'
    assert p['witness_artifact']==dict(id=12818748266,bytes=209587,sha256='707f168d61cb32d16ed7b97019c9669c6a11bb2661cd174a9298872078f49f82')
    assert p['original_calls']==2 and p['native_calls']==0 and not p['project_imported'] and p['new_host_original_only']
    assert p['failed_v4_observations_match_count']==16 and p['failed_v4_facts_unchanged'] and p['all_inputs_post_unchanged']
    assert len(p['runtime_wheels'])==len(p['runtime_pins'])==14
    assert p['source_contract_pins']['tests/level_set_contract.py']=='2d993c41d01d125bbee11d5c61df57bad0d4cf39d15d4997f141fa4fcc55d40a'
    for row in data['observations']:
        assert all(row['checks'].values()) and row['warnings']==row['events']==[]
        if row['result']:
            raw=zlib.decompress(base64.b64decode(row['result']['raw_zlib_base64']))
            assert hashlib.sha256(raw).hexdigest()==row['result']['raw_sha256'] and len(raw)==77
    assert data['observations'][0]['error']==['builtins','OverflowError','int too big to convert']


@pytest.mark.parametrize('version,scalar',(('1.24.4',False),('1.24.4',True),('2.5.4',False),('2.5.4',True)))
def test_all79_linux_rows_remain_identical_in_each_existing_mode(monkeypatch,version,scalar):
    monkeypatch.setattr(r.platform,'system',lambda:'Linux')
    if scalar:monkeypatch.setenv('SKIMAGEXX_DISABLE_SIMD','1')
    else:monkeypatch.delenv('SKIMAGEXX_DISABLE_SIMD',raising=False)
    record=json.loads(LINUX.read_bytes());before=copy.deepcopy(record)
    assert len(record['original_observations'][version])==79
    for row in record['original_observations'][version]:assert r.reference_observation(record,row['id'],SimpleNamespace(__version__=version)) is row
    assert record==before


def test_only_two_windows_records_change_and_other77_never_probe_abi(monkeypatch):
    monkeypatch.setattr(r.platform,'system',lambda:'Windows')
    monkeypatch.setattr(r,'_windows_abi',lambda np:pytest.fail('77 unchanged records must not probe ABI'))
    record=json.loads(LINUX.read_bytes());rows=record['original_observations']['2.5.4']
    unchanged=[row for row in rows if row['id'] not in r.WINDOWS_IDS]
    assert len(unchanged)==77
    for row in unchanged:assert r.reference_observation(record,row['id'],SimpleNamespace(__version__='2.5.4')) is row


@pytest.mark.parametrize('case_id',r.WINDOWS_IDS)
def test_exact_windows_abi_selects_full_untouched_original_row(monkeypatch,case_id):
    data=json.loads(SUPPLEMENT.read_bytes());record=json.loads(LINUX.read_bytes())
    monkeypatch.setattr(r.platform,'system',lambda:'Windows');monkeypatch.setattr(r,'_windows_abi',lambda np:data['abi'])
    expected=next(row for row in data['observations'] if row['id']==case_id)
    assert r.reference_observation(record,case_id,SimpleNamespace(__version__='2.5.4'))==expected


def test_actual_abi_getter_reads_type_identity_widths_versions_and_source(monkeypatch,tmp_path):
    module=type(sys)('skimage.segmentation.morphsnakes');source=tmp_path/'morphsnakes.py';source.write_bytes(b'fixed\r\nsource\r\n');module.__file__=str(source)
    scipy=type(sys)('scipy');scipy.__version__='1.18.1'
    skimage=type(sys)('skimage');skimage.__version__='0.26.0';skimage.__path__=[]
    segmentation=type(sys)('skimage.segmentation');segmentation.morphsnakes=module;skimage.segmentation=segmentation
    for name,value in (('scipy',scipy),('skimage',skimage),('skimage.segmentation',segmentation),('skimage.segmentation.morphsnakes',module)):
        monkeypatch.setitem(sys.modules,name,value)
    longdouble=type('longdouble',(),{'__module__':'numpy'});float64=type('float64',(),{'__module__':'numpy'})
    class Dtype:
        itemsize=8
        type=longdouble
        def __str__(self):return 'float64'
    np=SimpleNamespace(__version__='2.5.4',longdouble=longdouble,float64=float64,dtype=lambda value:Dtype(),finfo=lambda value:SimpleNamespace(nmant=52))
    for name,value in (('system','Windows'),('machine','AMD64'),('python_implementation','CPython'),('python_version','3.13.16')):
        monkeypatch.setattr(r.platform,name,lambda value=value:value)
    monkeypatch.setattr(r.ctypes,'sizeof',lambda value:4 if value is r.ctypes.c_long else 8)
    expected=json.loads(SUPPLEMENT.read_bytes())['abi'];expected['original_source_sha256']=hashlib.sha256(b'fixed\nsource\n').hexdigest()
    assert r._windows_abi(np)==expected


@pytest.mark.parametrize('field',tuple(json.loads(SUPPLEMENT.read_bytes())['abi']))
def test_any_windows_abi_mismatch_explicitly_fails_without_linux_fallback(monkeypatch,field):
    data=json.loads(SUPPLEMENT.read_bytes());actual=dict(data['abi']);actual[field]='not the witnessed value'
    monkeypatch.setattr(r.platform,'system',lambda:'Windows');monkeypatch.setattr(r,'_windows_abi',lambda np:actual)
    with pytest.raises(AssertionError,match='Unassessed LevelSet Windows ABI'):
        r.reference_observation(json.loads(LINUX.read_bytes()),r.WINDOWS_IDS[0],SimpleNamespace(__version__='2.5.4'))


def test_other_platform_cannot_reuse_windows_or_linux_record_for_two_ids(monkeypatch):
    monkeypatch.setattr(r.platform,'system',lambda:'Darwin')
    with pytest.raises(AssertionError,match='Unassessed LevelSet reference platform'):
        r.reference_observation(json.loads(LINUX.read_bytes()),r.WINDOWS_IDS[1],SimpleNamespace(__version__='2.5.4'))


@pytest.mark.parametrize('bad',('schema','ids','observations'))
def test_malformed_supplement_cannot_expand_scope_or_lose_rows(monkeypatch,tmp_path,bad):
    data=json.loads(SUPPLEMENT.read_bytes());context=dict(data['abi'])
    if bad=='schema':data['schema']='unknown'
    elif bad=='ids':data['ids'].append('disk_level_set/default')
    else:data['observations'].pop()
    target=tmp_path/'supplement.json';target.write_text(json.dumps(data))
    monkeypatch.setattr(r,'SUPPLEMENT',target);monkeypatch.setattr(r.platform,'system',lambda:'Windows');monkeypatch.setattr(r,'_windows_abi',lambda np:context)
    with pytest.raises(AssertionError,match='LevelSet Windows'):
        r.reference_observation(json.loads(LINUX.read_bytes()),r.WINDOWS_IDS[0],SimpleNamespace(__version__='2.5.4'))


def test_linux_fixture_factory_and_observer_body_are_immutable():
    assert hashlib.sha256(LINUX.read_bytes()).hexdigest()=='8fb0b31cf701c96c75730d618adebcae37118f243972901a570ce26efdf5a706'
    assert hashlib.sha256((ROOT/'tests/level_set_contract.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest()=='2d993c41d01d125bbee11d5c61df57bad0d4cf39d15d4997f141fa4fcc55d40a'
    source=(ROOT/'tests/test_level_set_public.py').read_text()
    tree=ast.parse(source)
    observe=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='observe')
    assert hashlib.sha256(ast.get_source_segment(source,observe).encode()).hexdigest()=='7be7f9d4e89730c3c51eb93826f9ab8c30507ace2c903bd39c6d4216634bca85'
    for filename,function in (('test_level_set_public.py','test_level_set_native_public_contract'),('test_level_set_pipeline.py','test_level_set_pipeline')):
        tree=ast.parse((ROOT/'tests'/filename).read_bytes());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
        calls=[n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='reference_observation']
        assert len(calls)==1


def test_selector_import_has_no_scientific_calls_and_preserves_frozen_bytes(monkeypatch):
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        assert name.split('.')[0] not in ('numpy','scipy','skimage','skimagexx')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.WINDOWS_IDS==r.WINDOWS_IDS
    assert 'coverage/contracts/level-set-windows.json -text' in (ROOT/'.gitattributes').read_text()
