"""Inert original-only witness/authentication tests, never scientific calls."""
import ast, builtins, hashlib, importlib.util, io, json, types, zipfile
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('level_set_windows_reference_metadata',ROOT/'tools/level_set_windows_reference.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def test_plan_is_exact_original_only_two_cases_without_scientific_import(monkeypatch,capsys):
    original=builtins.__import__
    def guard(name,*args,**kwargs):
        assert name.split('.')[0] not in ('numpy','scipy','skimage','skimagexx')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guard)
    assert p.main(['plan','--output',str(ROOT/'build/original-plan-only')])==0
    plan=json.loads(capsys.readouterr().out)
    assert plan['ids']==['checkerboard_level_set/square-wide','disk_level_set/radius-longdouble']
    assert (plan['original_calls'],plan['native_calls'],plan['builds'],plan['timings'],plan['product_imported'])==(2,0,0,False,False)
    assert plan['artifact']==12818350704 and plan['donor_source']=='75bc34e5bb978361baeea8340dbeae2491c42086'
    assert plan['original_source_sha256']=='71ced09bbab3918523c38abeda1235a79ca34eb7467d76875e0ef5b13d2092d6'


@pytest.mark.parametrize('alias',(False,True))
def test_original_delegates_once_with_untouched_observations_and_actual_radius_types(alias):
    class Float64:pass
    class LongDouble:pass
    cls=Float64 if alias else LongDouble
    radius=cls();radius.dtype=types.SimpleNamespace(type=cls,itemsize=8)
    np=types.SimpleNamespace(float64=Float64,longdouble=cls)
    calls=[];records=[]
    def checker(shape,**kwargs):
        calls.append(('checker',shape,kwargs));raise OverflowError('untouched original error')
    def disk(shape,**kwargs):calls.append(('disk',shape,kwargs));return 'untouched original result'
    specs=[dict(id=p.IDS[0],api='checker'),dict(id=p.IDS[1],api='disk')]+[dict(id=str(i),api='unrequested') for i in range(77)]
    def observe(function,spec,np):
        kwargs=dict(square_size=1<<70) if spec['api']=='checker' else dict(radius=radius)
        try:result=function((7,11),**kwargs);error=None
        except OverflowError as exc:result=None;error=str(exc)
        record=dict(result=result,error=error,warnings=['original warning'],events=['original event']);records.append(record);return record
    rows,ids,arguments=p.capture_original(types.SimpleNamespace(CASES=specs,observe=observe),np,dict(checker=checker,disk=disk))
    assert ids==list(p.IDS) and rows==records and all(a is b for a,b in zip(rows,records))
    assert [c[0] for c in calls]==['checker','disk'] and calls[0][2]==dict(square_size=1<<70) and calls[1][2]['radius'] is radius
    assert arguments[1]['radius_type_is_longdouble'] and arguments[1]['radius_type_is_float64'] is alias
    assert rows[0]['error']=='untouched original error' and rows[1]['result']=='untouched original result'


def test_unapproved_census_or_duplicate_id_stops_before_original_call():
    harness=types.SimpleNamespace(CASES=[dict(id=p.IDS[0],api='unexpected')]*79)
    with pytest.raises(ValueError,match='two frozen79-spec IDs'):p.capture_original(harness,None,{})


def zip_bytes(files,comment=b''):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:
        for name,body in files.items():z.writestr(name,body)
        z.comment=comment
    return stream.getvalue()


@pytest.mark.parametrize('corruption',(None,'wheel','source','contract','project-runtime'))
def test_offline_runtime_only_auth_excludes_project_wheel_and_preserves_frozen_source(tmp_path,monkeypatch,corruption):
    original={name:('immutable '+name).encode() for name in p.CONTRACT_PINS};pins={n:p.H(v) for n,v in original.items()}
    source=zip_bytes(original,p.SOURCE.encode());source_pin=p.H(source)
    monkeypatch.setattr(p,'CONTRACT_PINS',pins);monkeypatch.setattr(p,'SOURCE_ZIP_SHA',source_pin)
    runtime={};members={}
    for i in range(14):
        name='dependency'+str(i)+'.whl';member='dependency'+str(i)+'.py'
        if i==0 and corruption=='project-runtime':member='skimagexx/__init__.py'
        raw=zip_bytes({member:b'original runtime'});runtime[name]=dict(sha256=p.H(raw));members[p.PREFIX+'runtime-wheels/'+name]=raw
    state=dict(source=dict(revision=p.SOURCE,files_sha256=pins),runtime_wheels=runtime,runtime_pins={})
    package=dict(source_revision=p.SOURCE,source_archive_sha256=source_pin)
    members.update({p.PREFIX+'state.json':json.dumps(state).encode(),p.PREFIX+'package.json':json.dumps(package).encode(),p.PREFIX+'source.zip':source,
        p.PREFIX+'product-wheels/never-extract.whl':b'project core must remain unopened'})
    if corruption=='wheel':members[p.PREFIX+'runtime-wheels/dependency0.whl']+=b'changed'
    if corruption=='source':members[p.PREFIX+'source.zip']+=b'changed'
    if corruption=='contract':state['source']['files_sha256']=dict(pins,**{next(iter(pins)):'0'*64});members[p.PREFIX+'state.json']=json.dumps(state).encode()
    def wheel_records(raw):
        with zipfile.ZipFile(io.BytesIO(raw)) as z:return {n:dict(sha256=p.H(z.read(n)),bytes=len(z.read(n))) for n in z.namelist()}
    helper=types.SimpleNamespace(wheel_records=wheel_records,safe_member=Path)
    if corruption:
        with pytest.raises(ValueError):p.prepare(zip_bytes(members),tmp_path,helper)
    else:
        p.prepare(zip_bytes(members),tmp_path,helper);record=json.loads((tmp_path/'inputs.json').read_text())
        assert len(record['wheels'])==14 and not (tmp_path/'original-prefix/skimagexx').exists()
        assert all((tmp_path/'reference-source'/name).read_bytes()==body for name,body in original.items())


def test_local_scientific_execution_refused(monkeypatch):
    monkeypatch.delenv('GITHUB_ACTIONS',raising=False)
    with pytest.raises(ValueError,match='no local scientific calls'):p.main(['all','--output',str(ROOT/'build/refused-local-original')])


def test_source_and_workflow_have_no_candidate_import_build_or_recapture():
    source=(ROOT/'tools/level_set_windows_reference.py').read_text();tree=ast.parse(source)
    imports=[n for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert not any((n.module or '').startswith('skimagexx') for n in imports if isinstance(n,ast.ImportFrom))
    assert not any(a.name.startswith('skimagexx') for n in imports if isinstance(n,ast.Import) for a in n.names)
    for text in ('longdouble_is_float64','longdouble_dtype_type_name','longdouble_itemsize','longdouble_nmant','c_long_bytes','radius_type_is_float64','before-original-freeze.json'):
        assert text in source
    workflow=(ROOT/'.github/workflows/level-set-windows-reference.yml').read_text()
    assert 'if: always()' in workflow and "python-version: '3.13.16'" in workflow
    assert 'pip install' not in workflow and 'cmake --build' not in workflow
    assert 'ORIGINAL ONLY' in workflow and 'level-set-Windows-original-only-NEW-HOST' in workflow
