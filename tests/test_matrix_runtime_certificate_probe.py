"""Inert metadata guards, never certificate/native acceptance or image calls."""
import ast,base64,csv,hashlib,importlib.util,io,json,stat,sys,types,zipfile
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('certificate_probe',ROOT/'tools/matrix_runtime_certificate_probe.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


@pytest.mark.parametrize('name',('../escape','/absolute','C:/outside','safe/../../outside','a\\b'))
def test_archive_paths_fail_closed(name):
    with pytest.raises(ValueError):c.safe_member(name)


def wheel(payload=b'exact source',symlink=False):
    out=io.BytesIO();name='skimagexx/__init__.py';record='pkg.dist-info/RECORD'
    encoded='sha256='+base64.urlsafe_b64encode(hashlib.sha256(b'exact source').digest()).decode().rstrip('=')
    with zipfile.ZipFile(out,'w') as z:
        info=zipfile.ZipInfo(name)
        if symlink:info.external_attr=(stat.S_IFLNK|0o777)<<16
        z.writestr(info,payload);z.writestr(record,f'{name},{encoded},12\n{record},,\n')
    return out.getvalue()


def test_exact_original_record_metadata_is_required():
    result=c.wheel_records(wheel())
    assert result['skimagexx/__init__.py']['sha256']==hashlib.sha256(b'exact source').hexdigest()
    with pytest.raises(ValueError,match='payload'):c.wheel_records(wheel(b'wrong'))
    with pytest.raises(ValueError,match='symlink'):c.wheel_records(wheel(symlink=True))


def test_natural_first_return_traces_unchanged_metadata_function(tmp_path):
    source=tmp_path/'helper.py'
    text="def _runtime_profile():\n    corename='NotAudited'\n    if corename!='Known':\n        return None\n    return 'unexpected'\n"
    source.write_text(text);ns={};exec(compile(text,str(source),'exec'),ns)
    original=ns['_runtime_profile'];code=original.__code__
    wrapper=types.SimpleNamespace(__file__=str(source),_runtime_profile=original)
    value,events=c.trace_natural_profile(wrapper)
    assert value is None and original.__code__ is code and source.read_text()==text
    event=next(e for e in events if e['function']=='_runtime_profile' and e['event']=='return')
    assert event['line']==4 and event['return'] is None and event['locals']['corename']=='NotAudited'
    assert any("corename!='Known'" in line for line in event['source_context'])
    assert sys.gettrace() is None


def test_payload_never_retries_to_turn_refusal_into_success(monkeypatch):
    calls=[];wrapper=object()
    def natural(value):
        assert value is wrapper;calls.append('natural')
        return None,[{'function':'_authenticate_runtime','event':'return','line':42,'return':None,'locals':{}}]
    def post(value,np):assert value is wrapper;calls.append('post-read-only');return {'late_threads':1}
    monkeypatch.setattr(c,'trace_natural_profile',natural);monkeypatch.setattr(c,'post_attempt_getters',post)
    result=c.certificate_payload(wrapper,None)
    assert calls==['natural','post-read-only']
    assert result['natural_profile'] is None and result['certificate_retried'] is False
    assert result['first_refusal']['line']==42


@pytest.mark.parametrize('core,detail',(('Unknown','unassessed'),('Known','threads()!=1')))
def test_refusal_does_not_relabel_later_threads_as_first_attempt_success(monkeypatch,core,detail):
    wrapper=types.SimpleNamespace(_RUNTIME_CORES={'pinned':{'Known'}})
    event={'function':'_runtime_profile','event':'return','line':100,'return':None,
           'locals':{'corename':core,'digest':'pinned'}}
    monkeypatch.setattr(c,'trace_natural_profile',lambda w:(None,[event]))
    monkeypatch.setattr(c,'post_attempt_getters',lambda w,np:{'later_threads':1})
    result=c.certificate_payload(wrapper,None)
    assert detail in result['predicate_detail'] and result['natural_profile'] is None


def test_payload_does_not_invoke_arbitrary_repr_or_str():
    class Poison:
        def __repr__(self):raise AssertionError('provider side effect')
        def __str__(self):raise AssertionError('provider side effect')
    assert c.safe_value(Poison())=={'type':'Poison'}
    assert c.safe_value(b'Haswell')['hex']==b'Haswell'.hex()


def test_source_pins_are_distinct_raw_and_canonical_artifact_facts():
    # The controller must be mergeable without the unqualified product source.
    # Actual donor bytes are checked by extract_inputs, not imported here.
    assert c.HELPER_RAW_SHA=='8f9937b65467265d4a6f55921af19c163083f02b08d45988f7b76e8904f235b7'
    assert c.HELPER_SHA=='deb219b0cfb0f38d04e53d13e09ebfc40196f29de97c1e3c4c3d4273e416e9b8'
    assert c.FAILED_SOURCE=='a567c5d775bdc82f3b3aba7a1e8ba9a33fc50679'
    assert c.ARTIFACT_ID==12569527308 and c.ARTIFACT_SIZE==400047839


def test_import_order_is_checked_from_archived_source_before_probe():
    source='def admission():\n    for module in '+repr(c.ADMISSION_IMPORTS)+':\n        pass\n'+''.join("    importlib.import_module('skimagexx."+name+"')\n" for name in c.ADMISSION_PUBLIC_IMPORTS)
    c.verify_admission_imports(source)
    with pytest.raises(ValueError):c.verify_admission_imports(source.replace("'_matrix_transform'","'unverified_context'"))
    with pytest.raises(ValueError,match='public admission'):
        c.verify_admission_imports(source.replace("'skimagexx.coverage'","'skimagexx.unverified_context'"))
    probe=next(n for n in ast.parse((ROOT/'tools/matrix_runtime_certificate_probe.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='probe')
    imports=[ast.unparse(n) for n in probe.body if isinstance(n,ast.Import)]
    assert imports==['import skimagexx as xx','import numpy as np']


def test_helper_validator_requires_both_byte_identities(monkeypatch):
    raw=b'# inert metadata fixture\r\n';canonical=raw.replace(b'\r\n',b'\n')
    monkeypatch.setattr(c,'HELPER_RAW_SHA',hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(c,'HELPER_SHA',hashlib.sha256(canonical).hexdigest())
    c.verify_helper_bytes(raw)
    with pytest.raises(ValueError,match='raw'):c.verify_helper_bytes(canonical)
    monkeypatch.setattr(c,'HELPER_SHA','0'*64)
    with pytest.raises(ValueError,match='canonical'):c.verify_helper_bytes(raw)


def test_no_algorithm_build_timer_force_or_library_replacement_calls():
    tree=ast.parse((ROOT/'tools/matrix_runtime_certificate_probe.py').read_text())
    forbidden={'matmul','dot','matrix_transform','matrix_transform_run','matrix_transform_run_v2',
               'set_num_threads','perf_counter','timeit','monotonic','process_time','cache_clear'}
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden for n in ast.walk(tree))
    top=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert not any('numpy' in ast.unparse(n) or 'skimage' in ast.unparse(n) for n in top)
    text=ast.unparse(tree)
    assert 'certificate_retried=False' in text and 'natural_profile=result' in text
    assert "environment['OPENBLAS_CORETYPE']" not in text


def test_hosted_gate_and_signed_download_keep_tokens_private():
    source=ast.parse((ROOT/'tools/matrix_runtime_certificate_probe.py').read_text())
    main=ast.unparse(next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='main'))
    assert main.index("'fresh hosted diagnostic only")<main.index('fetch_artifact(')
    download=ast.unparse(next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='fetch_artifact'))
    assert 'NoRedirect' in download and 'urllib.request.urlopen(url)' in download
    assert 'urllib.request.urlopen(url, headers' not in download


def test_workflow_is_manual_artifact_only_new_host_and_retains_failures():
    text=(ROOT/'.github/workflows/matrix-runtime-certificate-probe.yml').read_text()
    for token in ('workflow_dispatch:','actions: read',"python-version: '3.13.16'",'if: always()', 'NEW-HOST'):
        assert token in text
    for forbidden in ('pip install','cmake','pytest','OPENBLAS_CORETYPE:','gh workflow run','genuine-prefix/**'):
        assert forbidden not in text
