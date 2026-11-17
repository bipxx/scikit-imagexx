"""Saved-raw/source/inert private comparison controls, zero science."""
import ast,base64,copy,hashlib,importlib.util,json,os,struct,subprocess,sys,types,zlib
from datetime import datetime,timedelta,timezone
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];FILE=ROOT/'tests/mark_wide_outer_private_compare.py'
S=importlib.util.spec_from_file_location('wide_private_test',FILE);C=importlib.util.module_from_spec(S);S.loader.exec_module(C)
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def test_exact_admitted_eight_rows_and_unchanged_whole_pipeline_sources():
    transport,rows=C.load_saved();contract=json.loads((ROOT/'coverage/contracts/mark-wide-outer-private-bridge-v1.json').read_bytes())
    assert transport['original_budget_remaining']==0 and contract['root_GO']is contract['component']is None
    assert set(rows)=={'minimum','modern'}and all(len(v)==4 for v in rows.values())
    for profile,records in rows.items():
        assert [r['id']for r in records]==contract['case_ids']
        assert sum(len(zlib.decompress(base64.b64decode(r['result']['raw_zlib_base64'])))for r in records)==456
        for r in records:
            assert r['before']==r['after']and r['error']is None and not r['warnings']and not r['FP_events']
            assert r['result_type']=={'module':'numpy','name':'ndarray'}and r['result']['own']
            assert r['target_immediate']['after']['mxcsr']==r['target_immediate']['before']['mxcsr']-1
            assert r['target_immediate']['after']['status']==(32 if profile=='minimum'else 0)
    for name,pin in(contract['source_pins']|contract['existing_source_reuse']).items():
        raw=(ROOT/name).read_bytes()
        if name in ('src/find_boundaries.cpp','src/mark_boundaries.cpp','src/bind_mark_boundaries.cpp'):
            from tools.mark_lbp_fp_context import VI_PUBLIC_MATMUL_CONTEXT_SHA256
            from tools.vi_public_matmul_context import speed_predecessor_bytes
            raw=speed_predecessor_bytes(ROOT,name,raw,VI_PUBLIC_MATMUL_CONTEXT_SHA256)
        if name=='src/find_boundaries.cpp':
            phase=json.loads((ROOT/'coverage/contracts/mark-wide-comparison-fp-v1.json').read_bytes())
            assert hashlib.sha256(raw).hexdigest()==phase['kernel_current_raw_sha256']
            for f in phase['inverse_fragments']:
                assert raw.count(f['current'].encode())==1;raw=raw.replace(f['current'].encode(),f['predecessor'].encode(),1)
        assert hashlib.sha256(raw).hexdigest()==pin,name


@pytest.mark.parametrize('profile',('minimum','modern'))
def test_seed_only_captured_input_words_keeps_current_nonportable_state(profile):
    _,rows=C.load_saved();expected=rows[profile][0]['target_immediate']['before']
    current=bytearray(range(32));current[8:10]=b'\xff\xff'
    seeded=C.seed_bytes(bytes(current),expected)
    for name,off,width in [('control',0,2),('status',4,2),('mxcsr',28,4)]:assert int.from_bytes(seeded[off:off+width],'little')==expected[name]
    assert seeded[2:4]==current[2:4]and seeded[6:28]==current[6:28]


@pytest.mark.parametrize('fault',('none','result','FP-before','FP-after','handler-id-only','policy'))
def test_strict_assessment_does_not_mask_hardware_or_payload(fault):
    _,rows=C.load_saved();expected=rows['minimum'][0];actual=copy.deepcopy(expected);immediate=copy.deepcopy(expected['target_immediate'])
    if fault=='result':actual['result']['dtype']='uint8'
    elif fault=='FP-before':immediate['before']['status']^=1
    elif fault=='FP-after':immediate['after']['mxcsr']^=1
    elif fault=='handler-id-only':immediate['numpy_before']['handler_id']+=1
    elif fault=='policy':immediate['numpy_after']['policy']['invalid']='ignore'
    differences=C.assess(expected,actual,immediate)
    if fault in('none','handler-id-only'):assert not differences
    else:assert differences
    assert expected==rows['minimum'][0] # Original not normalized or mutated.


def test_actual_CLI_without_GO_cannot_import_science_or_create_output(tmp_path):
    out=tmp_path/'absent';p=subprocess.run([sys.executable,str(FILE),'--profile','minimum','--output',str(out)],capture_output=True,text=True)
    assert p.returncode!=0 and 'NO_ROOT_GO'in p.stderr and not out.exists()


@pytest.mark.parametrize('fault',('none','calls','original','worker','runtime','component','sourcepin','deadline','otherprofile','outputexists'))
def test_actual_bound_Gate_with_inert_component_only(tmp_path,monkeypatch,fault):
    transport,rows=C.load_saved();module=tmp_path/'_mark_boundaries_core.so';module.write_bytes(b'INERT_NON_BINARY')
    output=tmp_path/'out';go=dict(status='ROOT_AUTHORIZED_MARK_WIDE_PRIVATE_SAVED4_ONCE',profile='minimum',script_sha256=H(FILE),fixture_sha256=H(C.FIXTURE),case_ids=[r['id']for r in rows['minimum']],native_public_calls=4,workers=1,original_calls=0,timing=0,builds=0,runtime=transport['profiles']['minimum']['runtime'],helper=transport['profiles']['minimum']['helper'],output=str(output),deadline_utc=(datetime.now(timezone.utc)+timedelta(minutes=2)).isoformat(),source='a'*40,git_argv=['INERT'],source_files={'tests/mark_wide_outer_private_compare.py':H(FILE)},component=dict(path=str(module),bytes=module.stat().st_size,sha256=H(module)))
    if fault=='calls':go['native_public_calls']=5
    elif fault=='original':go['original_calls']=1
    elif fault=='worker':go['workers']=4
    elif fault=='runtime':go['runtime']['numpy']='not-retained'
    elif fault=='component':go['component']['sha256']='0'*64
    elif fault=='sourcepin':go['source_files']['tests/mark_wide_outer_private_compare.py']='0'*64
    elif fault=='deadline':go['deadline_utc']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
    elif fault=='otherprofile':go['profile']='modern'
    elif fault=='outputexists':output.mkdir()
    path=tmp_path/'GO.json';path.write_text(json.dumps(go));args=types.SimpleNamespace(root_go=path,root_go_sha256=H(path),profile='minimum',output=output)
    for n in('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS','PYTHONDONTWRITEBYTECODE'):monkeypatch.setenv(n,'1')
    monkeypatch.setattr(C.subprocess,'check_output',lambda a,**k:'a'*40 if a[-1]=='HEAD'else'')
    if fault=='none':assert C.validate(args)[0]==go
    else:
        with pytest.raises(AssertionError):C.validate(args)


def test_no_aftercall_flag_clear_or_recreated_overlay_in_driver():
    source=FILE.read_text();tree=ast.parse(source)
    assert 'feclearexcept'not in source and '_mm_setcsr'not in source and 'seterr(all='not in source
    assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr in('where','paint','mark_boundaries_paint','mark_boundaries_mask')for n in ast.walk(tree))
    assert "mark.mark_boundaries"in source and "find.find_boundaries"in source
