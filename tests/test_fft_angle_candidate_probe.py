"""Metadata and inert private-overlay checks; no scientific calls or builds."""
import ast, datetime, hashlib, importlib.util, json, math, sys, types
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('angle_candidate_probe',ROOT/'tools/fft_angle_candidate_probe.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


@pytest.mark.parametrize('fail',(False,True))
def test_overlay_restores_actual_entries_without_spoofing(fail):
    old={name:object() for name in p.ENTRIES};core=types.SimpleNamespace(**old)
    module=types.SimpleNamespace(__name__='math',**{name:math.sin for name in p.ENTRIES})
    try:
        with p.overlay(core,module):
            assert all(getattr(core,n) is math.sin for n in p.ENTRIES)
            assert math.sin.__module__=='math'
            if fail:raise RuntimeError('inert failure')
    except RuntimeError:
        assert fail
    assert all(getattr(core,n) is value for n,value in old.items())


@pytest.mark.parametrize('bad',('python_wrapper','wrong_origin'))
def test_overlay_rejects_nonbuiltin_or_spoof_before_mutation(bad):
    old={name:object() for name in p.ENTRIES};core=types.SimpleNamespace(**old)
    module=types.SimpleNamespace(__name__='math',**{name:math.sin for name in p.ENTRIES})
    if bad=='python_wrapper':module.fft_c2c=lambda:None
    else:module.__name__='_fft_angle_candidate'
    with pytest.raises(ValueError,match='true builtin origin'):
        with p.overlay(core,module):pytest.fail('invalid overlay accepted')
    assert all(getattr(core,n) is v for n,v in old.items())


def test_pre_candidate_boundary_refs_are_fixed_and_independent():
    model=p.load('metadata_angle_model',ROOT/'tools/fft_angle_contract.py');ns=set(model.DENOMINATORS)
    for b in range(1,65):ns.update(n for d in (-2,-1,0,1,2) if 1<=(n:=(1<<b)+d)<1<<64)
    import struct
    rows=[]
    for n in sorted(ns):
        exact=model.angle_fraction(n,64);assert exact==model.Fraction(model.angle_decimal(n,64))
        rows.append(str(n)+' '+struct.pack('>d',float(exact)).hex()+'\n')
    assert len(rows)==363 and hashlib.sha256(''.join(rows).replace('\n','\r\n').encode()).hexdigest()==p.REF_SHA


def test_plan_has_private_scope_no_build_or_scientific_import(tmp_path,capsys):
    before=set(sys.modules)
    assert p.main(['plan','--output',str(ROOT/'build/angle-plan-only')])==0
    plan=json.loads(capsys.readouterr().out)
    assert plan['candidate']=='08c83244c3cbc9f93b44a051b3a56452b4d5213e'
    assert plan['leaf_tus']==2 and plan['tiny_init_tus']==1 and plan['lpi_specs']==80 and plan['bw_specs']==87
    assert plan['experimental_overlay'] and not plan['whole_product_acceptance']
    assert not any(n.split('.')[0] in {'numpy','scipy','skimage','skimagexx'} for n in set(sys.modules)-before)


def test_local_scientific_execution_refused(monkeypatch):
    monkeypatch.delenv('GITHUB_ACTIONS',raising=False)
    with pytest.raises(ValueError,match='hosted private experiment only'):
        p.main(['all','--output',str(ROOT/'build/angle-refused-local')])


def test_expiry_retains_false_decision_before_transport(monkeypatch,tmp_path):
    monkeypatch.setenv('GITHUB_ACTIONS','true');monkeypatch.setattr(p,'ROOT',tmp_path)
    monkeypatch.setattr(p,'DEADLINE','2000-01-01T00:00:00+00:00')
    monkeypatch.setattr(p.p,'transport',lambda:pytest.fail('expired candidate fetched'))
    out=tmp_path/'build/expired'
    with pytest.raises(ValueError,match='deadline expired'):
        p.main(['all','--output',str(out)])
    assert not json.loads((out/'decision.json').read_text())['keep']
    assert 'deadline expired' in (out/'failure.txt').read_text()


def test_source_uses_frozen_predicates_true_origin_and_fixed_compile_scope():
    source=(ROOT/'tools/fft_angle_candidate_probe.py').read_text()
    for text in ('lpi.run_case(', 'lpi.validate_reference(', 'lpi.compare_steps(', 'bw.reference_case(', 'bw.compare_result(', 'p.observe(', '19.51.36260', '/REMOVE:skimagexx.dir\\\\Release\\\\fft.obj'):
        assert text in source,text
    assert not any(isinstance(n,ast.Attribute) and n.attr=='__module__' and isinstance(n.ctx,ast.Store) for n in ast.walk(ast.parse(source)))
    assert "for name in ('fft','simd_fft_avx2')" in source
    assert 'full_product_acceptance=True' not in source


def test_workflow_retains_failure_and_uses_matching_host_compiler():
    text=(ROOT/'.github/workflows/fft-angle-candidate-probe.yml').read_text()
    assert 'if: always()' in text and 'fetch-depth: 0' in text
    assert 'VsDevCmd.bat' in text and 'python-version: \'3.13.16\'' in text
    assert 'fft-angle-single-candidate-NEW-HOST' in text
    assert 'cmake --build' not in text and 'pip install' not in text
