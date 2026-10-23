"""Namespace growth must preserve the image-operator benchmark's workloads.

The timer is replaced explicitly; this test produces no performance evidence.
"""
import importlib
import json
from pathlib import Path
import sys


def test_namespace_growth_retains_all_operator_workloads_without_timing(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'benchmarks'))
    harness=importlib.import_module('benchmarks.morphology')
    expected={'erosion','dilation','opening','closing','white_tophat','black_tophat',
              'binary_erosion','binary_dilation','binary_opening','binary_closing'}
    calls=[]

    def forbidden(*args,**kwargs):
        raise AssertionError('namespace-only function or real timer reached')

    def untimed_stub(reference,native,repeat,target_ms):
        calls.append((repeat,target_ms))
        return {'reference_ms':0.,'cpp_ms':0.,'speedup':1.,'unit_stub_no_measurement':True}

    monkeypatch.setattr(harness.xx.morphology,'__all__',
                        [*harness.xx.morphology.__all__,'future_generator'])
    monkeypatch.setattr(harness.xx.morphology,'future_generator',forbidden,raising=False)
    monkeypatch.setattr(harness.reference,'footprint_rectangle',forbidden)
    monkeypatch.setattr(harness,'measure',untimed_stub)
    run=importlib.import_module('run')
    monkeypatch.setattr(run,'batch',forbidden)
    output=tmp_path/'unit-only.json'
    monkeypatch.setattr(sys,'argv',['morphology.py','--size','3','--threads','1',
                                   '--repeat','1','--output',str(output)])
    old_workers=harness.xx.get_num_threads()
    old_policy=harness.xx.set_fallback_enabled(True)
    try:
        harness.main()
    finally:
        harness.xx.set_num_threads(old_workers)
        harness.xx.set_fallback_enabled(old_policy)
    report=json.loads(output.read_text(encoding='utf-8'))
    rows=report['results']
    assert len(rows)==len(calls)==120
    assert {r['api'] for r in rows}=={'morphology.'+name for name in expected}
    assert {r['dtype'] for r in rows}=={'bool','uint8','float32','float64'}
    assert {r['workload'] for r in rows}=={'rectangle-31','even-asymmetric','disk-5'}
    assert all(r['unit_stub_no_measurement'] and r['threads']==1 for r in rows)
    assert all(r['out_reused']==(r['workload']=='even-asymmetric') for r in rows)
