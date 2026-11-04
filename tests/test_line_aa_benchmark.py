"""Frozen benchmark inputs and import-only checks; no image execution."""
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_line_aa_benchmark_is_source_safe_and_prospective():
    raw = (ROOT/'coverage/contracts/line-aa.json').read_bytes()
    reference = json.loads(raw)
    contract = json.loads((ROOT/'coverage/contracts/line-aa-benchmark.json').read_bytes())
    assert contract['reference_contract_sha256'] == hashlib.sha256(raw).hexdigest()
    assert contract['cases'] == reference['benchmark_cases'] and len(contract['cases']) == 9
    assert contract['threads'] == 1 and contract['repeat'] == 7
    assert contract['raw_samples'] == 9*2*7 and contract['warmup_groups'] == 18
    assert contract['target_ms'] == 100 and contract['warmup_min_ms'] == 250
    method = json.loads((ROOT/'coverage/contracts/draw-coordinates-benchmark.json').read_bytes())
    assert contract['method_sha256'] == method['method_sha256']
    for case in contract['cases']:
        assert case['api'] == 'draw.line_aa' and not case['kwargs']
        r0,c0,r1,c1 = case['args']
        assert all(-(1<<31)+2 <= v <= (1<<31)-2 for v in case['args'])
        dr,dc = abs(r1-r0),abs(c1-c0)
        assert dr*dr+dc*dc <= (1<<31)-1
        assert 9*(dr+dc+1) <= contract['max_output_elements']
    assert max(9*(abs(s['args'][2]-s['args'][0])+abs(s['args'][3]-s['args'][1])+1)
               for s in contract['cases']) > reference['max_output_elements']


def test_line_aa_benchmark_import_has_no_scientific_side_effects(monkeypatch):
    original = builtins.__import__

    def checked(name,*args,**options):
        assert name.split('.')[0] not in {'numpy','scipy','skimage','skimagexx'}
        return original(name,*args,**options)

    monkeypatch.setattr(builtins,'__import__',checked)
    spec = importlib.util.spec_from_file_location('line_aa_benchmark_import_probe',ROOT/'benchmarks/line_aa.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
