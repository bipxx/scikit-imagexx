"""Prospective corpus and import-context checks without image execution."""
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_circle_benchmark_cases_and_bound_are_prospective():
    reference_raw = (ROOT/'coverage/contracts/circle-perimeter.json').read_bytes()
    reference = json.loads(reference_raw)
    contract = json.loads((ROOT/'coverage/contracts/circle-perimeter-benchmark.json').read_bytes())
    assert contract['cases'] == reference['benchmark_cases'] and len(contract['cases']) == 8
    assert contract['reference_contract_sha256'] == hashlib.sha256(reference_raw).hexdigest()
    assert {case['method'] for case in contract['cases']} == {'bresenham', 'andres'}
    assert max(16*(case['radius']+1) for case in contract['cases']) > 100_000
    assert contract['threads'] == 1 and contract['repeat'] == 7 and contract['raw_samples'] == 112
    assert contract['warmup_groups'] == 16 and contract['target_ms'] == 100
    method = json.loads((ROOT/'coverage/contracts/draw-coordinates-benchmark.json').read_bytes())
    assert contract['method_sha256'] == method['method_sha256']
    workflow = (ROOT/'.github/workflows/ci.yml').read_text()
    measure_job = workflow.split('  measure-minimum:', 1)[1].split('  debug-gil:', 1)[0]
    for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS'):
        assert name+": '1'" in measure_job



def test_circle_benchmark_module_import_has_no_scientific_side_effects(monkeypatch):
    original = builtins.__import__

    def checked(name, *args, **options):
        assert name.split('.')[0] not in {'numpy', 'scipy', 'skimage', 'skimagexx'}
        return original(name, *args, **options)

    monkeypatch.setattr(builtins, '__import__', checked)
    spec = importlib.util.spec_from_file_location('circle_benchmark_import_probe', ROOT/'benchmarks/circle_perimeter.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
