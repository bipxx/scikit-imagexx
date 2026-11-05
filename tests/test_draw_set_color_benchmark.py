import importlib.util,json,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_prospective_import_and_bounds():
    spec=importlib.util.spec_from_file_location('setcolor_benchmark',ROOT/'benchmarks/draw_set_color.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    contract=json.loads((ROOT/'coverage/contracts/draw-set-color-benchmark.json').read_text())
    assert len(contract['cases'])==8 and contract['raw_samples']==112
    assert all(module.source_safe(x)['image_elements']<=442368 for x in contract['cases'])
def test_timer_boundary():
    source=(ROOT/'benchmarks/draw_set_color.py').read_text();tree=ast.parse(source)
    assert 'sys.setprofile(prior)'in source and 'actual.tobytes()==expected.tobytes()'in source
    assert len([n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='measure'])==1

def test_source_closure():
    import hashlib
    contract=json.loads((ROOT/'coverage/contracts/draw-set-color-benchmark.json').read_text())
    for name,pin in contract['source_closure'].items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin
