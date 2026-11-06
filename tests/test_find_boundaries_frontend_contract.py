"""Source-only boundary/closure checks; no NumPy or target imports."""
import ast
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_find_boundaries_frontend_source_closure():
    value=json.loads((ROOT/'coverage/contracts/find-boundaries-frontend-v1.json').read_bytes())
    assert value['api']=='segmentation.find_boundaries'
    for name,pin in value['source_canonical_pins'].items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin
    assert value['reference_cases']==32 and value['admitted_returns_per_profile']==28
    assert value['source_error_rows_per_profile']==3 and value['rounded_control_only_per_profile']==1

def test_find_boundaries_frontend_no_original_or_array_arithmetic_forwarding():
    tree=ast.parse((ROOT/'python/skimagexx/_find_boundaries.py').read_bytes())
    assert any(isinstance(n,ast.FunctionDef)and n.name=='find_boundaries'for n in tree.body)
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):assert not any(a.name.split('.')[0]in('skimage','scipy')for a in node.names)
        if isinstance(node,ast.ImportFrom):assert (node.module or'').split('.')[0]not in('skimage','scipy')
        if isinstance(node,ast.Call)and isinstance(node.func,ast.Attribute)and isinstance(node.func.value,ast.Name)and node.func.value.id=='np':
            assert node.func.attr in('empty','empty_like')
    text=(ROOT/'src/bind_find_boundaries.cpp').read_text(encoding='utf-8')
    assert 'find_boundaries_into(view,parameters,destination,strides,workers)'in text
    assert 'check_array_storage(a)'in text and 'py::gil_scoped_release'in text
