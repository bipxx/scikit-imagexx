"""Inert cold/cached request guards. No actual scientific runtime prefix."""
import ast
import builtins
from pathlib import Path
import sys
import types

import pytest
from flood_fill_contract import invoke_reference, require_reference_authority


@pytest.mark.parametrize('api', ['flood', 'flood_fill'])
@pytest.mark.parametrize('cached', [False, True])
def test_refused_before_any_request(api, cached, monkeypatch):
    requests = []
    old = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in ('numpy', 'scipy', 'skimage', 'skimagexx'):
            requests.append(name)
            raise AssertionError('target/provider request forbidden')
        return old(name, *args, **kwargs)

    if cached:
        for name in ('numpy', 'scipy', 'skimage', 'skimagexx'):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setattr(builtins, '__import__', guarded)
    with pytest.raises(RuntimeError, match='UNBOUND'):
        invoke_reference(api + ':dtype-uint8', binding={'root_GO': 'invented'})
    assert requests == []


@pytest.mark.parametrize('binding', [None, {}, {'original_budget': 192}, {'state': 'PASS'}])
def test_truthy_binding_does_not_confer_capture_authority(binding):
    with pytest.raises(RuntimeError, match='UNBOUND'):
        require_reference_authority(binding)


def test_factory_is_lazy_and_no_target_resolution():
    text = Path(__file__).with_name('flood_fill_contract.py').read_text()
    tree = ast.parse(text)
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert [ast.unparse(n) for n in imports] == ['from math import prod']
    assert 'skimage' not in text and 'scipy' not in text
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'factory')
    assert ast.unparse(factory.body[1]) == 'import numpy as np'
