"""Installed public identity; scientific cases live in unchanged frozen suites."""
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path

import pytest
import skimagexx as xx
from skimagexx.coverage import native_functions, matrix_transform_contract

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def public_matrix():
    assert not os.environ.get('MATRIX_TRANSFORM_PRIVATE_BACKEND'), 'private adapter forbidden in public proof'
    module=importlib.import_module('skimagexx._matrix_transform');core=xx._core
    assert module._core is core
    assert xx.set_num_threads is core.set_num_threads and xx.get_num_threads is core.get_num_threads
    assert core.get_num_threads()==1
    assert module.matrix_transform is xx.transform.matrix_transform
    assert module.matrix_transform.__globals__['_core'] is core
    assert inspect.signature(module.matrix_transform)==inspect.Signature([
        inspect.Parameter('coords',inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter('matrix',inspect.Parameter.POSITIONAL_OR_KEYWORD)])
    assert tuple(native_functions()).count('transform.matrix_transform')==1
    assert xx.transform.__all__.count('matrix_transform')==1
    package=Path(xx.__file__).resolve().parent
    assert Path(module.__file__).resolve().parent==Path(core.__file__).resolve().parent==package
    for name in ('matrix_transform_run','matrix_transform_run_v2','matrix_transform_completed','matrix_transform_v2_completed'):
        entry=getattr(core,name)
        assert inspect.isbuiltin(entry) and entry.__module__==core.__name__ and entry.__name__==name
    assert core.matrix_transform_interface=='matrix-transform-math-v1' and type(core.matrix_transform_capabilities) is int and core.matrix_transform_capabilities==31
    assert core.matrix_transform_v2_interface=='matrix-transform-math-v2' and type(core.matrix_transform_v2_capabilities) is int and core.matrix_transform_v2_capabilities==63
    return module,core


def test_actual_public_source_is_frozen_520_and_scope_is_partial(public_matrix):
    module,core=public_matrix
    migration=json.loads((ROOT/'coverage/evidence/matrix-transform-public-assertion-migration.json').read_bytes())
    assert hashlib.sha256(Path(module.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==migration['frozen_implementation']['python/skimagexx/_matrix_transform.py']
    scope=matrix_transform_contract()
    assert scope['implementation']=='native-partial'
    assert scope['actual_runtime_evidence']==('Haswell','Zen')
    assert scope['prepared_only_runtime_evidence']==('Cooperlake','SkylakeX')


@pytest.mark.parametrize('enabled',(False,True))
def test_fallback_policy_does_not_replace_public_matrix_identity(public_matrix,enabled):
    module,_=public_matrix;old=xx.set_fallback_enabled(enabled)
    try:assert xx.transform.matrix_transform is module.matrix_transform
    finally:xx.set_fallback_enabled(old)
