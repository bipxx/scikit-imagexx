"""Inert metadata preflights only; never scientific/native/corpus acceptance."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import rank_corpus_sdk_scope as c
from tools import public_qualification as q

SCRIPTS = ('rank_statistics', 'rank_percentile', 'rank_bilateral', 'rank_decisions')


@pytest.fixture
def current_capsule(tmp_path):
    names = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files'], text=True).splitlines()
    layout = q.current_install_layout(lambda n: (ROOT / n).read_bytes(), names, 'modern')
    # This capsule models current source install rules, not historical71 acceptance.
    assert layout['SDK_count'] == 84
    runtime = dict(source=dict(revision='a' * 40, files_sha256=dict.fromkeys(names, 'inert-authenticated-source')),
                   runtime=dict(sdk_sha256=dict.fromkeys(layout['SDK_members'], 'inert-authenticated-sdk')))
    marker = dict(schema='rank-corpus-sdk-scope-v1', scope=c.CURRENT_PRODUCT_SCOPE,
                  profile='modern', source_revision=runtime['source']['revision'])
    (tmp_path / c.MARKER).write_text(json.dumps(marker))
    return tmp_path, runtime, marker


@pytest.mark.parametrize('profile', ('windows', 'modern', 'minimum', 'debug'))
def test_current_scope_uses_exact_source_derived_sdk_names(current_capsule, profile):
    capsule, runtime, marker = current_capsule
    marker['profile'] = profile; (capsule / c.MARKER).write_text(json.dumps(marker))
    layout = q.current_install_layout(lambda n: (ROOT / n).read_bytes(), runtime['source']['files_sha256'], profile)
    runtime['runtime']['sdk_sha256'] = dict.fromkeys(layout['SDK_members'], 'inert-authenticated-sdk')
    assert c.validate_sdk_scope(ROOT, capsule, runtime) == c.CURRENT_PRODUCT_SCOPE


@pytest.mark.parametrize('fault', ('old68', 'old71', 'old83', 'stale247', 'foreign-name', 'extra-name', 'wrong-profile', 'missing-marker', 'missing-marker-false68', 'revision'))
def test_current_wrong_sdk_or_scope_cannot_pass(current_capsule, fault):
    capsule, runtime, marker = current_capsule; sdk = runtime['runtime']['sdk_sha256']
    if fault in ('old68', 'old71', 'old83', 'missing-marker-false68'):
        # Synthetic stale-size subsets; these are not historical SDK manifests.
        historical_count = {'old71':71, 'old83':83}.get(fault,68)
        for name in list(sdk)[historical_count:]: del sdk[name]
        assert len(sdk) == historical_count
    if fault == 'foreign-name': sdk['include/skimagexx/foreign.hpp'] = sdk.pop(next(iter(sdk)))
    if fault == 'extra-name': sdk['include/skimagexx/foreign.hpp'] = 'inert'
    if fault == 'wrong-profile': marker['profile'] = 'windows'
    if fault == 'stale247': marker['scope'] = 'current247'
    if fault == 'revision': marker['source_revision'] = 'b' * 40
    if fault.startswith('missing-marker'): (capsule / c.MARKER).unlink()
    else: (capsule / c.MARKER).write_text(json.dumps(marker))
    with pytest.raises(AssertionError): c.validate_sdk_scope(ROOT, capsule, runtime)


@pytest.mark.parametrize('marker', ([], {}, {'schema': 'rank-corpus-sdk-scope-v1'},
    {'schema': 'rank-corpus-sdk-scope-v1', 'scope': 'current236', 'profile': [], 'source_revision': 'a' * 40},
    {'schema': 'rank-corpus-sdk-scope-v1', 'scope': 'current236', 'profile': 'modern', 'source_revision': '../outside'}))
def test_marker_boundary_rejects_malformed_values(current_capsule, marker):
    capsule, runtime, _ = current_capsule; (capsule / c.MARKER).write_text(json.dumps(marker))
    with pytest.raises(AssertionError): c.validate_sdk_scope(ROOT, capsule, runtime)


def test_legacy68_is_preserved_and_wrong_legacy_count_rejected(tmp_path):
    runtime = dict(source=dict(files_sha256={}), runtime=dict(sdk_sha256=dict.fromkeys(map(str, range(68)), 'inert')))
    assert c.SDK_MEMBERS == 68 and c.validate_sdk_scope(ROOT, tmp_path, runtime) == 'h_extrema23'
    for wrong_count in (67, 71, 83, 84):
        runtime['runtime']['sdk_sha256'] = dict.fromkeys(map(str, range(wrong_count)), 'inert')
        with pytest.raises(AssertionError, match='original68'): c.validate_sdk_scope(ROOT, tmp_path, runtime)


@pytest.mark.parametrize('script', SCRIPTS)
@pytest.mark.parametrize('wrong_sdk', (False, True))
def test_actual_four_verifier_preflights_use_scope_after_runtime_authentication(current_capsule, script, wrong_sdk):
    capsule, runtime, _ = current_capsule
    source = (ROOT / 'benchmarks' / (script + '.py')).read_text(); tree = ast.parse(source)
    function = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'verify'))
    scope_index = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == 'validate_sdk_scope')
    function.body = function.body[:scope_index + 1]
    label = script.replace('_', '-'); events = []
    info = dict(schema=label + '-exact-offline-v1', files_sha256={}, corpus_manifest_sha256='inert',
                product_source='inert', runtime_manifest_sha256='inert')
    (capsule / (label + '-artifacts.json')).write_text(json.dumps(info))
    def authenticate(*_): events.append('authenticated-runtime'); return runtime
    def scoped(root, path, data):
        assert events == ['authenticated-runtime']; events.append('scope')
        return c.validate_sdk_scope(root, path, data)
    ns = dict(json=json, capsule_tree=lambda _: {}, artifacts=types.SimpleNamespace(sha=lambda _: 'inert', verify=authenticate),
              MANIFEST=None, source_fingerprint=lambda _: 'inert', ROOT=ROOT, validate_sdk_scope=scoped)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ROOT / 'benchmarks' / (script + '.py')), 'exec'), ns)
    if wrong_sdk:
        for name in list(runtime['runtime']['sdk_sha256'])[:3]: del runtime['runtime']['sdk_sha256'][name]
        with pytest.raises(AssertionError, match='file closure'): ns['verify'](capsule)
    else: ns['verify'](capsule)
    assert events == ['authenticated-runtime', 'scope']


@pytest.mark.parametrize('current', (False, True))
def test_actual_capsule_writer_marks_current_without_relabeling_legacy(monkeypatch, tmp_path, current):
    for name in ('runtime-wheels', 'product-wheels'): (tmp_path / name).mkdir()
    (tmp_path / 'runtime-wheels/numpy.whl').write_bytes(b'inert numpy')
    (tmp_path / 'product-wheels/product.whl').write_bytes(b'inert product')
    (tmp_path / 'package.json').write_text(json.dumps({'product_scope': c.CURRENT_PRODUCT_SCOPE if current else c.PRODUCT_SCOPE}))
    monkeypatch.setattr(q, 'wheel_metadata', lambda p: dict(name='numpy' if p.name == 'numpy.whl' else 'scikit-imagexx', version='inert', sha256=q.sha(p)))
    state = dict(runtime_pins={'numpy': 'inert'}, profile='modern', source=dict(revision='a' * 40))
    capsule = q.create_capsule(tmp_path, 'statistics', state)
    marker = capsule / c.MARKER
    assert marker.exists() == current
    if current:
        value = json.loads(marker.read_text())
        assert value == dict(schema='rank-corpus-sdk-scope-v1', scope=c.CURRENT_PRODUCT_SCOPE, profile='modern', source_revision='a' * 40)
