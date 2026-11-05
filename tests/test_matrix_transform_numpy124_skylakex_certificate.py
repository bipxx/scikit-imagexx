"""Source-only exact-binary certificate and synthetic metadata refusal controls."""
import ast
import hashlib
import json
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'python/skimagexx/_matrix_transform.py'
CERT = ROOT / 'coverage/evidence/matrix-numpy124-skylakex-certificate-v1.json'
DIGEST = '554bde1d8a0c71d8dc21ae74de05c44da4fff5dbc6791a819f6acf5adfe90bd9'
COMMENT = '    # Exact F64 equivalence: coverage/evidence/matrix-numpy124-skylakex-certificate-v1.json\n'

def test_exact_inverse_preserves_all_math_and_runtime_guards():
    cert = json.loads(CERT.read_text())
    source = SOURCE.read_bytes().replace(b'\r\n', b'\n')
    assert hashlib.sha256(source).hexdigest() == cert['source_after_canonical_lf_sha256']
    assert source.count(COMMENT.encode()) == 1
    restored = source.replace(COMMENT.encode(), b'').replace(
        b"frozenset(('Haswell', 'Cooperlake', 'Zen', 'SkylakeX'))",
        b"frozenset(('Haswell', 'Cooperlake', 'Zen'))")
    assert hashlib.sha256(restored).hexdigest() == cert['source_before_canonical_lf_sha256']
    migration = json.loads((ROOT/'coverage/evidence/matrix-transform-public-assertion-migration.json').read_text())
    record = migration['qualified_implementation_updates']['python/skimagexx/_matrix_transform.py']
    assert record['current_canonical_lf_sha256'] == cert['source_after_canonical_lf_sha256']
    assert record['original_canonical_lf_sha256'] == 'deb219b0cfb0f38d04e53d13e09ebfc40196f29de97c1e3c4c3d4273e416e9b8'
    assert cert['independent_review']['sha256'] == '796729f6f3e65da4d0c90a7fb7726dfe2e8d54be32be3078b2549e324a514b9e'
    assert cert['scope']['natural_SkylakeX_execution_observed_by_this_packet'] is False

def scope(core, threads=1, auth=True, unforced=True):
    tree = ast.parse(SOURCE.read_text())
    nodes = [n for n in tree.body if
             isinstance(n, ast.FunctionDef) and n.name == '_runtime_profile' or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_RUNTIME_CORES' for t in n.targets)]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    def forbidden(*args):
        raise AssertionError('Windows driver rescue must not admit this Linux certificate')
    namespace.update(_unforced=lambda: unforced,
                     _authenticate_runtime=lambda: ('1.24.4', DIGEST, lambda: core, lambda: threads, None, ()) if auth else None,
                     _windows_skylakex_driver=forbidden)
    return namespace

@pytest.mark.parametrize('core', [b'Haswell', b'Cooperlake', b'Zen', b'SkylakeX'])
def test_exact_binary_synthetic_core_admission(core):
    state = scope(core)
    assert state['_runtime_profile']() == '1.24.4'
    assert state['_RUNTIME_CORENAME'] == core.decode('ascii')
    assert state['_THREAD_QUERY']() == 1
    assert state['_RUNTIME_DRIVER'] is None

@pytest.mark.parametrize('core,threads,auth,unforced', [
    (b'Unknown', 1, True, True), (None, 1, True, True),
    (b'SkylakeX', 2, True, True), (b'SkylakeX', 4, True, True),
    (b'SkylakeX', 1, False, True), (b'SkylakeX', 1, True, False)])
def test_synthetic_refusals_preserve_cleared_snapshot(core, threads, auth, unforced):
    state = scope(core, threads, auth, unforced)
    assert state['_runtime_profile']() is None
    assert state['_THREAD_QUERY'] is None
    assert state['_RUNTIME_CORENAME'] is None
    assert state['_RUNTIME_DRIVER'] is None
