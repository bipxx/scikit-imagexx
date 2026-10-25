"""Execute the real provenance assertion with metadata stand-ins only."""
import ast
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def test_real_source_migration_assertion_without_scientific_imports():
    path = ROOT / 'tests/test_reconstruction_contracts.py'
    matrix = ROOT / 'coverage/contracts/reconstruction.json'
    data = json.loads(matrix.read_bytes())
    tree = ast.parse(path.read_bytes())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'test_source_signature_and_fixed_scope')
    def original(seed, mask, method='dilation', footprint=None, offset=None):
        raise AssertionError('metadata test must never call an image operation')
    namespace = dict(ROOT=ROOT, MATRIX=matrix, __file__=str(path), json=json,
                     hashlib=hashlib, Path=Path, inspect=inspect, ORIGINAL=original,
                     skimage=SimpleNamespace(__version__='0.26.0'),
                     cases=lambda: data['cases'])
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    namespace[node.name]()


def test_scientific_AST_and_frozen_source_mapping_are_exact():
    record = json.loads((ROOT / 'coverage/evidence/reconstruction-public-assertion-migration.json').read_bytes())
    raw = record['original_source_UTF8'].encode()
    assert hashlib.sha256(raw).hexdigest() == record['original_raw_sha256']
    assert hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest() == record['original_canonical_lf_sha256']
    assert record['original_raw_sha256'] != record['original_canonical_lf_sha256']
