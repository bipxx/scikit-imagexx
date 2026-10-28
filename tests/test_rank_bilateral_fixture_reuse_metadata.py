"""Inert source-inverse guards for bilateral fixture reuse; no scientific imports."""
import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/test_rank_bilateral_contracts.py"
EVIDENCE = ROOT / "coverage/evidence/rank-bilateral-fixture-assertion-migration.json"
WITNESS = ROOT / "coverage/evidence/rank-bilateral-fixture-reuse.json"


def validate_inverse(source):
    # Execute only the actual new source-proof block, without importing the test.
    tree = ast.parse(SOURCE.read_text())
    guard = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "test_integrity_safe_scope_native0")
    start = next(i for i, n in enumerate(guard.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "reuse")
    end = next(i for i, n in enumerate(guard.body) if i > start and isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "current")
    block = ast.Module(body=guard.body[start:end+1], type_ignores=[])
    namespace = dict(ROOT=ROOT, current=source, evidence=json.loads(EVIDENCE.read_text()), ast=ast, copy=copy, hashlib=hashlib, json=json,
                     text_sha=lambda p: hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest())
    exec(compile(block, str(SOURCE), "exec"), namespace)


def test_qualified_source_and_historical_evidence_are_unchanged():
    witness = json.loads(WITNESS.read_text())
    assert witness["qualified_source_revision"] == "4e657ee64fb71e4d707d32e9539fdb4d7cc528af"
    assert hashlib.sha256(witness["qualified_fixture_utf8"].encode()).hexdigest() == "9fb6411bb69a92bbc40bc4ebab37f553328476a19d010c895ea7c0c2dc9d1f16"
    assert hashlib.sha256(EVIDENCE.read_bytes()).hexdigest() == "860dcf68c2a37c18b6a3ecea3385f6cb6d55490400a311f71502473bcf4c9800"
    validate_inverse(SOURCE.read_text())


@pytest.mark.parametrize("before,after", [
    ("def observe(function,name,p):", "def observe(function,name,p):\n pass"),
    ("rows=[]", "rows=[None]"),
    ("@pytest.mark.parametrize('workers',(1,4))", "@pytest.mark.parametrize('workers',(1,))"),
    ("\n   assert actual==frozen['observations']", "\n   assert actual!=frozen['observations']"),
    ("\ndef test_strict_pending_never_counts_forwarding(case,namespace,workers,frozen):", "\ndef test_strict_pending_never_counts_forwarding(case,namespace,workers):"),
])
def test_inverse_rejects_observer_factory_cases_assertion_or_dependency_changes(before, after):
    source = SOURCE.read_text()
    assert before in source
    with pytest.raises(AssertionError):
        validate_inverse(source.replace(before, after, 1))
