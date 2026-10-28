"""Exact qualified-source projection for the bilateral test fixture only."""
import ast
import copy
import hashlib
import json
from pathlib import Path


def project_qualified_fixture(root):
    root = Path(root)
    current = (root / "tests/test_rank_bilateral_contracts.py").read_text()
    reuse = json.loads((root / "coverage/evidence/rank-bilateral-fixture-reuse.json").read_text())
    evidence = root / "coverage/evidence/rank-bilateral-fixture-assertion-migration.json"
    assert hashlib.sha256(evidence.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == reuse["historical_migration_sha256"]
    actual_digest = hashlib.sha256(current.encode()).hexdigest()
    assert actual_digest == reuse["optimized_fixture_canonical_lf_sha256"]
    qualified = reuse["qualified_fixture_utf8"]
    assert hashlib.sha256(qualified.encode()).hexdigest() == reuse["qualified_fixture_canonical_lf_sha256"] == json.loads(evidence.read_text())["current_fixture_canonical_lf_sha256"]
    signature = "\ndef test_strict_pending_never_counts_forwarding(case,namespace,workers,frozen):"
    assertion = "\n   assert actual==frozen['observations'][frozen['snapshots'][profile()][key]],(path,key)"
    assert current.count(signature) == current.count(assertion) == 1
    inverse = current.replace(signature, signature.replace(",frozen", ""), 1).replace(assertion, "\n   frozen=json.loads(MATRIX.read_text())" + assertion, 1)
    restored = ast.parse(inverse)
    prior = ast.parse(qualified)
    assert len(restored.body) == len(prior.body)
    for actual, previous in zip(restored.body, prior.body):
        if isinstance(previous, ast.FunctionDef) and previous.name == "test_integrity_safe_scope_native0":
            assert isinstance(actual, ast.FunctionDef) and actual.name == previous.name
            actual.body = copy.deepcopy(previous.body)
    assert ast.dump(restored) == ast.dump(prior), "fixture reuse changed more than its dependency and integrity proof"
    return qualified, actual_digest
