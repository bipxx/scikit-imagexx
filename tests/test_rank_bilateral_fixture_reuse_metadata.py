"""Inert guards for the exact bilateral test-fixture source projection."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = "tests/test_rank_bilateral_contracts.py"
OLD_EVIDENCE = "coverage/evidence/rank-bilateral-fixture-assertion-migration.json"
WITNESS = "coverage/evidence/rank-bilateral-fixture-reuse.json"
spec = importlib.util.spec_from_file_location("bilateral_fixture_projection", ROOT / "tools/rank_bilateral_fixture_source.py")
projection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(projection)


def test_qualified_source_and_historical_evidence_are_unchanged():
    qualified, actual = projection.project_qualified_fixture(ROOT)
    assert hashlib.sha256(qualified.encode()).hexdigest() == "9fb6411bb69a92bbc40bc4ebab37f553328476a19d010c895ea7c0c2dc9d1f16"
    assert hashlib.sha256((ROOT / OLD_EVIDENCE).read_bytes()).hexdigest() == "860dcf68c2a37c18b6a3ecea3385f6cb6d55490400a311f71502473bcf4c9800"
    assert actual == hashlib.sha256((ROOT / FIXTURE).read_text().encode()).hexdigest()
    assert actual != hashlib.sha256(qualified.encode()).hexdigest()


@pytest.mark.parametrize("before,after", [
    ("def observe(function,name,p):", "def observe(function,name,p):\n pass"),
    ("rows=[]", "rows=[None]"),
    ("@pytest.mark.parametrize('workers',(1,4))", "@pytest.mark.parametrize('workers',(1,))"),
    ("\n   assert actual==frozen['observations']", "\n   assert actual!=frozen['observations']"),
    ("\ndef test_strict_pending_never_counts_forwarding(case,namespace,workers,frozen):", "\ndef test_strict_pending_never_counts_forwarding(case,namespace,workers):"),
])
def test_inverse_rejects_observer_factory_cases_assertion_or_dependency_changes(tmp_path, before, after):
    for filename in (FIXTURE, OLD_EVIDENCE, WITNESS):
        target = tmp_path / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / filename).read_bytes())
    source = (tmp_path / FIXTURE).read_text()
    assert before in source
    changed = source.replace(before, after, 1)
    (tmp_path / FIXTURE).write_text(changed)
    witness = json.loads((tmp_path / WITNESS).read_text())
    # Authenticate the mutated source so rejection exercises the exact inverse.
    witness["optimized_fixture_canonical_lf_sha256"] = hashlib.sha256(changed.encode()).hexdigest()
    (tmp_path / WITNESS).write_text(json.dumps(witness))
    with pytest.raises(AssertionError):
        projection.project_qualified_fixture(tmp_path)


@pytest.mark.parametrize("wrong_pin", [False, True])
def test_actual_benchmark_preflight_keeps_historical_pin_and_reports_current_hash(wrong_pin):
    tree = ast.parse((ROOT / "benchmarks/rank_bilateral.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    guarded = next(n for n in main.body if isinstance(n, ast.Try))
    start = next(i for i, n in enumerate(guarded.body) if isinstance(n, ast.ImportFrom) and n.module == "tools.rank_bilateral_fixture_source")
    block = ast.Module(body=guarded.body[start:start+4], type_ignores=[])
    manifest = json.loads((ROOT / "coverage/contracts/rank-bilateral-benchmark.json").read_text())
    if wrong_pin:
        manifest["fixture_source_sha256"][FIXTURE] = "0" * 64
    namespace = dict(ROOT=ROOT, manifest=manifest, report={"metadata": {}}, hashlib=hashlib,
                     text_sha=lambda p: hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest())
    import sys
    sys.path.insert(0, str(ROOT))
    try:
        if wrong_pin:
            with pytest.raises(AssertionError):
                exec(compile(block, "actual-benchmark-preflight", "exec"), namespace)
        else:
            exec(compile(block, "actual-benchmark-preflight", "exec"), namespace)
            receipt = namespace["report"]["metadata"]["historical_fixture_projection"]
            assert receipt["actual_current_sha256"] == projection.project_qualified_fixture(ROOT)[1]
            assert receipt["qualified_sha256"] != receipt["actual_current_sha256"]
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize('change_observer', (False, True))
def test_actual_masked_center_guard_preserves_source_projection_and_observer_derivation(tmp_path, change_observer):
    # Compile only the actual metadata guard and two uncalled observers. Their
    # numerical globals are deliberately absent; no image/fixture call can run.
    import inspect
    import sys
    path=ROOT/'tests/test_rank_bilateral_masked_center_public.py'
    source=path.read_text(encoding='utf-8')
    if change_observer:
        assert source.count('np.arange(9,dtype=np.uint16)%4')==1
        source=source.replace('np.arange(9,dtype=np.uint16)%4','np.arange(9,dtype=np.uint16)%5',1)
        path=tmp_path/'changed_masked_observer.py';path.write_text(source,encoding='utf-8')
    tree=ast.parse(source)
    names={'source_observe','source_mask_feedback_observe','test_frozen_source_observer_derivation_and_scope'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names]
    assert len(nodes)==3
    namespace=dict(ROOT=ROOT,CASES=(None,)*396,ast=ast,hashlib=hashlib,json=json,inspect=inspect)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    sys.path.insert(0,str(ROOT))
    try:
        guard=namespace['test_frozen_source_observer_derivation_and_scope']
        if change_observer:
            with pytest.raises(AssertionError):guard()
        else:guard()
    finally:sys.path.pop(0)
