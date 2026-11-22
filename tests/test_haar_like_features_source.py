"""Source-only authority and scope checks for the Haar feature family."""

import ast
import hashlib
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "coverage/contracts/haar-like-features-v1.json").read_bytes())


def test_frozen_safe_recipe_and_timing_boundary():
    fixture = ROOT / CONTRACT["upstream"]["coordinate_and_value_fixture"]
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == CONTRACT["upstream"]["minimum_fixture_sha256"]
    reference = json.loads(fixture.read_bytes())
    assert reference["original_calls"] == CONTRACT["upstream"]["common_safe_cases"] == len(reference["cases"])
    assert reference["original_sha256"] == CONTRACT["upstream"]["python_source_sha256"]
    assert reference["unsafe_exclusions"] == CONTRACT["upstream"]["unsafe_exclusions"]
    cases, recipe = runpy.run_path(str(ROOT / "benchmarks/haar_like_features.py"))["fixed_cases"]()
    assert hashlib.sha256(Path(recipe["__file__"]).read_bytes()).hexdigest() == reference["recipe_sha256"]
    fixed = CONTRACT["timing"]["case_ids"]
    assert [case["id"] for case in cases] == fixed
    assert set(fixed) <= {row["id"] for row in reference["cases"] if "output" in row}
    assert CONTRACT["timing"]["workers"] == 1
    assert CONTRACT["timing"]["execution_authority"] is None


def test_native_owner_and_no_unsafe_value_fallback():
    tree = ast.parse((ROOT / "python/skimagexx/_haar_like_features.py").read_bytes())
    imports = {name.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for name in node.names}
    assert "skimage" not in imports and "fallback" not in imports
    native = {
        node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "_core"
    }
    assert native == {"haar_coordinates", "haar_generated_values", "haar_selected_values"}
