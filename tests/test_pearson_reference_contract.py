"""Source and frozen-data checks for the first Pearson reference contract."""

import json
import math
from pathlib import Path
import struct

import pytest

from fixtures.pearson_reference_v1 import (
    BENCHMARK_SPECS, SOURCE_HASHES, VERSIONS, benchmark_cases, descriptor,
    semantic_cases, sha256,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "coverage/contracts/pearson-reference-v1.json"
FIXTURE = ROOT / "tests/fixtures/pearson_original_windows_v1.json"
OBSERVER = ROOT / "tests/fixtures/pearson_reference_v1.py"


def test_frozen_pearson_capture():
    contract = json.loads(CONTRACT.read_bytes())
    assert contract["reference"]["fixture"] == FIXTURE.relative_to(ROOT).as_posix()
    assert sha256(FIXTURE.read_bytes()) == contract["reference"]["sha256"]
    fixture = json.loads(FIXTURE.read_bytes())
    assert fixture["metadata"]["versions"] == VERSIONS
    assert fixture["metadata"]["source_sha256"] == SOURCE_HASHES
    assert fixture["metadata"]["observer_sha256"] == sha256(OBSERVER.read_bytes())
    assert set(fixture["metadata"]["pool_limits"].values()) == {"1"}
    assert len(fixture["rows"]) == contract["reference"]["case_count"]
    assert sum("error" in row for row in fixture["rows"]) == contract["reference"]["errors"]
    assert sum(bool(row["warnings"]) for row in fixture["rows"]) == contract["reference"]["warned_rows"]

    reproduced = {}
    for group, cases in (("semantics", semantic_cases()), ("benchmark-input", benchmark_cases())):
        for name, left, right, mask, count, warning_action in cases:
            assert name not in reproduced
            arrays = [left, right] if mask is None else [left, right, mask]
            reproduced[name] = group, count, warning_action, [descriptor(array) for array in arrays]
    assert list(reproduced) == [row["id"] for row in fixture["rows"]]

    for row in fixture["rows"]:
        assert reproduced[row["id"]] == (row["group"], row["selected_count"],
                                          row["warning_action"], row["inputs"])
        if "error" in row:
            assert set(row["error"]) == {"class", "message"}
            assert row["error"]["message"] and "components" not in row
            continue
        assert row["type"] == "builtins.tuple"
        assert [value["name"] for value in row["components"]] == ["r", "p"]
        for value in row["components"]:
            number = struct.unpack(">d", bytes.fromhex(value["ieee754_be"]))[0]
            assert value["type"] == "builtins.float" and number.hex() == value["hex"]
            if not math.isnan(number):
                limit = -1 if value["name"] == "r" else 0
                assert limit <= number <= 1


@pytest.mark.parametrize(("name", "expected_r"), (("two-positive", 1.0), ("two-negative", -1.0)))
def test_two_sample_pearson_is_exact(name, expected_r):
    row = next(row for row in json.loads(FIXTURE.read_bytes())["rows"] if row["id"] == name)
    assert not row["warnings"] and row["selected_count"] == 2
    assert [float.fromhex(component["hex"]) for component in row["components"]] == [expected_r, 1.0]


def test_paired_pearson_benchmark_design():
    contract = json.loads(CONTRACT.read_bytes())["benchmark"]
    assert [(row["id"], tuple(row["shape"]), row["mask_percent"]) for row in contract["cases"]] == list(BENCHMARK_SPECS)
    assert contract["native_workers"] == contract["reference_pool_workers"] == 1
    assert contract["accepted_pairs"] == 7 and contract["minimum_batch_ms"] == 100
    assert contract["warmup_ms_per_side"] == 250
    assert contract["timed"] is False
    frozen = {row["id"]: row for row in json.loads(FIXTURE.read_bytes())["rows"]}
    for spec in contract["cases"]:
        saved = frozen[spec["id"]]
        assert saved["group"] == "benchmark-input" and not saved["warnings"]
        assert saved["selected_count"] == spec["selected_count"]
        r, probability = [float.fromhex(value["hex"]) for value in saved["components"]]
        assert math.isfinite(r) and math.isfinite(probability)
        assert abs(r) <= 0.95 and probability >= 1e-12
