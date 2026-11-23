"""Verify completed fixed-case preparation and the exact four-file ELF transport."""
from hashlib import sha256
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_bytes())


def seal(raw):
    return {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}


def collect(reports, transport):
    assert reports.resolve() != transport.resolve()
    manifest = read(transport / "prepared.json")
    terminal = read(reports / "terminal.json")
    capacity = read(reports / "authority-capacity.json")
    plan = read(HERE / "speed-prep-v1.json")
    authority = manifest["authority"]
    assert terminal["exit"] == 0 and terminal["stage"] == "transport"
    assert not manifest["timing"] and terminal["purpose"] == authority["purpose"] == plan["purpose"]
    assert capacity["authority"] == authority and authority["prior_modern_admission_sha256"] == plan["prior_modern_admission_sha256"]
    assert authority["preparation"] == seal((HERE / "speed-prep-v1.json").read_bytes())["sha256"]
    assert len(capacity["observed_free_bytes"]) == 2
    assert all(free >= capacity["required_bytes"] == 30 * 2**30 for free in capacity["observed_free_bytes"].values())
    for stage in ("source-proof", "venv", "stock-install", "pybind-install", "runtime-proof"):
        assert read(reports / (stage + ".json"))["exit"] == 0
    for stage, path in (("toolchain", "toolchain.json"), ("actual_runtime_record", "runtime-proof.log"),
                        ("source_proof", "source-proof.log"), ("donor_transfer", "donor-transfer.json")):
        assert manifest[stage] == seal((reports / path).read_bytes())
    assert read(reports / "source-proof.log") == {"families": 3, "owned_paths": 30, "windows_local_preflight": False}
    assert read(reports / "runtime-proof.log") == manifest["runtime"]
    assert manifest["runtime"]["donor_core"] == plan["core"]["sha256"] and manifest["runtime"]["selected_provider_members"] == 24
    assert set(manifest["products"]) == set(plan["families"]) and len(manifest["products"]) == 3
    assert not any("-sdk" in path.name or "-scalar" in path.name for path in reports.iterdir())
    assert sorted(path.name for path in reports.glob("*-cases.json")) == sorted(f"{n}-fixed-cases.json" for n in plan["families"])
    allowed = {"prepared.json"}
    for family, product in manifest["products"].items():
        expected = plan["families"][family]
        assert (product["source"], product["tree"], product["fixed_cases"]) == (expected["source"], expected["tree"], expected["frozen_cases"])
        for stage in ("leaf", "link"):
            log = reports / f"{family}-{stage}.json"
            assert read(log)["exit"] == 0 and product[stage + "_command"] == seal(log.read_bytes())
        record = product["elf"]
        name = f"_{family}_private.cpython-313-x86_64-linux-gnu.so"
        assert record["name"] == name
        allowed.add(name)
        binary = transport / name
        assert binary.is_file() and not binary.is_symlink() and binary.stat().st_size == record["bytes"] <= 2**20
        raw = binary.read_bytes()
        assert record["sha256"] == sha256(raw).hexdigest() and raw[:6] == b"\x7fELF\x02\x01" and raw[18:20] == b"\x3e\x00"
        path = reports / f"{family}-fixed-cases.json"
        assert product["witness"] == seal(path.read_bytes()) and read(reports / f"{family}-fixed.json")["exit"] == 0
        witness = read(path)
        assert witness["mode"] == "normal" and witness["workers"] == 1
        assert (witness["private_sha256"], witness["core_sha256"]) == (record["sha256"], plan["core"]["sha256"])
        rows = witness["rows"]
        assert len(rows) == 4 and all(row["native_hits"] == 1 and row["failure"] is None for row in rows)
        assert [(row["id"], row["expected_stock_row_sha256"]) for row in rows] == [
            (row["id"], row["canonical_row_sha256"]) for row in expected["frozen_cases"]]
    assert {path.name for path in transport.iterdir()} == allowed
    assert sum((transport / name).stat().st_size for name in allowed) <= plan["transport_size_limit_bytes"]
    return {"run": manifest["run"], "carrier": authority["carrier"], "authority_sha256": manifest["authority_sha256"],
            "bounded_transport_members": {path.name: seal(path.read_bytes()) for path in transport.iterdir()},
            "named_candidate_checks": 12, "timing": False, "sdk_runs": 0, "whole_stock_replay": 0}


if __name__ == "__main__":
    reports, transport = map(Path, sys.argv[1:])
    print(json.dumps(collect(reports, transport), indent=2))
