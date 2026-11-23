"""Stdlib guard for the exact carrier projection, authority and collector."""
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

HERE = Path(__file__).resolve().parent


def rejected(check):
    try:
        check()
    except (AssertionError, KeyError):
        return
    raise AssertionError("guard accepted an incomplete or unauthorized input")


def synthetic_collector(driver, plan, historical):
    collect = runpy.run_path(str(HERE / "collect.py"))["collect"]
    seal = driver["seal"]
    with TemporaryDirectory(prefix="medial-collector-") as root:
        reports, payload = Path(root) / "reports", Path(root) / "product"
        reports.mkdir()
        payload.mkdir()

        def save(name, data):
            (reports / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")

        for step in ("git-fetch", "git-checkout", "stock-install", "pybind-install",
                     "verify-portable", "leaf", "link", "scalar"):
            save(step, {"exit": 0})
        purpose = driver["PURPOSE"]
        save("terminal", {"stage": "transport", "exit": 0, "purpose": purpose})
        floor = driver["FLOOR"]
        save("authority-capacity", {"authority_sha256": "e" * 64, "floor": floor,
             "observed_free_bytes": {"work": floor, "temp": floor}})
        source = {key: plan[key] for key in ("source", "tree")}
        proof = {**source, "paths": len(plan["source_paths"])}
        save("source-proof", proof)
        log = {name: {"wheel_sha256": pin["wheel_sha256"], "wheel_record_sha256": "b" * 64,
                      "retained_uv_record_sha256": pin["retained_uv_record_sha256"],
                      "host_record_sha256": "c" * 64, "immutable_members": pin["immutable_members"],
                      "immutable_digest": pin["retained_immutable_digest"]}
               for name, pin in plan["stock_distributions"].items()}
        members = {name: ["pseudo-digest", "0"] for name in plan["required_live_after_scalar"]}
        while len(members) < 3981:
            members[f"synthetic/{len(members)}"] = ["pseudo-digest", "0"]
        save("portable-installed", {"distributions": log, "member_pins": members,
             "python": plan["runtime"]["version"], "python_sha256": plan["runtime"]["executable_sha256"],
             "science_imports": 0})
        xml = reports / "scalar.xml"
        suite = ElementTree.Element("testsuite", tests="30", failures="0", errors="0", skipped="0")
        for label in historical["nodes"]:
            ElementTree.SubElement(suite, "testcase", classname="test_medial_axis_public", name=label)
        ElementTree.ElementTree(suite).write(xml, encoding="utf-8")
        name = "_medial_axis_private" + plan["runtime"]["extension_suffix"]
        binary = payload / name
        binary.write_bytes(b"\x7fELF\x02\x01" + bytes(12) + b"\x3e\x00" + b"synthetic")
        scalar = {**proof, "cases": 30, "exit": 0, "cpu": 0, "workers": 1, "avx2": False,
                  "bindings": ["medial_axis_prepare", "medial_axis_ordered"], "timing": 0,
                  "stock_replay": 0, "normal_replay": 0, "sdk_replay": 0,
                  "private_sha256": seal(binary)["sha256"], "donor_sha256": plan["donor"]["core_sha256"],
                  "portable": seal(reports / "portable-installed.json"), "xml": seal(xml),
                  "loaded": {key: members[key][0] for key in plan["required_live_after_scalar"]}}
        save("scalar-result", scalar)
        (payload / "medial-scalar.json").write_text(json.dumps({**source, "no_timing": True,
             "elf": {"name": name, **seal(binary)}, "scalar": seal(reports / "scalar-result.json"),
             "authority_sha256": "e" * 64}), encoding="utf-8")
        assert collect(reports, payload, plan, historical["nodes"])["scalar"] == 30
        rejected(lambda: collect(reports, payload, plan, historical["nodes"][:-1]))
        save("verify-portable", {"exit": 1})
        rejected(lambda: collect(reports, payload, plan, historical["nodes"]))


def main():
    assert sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode and not sys.flags.optimize
    git, repository = sys.argv[1:]
    driver = runpy.run_path(str(HERE / "run.py"))
    packet, plan, historical = driver["inputs"]()
    revision = plan["source"]

    def show(*args):
        return subprocess.check_output([git, "--no-optional-locks", "-C", repository, *args])

    assert show("rev-parse", f"{revision}^", f"{revision}^{{tree}}").decode().splitlines() == [
        plan["direct_parent"], plan["tree"]]
    added = show("diff-tree", "--no-commit-id", "--name-status", "-r", revision).decode().splitlines()
    assert set(added) == {"A\t" + path for path in plan["owned_paths"]}
    for path, pin in plan["source_paths"].items():
        assert sha256(show("show", f"{revision}:{path}")).hexdigest() == pin, path
    assert (packet["runtime"]["version"], packet["runtime"]["executable_sha256"]) == (
        plan["runtime"]["version"], plan["runtime"]["executable_sha256"])
    wheels = packet["donor"]["wheel_inputs"]
    for wanted in plan["stock_distributions"].values():
        result = wheels[wanted["wheel"]]
        assert (result["sha256"], result["bytes"]) == (wanted["wheel_sha256"], wanted["wheel_bytes"])
    rows = [json.loads(line) for line in (HERE / "linux-modern-local-cases-v1.jsonl").read_bytes().splitlines()]
    assert len(rows) == len({row["case_id"] for row in rows}) == 27
    assert len(historical["nodes"]) == len(set(historical["nodes"])) == 30
    assert {"test_frozen_public_contract[" + row["case_id"] + "]" for row in rows} <= set(historical["nodes"])
    assert "if: success()" in (HERE.parent / "workflows/report-json.yml").read_text()
    assert "retention-days: 1" in (HERE.parent / "workflows/report-json.yml").read_text()
    payload_hash = driver["seal"](HERE / "payload.json")["sha256"]
    now = datetime.now(timezone.utc)
    notice = {"carrier": "3" * 40, "payload": payload_hash, "purpose": driver["PURPOSE"], "cpu": 0,
              "deadline_utc": (now + timedelta(minutes=5)).isoformat()}
    raw = json.dumps(notice)
    env = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": driver["REPOSITORY"],
           "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_REF": driver["REF"],
           "GITHUB_SHA": notice["carrier"], "MEDIAL_AUTHORITY": raw,
           "MEDIAL_AUTHORITY_SHA256": sha256(raw.encode()).hexdigest()}
    assert driver["authority"](env, now, payload_hash) == notice
    rejected(lambda: driver["authority"]({}, now, payload_hash))
    rejected(lambda: driver["authority"](env, now, "0" * 64))
    rejected(lambda: driver["authority"](env, now + timedelta(minutes=8), payload_hash))
    synthetic_collector(driver, plan, historical)
    for file in HERE.iterdir():
        content = file.read_bytes()
        assert b"\x00" not in content
        personal = rb"(?i)(/mnt/[cd]/(?:users|scratch)/[a-z0-9_.-]+/|/home/[a-z0-9_.-]+/|[a-z]:[/\\]+users[/\\]+[a-z0-9_.-]+[/\\])"
        assert not re.search(personal, content), file.name
    assert not {"skimagexx", "numpy", "scipy", "skimage", "pytest"} & sys.modules.keys()
    print(f"{sys.version.split()[0]} source pins, exact nodes, authority refusal and collector passed; no science")


if __name__ == "__main__":
    main()
