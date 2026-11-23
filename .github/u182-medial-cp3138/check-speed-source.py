"""Stdlib-only exact timing ingress and raw-sample collector checks."""
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import runpy
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def must_refuse(function):
    try:
        function()
    except (AssertionError, KeyError):
        return
    raise AssertionError("accepted denied authorization or invalid evidence")


def synthetic(collector, frozen, index):
    verify_batches, verify_pair = collector["verify_batches"], collector["verify_pair"]
    pair = {key: frozen.get(key) for key in ("case_id", "input", "mask", "skeleton", "distance")}
    pair.update(fallback_enabled=False, seed=int(frozen["rng"]), stock_order=frozen["loop"][0]["order"],
                stock_calls={"public": 1, "label": 1024, "edt": 1},
                native_calls=["medial_axis_prepare", "edt", "medial_axis_ordered"])
    verify_pair(pair, frozen)
    must_refuse(lambda: verify_pair(dict(pair, fallback_enabled=True), frozen))
    orders = [[0, 1] if (turn + index + 1) % 2 == 0 else [1, 0] for turn in range(7)]
    batches = [{"function": side, "loops": 1, "elapsed_ns": 250_000_000, "complete": True,
                "failure": None, "phase": "warmup", "accepted": False} for side in (0, 1)]
    for turn, order in enumerate(orders):
        batches.extend({"function": side, "loops": 1, "elapsed_ns": 100_000_000 * (side + 1),
                        "complete": True, "failure": None, "phase": "sample", "accepted": True,
                        "round": turn, "ordinal": ordinal} for ordinal, side in enumerate(order))
    report = {"sample_labels": ["scikit-image", "current"], "batch_records": batches,
              "samples_ms": [[100.] * 7, [200.] * 7], "loops": [[1] * 7, [1] * 7],
              "warmup_elapsed_ns": [250_000_000] * 2, "sample_order": orders,
              "reference_ms": 100., "cpp_ms": 200., "speedup": .5}
    verify_batches(report, batches, index)
    must_refuse(lambda: verify_batches(dict(report, speedup=1.), batches, index))
    must_refuse(lambda: verify_batches(report, batches[:-1], index))


def main():
    assert sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode and not sys.flags.optimize
    git, repo = sys.argv[1:]
    stage = runpy.run_path(str(HERE / "speed-stage.py"))
    packet, plan, prior, admission = stage["inputs"]()
    for file in HERE.glob("*.py"):
        compile(file.read_bytes(), str(file), "exec")
    assert {name: pin for name, pin in plan["runtime"].items()} == {
        "version": packet["runtime"]["version"], "executable_sha256": packet["runtime"]["executable_sha256"],
        "extension_suffix": packet["runtime"]["ext_suffix"]}
    assert admission["envelope"]["id"] == 14788827504 and admission["elf"]["bytes"] == 296088
    for path, pin in plan["source_paths"].items():
        raw = subprocess.check_output([git, "--no-optional-locks", "-C", repo, "show", f"{plan['source']}:{path}"])
        assert sha256(raw).hexdigest() == pin, path
    providers = stage["read"](HERE / "timing-providers.json")
    assert len(providers) == 13 and set(providers.values()) <= set(plan["selected_retained_live_files_sha256"])
    collector = runpy.run_path(str(HERE / "collect-speed.py"))
    cases = collector["rows"](HERE / "linux-modern-local-cases-v1.jsonl")
    assert len(cases) == len({row["case_id"] for row in cases}) == 27
    assert [row["case_id"] for row in cases[:4]] == stage["CASES"] == collector["CASE_IDS"]
    for index, case in enumerate(cases[:4]):
        synthetic(collector, case, index)
    now = datetime(2026, 11, 23, 4, tzinfo=timezone.utc)
    authority = {"carrier": "c" * 40, "payload": stage["seal"](HERE / "speed-payload.json")["sha256"],
                 "purpose": stage["PURPOSE"], "scalar_admission": stage["ADMISSION"], "artifact": 14788827504,
                 "cpu": 0, "workers": 1, "deadline_utc": (now + timedelta(minutes=30)).isoformat()}
    raw = json.dumps(authority)
    env = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": stage["DRIVER"]["REPOSITORY"],
           "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_REF": stage["REF"], "GITHUB_SHA": authority["carrier"],
           "MEDIAL_AUTHORITY": raw, "MEDIAL_AUTHORITY_SHA256": sha256(raw.encode()).hexdigest()}
    assert stage["authority"](env, now)[0] == authority
    must_refuse(lambda: stage["authority"]({}, now))
    must_refuse(lambda: stage["authority"](dict(env, MEDIAL_AUTHORITY_SHA256="0" * 64), now))
    must_refuse(lambda: stage["authority"](env, now + timedelta(hours=1)))
    peers = runpy.run_path(str(HERE / "process_guard.py"))["check_processes"]
    assert peers("1 0 S runner runner\n2 1 S python python quiet-speed.py\n3 1 Z git git fetch", 2, 3) == []
    assert peers("1 0 S runner runner\n2 1 S python python quiet-speed.py\n3 1 S git git fetch", 2, 3) == [
        {"pid": 3, "name": "git"}]
    workflow = (HERE.parent / "workflows/report-json.yml").read_text()
    assert "if: always()" in workflow and "speed-stage.py" in workflow and "quiet-speed.py" in workflow
    assert "scalar-only" not in workflow and "install" not in workflow
    assert not {"skimagexx", "numpy", "scipy", "skimage", "pytest"} & sys.modules.keys()
    print(sys.version.split()[0], "exact fixed cases, source pins, quiet/authority and synthetic loss collector passed; zero science")


if __name__ == "__main__":
    main()
