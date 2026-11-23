"""Verify the actual stage exit and exclusive quiet process boundary."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
COMPETING = re.compile(r"(?:^|[/\s])(?:g\+\+|gcc|cc1plus|cmake|ninja|pytest|ctest|curl|wget|git|gh|stage\.py|quiet-timing\.py|pip)(?:[\s]|$)")


def read(path):
    return json.loads(path.read_bytes())


def seal(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def check_processes(raw, own_pid, staged_pid):
    processes = {}
    for line in raw.splitlines():
        pid, parent, status, name, *detail = line.split(maxsplit=4)
        processes[int(pid)] = (int(parent), status, name, detail[0] if detail else "")
    ancestor = own_pid
    allowed = set()
    while ancestor in processes and ancestor not in allowed:
        allowed.add(ancestor)
        ancestor = processes[ancestor][0]
    active = []
    for pid, (_, status, name, arguments) in processes.items():
        if pid in allowed or status.startswith("Z") or name == "ps":
            continue
        if pid == staged_pid or COMPETING.search(arguments) or COMPETING.search(name) or " -m pytest" in arguments:
            active.append({"pid": pid, "name": name})
    return active


def main():
    assert sys.platform == "linux" and sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode and not sys.flags.optimize
    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve(strict=True)
    temp = Path(os.environ["RUNNER_TEMP"]).resolve(strict=True)
    reports = workspace / "hough-timing-stage"
    assert (reports / "stage-process.exit").read_text().strip() == "0"
    ready = read(reports / "stage-ready.json")
    work = Path(ready["work"]).resolve(strict=True)
    assert work.parent == temp and work.name.startswith("u182-hough-timing-v1-")
    outputs = workspace / "hough-timing-results"
    try:
        assert ready["authority_sha256"] == os.environ["HOUGH_AUTHORITY_SHA256"]
        stage = runpy.run_path(str(HERE / "stage.py"))
        grant = stage["authority"](os.environ, datetime.now(timezone.utc))
        authority = work / "authority.json"
        assert seal(authority)["sha256"] == ready["authority_sha256"] and read(authority) == grant
        draft = work / "binding-draft.json"
        assert {"path": str(draft), **seal(draft)} == ready["binding_draft"]
        binding = read(draft)
        assert binding["authority_sha256"] == ready["authority_sha256"] and binding["transport"] == grant["transport"]
        python = Path(ready["interpreter"]).resolve(strict=True)
        assert python.is_relative_to(work)
        assert 0 in os.sched_getaffinity(0) and shutil.disk_usage(work).free >= 30 * 2**30
        raw = subprocess.check_output(["/usr/bin/ps", "-eo", "pid=,ppid=,stat=,comm=,args="], text=True)
        competitors = check_processes(raw, os.getpid(), ready["pid"])
        quiet = reports / "quiet.json"
        write(quiet, {"authority_sha256": ready["authority_sha256"], "stage_process_exit": 0, "cpu": 0,
                      "checked_utc": datetime.now(timezone.utc).isoformat(),
                      "active_competing_local_jobs": competitors, "transfers_and_builds_terminal": not competitors,
                      "visible_process_count": len(raw.splitlines())})
        assert not competitors
        binding["quiet"] = {"path": str(quiet), **seal(quiet)}
        binding_path = work / "binding.json"
        write(binding_path, binding)
        argv = ["taskset", "-c", "0", str(python), "-I", "-S", "-B", str(HERE / "quiet-timing.py"),
                "--authority", str(authority), "--authority-sha256", ready["authority_sha256"], "--binding", str(binding_path),
                "--binding-sha256", seal(binding_path)["sha256"], "--out", str(outputs)]
        child = dict(os.environ)
        child.pop("GH_TOKEN", None)
        child["TMPDIR"] = str(work)
        with (reports / "timing-process.log").open("xb") as stdout, (reports / "timing-process.stderr.log").open("xb") as stderr:
            result = subprocess.run(argv, env=child, stdout=stdout, stderr=stderr, check=False)
        write(reports / "timing-process.json", {"argv": argv, "exit": result.returncode, "binding_sha256": seal(binding_path)["sha256"]})
        collection = subprocess.run([sys.executable, "-I", "-S", "-B", HERE / "collect-timing.py", reports, outputs],
                                    check=False)
        return result.returncode or collection.returncode
    finally:
        shutil.rmtree(work)


if __name__ == "__main__":
    raise SystemExit(main())
