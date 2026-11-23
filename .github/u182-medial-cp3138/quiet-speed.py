"""Check real stage exit and run one isolated worker without setup work."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    assert sys.platform == "linux" and sys.flags.isolated and sys.flags.no_site
    assert sys.dont_write_bytecode and not sys.flags.optimize
    stage = runpy.run_path(str(HERE / "speed-stage.py"))
    grant, raw = stage["authority"](os.environ, datetime.now(timezone.utc))
    read, write, seal = (stage[name] for name in ("read", "write", "seal"))
    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve(strict=True)
    reports = workspace / "medial-timing-stage"
    outputs = workspace / "medial-timing-results"
    assert (reports / "stage-process.exit").read_text().strip() == "0"
    binding = read(reports / "stage-ready.json")
    work = Path(binding["work"]).resolve(strict=True)
    assert work.parent == Path(os.environ["RUNNER_TEMP"]).resolve(strict=True)
    assert work.name.startswith("medial-quiet-speed-") and not work.is_relative_to(workspace)
    try:
        assert (work / "authority.json").read_bytes() == raw
        assert datetime.now(timezone.utc) < datetime.fromisoformat(grant["deadline_utc"])
        assert 0 in os.sched_getaffinity(0) and shutil.disk_usage(work).free >= 30 * 2**30
        processes = subprocess.check_output(["/usr/bin/ps", "-eo", "pid=,ppid=,stat=,comm=,args="], text=True)
        peers = runpy.run_path(str(HERE / "process_guard.py"))["check_processes"](
            processes, os.getpid(), binding["pid"])
        quiet = reports / "quiet.json"
        write(quiet, {"authority_sha256": binding["authority_sha256"], "stage_exit": 0, "cpu": 0,
                      "checked_utc": datetime.now(timezone.utc).isoformat(), "active_competing_local_jobs": peers,
                      "visible_process_count": len(processes.splitlines())})
        assert not peers
        binding["quiet"] = {"path": str(quiet), **seal(quiet)}
        binding["portable_path"] = str(reports / "portable-installed.json")
        binding["transfer_path"] = str(reports / "elf-transfer.json")
        pin = work / "binding.json"
        write(pin, binding)
        interpreter = Path(binding["interpreter"]).resolve(strict=True)
        assert interpreter.is_relative_to(work)
        args = ["taskset", "-c", "0", str(interpreter), "-I", "-S", "-B", str(HERE / "quiet-benchmark.py"),
                str(pin), seal(pin)["sha256"], str(outputs)]
        clean = dict(os.environ, TMPDIR=str(work))
        clean.pop("GH_TOKEN", None)
        with (reports / "timing.stdout.log").open("xb") as stdout, (reports / "timing.stderr.log").open("xb") as stderr:
            status = subprocess.run(args, env=clean, stdout=stdout, stderr=stderr, check=False).returncode
        write(reports / "timing-process.json", {"exit": status, "argv": args, "binding_sha256": seal(pin)["sha256"]})
        with (reports / "timing-collection.json").open("xb") as target:
            parsed = subprocess.run([sys.executable, "-I", "-S", "-B", HERE / "collect-speed.py", reports, outputs],
                                    stdout=target, check=False).returncode
        return status or parsed
    finally:
        shutil.rmtree(work)


if __name__ == "__main__":
    raise SystemExit(main())
