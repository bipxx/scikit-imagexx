"""Stage the admitted Medial Axis ELF and provably identical stock without science."""
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
from tempfile import mkdtemp
from urllib import error, parse, request
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
DRIVER = runpy.run_path(str(HERE / "run.py"))
BOOT = DRIVER["BOOT"]
read, write, seal, command = (DRIVER[name] for name in ("read", "write", "seal", "command"))
REF = "refs/heads/diagnostics/u182-medial-cp3138-speed-v1"
PURPOSE = "medial-axis-modern-quiet-speed-v1"
ADMISSION = "efa70b777050805e8b9bd38dda0e389f7822b384b78ed5d45fdeebd94bca7a34"
CASES = ["symmetric-thick", "perforated-distance", "masked-components-distance", "seeded-corridors"]


def inputs():
    packet, plan, prior = DRIVER["inputs"]()
    pins = read(HERE / "speed-payload.json")
    for path, pin in pins.items():
        assert seal(HERE / path) == pin, path
    assert seal(HERE / "scalar-admission.json")["sha256"] == ADMISSION
    admission = read(HERE / "scalar-admission.json")
    assert (admission["source"], admission["tree"], admission["exact_cases"]) == (plan["source"], plan["tree"], 30)
    assert admission["expected_immutable_files"] == 3981 and admission["actual_generated_wrappers"] == 11
    assert admission["elf_transferred"] is False and admission["timing_authorized"] is False
    return packet, plan, prior, admission


def authority(env, now):
    packet, plan, prior, admission = inputs()
    assert env.get("GITHUB_ACTIONS") == "true" and env.get("GITHUB_REPOSITORY") == DRIVER["REPOSITORY"]
    assert env.get("GITHUB_REF") == REF and env.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
    raw = env["MEDIAL_AUTHORITY"].encode()
    assert len(raw) <= 1536 and sha256(raw).hexdigest() == env["MEDIAL_AUTHORITY_SHA256"]
    expected = {"carrier": env["GITHUB_SHA"], "payload": seal(HERE / "speed-payload.json")["sha256"],
                "purpose": PURPOSE, "scalar_admission": ADMISSION, "artifact": admission["envelope"]["id"],
                "cpu": 0, "workers": 1}
    value = json.loads(raw)
    assert set(value) == set(expected) | {"deadline_utc"}
    assert all(value[key] == val for key, val in expected.items())
    deadline = datetime.fromisoformat(value["deadline_utc"])
    expiry = datetime.fromisoformat(admission["envelope"]["expires_at"])
    assert now < deadline < expiry and (deadline - now).total_seconds() <= 5400
    return value, raw


def transfer_elf(admission, work, out):
    envelope = admission["envelope"]
    api = f"https://api.github.com/repos/{DRIVER['REPOSITORY']}/actions/"
    headers = {"Authorization": "Bearer " + os.environ["GH_TOKEN"], "Accept": "application/vnd.github+json"}

    def get(path, limit):
        with request.urlopen(request.Request(api + path, headers=headers), timeout=30) as response:
            return BOOT["chunks"](response, limit)

    artifact = json.loads(get(f"artifacts/{envelope['id']}", 2**20))
    assert artifact["name"] == "u182-medial-scalar-elf-v2" and artifact["workflow_run"]["id"] == admission["run"]
    assert artifact["workflow_run"]["head_sha"] == admission["carrier"]
    assert all(artifact[key] == envelope[key] for key in ("digest", "size_in_bytes", "expires_at", "expired"))
    assert 0 < envelope["size_in_bytes"] < 2**20 and not artifact["expired"]
    tools = runpy.run_path(str(HERE / "range_helper.py"))
    query = request.Request(api + f"artifacts/{envelope['id']}/zip", headers=headers)
    try:
        request.build_opener(tools["NoRedirect"]).open(query, timeout=20).close()
    except error.HTTPError as exc:
        assert exc.code in (301, 302, 303, 307, 308)
        location = exc.headers["Location"]
    else:
        raise RuntimeError("expected exact ELF artifact signed redirect")
    assert parse.urlparse(location).scheme == "https"
    with request.urlopen(location, timeout=30) as response:
        raw = BOOT["chunks"](response, envelope["size_in_bytes"])
    assert len(raw) == envelope["size_in_bytes"] and "sha256:" + sha256(raw).hexdigest() == envelope["digest"]
    name = admission["elf"]["name"]
    with ZipFile(BytesIO(raw)) as archive:
        assert sorted(archive.namelist()) == sorted([name, "medial-scalar.json"])
        info = archive.getinfo(name)
        assert info.file_size == admission["elf"]["bytes"] and info.file_size < 2**20
        assert archive.getinfo("medial-scalar.json").file_size < 4096
        receipt = json.loads(archive.read("medial-scalar.json"))
        binary = archive.read(info)
    assert receipt["source"] == admission["source"] and receipt["tree"] == admission["tree"]
    assert receipt["elf"] == admission["elf"] and receipt["no_timing"] is True
    assert receipt["scalar"]["sha256"] == "d7d9e5031641a2b106aa62328fcbb3206dcfef751defe57a34ee0ad0677bd8f0"
    assert sha256(binary).hexdigest() == admission["elf"]["sha256"]
    assert binary[:6] == b"\x7fELF\x02\x01" and binary[18:20] == b"\x3e\x00"
    path = work / name
    with path.open("xb") as handle:
        handle.write(binary)
    path.chmod(0o600)
    write(out / "elf-transfer.json", {"envelope": envelope, "elf": admission["elf"], "archive_retained": False,
                                      "candidate_recompiled": False, "scalar_replayed": False})
    return path


def main():
    grant, raw = authority(os.environ, datetime.now(timezone.utc))
    packet, plan, prior, admission = inputs()
    if sys.argv[1:] and sys.argv[1] == "verify-portable":
        return DRIVER["stage_portable"](packet, plan, *map(Path, sys.argv[2:]))
    assert not sys.argv[1:]
    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve(strict=True)
    temp = Path(os.environ["RUNNER_TEMP"]).resolve(strict=True)
    assert temp != workspace and not temp.is_relative_to(workspace)
    assert 0 in os.sched_getaffinity(0)
    assert not any(os.environ.get(name) for name in DRIVER["CLEAN_ENV"] + ("SKIMAGEXX_DISABLE_SIMD",))
    out = workspace / "medial-timing-stage"
    out.mkdir()
    free = {str(path): shutil.disk_usage(path).free for path in (workspace, temp)}
    write(out / "capacity.json", {"actual_free": free, "floor": DRIVER["FLOOR"], "staging_cap": 2 * 2**30,
                                  "sampler_reserve": 32 * 2**20})
    assert min(free.values()) >= DRIVER["FLOOR"]
    work = Path(mkdtemp(dir=temp, prefix="medial-quiet-speed-")).resolve()
    stage = "source"
    try:
        source = DRIVER["stage_source"](out, work, plan, prior)
        stage = "runtime"
        python = BOOT["provision_python"](packet, work)
        stage = "donor"
        wheels, package = BOOT["donor_inputs"](packet, work, out)
        stock, pybind, donor = BOOT["install_sites"](packet, wheels, package, python, out, work)
        stage = "portable-record"
        command(out, "verify-portable", ["taskset", "-c", "0", python, "-I", "-S", "-B",
                                         Path(__file__).resolve(), "verify-portable", out, wheels, stock, pybind, donor])
        for wheel in wheels.iterdir():
            wheel.unlink()
        wheels.rmdir()
        stage = "elf-transfer"
        elf = transfer_elf(admission, work, out)
        (work / "authority.json").write_bytes(raw)
        write(out / "stage-ready.json", {"work": str(work), "interpreter": str(python), "stock": str(stock),
              "pybind": str(pybind), "donor": str(donor), "source": str(source), "elf": str(elf), "pid": os.getpid(),
              "portable": seal(out / "portable-installed.json"), "transfer": seal(out / "elf-transfer.json"),
              "authority_sha256": sha256(raw).hexdigest(), "purpose": grant["purpose"]})
    except Exception as exc:
        write(out / "stage-failure.json", {"stage": stage, "error": str(exc)})
        shutil.rmtree(work)
        raise
    return 0


if __name__ == "__main__":
    if not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode and not sys.flags.optimize):
        raise SystemExit("isolated nonoptimized Python required")
    raise SystemExit(main())
