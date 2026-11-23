"""Stage the exact previously tested circle-peaks ELF and genuine timing sites."""
from base64 import urlsafe_b64encode
import csv
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO, StringIO
import json
import os
from pathlib import Path, PurePosixPath
import runpy
import shutil
import stat
import subprocess
import sys
import tarfile
from tempfile import mkdtemp
from urllib import error, parse, request
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
REPO = "bipxx/scikit-imagexx"
REF = "refs/heads/diagnostics/u182-circle-peaks-modern-timing-v2"
FLOOR = 30 * 2**30


def digest(raw):
    return sha256(raw).hexdigest()


def seal(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": digest(raw)}


def ref(path):
    return {"path": str(path), **seal(path)}


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def command(out, label, argv):
    cmd = [str(item) for item in argv]
    env = {name: value for name, value in os.environ.items() if name != "GH_TOKEN"}
    with (out / (label + ".log")).open("xb") as stdout:
        status = subprocess.run(cmd, env=env, stdout=stdout, stderr=subprocess.STDOUT).returncode
    write(out / (label + ".json"), {"argv": cmd, "exit": status, "log": seal(out / (label + ".log"))})
    assert status == 0, (label, status)


def chunks(stream, limit):
    data = bytearray()
    while block := stream.read(2**20):
        data.extend(block)
        assert len(data) <= limit
    return bytes(data)


def donor_inputs(packet, work, out):
    donor = packet["donor"]
    artifact = donor["full_qualification_artifact"]
    token = os.environ["GH_TOKEN"]
    assert token
    actions = f"https://api.github.com/repos/{REPO}/actions/"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

    def api(path):
        with request.urlopen(request.Request(actions + path, headers=headers), timeout=20) as response:
            return json.load(response)

    meta = api(f"artifacts/{artifact}")
    job = api(f"jobs/{donor['job']}")
    assert (job["id"], job["run_id"], job["head_sha"], job["status"], job["conclusion"]) == (
        donor["job"], donor["run"], donor["source"], "completed", "success")
    assert (meta["id"], meta["name"], meta["expired"], meta["digest"]) == (
        artifact, "public-qualification-modern", False, donor["full_qualification_outer_digest"])
    historical = json.loads((HERE / "timing-historical.json").read_bytes())
    assert (meta["workflow_run"]["id"], meta["workflow_run"]["head_sha"], meta["workflow_run"]["head_branch"]) == (
        donor["run"], donor["source"], historical["head_branch"])
    assert meta["size_in_bytes"] == historical["size_in_bytes"]
    helper = HERE / "range_helper.py"
    pin = packet["transport"]["exact_helper"]
    assert seal(helper) == {key: pin[key] for key in ("bytes", "sha256")}
    tools = runpy.run_path(str(helper))
    req = request.Request(actions + f"artifacts/{artifact}/zip", headers=headers)
    try:
        request.build_opener(tools["NoRedirect"]).open(req, timeout=20).close()
    except error.HTTPError as exc:
        if exc.code not in (301, 302, 303, 307, 308):
            raise RuntimeError(f"Donor redirect HTTP {exc.code}") from None
        url = exc.headers.get("Location", "")
    else:
        raise RuntimeError("Expected signed donor redirect")
    assert parse.urlparse(url).scheme == "https"
    transport = tools["RemoteZip"](url, meta["size_in_bytes"])
    wheels = work / "wheels"
    wheels.mkdir()
    names = []
    with transport, ZipFile(transport) as archive:
        entries = archive.namelist()
        prefix = historical["namespace"]

        def selected(path, pin):
            member = prefix + path
            info = archive.getinfo(member)
            assert entries.count(member) == 1 and info.file_size == pin["bytes"], member
            compressed = historical["compressed_member_sizes"][path]
            assert info.compress_size == compressed and sum(historical["compressed_member_sizes"].values()) < 96 * 2**20
            with archive.open(member) as file:
                raw = chunks(file, pin["bytes"])
            assert len(raw) == pin["bytes"] and digest(raw) == pin["sha256"], member
            names.append(member)
            return raw

        manifest = json.loads(selected("manifest.json", donor["qualification_manifest"]))
        package = json.loads(selected("package.json", donor["package_manifest"]))
        assert manifest["source"]["revision"] == package["source_revision"] == donor["source"]
        assert manifest["profile"] == "modern" and manifest["passed"] and package["record_verified"]
        assert manifest["records"]["package.json"] == seal_dict(donor["package_manifest"])
        wheel_pins = donor["wheel_inputs"]
        for path, pin in [(row["full_qualification"], row) for row in wheel_pins.values()]:
            assert manifest["records"][path] == seal_dict(pin)
            (wheels / PurePosixPath(path).name).write_bytes(selected(path, pin))
    write(out / "donor-transfer.json", {"artifact": artifact, "members": names, "bytes_read": transport.bytes_read,
                                        "listed_outer_digest": meta["digest"], "whole_outer_verified": False})
    return wheels, package


def seal_dict(pin):
    return {key: pin[key] for key in ("bytes", "sha256")}


def provision_python(packet, work):
    asset, = [row for row in packet["runtime"]["python_release_assets_metadata_only"]
              if row["name"].endswith("-install_only_stripped.tar.gz")]
    url = "https://github.com/astral-sh/python-build-standalone/releases/download/20251010/" + parse.quote(asset["name"])
    with request.urlopen(url, timeout=30) as response:
        raw = chunks(response, asset["bytes"])
    assert len(raw) == asset["bytes"] and digest(raw) == asset["sha256"]
    destination = work / "cp3138"
    destination.mkdir()
    with tarfile.open(fileobj=BytesIO(raw), mode="r:gz") as archive:
        infos = archive.getmembers()
        assert sum(member.size for member in infos) < 768 * 2**20
        assert all(PurePosixPath(member.name).parts[0] == "python" for member in infos)
        archive.extractall(destination, members=infos, filter="data")
    python = destination / "python/bin/python3.13"
    runtime = packet["runtime"]
    assert seal(python)["sha256"] == runtime["executable_sha256"]
    assert subprocess.check_output([python, "-I", "-S", "-B", "-c", "import sys; print(sys.version)"], text=True).strip() == runtime["version"]
    return python


def install_sites(packet, wheels, package, python, out, work):
    candidate = work / "candidate"
    candidate.mkdir()
    name, = [name for name in packet["donor"]["wheel_inputs"] if name.startswith("scikit_imagexx-")]
    raw = (wheels / name).read_bytes()
    assert digest(raw) == package["wheel_sha256"]
    pins = package["members_sha256"]
    with ZipFile(BytesIO(raw)) as archive:
        entries = archive.infolist()
        names = [item.filename for item in entries]
        record, = [name for name in names if name.endswith(".dist-info/RECORD")]
        rows = list(csv.reader(StringIO(archive.read(record).decode(), newline="")))
        row_map = {row[0]: row[1:] for row in rows}
        assert len(entries) == len(row_map) == len(rows) == len(pins) == 260
        assert set(names) == set(row_map) == set(pins) and row_map[record] == ["", ""]
        assert sum(item.file_size for item in entries) <= 128 * 2**20
        for item in entries:
            name = item.filename
            paths = PurePosixPath(name)
            assert paths.parts and not paths.is_absolute() and ".." not in paths.parts and "\\" not in name and ":" not in name
            assert not item.is_dir() and not stat.S_ISLNK(item.external_attr >> 16)
            data = archive.read(item)
            assert digest(data) == pins[name]
            if name != record:
                code = urlsafe_b64encode(bytes.fromhex(pins[name])).decode().rstrip("=")
                assert row_map[name] == ["sha256=" + code, str(len(data))]
            if not (name.startswith("skimagexx/") or name == record):
                continue
            file = candidate.joinpath(*paths.parts)
            file.parent.mkdir(exist_ok=True, parents=True)
            with file.open("xb") as output:
                output.write(data)
    assert package["sole_core"] == packet["donor"]["core"]["member"]
    write(out / "donor-wheel.json", {"members": len(pins), "record_sha256": pins[record], "core": packet["donor"]["core"]})
    stock, venv = (work / name for name in ("stock", "installer"))
    command(out, "venv", [python, "-I", "-B", "-m", "venv", venv])
    install = [venv / "bin/python", "-I", "-B", "-m", "pip", "--isolated", "install", "--no-cache-dir", "--no-index", "--no-deps"]
    command(out, "stock-install", [*install, "--target", stock, *[wheels / name for name in packet["donor"]["wheel_inputs"]
                                                                  if not name.startswith("scikit_imagexx-")]])
    for wheel in wheels.iterdir():
        wheel.unlink()
    wheels.rmdir()
    return stock, candidate


def prepared_transport(packet, grant, work, reports):
    expected = grant["transport"]
    token = os.environ["GH_TOKEN"]
    assert token
    actions = f"https://api.github.com/repos/{REPO}/actions/"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

    def api(label):
        req = request.Request(actions + label, headers=headers)
        with request.urlopen(req, timeout=20) as response:
            return json.load(response)

    key = f"artifacts/{expected['artifact_id']}"
    actual = api(key)
    job = api(f"jobs/{expected['job']}")
    assert (job["id"], job["run_id"], job["head_sha"], job["status"], job["conclusion"]) == (
        expected["job"], expected["run"], grant["producer_source"], "completed", "success")
    assert (actual["id"], actual["name"], actual["expired"], actual["digest"], actual["expires_at"], actual["size_in_bytes"]) == (
        expected["artifact_id"], expected["artifact_name"], False, expected["artifact_digest"],
        expected["expires_at"], expected["verified_zip_bytes"])
    assert (actual["workflow_run"]["id"], actual["workflow_run"]["head_sha"]) == (expected["run"], grant["producer_source"])
    helper = HERE / "range_helper.py"
    assert seal(helper) == seal_dict(packet["transport"]["exact_helper"])
    no_redirect = runpy.run_path(str(helper))["NoRedirect"]
    req = request.Request(actions + key + "/zip", headers=headers)
    try:
        request.build_opener(no_redirect).open(req, timeout=20).close()
    except error.HTTPError as exc:
        if exc.code not in (301, 302, 303, 307, 308):
            raise RuntimeError(f"Prepared artifact redirect HTTP {exc.code}") from None
        url = exc.headers.get("Location", "")
    else:
        raise RuntimeError("Expected a prepared artifact redirect")
    assert parse.urlparse(url).scheme == "https"
    size = expected["verified_zip_bytes"]
    assert 0 < size <= 4 * 2**20
    with request.urlopen(url, timeout=20) as response:
        raw = chunks(response, size)
    assert len(raw) == size and "sha256:" + digest(raw) == expected["artifact_digest"]
    names = expected["members"]
    destination = work / "prepared"
    destination.mkdir()
    with ZipFile(BytesIO(raw)) as archive:
        infos = archive.infolist()
        assert len(infos) == len(names) == 2 and {entry.filename for entry in infos} == set(names)
        assert sum(entry.file_size for entry in infos) <= 4 * 2**20
        for info in infos:
            label = info.filename
            assert PurePosixPath(label).name == label and not info.is_dir()
            assert not stat.S_ISLNK(info.external_attr >> 16)
            content = archive.read(info)
            assert {"bytes": len(content), "sha256": digest(content)} == names[label]
            with (destination / label).open("xb") as stream:
                stream.write(content)
    write(reports / "prepared-transport.json", {"expected": expected, "transfer_bytes": len(raw), "archive_retained": False})
    return destination


def authority(env, now):
    if not env.get("CIRCLE_TIMING_AUTHORITY") or not env.get("CIRCLE_TIMING_AUTHORITY_SHA256"):
        raise SystemExit(2)
    assert (env["GITHUB_ACTIONS"], env["GITHUB_EVENT_NAME"], env["GITHUB_REPOSITORY"], env["GITHUB_REF"]) == (
        "true", "workflow_dispatch", REPO, REF)
    raw = env["CIRCLE_TIMING_AUTHORITY"].encode()
    assert len(raw) <= 2048 and digest(raw) == env["CIRCLE_TIMING_AUTHORITY_SHA256"]
    value = json.loads(raw)
    assert set(value) == {"purpose", "carrier", "plan_sha256", "producer_source", "transport", "deadline_utc"}
    plan = (HERE / "timing-plan.json").read_bytes()
    assert value["plan_sha256"] == digest(plan)
    expected = json.loads(plan)
    assert (value["purpose"], value["carrier"], value["producer_source"]) == (
        "circle-peaks-four-public-modern-quiet", env["GITHUB_SHA"], expected["producer"])
    assert value["transport"] == expected["transport"]
    deadline = datetime.fromisoformat(value["deadline_utc"])
    assert 0 < (deadline - now).total_seconds() <= 5400
    assert deadline < datetime.fromisoformat(value["transport"]["expires_at"])
    return value


def verify_source(workspace, plan, grant):
    git = ["/usr/bin/git", "--no-optional-locks", "--no-replace-objects", "-C", str(workspace)]
    head, tree = subprocess.check_output([*git, "rev-parse", "HEAD", "HEAD^{tree}"], text=True).splitlines()
    assert head == grant["carrier"]
    assert not subprocess.check_output([*git, "status", "--porcelain=v1", "--untracked-files=all"])
    for name, value in plan["files"].items():
        rel = (HERE / name).relative_to(workspace).as_posix()
        raw = subprocess.check_output([*git, "cat-file", "blob", f"{head}:{rel}"])
        assert seal(HERE / name) == value and digest(raw) == value["sha256"] and len(raw) == value["bytes"]
    return {"head": head, "tree": tree, "verified_files": len(plan["files"])}


def main():
    assert not sys.flags.optimize
    now = datetime.now(timezone.utc)
    grant = authority(os.environ, now)
    plan = json.loads((HERE / "timing-plan.json").read_bytes())
    workspace, temp = (Path(os.environ[name]).resolve(strict=True) for name in ("GITHUB_WORKSPACE", "RUNNER_TEMP"))
    assert workspace != temp and not temp.is_relative_to(workspace)
    assert not any(os.environ.get(name) for name in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_AUDIT",
                                                     "NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE", "SKIMAGEXX_DISABLE_SIMD",
                                                     "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"))
    source = verify_source(workspace, plan, grant)
    packet = json.loads((HERE / "timing-bootstrap.json").read_bytes())
    assert all(os.environ.get(name) == "1" for name in packet["runtime"]["thread_environment"])
    assert 0 in os.sched_getaffinity(0)
    reports = workspace / "circle-timing-stage"
    reports.mkdir()
    available = {str(path): shutil.disk_usage(path).free for path in (workspace, temp)}
    write(reports / "capacity.json", {"available_bytes": available, "required_bytes": FLOOR, "provision_bound_bytes": 2 * 2**30})
    assert all(item >= FLOOR for item in available.values())
    work = Path(mkdtemp(dir=temp, prefix="u182-circle-timing-v1-")).resolve(strict=True)
    assert work.parent == temp
    stage = "prepared-transport"
    try:
        prepared = prepared_transport(packet, grant, work, reports)
        record = json.loads((prepared / "prepared.json").read_bytes())
        assert record["source"] == plan["source"] and record["tree"] == plan["tree"]
        assert record["stock_rows_sha256"] == plan["stock_rows_sha256"] and record["sdk_exits"] == [0, 0]
        assert record["fixed"] == [pin["id"] for pin in plan["fixed"]] and record["elf"] == plan["elf"]
        assert all((pin["native"], pin["controlled_provider"]) == (19, 10) for pin in record["modes"].values())
        stage = "runtime"
        python = provision_python(packet, work)
        stage = "donor"
        wheels, package = donor_inputs(packet, work, reports)
        assert package["wheel_sha256"] == record["base_wheel_sha256"]
        stage = "stock-and-base"
        stock, base = install_sites(packet, wheels, package, python, reports, work)
        assert not (work / "wheels").exists()
        stage = "bind"
        with (work / "authority.json").open("xb") as output:
            output.write(os.environ["CIRCLE_TIMING_AUTHORITY"].encode())
        binding = {"authority_sha256": os.environ["CIRCLE_TIMING_AUTHORITY_SHA256"], "transport": grant["transport"],
                   "source": source, "stock": str(stock), "base": str(base), "elf": ref(prepared / plan["elf"]["name"]),
                   "prepared": ref(prepared / "prepared.json")}
        draft = work / "binding-draft.json"
        write(draft, binding)
        write(reports / "stage-ready.json", {"work": str(work), "pid": os.getpid(), "interpreter": str(python),
                                             "authority_sha256": binding["authority_sha256"], "binding_draft": ref(draft)})
    except Exception as exc:
        write(reports / "stage-failure.json", {"stage": stage, "error": type(exc).__name__ + ": " + str(exc)})
        shutil.rmtree(work)
        raise


if __name__ == "__main__":
    main()
