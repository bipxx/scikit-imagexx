"""Provision the admitted Hough donor and run one private circle-peaks check."""
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
from tempfile import TemporaryDirectory
from urllib import error, parse, request
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
REPO = "bipxx/scikit-imagexx"
REF = "refs/heads/diagnostics/u182-circle-peaks-modern-component-v2"
FLOOR = 30 * 2**30


def digest(raw):
    return sha256(raw).hexdigest()


def seal(path):
    return {"bytes": path.stat().st_size, "sha256": digest(path.read_bytes())}


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def command(out, label, argv, *, env=None, check=True):
    argv = [str(item) for item in argv]
    if env is None:
        env = dict(os.environ)
    env.pop("GH_TOKEN", None)
    with (out / (label + ".log")).open("xb") as stream:
        status = subprocess.run(argv, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
    write(out / (label + ".json"), {"argv": argv, "exit": status, "log_sha256": digest((out / (label + ".log")).read_bytes())})
    if check and status:
        raise RuntimeError(f"{label} exited {status}")
    return status


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
    historical = json.loads((HERE / "historical-transport.json").read_bytes())
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
        pybind = donor["pybind_input"]
        for path, pin in [(row["full_qualification"], row) for row in wheel_pins.values()] + [(pybind["full_qualification"], pybind)]:
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
            file = candidate.joinpath(*paths.parts)
            file.parent.mkdir(exist_ok=True, parents=True)
            with file.open("xb") as output:
                output.write(data)
    assert package["sole_core"] == packet["donor"]["core"]["member"]
    write(out / "donor-wheel.json", {"members": len(pins), "record_sha256": pins[record], "core": packet["donor"]["core"]})
    stock, pybind_site, venv = [work / name for name in ("stock", "pybind", "installer")]
    command(out, "venv", [python, "-I", "-B", "-m", "venv", venv])
    install = [venv / "bin/python", "-I", "-B", "-m", "pip", "--isolated", "install", "--no-cache-dir", "--no-index", "--no-deps"]
    command(out, "stock-install", [*install, "--target", stock, *[wheels / name for name in packet["donor"]["wheel_inputs"]
                                                                  if not name.startswith("scikit_imagexx-")]])
    command(out, "pybind-install", [*install, "--target", pybind_site, wheels / PurePosixPath(packet["donor"]["pybind_input"]["full_qualification"]).name])
    for wheel in wheels.iterdir():
        wheel.unlink()
    wheels.rmdir()
    return stock, pybind_site / "pybind11/include", candidate


def authority(env, now):
    if not env.get("CIRCLE_COMPONENT_AUTHORITY") or not env.get("CIRCLE_COMPONENT_AUTHORITY_SHA256"):
        raise SystemExit(2)
    assert env["GITHUB_ACTIONS"] == "true" and env["GITHUB_EVENT_NAME"] == "workflow_dispatch"
    assert (env["GITHUB_REPOSITORY"], env["GITHUB_REF"]) == (REPO, REF)
    raw = env["CIRCLE_COMPONENT_AUTHORITY"].encode()
    assert len(raw) <= 1024 and digest(raw) == env["CIRCLE_COMPONENT_AUTHORITY_SHA256"]
    grant = json.loads(raw)
    assert set(grant) == {"purpose", "carrier", "manifest_sha256", "deadline_utc"}
    assert (grant["purpose"], grant["carrier"]) == ("circle-peaks-modern-component", env["GITHUB_SHA"])
    deadline = datetime.fromisoformat(grant["deadline_utc"])
    assert 0 < (deadline - now).total_seconds() <= 5400
    return grant


def retain_for_timing(out, work, workspace, grant, plan):
    """Retain the exact already-tested ELF after the existing full collector passes."""
    frozen = HERE / "frozen/linux-modern-rows-v1.jsonl"
    raw = frozen.read_bytes()
    assert digest(raw) == plan["stock_rows_sha256"]
    rows = [json.loads(line) for line in raw.splitlines()]
    collector = runpy.run_path(str(HERE / "check-report.py"))
    component = out / "component"
    proof = collector["collect"](component, plan, rows)
    assert proof["sdk_exits"] == [0, 0] and proof["stock_peak_calls"] == proof["timings"] == 0
    assert {name: (mode["native"], mode["controlled_provider"]) for name, mode in proof["modes"].items()} == {
        "normal": (19, 10), "scalar": (19, 10)}
    name = "_hough_circle_peaks_private.cpython-313-x86_64-linux-gnu.so"
    binary = work / "compiled" / name
    assert binary.is_file() and not binary.is_symlink() and 0 < binary.stat().st_size <= 8 * 2**20
    elf = binary.read_bytes()
    assert elf[:4] == b"\x7fELF" and digest(elf) == proof["products"]["private"]
    write(out / "collection.json", proof)
    receipts = {}
    for label in ("entry", "terminal", "products", "normal-rows", "scalar-rows", "sdk-link", "sdk"):
        path = component / (label + ".json")
        receipts[path.name] = seal(path)
    assert receipts["sdk.json"]["sha256"] == proof["processes"]["sdk"]
    assert receipts["sdk-link.json"]["sha256"] == proof["processes"]["sdk-link"]
    assert receipts["normal-rows.json"]["sha256"] == proof["modes"]["normal"]["rows_sha256"]
    assert receipts["scalar-rows.json"]["sha256"] == proof["modes"]["scalar"]["rows_sha256"]
    entry = json.loads((component / "entry.json").read_bytes())
    prepared = {
        "scope": "exact-private-circle-peaks-linux-modern-leaf-for-later-quiet-timing",
        "carrier": grant["carrier"], "manifest_sha256": grant["manifest_sha256"],
        "source": plan["source"]["revision"], "tree": plan["source"]["tree"],
        "runtime": plan["runtime"]["version"], "runtime_executable_sha256": plan["runtime"]["python_sha256"],
        "stock_rows_sha256": plan["stock_rows_sha256"], "fixed": [
            "peak-default-real", "peak-normalized-ranking", "peak-cross-radius-pruning", "peak-capped-equal"],
        "provider_records": {key: {field: value[field] for field in ("sha256", "members")}
                             for key, value in entry["provider_records"].items()},
        "base_wheel_sha256": plan["base"]["wheel_sha256"],
        "base_core_sha256": plan["base"]["members_sha256"][plan["base"]["core_member"]],
        "static_sdk_dependency_sha256": plan["base"]["members_sha256"]["lib/libskimagexx.a"],
        "leaf_sha256": proof["products"]["leaf"], "sdk_product_sha256": proof["products"]["sdk"],
        "modes": proof["modes"], "sdk_exits": proof["sdk_exits"], "row_ids": plan["row_ids"],
        "receipt_pins": receipts, "collection": seal(out / "collection.json"),
        "elf": {"name": name, "bytes": len(elf), "sha256": digest(elf)},
        "stock_peak_calls": 0, "timing_calls": 0,
    }
    target = workspace / "circle-peaks-component-transport"
    target.mkdir()
    with (target / name).open("xb") as stream:
        stream.write(elf)
    write(target / "prepared.json", prepared)


def main():
    assert not sys.flags.optimize
    grant = authority(os.environ, datetime.now(timezone.utc))
    manifest = HERE / "manifest-v1.json"
    assert digest(manifest.read_bytes()) == grant["manifest_sha256"]
    packet = json.loads((HERE / "bootstrap.json").read_bytes())
    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve(strict=True)
    temp = Path(os.environ["RUNNER_TEMP"]).resolve(strict=True)
    assert temp != workspace and not temp.is_relative_to(workspace)
    git = ["/usr/bin/git", "--no-optional-locks", "--no-replace-objects", "-C", str(workspace)]
    revision, tree = subprocess.check_output([*git, "rev-parse", "HEAD", "HEAD^{tree}"], text=True).splitlines()
    assert revision == grant["carrier"]
    assert not subprocess.check_output([*git, "status", "--porcelain=v1", "--untracked-files=all"])
    assert all(os.environ.get(key) == "1" for key in packet["runtime"]["thread_environment"])
    assert not any(os.environ.get(key) for key in ("PYTHONHOME", "PYTHONPATH", "LD_PRELOAD", "LD_AUDIT",
                                                   "NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE", "SKIMAGEXX_DISABLE_SIMD",
                                                   "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"))
    assert 0 in os.sched_getaffinity(0)
    out = workspace / "circle-peaks-component-results"
    out.mkdir()
    capacity = {str(path): shutil.disk_usage(path).free for path in (workspace, temp)}
    write(out / "authority-capacity.json", {"authority": grant, "tree": tree, "free": capacity, "floor": FLOOR,
                                           "staging_bound_bytes": 2 * 2**30})
    stage = "capacity"
    try:
        assert all(free >= FLOOR for free in capacity.values())
        frozen = HERE / "frozen"
        args = [HERE / "component.py", "--manifest", manifest, "--manifest-sha256", grant["manifest_sha256"],
                "--git", git[0], "--git-repo", workspace, "--source", HERE / "source", "--tree", tree,
                "--rows", frozen / "linux-modern-rows-v1.jsonl", "--prepared", frozen,
                "--bridge", HERE / "helpers/private_bridge.cpp", "--storage-source", HERE / "helpers/storage_projection.cpp"]
        stage = "family-source"
        command(out, stage, [sys.executable, "-I", "-S", "-B", *args, "--check-inputs"])
        with TemporaryDirectory(dir=temp, prefix="u182-circle-peaks-component-") as fixture:
            work = Path(fixture)
            stage = "python"
            python = provision_python(packet, work)
            stage = "donor"
            wheels, package = donor_inputs(packet, work, out)
            stage = "donor-install"
            stock, pybind, base = install_sites(packet, wheels, package, python, out, work)
            stage = "private-component"
            command(out, stage, ["/usr/bin/taskset", "-c", "0", python, "-I", "-S", "-B", *args,
                                 "--stock", stock, "--pybind", pybind, "--base", base,
                                 "--output", out / "component", "--scratch", work / "compiled"])
            stage = "retain-verified-elf"
            retain_for_timing(out, work, workspace, grant, json.loads(manifest.read_bytes()))
    except Exception as exc:
        write(out / "terminal.json", {"stage": stage, "exit": 1, "error": type(exc).__name__ + ": " + str(exc)})
        raise
    write(out / "terminal.json", {"stage": stage, "exit": 0, "purpose": grant["purpose"]})


if __name__ == "__main__":
    main()
