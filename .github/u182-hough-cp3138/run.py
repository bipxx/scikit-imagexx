"""Authorized hosted Hough input and direct-component diagnostic."""
from base64 import urlsafe_b64encode
import csv
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO, StringIO
import json
import os
from pathlib import Path, PurePosixPath
import re
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
BOOT = "1069660c7a185703c4f75875ebdf865cbd32690ec50a3cc59d0a403e189dd703"
FLOOR = 30 * 2**30
FLAGS = ["-std=c++17", "-O3", "-fPIC", "-fvisibility=hidden", "-fno-fast-math", "-ffp-contract=off", "-frounding-math"]
CAP = 150 * 2**20


def digest(data):
    return sha256(data).hexdigest()


def seal(path):
    return {"bytes": path.stat().st_size, "sha256": digest(path.read_bytes())}


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def authority(env, now):
    assert env["GITHUB_ACTIONS"] == "true" and env["GITHUB_REPOSITORY"] == REPO
    assert env["GITHUB_REF"] == "refs/heads/diagnostics/u182-hough-cp3138-direct-v1"
    assert env["GITHUB_EVENT_NAME"] == "workflow_dispatch"
    raw = env["HOUGH_AUTHORITY"].encode()
    assert len(raw) < 1024 and digest(raw) == env["HOUGH_AUTHORITY_SHA256"]
    value = json.loads(raw)
    assert set(value) == {"carrier", "packet", "purpose", "deadline_utc"}
    assert (value["carrier"], value["packet"]) == (env["GITHUB_SHA"], BOOT)
    assert value["purpose"] in {"bootstrap", "components"}
    deadline = datetime.fromisoformat(value["deadline_utc"])
    assert 0 < (deadline - now).total_seconds() <= 5400
    return value


def command(out, label, argv, *, env=None, check=True):
    argv = [str(item) for item in argv]
    with (out / (label + ".log")).open("xb") as file:
        status = subprocess.run(argv, env=env, stdout=file, stderr=subprocess.STDOUT, check=False).returncode
    write(out / (label + ".json"), {"argv": argv, "exit": status})
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


def stage_source(packet, out, work):
    mirror = work / "source.git"
    git = "/usr/bin/git"
    command(out, "git-init", [git, "init", "--bare", mirror])
    repo_git = [git, "-C", mirror]
    command(out, "git-origin", [*repo_git, "remote", "add", "origin", f"https://github.com/{REPO}.git"])
    for key, value in (
        ("remote.origin.promisor", "true"), ("remote.origin.partialclonefilter", "blob:none")
    ):
        command(out, "git-" + key.rsplit(".", 1)[-1], [*repo_git, "config", key, value])
    refs = [row["ref"] for row in packet["families"].values()]
    command(out, "git-fetch", [*repo_git, "fetch", "--no-tags", "--depth=2", "--filter=blob:none", "origin",
                               *[f"{ref}:{ref}" for ref in refs]])
    command(out, "source-proof", [sys.executable, "-I", "-S", "-B", HERE / "verify-inputs.py", "--packet",
                                  HERE / "bootstrap-v1.json", "--packet-sha256", BOOT, "source", "--repo", mirror, "--git", git])
    source_roots = {}
    for name, family in packet["families"].items():
        name = name.rsplit(".", 1)[1]
        destination = work / name
        destination.mkdir()
        for path in family["owned_git_files"] | family["verified_shared_git_files"]:
            raw = subprocess.check_output([*repo_git, "cat-file", "blob", f"{family['source']}:{path}"])
            path = destination / path
            path.parent.mkdir(exist_ok=True, parents=True)
            with path.open("xb") as output:
                output.write(raw)
        source_roots[name] = destination
    return source_roots


def verify_tools(packet, out):
    tools = packet["runtime"]["compiler_reference"]
    compiler = Path("/usr/bin/g++")
    actual = {key: seal(Path(row["path"])) for key, row in tools["inputs"].items()
              if key in {"/usr/bin/g++", "/usr/bin/readelf"}}
    for key in ("cc1plus", "ld"):
        row = tools[key]
        actual[row["path"]] = seal(Path(row["path"]))
    actual["version"] = subprocess.check_output([compiler, "--version"], text=True).splitlines()[0]
    cc1plus = subprocess.check_output([compiler, "-print-prog-name=cc1plus"], text=True).strip()
    actual["cc1plus"] = cc1plus
    write(out / "toolchain.json", actual)
    for key, pin in tools["inputs"].items():
        if key in actual:
            assert actual[key] == seal_dict(pin)
    assert all(actual[tools[key]["path"]] == seal_dict(tools[key]) for key in ("cc1plus", "ld"))
    assert actual["version"] == tools["gnu"] and Path(cc1plus).resolve() == Path(tools["cc1plus"]["path"]).resolve()
    return compiler


def components(packet, compiler, out, work, source_roots, python, stock, pybind, candidate):
    include = subprocess.check_output([python, "-I", "-S", "-B", "-c", "import sysconfig; print(sysconfig.get_path('include'))"], text=True).strip()
    failures = []
    children_env = {**os.environ, "TMPDIR": str(work)}
    children_env.pop("GH_TOKEN", None)
    for name, source in source_roots.items():
        try:
            leaf = source / "src" / (name + ".cpp")
            binary = work / ("_" + name + "_private" + packet["runtime"]["ext_suffix"])
            cpp = [compiler, *FLAGS, "-I" + str(source / "include")]
            obj = work / (name + ".o")
            command(out, name + "-leaf", [*cpp, "-c", leaf, "-o", obj], env=children_env)
            command(out, name + "-link", [*cpp, "-I" + str(source / "src"), "-I" + include, "-I" + str(pybind),
                    "-shared", source / "src" / ("bind_" + name + ".cpp"), HERE / "storage_projection.cpp",
                    HERE / (name + "-bridge.cpp"), obj, "-o", binary], env=children_env)
            exports = subprocess.check_output(["/usr/bin/readelf", "-hWs", binary], text=True)
            assert "ELF64" in exports and "X86-64" in exports and re.search(r"\bPyInit__" + name + r"_private\b", exports)
            write(out / (name + "-elf.json"), seal(binary))
            assert sum(path.stat().st_size for path in work.iterdir() if path.is_file()) <= CAP
            program = work / (name + "-sdk")
            command(out, name + "-sdk-link", [*cpp, source / "tests/cmake_consumer" / (name + "_consumer.cpp"), obj, "-o", program],
                    env=children_env)
            for mode in ("normal", "scalar"):
                env = dict(children_env)
                env.pop("SKIMAGEXX_DISABLE_SIMD", None)
                if mode == "scalar":
                    env["SKIMAGEXX_DISABLE_SIMD"] = "1"
                if command(out, name + "-" + mode + "-sdk", ["taskset", "-c", "0", program], env=env, check=False):
                    failures.append((name, mode, "sdk"))
                if command(out, name + "-" + mode, ["taskset", "-c", "0", python, "-I", "-S", "-B", HERE / "parity.py",
                           name, mode, source, binary, stock, candidate, out / (name + "-" + mode + "-cases.json")],
                           env=env, check=False):
                    failures.append((name, mode, "parity"))
        except (RuntimeError, AssertionError, subprocess.CalledProcessError) as exc:
            failures.append((name, "build/setup", str(exc)))
    return failures


def main():
    value = authority(os.environ, datetime.now(timezone.utc))
    raw = (HERE / "bootstrap-v1.json").read_bytes()
    assert digest(raw) == BOOT
    packet = json.loads(raw)
    pins = json.loads((HERE / "carrier-inputs.json").read_bytes())
    for path, pin in pins.items():
        assert seal(HERE / path) == pin, path
    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve(strict=True)
    temp = Path(os.environ["RUNNER_TEMP"]).resolve(strict=True)
    assert temp != workspace and not temp.is_relative_to(workspace)
    out = workspace / "hough-component-results"
    out.mkdir()
    capacity = {str(path): shutil.disk_usage(path).free for path in (workspace, temp)}
    write(out / "authority-capacity.json", {"authority": value, "observed_free_bytes": capacity,
                                            "required_bytes": FLOOR, "staging_bound_bytes": 2 * 2**30})
    assert all(size >= FLOOR for size in capacity.values())
    assert all(os.environ.get(name) == "1" for name in packet["runtime"]["thread_environment"])
    assert 0 in os.sched_getaffinity(0)
    stage = "compiler-proof"
    try:
        if value["purpose"] == "components":
            compiler = verify_tools(packet, out)
        with TemporaryDirectory(dir=temp, prefix="u182-hough-cp3138-") as fixture:
            work = Path(fixture)
            stage = "sources"
            sources = stage_source(packet, out, work)
            stage = "runtime"
            python = provision_python(packet, work)
            stage = "donor"
            wheels, package = donor_inputs(packet, work, out)
            stock, pybind, candidate = install_sites(packet, wheels, package, python, out, work)
            stage = "runtime-proof"
            command(out, "runtime-proof", ["taskset", "-c", "0", python, "-I", "-S", "-B", HERE / "verify-inputs.py",
                    "--packet", HERE / "bootstrap-v1.json", "--packet-sha256", BOOT, "runtime", "--stock", stock,
                    "--pybind", pybind, "--candidate", candidate])
            if value["purpose"] == "components":
                stage = "components"
                assert shutil.disk_usage(work).free >= FLOOR
                failures = components(packet, compiler, out, work, sources, python, stock, pybind, candidate)
                assert not failures, failures
    except Exception as exc:
        write(out / "terminal.json", {"stage": stage, "exit": 1, "error": type(exc).__name__ + ": " + str(exc)})
        raise
    write(out / "terminal.json", {"stage": stage, "exit": 0, "purpose": value["purpose"], "science_scope": "private component only"})


if __name__ == "__main__":
    main()
