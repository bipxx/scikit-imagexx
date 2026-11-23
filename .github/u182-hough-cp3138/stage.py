"""Verified staging for a separate Hough timing step."""
from base64 import urlsafe_b64encode
import csv
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO, StringIO
from datetime import datetime, timezone
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
BOOT = "cc4c9890b8a48baac2862a6dd5524a84d8e75607cf06d2eafe05c4721ca4a8c8"
PREP_ADMISSION = "feb6438b619cfd8050dfdd0f637f3615b58e873849a43cb1c9c99ff4bb4c23b6"
FAMILY = "hough_line"
FLOOR = 30 * 2**30


def digest(data):
    return sha256(data).hexdigest()


def seal(path):
    return {"bytes": path.stat().st_size, "sha256": digest(path.read_bytes())}


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


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


def check_sources(packet, variant, repo, git, local_baseline):
    family_id = "transform." + FAMILY
    verifier = runpy.run_path(str(HERE / "verify-inputs.py"))["source_proof"]
    result = verifier(packet, repo, Path(git), local_baseline, family_id)
    assert result == {"families": 1, "owned_paths": 10, "windows_local_preflight": local_baseline}
    args = [git, "--no-optional-locks", "--no-replace-objects", "-C", repo]

    def read(*argv):
        return subprocess.check_output([*args, *argv])

    source = read("rev-parse", "--verify", variant["local_ref"]).decode().strip()
    assert source == variant["source"]
    assert read("rev-list", "--parents", "-n", "1", source).decode().split() == [source, variant["parent"]]
    assert read("rev-parse", f"{source}^{{tree}}").decode().strip() == variant["tree"]
    path = variant["sole_changed_source"]
    assert read("diff-tree", "--no-commit-id", "--name-status", "-r", variant["parent"], source).decode().splitlines() == [f"M\t{path}"]
    before = read("cat-file", "blob", f"{variant['parent']}:{path}")
    wide = read("cat-file", "blob", f"{source}:{path}")
    assert digest(before) == variant["old_source_sha256"] and digest(wide) == variant["new_source_sha256"]
    old_line, new_line = (variant[key].encode() for key in ("old_line", "new_line"))
    assert before.count(old_line) == 1 and before.replace(old_line, new_line) == wide
    return read, before, wide


def stage_source(packet, variant, out, work):
    mirror = work / "source.git"
    git = "/usr/bin/git"
    command(out, "git-init", [git, "init", "--bare", mirror])
    repo_git = [git, "-C", mirror]
    command(out, "git-origin", [*repo_git, "remote", "add", "origin", f"https://github.com/{REPO}.git"])
    for key, value in (
        ("remote.origin.promisor", "true"), ("remote.origin.partialclonefilter", "blob:none")
    ):
        command(out, "git-" + key.rsplit(".", 1)[-1], [*repo_git, "config", key, value])
    family_id = "transform." + FAMILY
    family = packet["families"][family_id]
    refs = (family["ref"], variant["local_ref"])
    command(out, "git-fetch", [*repo_git, "fetch", "--no-tags", "--depth=2", "--filter=blob:none", "origin",
                               *[f"{ref}:{ref}" for ref in refs]])
    read, before, wide = check_sources(packet, variant, mirror, git, False)
    source = work / FAMILY
    source.mkdir()
    for path in family["owned_git_files"] | family["verified_shared_git_files"]:
        raw = wide if path == variant["sole_changed_source"] else read("cat-file", "blob", f"{variant['source']}:{path}")
        path = source / path
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("xb") as output:
            output.write(raw)
    write(out / "source-proof.json", {"before": variant["parent"], "wide": variant["source"], "tree": variant["tree"],
                                     "file": variant["sole_changed_source"], "other_source_files_unchanged": True,
                                     "before_sha256": digest(before), "wide_sha256": digest(wide)})
    return source


def authority(env, now):
    assert env["GITHUB_ACTIONS"] == "true" and env["GITHUB_REPOSITORY"] == REPO
    assert env["GITHUB_REF"] == "refs/heads/diagnostics/u182-hough-line-wide-speed-v1"
    assert env["GITHUB_EVENT_NAME"] == "workflow_dispatch"
    raw = env["HOUGH_AUTHORITY"].encode()
    assert 0 < len(raw) <= 8192 and digest(raw) == env["HOUGH_AUTHORITY_SHA256"]
    value = json.loads(raw)
    assert set(value) == {"carrier", "purpose", "deadline_utc", "prep_admission_sha256", "producer_source",
                          "transport", "producer_manifest_sha256", "elf_sha256", "bootstrap_sha256",
                          "fixed_case_plan_sha256", "sampler_sha256", "threadpoolctl_sha256", "timing_entry_sha256"}
    assert (value["carrier"], value["purpose"], value["prep_admission_sha256"]) == (
        env["GITHUB_SHA"], "hough-line-wide-modern-fixed-speed", PREP_ADMISSION)
    deadline = datetime.fromisoformat(value["deadline_utc"])
    assert 0 < (deadline - now).total_seconds() <= 5400
    admitted = (HERE / "line-wide-admission.json").read_bytes()
    assert digest(admitted) == PREP_ADMISSION
    record = json.loads(admitted)
    transport_raw = (HERE / "wide-transport.json").read_bytes()
    plan = json.loads((HERE / "wide-speed-plan.json").read_bytes())
    assert digest(transport_raw) == plan["admitted_transport_sha256"]
    assert plan["wide_component_admission_sha256"] == PREP_ADMISSION
    transport = json.loads(transport_raw)
    assert (transport["run"], transport["job"], transport["artifact_id"], transport["members"]) == (
        record["run"], record["job"], record["transport_artifact"], record["verified"]["transport"])
    assert value["transport"] == transport and deadline < datetime.fromisoformat(transport["expires_at"])
    assert value["producer_source"] == record["source"]
    assert value["producer_manifest_sha256"] == transport["members"]["line-wide.json"]["sha256"]
    assert value["elf_sha256"] == {FAMILY: record["verified"]["elf"]["sha256"]}
    for key, path in (
        ("bootstrap_sha256", "bootstrap-v1.json"), ("fixed_case_plan_sha256", "wide-speed-plan.json"),
        ("sampler_sha256", "sampler.py"), ("threadpoolctl_sha256", "threadpoolctl.py"), ("timing_entry_sha256", "quiet-timing.py")
    ):
        assert value[key] == seal(HERE / path)["sha256"]
    return value


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
        expected["job"], expected["run"], "58894db60ba6a526fd379f1417d31dcc07574902", "completed", "success")
    assert (actual["id"], actual["name"], actual["expired"], actual["digest"], actual["expires_at"], actual["size_in_bytes"]) == (
        expected["artifact_id"], expected["artifact_name"], False, expected["artifact_digest"],
        expected["expires_at"], expected["verified_zip_bytes"])
    assert (actual["workflow_run"]["id"], actual["workflow_run"]["head_sha"]) == (expected["run"], job["head_sha"])
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


def entry():
    assert sys.platform == "linux" and sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode and not sys.flags.optimize
    grant = authority(os.environ, datetime.now(timezone.utc))
    packet_raw = (HERE / "bootstrap-v1.json").read_bytes()
    assert digest(packet_raw) == BOOT
    packet = json.loads(packet_raw)
    for label, pin in json.loads((HERE / "timing-inputs.json").read_bytes()).items():
        assert seal(HERE / label) == pin
    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve(strict=True)
    temp = Path(os.environ["RUNNER_TEMP"]).resolve(strict=True)
    assert temp != workspace and not temp.is_relative_to(workspace)
    reports = workspace / "hough-timing-stage"
    reports.mkdir()
    available = {str(path): shutil.disk_usage(path).free for path in (workspace, temp)}
    write(reports / "capacity.json", {"available_bytes": available, "required_bytes": FLOOR})
    assert all(size >= FLOOR for size in available.values())
    assert 0 in os.sched_getaffinity(0)
    assert all(os.environ.get(key) == "1" for key in packet["runtime"]["thread_environment"])
    work = Path(mkdtemp(dir=temp, prefix="u182-hough-timing-v1-"))
    try:
        variant = json.loads((HERE / "wide-line.json").read_bytes())
        source = stage_source(packet, variant, reports, work)
        transport = prepared_transport(packet, grant, work, reports)
        python = provision_python(packet, work)
        wheels, package = donor_inputs(packet, work, reports)
        stock, pybind, candidate = install_sites(packet, wheels, package, python, reports, work)
        command(reports, "runtime-proof", ["taskset", "-c", "0", python, "-I", "-S", "-B",
                HERE / "verify-inputs.py", "--packet", HERE / "bootstrap-v1.json", "--packet-sha256", BOOT,
                "runtime", "--stock", stock, "--pybind", pybind, "--candidate", candidate])
        manifest = json.loads((transport / "line-wide.json").read_bytes())
        assert manifest["no_timing"] and (manifest["source"], manifest["tree"]) == (variant["source"], variant["tree"])
        runtime = reports / "runtime-proof.log"
        assert json.loads(runtime.read_bytes()) == json.loads((HERE / "v3-reuse-v1.json").read_bytes())["runtime"]

        def ref(path):
            return {"path": str(path), **seal(path)}

        owned = packet["families"]["transform." + FAMILY]["owned_git_files"]
        contract, = [path for path in owned if path.startswith("coverage/contracts/") and path.endswith(".json")]
        fixed = {"source_root": str(source), "elf": ref(transport / manifest["elf"]["name"]),
                 "recipe": ref(source / "benchmarks/hough_line.py"),
                 "fixture": ref(source / "tests/hough_line_reference.py"),
                 "contract": ref(source / contract),
                 "facade": ref(source / "python/skimagexx/_hough_line.py"),
                 "stock_rows": ref(HERE / "hough_line-stock-0.json")}
        binding = {"authority_sha256": os.environ["HOUGH_AUTHORITY_SHA256"], "transport": grant["transport"],
                   "bootstrap": ref(HERE / "bootstrap-v1.json"), "fixed_case_plan": ref(HERE / "wide-speed-plan.json"),
                   "runtime_proof": ref(runtime), "producer_manifest": ref(transport / "line-wide.json"),
                   "sampler": ref(HERE / "sampler.py"), "threadpoolctl": ref(HERE / "threadpoolctl.py"),
                   "candidate_site": str(candidate), "stock_site": str(stock), "families": {FAMILY: fixed}}
        with (work / "authority.json").open("xb") as stream:
            stream.write(os.environ["HOUGH_AUTHORITY"].encode())
        write(work / "binding-draft.json", binding)
        write(reports / "stage-ready.json", {"authority_sha256": binding["authority_sha256"], "work": str(work),
                                             "interpreter": str(python), "pid": os.getpid(),
                                             "binding_draft": ref(work / "binding-draft.json")})
    except Exception:
        shutil.rmtree(work)
        raise


if __name__ == "__main__":
    entry()
