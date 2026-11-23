"""Run only the exact modern private circle-peaks ABI and SDK on a prepared host."""
from argparse import ArgumentParser
from base64 import urlsafe_b64encode
import csv
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import sysconfig

HERE = Path(__file__).resolve().parent
FLOOR = 30 * 2**30
CAP = 150 * 2**20
FLAGS = ("-std=c++17", "-O3", "-fPIC", "-fvisibility=hidden", "-fno-fast-math", "-ffp-contract=off",
         "-frounding-math", "-fopenmp")


def digest(raw):
    return sha256(raw).hexdigest()


def check(path, pin):
    raw = path.read_bytes()
    assert digest(raw) == pin["sha256"] and len(raw) == pin["bytes"], path
    return raw


def write(path, row):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(row, stream, indent=2)
        stream.write("\n")


def source_gate(args, plan, require_checkout_files):
    git = [str(args.git), "--no-optional-locks", "--no-replace-objects", "-C", str(args.git_repo)]

    def read(*words):
        return subprocess.check_output([*git, *words])

    src = plan["source"]
    tree = read("rev-parse", args.tree).decode().strip()
    if not args.check_inputs:
        assert tree == read("rev-parse", "HEAD^{tree}").decode().strip()
    for name, pin in src["files"].items():
        git_path = ".github/u182-circle-peaks-component-v1/source/" + name
        assert read("rev-parse", f"{tree}:{git_path}").decode().strip() == pin["blob"], name
        raw = read("cat-file", "blob", f"{tree}:{git_path}")
        assert digest(raw) == pin["sha256"] and len(raw) == pin["bytes"], name
        assert (args.source / name).read_bytes().replace(b"\r\n", b"\n") == raw.replace(b"\r\n", b"\n"), name
    data = check(args.rows, plan["inputs"]["linux-modern-rows-v1.jsonl"])
    assert digest(data) == plan["stock_rows_sha256"]
    rows = [json.loads(line) for line in data.splitlines()]
    assert [row["id"] for row in rows] == plan["row_ids"] and len(rows) == len(set(plan["row_ids"])) == 29
    assert sum(row["scope"] not in ("compatibility", "error") for row in rows) == 19
    for row in rows:
        assert row["inputs_unchanged"] is True
        if row["scope"] == "compatibility":
            assert len(row["result"]) == 4 and all("data_hex" in output for output in row["result"])
    for name, pin in plan["inputs"].items():
        if name == "linux-modern-rows-v1.jsonl":
            continue
        if name.endswith(".npy"):
            path = args.prepared / name
        else:
            path = {"private_bridge.cpp": args.bridge, "storage_projection.cpp": args.storage_source}[name]
        check(path, pin)
    check(HERE / "parity.py", plan["parity"])


def scientific_inputs(args, plan):
    assert sys.platform == "linux" and sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
    assert not sys.flags.optimize and not {"numpy", "scipy", "skimage", "skimagexx"} & sys.modules.keys()
    assert not any(os.environ.get(name) for name in ("PYTHONHOME", "PYTHONPATH", "LD_PRELOAD", "LD_AUDIT",
                                                    "NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE", "SKIMAGEXX_DISABLE_SIMD",
                                                    "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"))
    assert sys.version == plan["runtime"]["version"] and os.sched_getaffinity(0) == {0}
    assert digest(Path(sys.executable).resolve().read_bytes()) == plan["runtime"]["python_sha256"]
    assert sysconfig.get_config_var("EXT_SUFFIX") == ".cpython-313-x86_64-linux-gnu.so"
    assert all(os.environ.get(name) == "1" for name in plan["runtime"]["threads"])
    include = Path(sysconfig.get_path("include"))
    for name, pin in plan["runtime"]["python_headers"].items():
        check(include / name, pin)
    for name, pin in plan["runtime"]["pybind_headers"].items():
        check(args.pybind / "pybind11" / name, pin)
    tools = plan["tools"]
    for name, pin in tools.items():
        check(Path(name), pin)
    cc1 = subprocess.check_output(["/usr/bin/g++", "-print-prog-name=cc1plus"], text=True).strip()
    ld = subprocess.check_output(["/usr/bin/g++", "-print-prog-name=ld"], text=True).strip()
    assert cc1 == "/usr/libexec/gcc/x86_64-linux-gnu/13/cc1plus"
    assert Path(shutil.which(ld) or ld).resolve() == Path("/usr/bin/ld").resolve()
    providers = tuple(metadata.distributions(path=[str(args.stock)]))
    proofs = {}
    for key, group in plan["runtime"]["distributions"].items():
        dist, = [dist for dist in providers if dist.metadata["Name"].lower().replace("-", "_") == key]
        assert dist.version == group["version"]
        fp, = [name for name in dist.files if name.as_posix().endswith(".dist-info/RECORD")]
        record = Path(dist.locate_file(fp)).resolve(strict=True)
        assert record.is_relative_to(args.stock)
        raw = record.read_bytes()
        rows = list(csv.reader(raw.decode().splitlines()))
        entries = {row[0]: row[1:] for row in rows}
        assert len(rows) == len(entries)
        for name in group["names"]:
            path = Path(dist.locate_file(name)).resolve(strict=True)
            assert path.is_relative_to(args.stock)
            pin = plan["runtime"]["provider_files"][name]
            check(path, pin)
            token = urlsafe_b64encode(bytes.fromhex(pin["sha256"])).decode().rstrip("=")
            assert entries[name] == ["sha256=" + token, str(pin["bytes"])]
        proofs[key] = {"record": str(record), "sha256": digest(raw), "members": len(group["names"])}
    for name, pin in plan["base"]["members_sha256"].items():
        path = (args.base / name).resolve(strict=True)
        assert path.is_relative_to(args.base) and digest(path.read_bytes()) == pin
    assert not (args.stock / "skimagexx").exists() and args.base != args.stock
    return include, proofs


def command(out, label, argv, work, env, check_exit=True):
    argv = [str(item) for item in argv]
    with (out / (label + ".log")).open("xb") as stream:
        result = subprocess.run(argv, cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT)
    write(out / (label + ".json"), {"argv": argv, "exit": result.returncode,
                                   "log_sha256": digest((out / (label + ".log")).read_bytes())})
    if check_exit and result.returncode:
        raise RuntimeError(f"{label} exited {result.returncode}")
    return result.returncode


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("manifest", "git", "git-repo", "source", "rows", "prepared", "bridge", "storage-source"):
        parser.add_argument("--" + name, required=True, type=lambda value: Path(value).resolve(strict=True))
    parser.add_argument("--tree", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--check-inputs", action="store_true")
    for name in ("pybind", "stock", "base", "output", "scratch"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve())
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    assert digest(raw) == args.manifest_sha256
    plan = json.loads(raw)
    if args.check_inputs:
        source_gate(args, plan, False)
        print(json.dumps({"source_files": len(plan["source"]["files"]), "rows": len(plan["row_ids"]),
                          "native_expected": 19, "controlled_compatibility": 10, "science_imports": 0}))
        return 0
    if not os.environ.get("CIRCLE_COMPONENT_AUTHORITY") or not os.environ.get("CIRCLE_COMPONENT_AUTHORITY_SHA256"):
        print("Root authority is required", file=sys.stderr)
        return 2
    grant_raw = os.environ["CIRCLE_COMPONENT_AUTHORITY"].encode()
    assert len(grant_raw) <= 1024 and digest(grant_raw) == os.environ["CIRCLE_COMPONENT_AUTHORITY_SHA256"]
    grant = json.loads(grant_raw)
    assert set(grant) == {"purpose", "carrier", "manifest_sha256", "deadline_utc"}
    assert os.environ["GITHUB_ACTIONS"] == "true" and os.environ["GITHUB_EVENT_NAME"] == "workflow_dispatch"
    assert os.environ["GITHUB_REPOSITORY"] == "bipxx/scikit-imagexx" and args.git == Path("/usr/bin/git").resolve()
    assert (grant["purpose"], grant["carrier"], grant["manifest_sha256"]) == (
        "circle-peaks-modern-component", os.environ["GITHUB_SHA"], args.manifest_sha256)
    actual_carrier = subprocess.check_output(["/usr/bin/git", "-C", os.environ["GITHUB_WORKSPACE"], "rev-parse", "HEAD"], text=True).strip()
    assert actual_carrier == grant["carrier"]
    deadline = datetime.fromisoformat(grant["deadline_utc"])
    assert 0 < (deadline - datetime.now(timezone.utc)).total_seconds() <= 5400
    assert all(path.is_dir() for path in (args.pybind, args.stock, args.base))
    assert args.rows.parent == args.prepared  # child sees one immutable stock fixture directory
    assert not args.output.exists() and not args.scratch.exists()
    assert args.output.parent.is_dir() and args.scratch.parent.is_dir() and args.output != args.scratch
    assert not args.output.is_relative_to(args.scratch) and not args.scratch.is_relative_to(args.output)
    capacity = {str(path): shutil.disk_usage(path).free for path in (args.output.parent, args.scratch.parent)}
    assert all(free >= FLOOR for free in capacity.values())
    source_gate(args, plan, True)
    include, proofs = scientific_inputs(args, plan)
    out, work = args.output, args.scratch
    out.mkdir()
    work.mkdir()
    env = dict(os.environ, TMPDIR=str(work))
    env.pop("GH_TOKEN", None)
    write(out / "entry.json", {"authority": grant, "capacity": capacity, "floor": FLOOR,
                               "compiled_cap": CAP, "provider_records": proofs, "runtime": sys.version})
    flags = ["/usr/bin/g++", *FLAGS, "-I" + str(args.source / "include")]
    obj = work / "hough_circle_peaks.o"
    binary = work / "_hough_circle_peaks_private.cpython-313-x86_64-linux-gnu.so"
    lib = args.base / "lib/libskimagexx.a"
    client = work / "hough_circle_peaks_client"
    command(out, "leaf", [*flags, "-c", args.source / "src/hough_circle_peaks.cpp", "-o", obj], work, env)
    command(out, "bridge", [*flags, "-I" + str(args.source / "src"), "-I" + str(include), "-I" + str(args.pybind),
                            "-shared", args.source / "src/bind_hough_circle_peaks.cpp",
                            args.storage_source, args.bridge, obj, lib,
                            "-o", binary], work, env)
    command(out, "elf", ["/usr/bin/readelf", "-hWs", binary], work, env)
    exports = (out / "elf.log").read_text()
    assert "ELF64" in exports and "X86-64" in exports and re.search(r"\bPyInit__hough_circle_peaks_private\b", exports)
    command(out, "sdk-link", [*flags, args.source / "tests/cmake_consumer/hough_circle_peaks_consumer.cpp",
                             obj, lib, "-o", client], work, env)
    command(out, "sdk", [client], work, env)
    logical = sum(path.stat().st_size for path in work.rglob("*") if path.is_file())
    assert logical <= CAP
    write(out / "products.json", {"leaf": digest(obj.read_bytes()), "private": digest(binary.read_bytes()),
                                  "sdk": digest(client.read_bytes()), "scratch_logical_bytes": logical})
    status = {}
    for mode in ("normal", "scalar"):
        assert datetime.now(timezone.utc) < deadline
        child_env = dict(env)
        if mode == "scalar":
            child_env["SKIMAGEXX_DISABLE_SIMD"] = "1"
        status[mode] = command(out, mode, ["/usr/bin/taskset", "-c", "0", sys.executable, "-I", "-S", "-B",
                               HERE / "parity.py", mode, args.manifest, args.source, args.prepared, args.stock,
                               args.base, binary, out / (mode + "-rows.json")], work, child_env, check_exit=False)
    write(out / "terminal.json", {"source": plan["source"]["revision"], "status": status, "sdk_exit": 0,
                                  "stock_calls": 0, "timings": 0, "exit": int(any(status.values()))})
    return int(any(status.values()))


if __name__ == "__main__":
    raise SystemExit(main())
