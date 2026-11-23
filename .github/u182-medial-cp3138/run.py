"""Build one exact Medial Axis private component and verify its scalar pytest scope."""
from datetime import datetime, timezone
from hashlib import sha256
import inspect
from importlib import util
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import sysconfig
from tempfile import TemporaryDirectory
from xml.etree import ElementTree


HERE = Path(__file__).resolve().parent
FLOOR = 30 * 2**30
CAP = 96 * 2**20
PURPOSE = "medial-axis-modern-scalar-v1"
REF = "refs/heads/diagnostics/u182-medial-cp3138-scalar-v1"
REPOSITORY = "bipxx/scikit-imagexx"
BOOT = runpy.run_path(str(HERE / "bootstrap.py"))
PORTABLE = runpy.run_path(str(HERE / "portable_record.py"))
seal, command, write = (BOOT[name] for name in ("seal", "command", "write"))
CLEAN_ENV = ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_AUDIT", "LD_LIBRARY_PATH",
             "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "NPY_DISABLE_CPU_FEATURES",
             "CPATH", "CPLUS_INCLUDE_PATH", "COMPILER_PATH", "LIBRARY_PATH", "GCC_EXEC_PREFIX")


def read(path):
    return json.loads(path.read_bytes())


def authority(env, now, hashes):
    assert env.get("GITHUB_ACTIONS") == "true" and env.get("GITHUB_REPOSITORY") == REPOSITORY
    assert env.get("GITHUB_REF") == REF and env.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
    raw = env["MEDIAL_AUTHORITY"].encode()
    assert len(raw) <= 1024 and sha256(raw).hexdigest() == env["MEDIAL_AUTHORITY_SHA256"]
    value = json.loads(raw)
    assert set(value) == {"carrier", "payload", "purpose", "deadline_utc", "cpu"}
    assert (value["carrier"], value["payload"], value["purpose"], value["cpu"]) == (
        env["GITHUB_SHA"], hashes, PURPOSE, 0)
    deadline = datetime.fromisoformat(value["deadline_utc"])
    assert 0 < (deadline - now).total_seconds() <= 5400
    return value


def exact_source(source, plan):
    def git(*args):
        return subprocess.check_output(["/usr/bin/git", "--no-optional-locks", "-C", source, *args])
    assert git("rev-parse", "HEAD", "HEAD^{tree}", "HEAD^").decode().splitlines() == [
        plan["source"], plan["tree"], plan["direct_parent"]]
    assert not git("status", "--porcelain=v1", "--untracked-files=all")
    for path, sha in plan["source_paths"].items():
        assert seal(source / path)["sha256"] == sha == sha256(git("show", f"HEAD:{path}")).hexdigest()
    return {"source": plan["source"], "tree": plan["tree"], "paths": len(plan["source_paths"])}


def stage_source(out, work, plan, predecessor):
    source = work / "source"
    source.mkdir()
    git = ["/usr/bin/git", "-C", source]
    command(out, "git-init", [*git, "init", "--quiet"])
    command(out, "git-remote", [*git, "remote", "add", "origin", f"https://github.com/{REPOSITORY}.git"])
    command(out, "git-promisor", [*git, "config", "remote.origin.promisor", "true"])
    command(out, "git-partialclonefilter", [*git, "config", "remote.origin.partialclonefilter", "blob:none"])
    command(out, "git-fetch", [*git, "fetch", "--quiet", "--filter=blob:none", "--depth=2",
                                "origin", predecessor["source_ref"]])
    assert subprocess.check_output([*git, "rev-parse", "FETCH_HEAD"], text=True).strip() == plan["source"]
    command(out, "git-sparse", [*git, "sparse-checkout", "set", "--no-cone",
                                 *["/" + path for path in plan["source_paths"]]])
    command(out, "git-checkout", [*git, "checkout", "--quiet", "--detach", "FETCH_HEAD"])
    write(out / "source-proof.json", exact_source(source, plan))
    return source


def runtime(packet, plan, pybind, donor, *, scalar):
    assert not sys.flags.optimize and sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
    assert sys.platform == "linux" and os.sched_getaffinity(0) == {0}
    assert not any(os.environ.get(key) for key in CLEAN_ENV)
    assert all(os.environ.get(key) == "1" for key in packet["runtime"]["thread_environment"])
    assert os.environ.get("SKIMAGEXX_DISABLE_SIMD") == ("1" if scalar else None)
    assert os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1"
    profile = plan["runtime"]
    assert (sys.version, seal(Path(sys.executable).resolve())["sha256"], sysconfig.get_config_var("EXT_SUFFIX")) == (
        profile["version"], profile["executable_sha256"], profile["extension_suffix"])
    include = Path(sysconfig.get_path("include"))
    for name, pin in packet["runtime"]["cpython_headers"].items():
        assert seal(include / name) == BOOT["seal_dict"](pin)
    for name, pin in packet["runtime"]["pybind_headers"].items():
        assert seal(pybind / "pybind11" / name) == BOOT["seal_dict"](pin)
    assert seal(donor / packet["donor"]["core"]["member"]) == BOOT["seal_dict"](packet["donor"]["core"])
    assert not {"numpy", "scipy", "skimage", "pytest", "skimagexx"} & sys.modules.keys()
    return include


def stage_portable(packet, plan, out, wheels, stock, pybind, donor):
    include = runtime(packet, plan, pybind, donor, scalar=False)
    installed, members = PORTABLE["check_install"](plan, wheels, stock)
    assert len(installed) == 13 and len(members) == 3981
    write(out / "portable-installed.json", {"distributions": installed, "member_pins": members,
          "python": sys.version, "python_sha256": seal(Path(sys.executable).resolve())["sha256"],
          "include": str(include), "science_imports": 0})


def load(name, path):
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def scalar_only(packet, plan, out, stock, pybind, donor, source, private):
    runtime(packet, plan, pybind, donor, scalar=True)
    source_receipt = exact_source(source, plan)
    installed = read(out / "portable-installed.json")
    assert len(installed["distributions"]) == 13 and len(installed["member_pins"]) == 3981
    private_pin = seal(private)
    donor_core = donor / packet["donor"]["core"]["member"]
    sys.path[:0] = [str(donor), str(stock), str(source / "tests")]
    import pytest
    import skimagexx as xx
    from skimagexx import _core, morphology

    assert Path(_core.__file__).resolve() == donor_core and xx.get_num_threads() == 1 and not xx.has_avx2()
    assert Path(morphology.__file__).resolve() == donor / "skimagexx/morphology.py"
    component = load("_medial_axis_private", private)
    names = ("medial_axis_prepare", "medial_axis_ordered")
    for name in names:
        entry = getattr(component, name)
        assert name not in vars(_core) and inspect.isbuiltin(entry) and entry.__module__ == component.__name__
        setattr(_core, name, entry)
    wrapper = load("skimagexx._medial_axis", source / "python/skimagexx/_medial_axis.py")
    assert wrapper._core is _core
    morphology.medial_axis = wrapper.medial_axis
    xml = out / "scalar.xml"
    code = pytest.main(["-q", "-p", "no:cacheprovider", "--capture=sys", f"--junitxml={xml}",
                        str(source / "tests/test_medial_axis_public.py")])
    if code:
        return int(code)
    suite, = ElementTree.parse(xml).iter("testsuite")
    assert [int(suite.get(key, 0)) for key in ("tests", "failures", "errors", "skipped")] == [30, 0, 0, 0]
    provenance = PORTABLE["check_loaded"](stock, installed["member_pins"], plan["required_live_after_scalar"])
    assert xx.get_num_threads() == 1 and not xx.has_avx2() and os.sched_getaffinity(0) == {0}
    assert all(getattr(_core, name) is getattr(component, name) for name in names)
    assert seal(private) == private_pin and seal(donor_core) == BOOT["seal_dict"](packet["donor"]["core"])
    assert exact_source(source, plan) == source_receipt
    write(out / "scalar-result.json", {**source_receipt, "cases": 30, "exit": 0, "cpu": 0, "workers": 1,
          "avx2": False, "stock_replay": 0, "normal_replay": 0, "sdk_replay": 0, "timing": 0,
          "bindings": list(names), "private_sha256": private_pin["sha256"], "donor_sha256": seal(donor_core)["sha256"],
          "xml": seal(xml), "portable": seal(out / "portable-installed.json"), "loaded": provenance})
    return 0


def inputs():
    pins = read(HERE / "payload.json")
    for path, pin in pins.items():
        assert seal(HERE / path) == pin, path
    packet, plan, historical = [read(HERE / path) for path in
                                ("bootstrap.json", "portable-pins-v1.json", "predecessor.json")]
    assert not plan["execution_authority"] and plan["source"] == historical["source"]
    assert plan["tree"] == historical["tree"] and historical["local_binary_carried"] is False
    assert historical["stock_rows"] == historical["normal_passed"] - 3 == 27 and historical["sdk_exit"] == 0
    for path, key in (("stock-comparison-v2.json", "stock_comparison_sha256"),
                      ("linux-modern-local-cases-v1.jsonl", "stock_journal_sha256"),
                      ("normal.json", "normal_parallel_scope_sha256")):
        assert pins[path]["sha256"] == historical[key]
    return packet, plan, historical


def main():
    packet, plan, historical = inputs()
    value = authority(os.environ, datetime.now(timezone.utc), seal(HERE / "payload.json")["sha256"])
    if sys.argv[1:]:
        mode, *paths = sys.argv[1:]
        if mode == "verify-portable":
            return stage_portable(packet, plan, *map(Path, paths))
        assert mode == "scalar-only"
        return scalar_only(packet, plan, *map(Path, paths))

    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve(strict=True)
    temp = Path(os.environ["RUNNER_TEMP"]).resolve(strict=True)
    assert temp != workspace and not temp.is_relative_to(workspace)
    assert not any(os.environ.get(key) for key in CLEAN_ENV + ("SKIMAGEXX_DISABLE_SIMD",))
    assert 0 in os.sched_getaffinity(0)
    out = workspace / "medial-scalar-results"
    out.mkdir()
    free = {str(path): shutil.disk_usage(path).free for path in (workspace, temp)}
    write(out / "authority-capacity.json", {"authority": value, "observed_free_bytes": free,
          "authority_sha256": os.environ["MEDIAL_AUTHORITY_SHA256"],
          "floor": FLOOR, "fixture_limit": 2 * 2**30, "native_limit": CAP})
    stage = "capacity"
    try:
        assert min(free.values()) >= FLOOR
        stage = "compiler"
        compiler = BOOT["verify_tools"](packet, out)
        with TemporaryDirectory(dir=temp, prefix="medial-scalar-") as directory:
            work = Path(directory)
            stage = "source"
            source = stage_source(out, work, plan, historical)
            stage = "runtime"
            python = BOOT["provision_python"](packet, work)
            stage = "donor"
            wheels, package = BOOT["donor_inputs"](packet, work, out)
            stock, pybind, donor = BOOT["install_sites"](packet, wheels, package, python, out, work)
            stage = "portable-record"
            entry = ["taskset", "-c", "0", python, "-I", "-S", "-B", Path(__file__).resolve()]
            command(out, "verify-portable", [*entry, "verify-portable", out, wheels, stock, pybind, donor])
            verified = read(out / "portable-installed.json")
            assert len(verified["distributions"]) == 13 and len(verified["member_pins"]) == 3981
            for wheel in wheels.iterdir():
                wheel.unlink()
            wheels.rmdir()

            stage = "native"
            assert datetime.now(timezone.utc) < datetime.fromisoformat(value["deadline_utc"])
            assert shutil.disk_usage(work).free >= FLOOR
            native = work / "native"
            native.mkdir()
            common = [compiler, "-std=c++17", "-O3", "-fPIC", "-fvisibility=hidden", "-Wall", "-Wextra",
                      f"-ffile-prefix-map={work}=/usr/src/medial-scalar",
                      f"-ffile-prefix-map={HERE}=/usr/src/medial-diagnostic", "-I" + str(source / "include")]
            leaf = native / "medial_axis.o"
            private = native / ("_medial_axis_private" + plan["runtime"]["extension_suffix"])
            command(out, "leaf", [*common, "-c", source / "src/medial_axis.cpp", "-o", leaf])
            command(out, "link", [*common, "-I" + verified["include"], "-I" + str(pybind), "-I" + str(source / "src"),
                                   "-shared", source / "src/bind_medial_axis.cpp", HERE / "storage_projection.cpp",
                                   HERE / "private_bridge.cpp", leaf, "-o", private])
            binary = private.read_bytes()
            assert binary[:6] == b"\x7fELF\x02\x01" and binary[18:20] == b"\x3e\x00"
            assert len(binary) < 2**20 and sum(file.stat().st_size for file in native.iterdir()) <= CAP
            assert all(prefix not in binary for prefix in (b"/home/", b"/mnt/c/Users/", b"C:\\Users\\"))
            stage = "scalar"
            assert datetime.now(timezone.utc) < datetime.fromisoformat(value["deadline_utc"])
            command(out, "scalar", [*entry, "scalar-only", out, stock, pybind, donor, source, private],
                    env=dict(os.environ, SKIMAGEXX_DISABLE_SIMD="1"))
            result = read(out / "scalar-result.json")
            assert result["cases"] == 30 and result["exit"] == 0 and result["private_sha256"] == seal(private)["sha256"]
            stage = "transport"
            transport = workspace / "medial-scalar-transport"
            transport.mkdir()
            with (transport / private.name).open("xb") as handle:
                handle.write(binary)
            write(transport / "medial-scalar.json", {"source": plan["source"], "tree": plan["tree"],
                  "elf": {"name": private.name, **seal(private)}, "scalar": seal(out / "scalar-result.json"),
                  "authority_sha256": os.environ["MEDIAL_AUTHORITY_SHA256"], "no_timing": True})
    except Exception as exc:
        write(out / "terminal.json", {"stage": stage, "exit": 1, "error": str(exc)})
        raise
    write(out / "terminal.json", {"stage": stage, "exit": 0, "purpose": PURPOSE})


if __name__ == "__main__":
    if sys.flags.optimize:
        raise SystemExit("Optimized Python is unsupported")
    raise SystemExit(main())
