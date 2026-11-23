"""Measure only the four original whole public peaks calls on an admitted quiet host."""
from argparse import ArgumentParser
from base64 import urlsafe_b64encode
import csv
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from importlib import metadata
from importlib.util import module_from_spec, spec_from_file_location
import inspect
import json
import os
from pathlib import Path
import sys
from traceback import format_exc

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source"
FROZEN = HERE / "frozen"


def digest(raw):
    return sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def readonly(item):
    path = Path(item["path"]).resolve(strict=True)
    raw = path.read_bytes()
    assert len(raw) == item["bytes"] and digest(raw) == item["sha256"]
    return path


def attach(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def journal(path, item):
    with path.open("a", encoding="utf8", newline="\n") as output:
        output.write(json.dumps(item, allow_nan=False) + "\n")


def verify(args):
    assert sys.platform == "linux" and sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode and not sys.flags.optimize
    assert not {"numpy", "skimage", "skimagexx", "scipy"} & sys.modules.keys()
    assert os.sched_getaffinity(0) == {0}
    assert not any(os.environ.get(name) for name in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_AUDIT",
                                                     "NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE", "SKIMAGEXX_DISABLE_SIMD"))
    raw = args.authority.read_bytes()
    assert digest(raw) == args.authority_sha256 == os.environ.get("CIRCLE_TIMING_AUTHORITY_SHA256")
    binding_raw = args.binding.read_bytes()
    assert digest(binding_raw) == args.binding_sha256
    grant, binding = json.loads(raw), json.loads(binding_raw)
    assert grant["purpose"] == "circle-peaks-four-public-modern-quiet" and binding["authority_sha256"] == args.authority_sha256
    assert binding["source"]["head"] == grant["carrier"] == os.environ["GITHUB_SHA"]
    assert binding["transport"] == grant["transport"]
    now = datetime.now(timezone.utc)
    assert now < datetime.fromisoformat(grant["deadline_utc"]) < datetime.fromisoformat(grant["transport"]["expires_at"])
    quiet = read(readonly(binding["quiet"]))
    assert quiet["authority_sha256"] == args.authority_sha256 and quiet["stage_process_exit"] == 0
    assert quiet["cpu"] == 0 and quiet["active_competing_local_jobs"] == [] and quiet["transfers_and_builds_terminal"]
    assert 0 <= (now - datetime.fromisoformat(quiet["checked_utc"])).total_seconds() <= 120
    plan_raw = (HERE / "timing-plan.json").read_bytes()
    assert digest(plan_raw) == grant["plan_sha256"]
    plan = json.loads(plan_raw)
    params = read(HERE / "manifest-v1.json")
    for name, info in plan["files"].items():
        readonly({"path": str(HERE / name), **info})
    assert sys.version == params["runtime"]["version"]
    assert digest(Path(sys.executable).resolve().read_bytes()) == params["runtime"]["python_sha256"]
    assert all(os.environ.get(name) == "1" for name in params["runtime"]["threads"])
    binary = readonly(binding["elf"])
    assert binding["elf"]["sha256"] == plan["elf"]["sha256"]
    prepared = read(readonly(binding["prepared"]))
    assert prepared["carrier"] == plan["producer"] and prepared["elf"] == plan["elf"]
    assert prepared["stock_rows_sha256"] == plan["stock_rows_sha256"]
    assert prepared["source"] == plan["source"] and prepared["tree"] == plan["tree"]
    stock, base = (Path(binding[key]).resolve(strict=True) for key in ("stock", "base"))
    assert stock != base and args.authority.parent == args.binding.parent == stock.parent == base.parent
    assert binary.is_relative_to(stock.parent) and not (stock / "skimagexx").exists()
    for name, hash_value in params["base"]["members_sha256"].items():
        if name != "lib/libskimagexx.a":
            assert digest((base / name).read_bytes()) == hash_value
    distributions = tuple(metadata.distributions(path=[str(stock)]))
    proofs = {}
    for name, group in params["runtime"]["distributions"].items():
        dist, = [entry for entry in distributions if entry.metadata["Name"].lower().replace("-", "_") == name]
        assert group["version"] == dist.version
        member, = [entry for entry in dist.files if entry.as_posix().endswith(".dist-info/RECORD")]
        file = Path(dist.locate_file(member)).resolve(strict=True)
        assert file.is_relative_to(stock)
        raw = file.read_bytes()
        entries = list(csv.reader(raw.decode().splitlines()))
        records = {row[0]: row[1:] for row in entries}
        assert len(records) == len(entries)
        for path in group["names"]:
            pin = params["runtime"]["provider_files"][path]
            resolved = Path(dist.locate_file(path)).resolve(strict=True)
            assert resolved.is_relative_to(stock)
            readonly({"path": str(resolved), **pin})
            token = urlsafe_b64encode(bytes.fromhex(pin["sha256"])).decode().rstrip("=")
            assert records[path] == ["sha256=" + token, str(pin["bytes"])]
        proofs[name] = {"record_sha256": digest(raw), "members": len(group["names"])}
    assert sum(pin["members"] for pin in proofs.values()) == 18
    return grant, params, plan, quiet, binary, stock, base, proofs


def execute(args, grant, params, plan, quiet, binary, stock, base, proofs):
    out = args.out
    out.mkdir(exist_ok=False)
    log = out / "journal.jsonl"
    results = []
    cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
    try:
        sys.path[:0] = [str(base), str(stock)]
        import numpy as np
        import scipy
        import skimage
        from scipy.spatial import _ckdtree
        from skimage import transform as original
        from skimage.transform import hough_transform, _hough_transform
        import skimagexx as xx
        from skimagexx import _core, transform

        assert (np.__version__, scipy.__version__, skimage.__version__) == ("2.5.4", "1.18.1", "0.26.0")
        loaded = {}
        for module in (np, scipy, skimage, _ckdtree, hough_transform, _hough_transform, sys.modules["numpy._core._multiarray_umath"]):
            path = Path(module.__file__).resolve()
            name = path.relative_to(stock).as_posix()
            pin = params["runtime"]["provider_files"][name]
            readonly({"path": str(path), **pin})
            loaded[name] = pin["sha256"]
        assert original.hough_circle_peaks is hough_transform.hough_circle_peaks
        assert hough_transform.cKDTree is _ckdtree.cKDTree
        assert Path(xx.__file__).resolve() == base / "skimagexx/__init__.py"
        assert Path(transform.__file__).resolve() == base / "skimagexx/transform.py"
        core = Path(_core.__file__).resolve()
        assert core == base / params["base"]["core_member"]
        assert digest(core.read_bytes()) == params["base"]["members_sha256"][params["base"]["core_member"]]
        attach("_hough_circle_peaks_private", binary).attach_to_core(_core)
        kernel = _core.hough_circle_peak_indices
        assert inspect.isbuiltin(kernel) and kernel.__module__ == "skimagexx._core"
        facade = attach("skimagexx._hough_circle_peaks", SOURCE / "python/skimagexx/_hough_circle_peaks.py")
        transform.hough_circle_peaks = facade.hough_circle_peaks
        assert xx.transform.hough_circle_peaks is facade.hough_circle_peaks
        xx.set_fallback_enabled(False)
        assert xx.get_num_threads() == 1 and xx.has_avx2()
        fixture = attach("_circle_peak_fixture", SOURCE / "tests/hough_circle_peaks_reference.py")
        recipe = attach("_circle_peak_fixed", SOURCE / "benchmarks/hough_circle_peaks.py")
        assert (recipe.METHOD["workers"], recipe.METHOD["samples"], recipe.METHOD["minimum_sample_ns"], recipe.METHOD["minimum_warmup_ns"]) == (1, 7, 100_000_000, 250_000_000)
        sampler = attach("_circle_peak_sampler", HERE / "sampler.py")
        pool_controller = attach("threadpoolctl", HERE / "threadpoolctl.py")
        fields = {}
        for name, item in params["fields"].items():
            field = FROZEN / (name + ".npy")
            assert digest(field.read_bytes()) == item["sha256"]
            array = np.load(field, allow_pickle=False)
            data = fixture.snapshot(array)
            delta = {key for key, value in item["input"].items() if data[key] != value}
            assert delta <= {"owns", "base"}
            fields[name] = array.copy(order="C") if delta else array
            assert fixture.snapshot(fields[name]) == item["input"]
        raw = (FROZEN / "linux-modern-rows-v1.jsonl").read_bytes()
        assert digest(raw) == plan["stock_rows_sha256"]
        saved = {row["id"]: row for row in map(json.loads, raw.splitlines()) if row["scope"] == "benchmark"}
        assert list(saved) == [pin["id"] for pin in plan["fixed"]]
        with pool_controller.threadpool_limits(limits=1):
            actual = pool_controller.threadpool_info()
            assert actual and all(row["num_threads"] == 1 for row in actual)
            policy = np.geterr(), np.geterrcall(), np.getbufsize()
            journal(log, {"event": "quiet_entry_verified", "cpu": cpu, "affinity": [0], "pools": actual,
                          "loaded": loaded, "provider_records": proofs})
            for index, ((label, inputs, options), pin) in enumerate(zip(recipe.make_cases(np, fields, "linux-minimum"), plan["fixed"], strict=True)):
                row = saved[label]
                assert label == pin["id"] and options == row["options"]
                canonical = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                assert digest(canonical) == pin["row_sha256"]
                before = [fixture.snapshot(arg, "data_hex" in expected) for arg, expected in zip(inputs, row["inputs"], strict=True)]
                assert before == row["inputs"]
                ref = partial(original.hough_circle_peaks, *inputs, **options)
                native = partial(transform.hough_circle_peaks, *inputs, **options)
                hits = []

                def observe_native(volume, x, y, threshold):
                    value = kernel(volume, x, y, threshold)
                    hits.append((volume is inputs[0], x, y, threshold, len(value)))
                    return value

                assert datetime.now(timezone.utc) < datetime.fromisoformat(grant["deadline_utc"])
                _core.hough_circle_peak_indices = observe_native
                try:
                    private = fixture.observe(transform.hough_circle_peaks, np, inputs, row)
                finally:
                    _core.hough_circle_peak_indices = kernel
                stock_check = fixture.observe(original.hough_circle_peaks, np, inputs, row)
                assert private == stock_check
                assert hits == [(True, options.get("min_xdistance", 1), options.get("min_ydistance", 1),
                                  options.get("threshold"), inputs[0].shape[0])]
                assert pool_controller.threadpool_info() == actual and os.sched_getaffinity(0) == {0}
                journal(log, {"event": "paired_exact_before_samples", "case_id": label, "native_hits": 1,
                              "private_sha256": plan["elf"]["sha256"], "saved_case_sha256": pin["row_sha256"],
                              "outputs": private, "inputs_unchanged": True})
                measured = sampler.measure(ref, native, 7, 100, journal=log, case_index=index, case_id=label, pass_index=1,
                                           deadline=grant["deadline_utc"])
                assert [fixture.snapshot(arg, "data_hex" in expected) for arg, expected in zip(inputs, row["inputs"], strict=True)] == before
                assert pool_controller.threadpool_info() == actual and os.sched_getaffinity(0) == {0}
                assert (np.geterr(), np.geterrcall(), np.getbufsize()) == policy
                assert xx.get_num_threads() == 1 and _core.hough_circle_peak_indices is kernel
                results.append(dict(case_id=label, **measured))
    except Exception:
        journal(log, {"event": "failure", "completed_cases": len(results), "traceback": format_exc()})
        raise
    finally:
        with (out / "report.json").open("x", encoding="utf8", newline="\n") as stream:
            json.dump({"source": plan["source"], "private_sha256": plan["elf"]["sha256"], "runtime": sys.version,
                       "authority_sha256": args.authority_sha256, "quiet": quiet,
                       "method": {"threads": 1, "samples_per_side": 7, "min_sample_ms": 100, "min_warmup_ms": 250, "affinity": [0], "cpu": cpu},
                       "results": results}, stream, indent=2, allow_nan=False)
            stream.write("\n")
    assert len(results) == 4 and all(list(map(len, row["samples_ms"])) == [7, 7] for row in results)
    return int(any(row["speedup"] < 1.0 for row in results))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("authority", "binding", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("authority-sha256", "binding-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    raise SystemExit(execute(args, *verify(args)))
