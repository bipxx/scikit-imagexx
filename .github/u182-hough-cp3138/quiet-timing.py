"""Unbound one-worker Hough timing entry for an already prepared isolated host."""
from argparse import ArgumentParser
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig
from traceback import format_exc
import warnings


BOOT = "cc4c9890b8a48baac2862a6dd5524a84d8e75607cf06d2eafe05c4721ca4a8c8"
PLAN = "d3e123b1b3c198315fc659e91bc7c5d980ddab929e712810858190dade3136cd"
SAMPLER = "29e5fec91dcc4b98f74b6de573ea943ef40f08c7f0187d133c06ad960ff3fd3d"
POOLS = "12fb9526b6a74d2e686b7ec148dc165c3587999b14fa86984aa180e10802400b"
PORTABLE_LINE_STOCK = "9c7243a68e5f14ee803c5f2845fb1b346b22b34c11fb577d1f592378fea65e22"
FAMILIES = {"hough_line": "hough_line_accumulate"}


def fingerprint(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}


def readonly(record):
    path = Path(record["path"])
    assert path.is_absolute() and path.is_file() and not path.is_symlink()
    assert fingerprint(path) == {key: record[key] for key in ("bytes", "sha256")}, path
    return path


def read_json(path):
    return json.loads(path.read_bytes())


def load(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def journal(path, item):
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(item, allow_nan=False) + "\n")
        output.flush()


def verify(args):
    assert not sys.flags.optimize and sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
    assert fingerprint(args.authority)["sha256"] == args.authority_sha256
    authority = read_json(args.authority)
    assert authority["purpose"] == "hough-line-wide-modern-fixed-speed"
    assert authority["sampler_sha256"] == SAMPLER and authority["threadpoolctl_sha256"] == POOLS
    assert fingerprint(Path(__file__))["sha256"] == authority["timing_entry_sha256"]
    assert authority["bootstrap_sha256"] == BOOT and authority["fixed_case_plan_sha256"] == PLAN
    now = datetime.now(timezone.utc)
    assert 0 < (datetime.fromisoformat(authority["deadline_utc"]) - now).total_seconds() <= 5400
    assert fingerprint(args.binding)["sha256"] == args.binding_sha256
    binding = read_json(args.binding)
    assert binding["authority_sha256"] == args.authority_sha256
    assert binding["transport"] == authority["transport"]
    quiet = read_json(readonly(binding["quiet"]))
    assert quiet["authority_sha256"] == args.authority_sha256 and quiet["stage_process_exit"] == 0
    assert quiet["cpu"] == 0 and quiet["active_competing_local_jobs"] == [] and quiet["transfers_and_builds_terminal"] is True
    assert 0 <= (now - datetime.fromisoformat(quiet["checked_utc"])).total_seconds() <= 120
    boot_path, plan_path = (readonly(binding[name]) for name in ("bootstrap", "fixed_case_plan"))
    assert fingerprint(boot_path)["sha256"] == BOOT and fingerprint(plan_path)["sha256"] == PLAN
    boot, plan = read_json(boot_path), read_json(plan_path)
    assert sys.platform == "linux" and sys.version == boot["runtime"]["version"]
    assert sysconfig.get_config_var("EXT_SUFFIX") == boot["runtime"]["ext_suffix"]
    assert fingerprint(Path(sys.executable).resolve())["sha256"] == boot["runtime"]["executable_sha256"]
    assert os.sched_getaffinity(0) == {0}
    assert all(os.environ.get(name) == "1" for name in boot["runtime"]["thread_environment"])
    assert not any(os.environ.get(name) for name in ("PYTHONPATH", "PYTHONHOME", "SKIMAGEXX_DISABLE_SIMD"))
    manifest_path = readonly(binding["producer_manifest"])
    assert fingerprint(manifest_path)["sha256"] == authority["producer_manifest_sha256"]
    manifest = read_json(manifest_path)
    assert manifest["source"] == authority["producer_source"] and manifest["no_timing"]
    runtime = read_json(readonly(binding["runtime_proof"]))
    assert runtime == {"donor_core": boot["donor"]["core"]["sha256"], "runtime": boot["runtime"]["version"],
                       "science_imports": 0, "selected_provider_members": 24}
    assert set(binding["families"]) == set(authority["elf_sha256"]) == set(FAMILIES)
    for name, row in binding["families"].items():
        assert manifest["elf"]["sha256"] == authority["elf_sha256"][name]
        assert fingerprint(readonly(row["elf"])) == {key: manifest["elf"][key] for key in ("bytes", "sha256")}
        assert (manifest["source"], manifest["tree"]) == (plan["families"][name]["source"], plan["families"][name]["tree"])
        owned = boot["families"]["transform." + name]["owned_git_files"]
        source = Path(row["source_root"]).resolve(strict=True)
        contract, = [path for path in owned if path.startswith("coverage/contracts/") and path.endswith((".json", ".jsonl"))]
        for label, relative in (("recipe", "benchmarks/" + name + ".py"), ("fixture", "tests/" + name + "_reference.py"),
                                ("facade", "python/skimagexx/_" + name + ".py"), ("contract", contract)):
            path = readonly(row[label])
            assert path.resolve() == source / relative and fingerprint(path)["sha256"] == owned[relative]["sha256"]
        assert owned["benchmarks/" + name + ".py"]["sha256"] == plan["families"][name]["benchmark_recipe"]["sha256"]
        assert fingerprint(readonly(row["stock_rows"]))["sha256"] == PORTABLE_LINE_STOCK
    assert fingerprint(readonly(binding["sampler"]))["sha256"] == SAMPLER
    assert fingerprint(readonly(binding["threadpoolctl"]))["sha256"] == POOLS
    return authority, binding, boot, plan, quiet


def input_state(inputs, options):
    arrays = (inputs[0], options.get("theta"))
    return [None if value is None else (value.shape, value.strides, value.dtype.str, value.flags.writeable,
                                        sha256(value.tobytes()).hexdigest()) for value in arrays]


def check_outcome(fixture, np, call, args, options, row):
    image = args[0]
    snapshot = fixture.snapshot
    assert snapshot(image) == row["image"]
    theta = options.get("theta")
    assert (None if theta is None else snapshot(theta)) == row["theta"]
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always")
        result = call()
    assert [[item.category.__module__, item.category.__name__, str(item.message)] for item in notices] == row["warnings"]
    saved = row["result"]
    assert type(result).__name__ == saved["container_type"]
    for array, section in zip(result, ("accumulator", "angles", "distances"), strict=True):
        assert snapshot(array) == saved[section]
    votes, angles, distances = result
    assert (angles is theta) is saved["theta_is_input"]
    assert (theta is not None and np.shares_memory(angles, theta)) is saved["theta_shares_input"]
    assert np.all(votes.sum(axis=0) == row["nonzero_count"])
    assert [float(distances[0]), float(distances[-1])] == saved["distance_endpoints"]
    assert snapshot(image) == row["image"] and (None if theta is None else snapshot(theta)) == row["theta"]


def execute(args, authority, binding, boot, plan, quiet):
    candidate, stock = (Path(binding[key]).resolve(strict=True) for key in ("candidate_site", "stock_site"))
    assert candidate != stock
    output = args.out
    output.mkdir(exist_ok=False)
    log = output / "journal.jsonl"
    results = []
    cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
               if line.startswith("model name"))
    try:
        sys.path[:0] = [str(candidate), str(stock)]
        import numpy as np
        import skimagexx as xx
        from skimage import transform as stock_public
        from skimagexx import _core, transform
        original = import_module("skimage.transform.hough_transform")
        for name in ("skimage.transform.hough_transform", "skimage.transform._hough_transform", "numpy", "numpy._core._multiarray_umath"):
            module = import_module(name)
            path = Path(module.__file__).resolve()
            relative = path.relative_to(stock).as_posix()
            namespace = "numpy" if name.startswith("numpy") else "scikit_image"
            assert fingerprint(path)["sha256"] == boot["runtime"]["providers"][namespace]["files_sha256"][relative]
        core = Path(_core.__file__).resolve()
        assert core == candidate / boot["donor"]["core"]["member"]
        assert fingerprint(core)["sha256"] == boot["donor"]["core"]["sha256"]
        xx.set_fallback_enabled(False)
        assert xx.get_num_threads() == 1
        sampler = load("_u182_hough_sampler", readonly(binding["sampler"]))
        pools = load("threadpoolctl", readonly(binding["threadpoolctl"]))
        index = 0
        with pools.threadpool_limits(limits=1):
            active_pools = pools.threadpool_info()
            assert active_pools and all(row["num_threads"] == 1 for row in active_pools)
            journal(log, {"event": "quiet_entry_verified", "cpu": cpu, "affinity": [0], "pools": active_pools})
            for name, symbol in FAMILIES.items():
                row = binding["families"][name]
                binary = readonly(row["elf"])
                load("_" + name + "_private", binary).attach_to_core(_core)
                native = getattr(_core, symbol)
                assert inspect.isbuiltin(native) and native.__module__ == "skimagexx._core"
                facade = getattr(load("skimagexx._" + name, readonly(row["facade"])), name)
                setattr(transform, name, facade)
                setattr(xx.transform, name, facade)
                original_call = getattr(original, name)
                assert getattr(stock_public, name) is original_call
                fixture = load("_u182_fixture_" + name, readonly(row["fixture"]))
                recipe = load("_u182_recipe_" + name, readonly(row["recipe"]))
                saved = read_json(readonly(row["stock_rows"]))["future_benchmark_endpoints"]
                fixed = plan["families"][name]["frozen_cases"]
                assert len(saved) == len(fixed) == 4
                for case, expected, pin in zip(recipe.make_cases(), saved, fixed, strict=True):
                    label, inputs, options, *_details = case
                    assert label == expected["id"] == pin["id"]
                    canonical = json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                    assert sha256(canonical).hexdigest() == pin["canonical_row_sha256"]
                    ref, current = partial(original_call, *inputs, **options), partial(getattr(xx.transform, name), *inputs, **options)
                    before = input_state(inputs, options)
                    hits = []

                    def witness(*values):
                        hits.append(values)
                        return native(*values)

                    setattr(_core, symbol, witness)
                    try:
                        check_outcome(fixture, np, current, inputs, options, expected)
                    finally:
                        setattr(_core, symbol, native)
                    assert len(hits) == 1 and hits[0][0] is inputs[0]
                    check_outcome(fixture, np, ref, inputs, options, expected)
                    assert pools.threadpool_info() == active_pools
                    journal(log, {"event": "paired_exact_before_samples", "family": name, "case_id": label,
                                  "private_sha256": fingerprint(binary)["sha256"], "native_hits": 1, "saved_case_sha256": pin["canonical_row_sha256"]})
                    measured = sampler.measure(ref, current, 7, 100, journal=log, case_index=index, case_id=label,
                                               pass_index=1, deadline=authority["deadline_utc"])
                    assert os.sched_getaffinity(0) == {0} and xx.get_num_threads() == 1 and pools.threadpool_info() == active_pools
                    assert input_state(inputs, options) == before and getattr(_core, symbol) is native
                    measured.update(family=name, case_id=label)
                    results.append(measured)
                    journal(log, {"event": "case_succeeded", "result": measured})
                    index += 1
    except Exception:
        journal(log, {"event": "failure", "reached_cases": len(results), "traceback": format_exc()})
        raise
    finally:
        (output / "report.json").write_text(json.dumps({"method": {"threads": 1, "samples_per_side": 7, "min_sample_ms": 100,
                                                                      "min_warmup_ms": 250, "platform": platform.platform(),
                                                                      "cpu": cpu, "affinity": [0]},
                                                      "authority_sha256": args.authority_sha256, "quiet": quiet, "prior_fixed": plan["prior_fixed"], "results": results},
                                                     indent=2, allow_nan=False) + "\n", encoding="utf-8")
    assert len(results) == 4 and all(list(map(len, row["samples_ms"])) == [7, 7] for row in results)
    assert all(row["speedup"] >= 1.0 for row in results), "At least one fixed case is below parity speed"


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("authority", "binding", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--binding-sha256", required=True)
    args = parser.parse_args()
    execute(args, *verify(args))
