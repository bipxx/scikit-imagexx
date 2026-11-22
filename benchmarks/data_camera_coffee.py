"""Measure the two fixed warm installed camera and coffee public calls."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib import import_module
from importlib.metadata import distribution, version
from importlib.resources import files
from importlib.util import module_from_spec, spec_from_file_location
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import warnings


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "coverage/contracts/data-camera-coffee-assets-v1.json"
WRAPPER = "python/skimagexx/_data_camera_coffee.py"
SAMPLER = "tools/montage_sustained.py"
ASSET_ROOT = "python/skimagexx/_data_assets/"
LIMITS = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def cpu_affinity():
    if os.name != "nt":
        return sorted(os.sched_getaffinity(0))
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
                                               ctypes.POINTER(ctypes.c_size_t))
    process, system = ctypes.c_size_t(), ctypes.c_size_t()
    assert kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(),
                                         ctypes.byref(process), ctypes.byref(system))
    return [bit for bit in range(process.value.bit_length()) if process.value & (1 << bit)]


def source_state(git):
    checkout = str(ROOT)
    if os.name != "nt":
        checkout = subprocess.check_output(["/usr/bin/wslpath", "-w", checkout], text=True).strip()
    command = [git, "-C", checkout]
    assert not subprocess.check_output([*command, "status", "--porcelain"], text=True).strip()
    return subprocess.check_output([*command, "rev-parse", "HEAD"], text=True).strip()


def require_authority(args):
    raw = args.go.read_bytes()
    assert sha256(raw).hexdigest() == args.go_sha256
    go = json.loads(raw)
    assert go["authority"] == "ROOT_AUTHORIZED_CAMERA_COFFEE_INSTALLED_ONCE"
    deadline = datetime.fromisoformat(go["deadline_utc"])
    assert deadline.tzinfo is not None and datetime.now(timezone.utc) < deadline
    assert Path(sys.executable).absolute() == Path(go["python"]).absolute()
    assert Path(go["output"]).resolve() == args.output.resolve()
    assert sys.flags.isolated and sys.dont_write_bytecode
    assert "PYTHONPATH" not in os.environ and "PYTHONHOME" not in os.environ
    assert all(os.environ[key] == "1" for key in LIMITS) and threading.active_count() == 1
    assert cpu_affinity() == go["cpu"] and len(go["cpu"]) == 1
    git = Path(go["git"])
    assert git.is_absolute() and git.is_file() and git.name.lower() == "git.exe"
    assert source_state(str(git)) == go["source"]
    manifest = json.loads((ROOT / CONTRACT).read_bytes())
    frozen_sha = digest(args.cases)
    assert frozen_sha == manifest["performance"]["freeze_sha256"] == go["frozen_case_sha256"]
    frozen = json.loads(args.cases.read_bytes())
    profile = frozen["profiles"][go["profile"]]
    assert Path(profile["interpreter"]).absolute() == Path(sys.executable).absolute()
    assert sys.version == profile["python"]
    assert frozen["case_ids"] == manifest["performance"]["cases"] == go["cases"]
    paths = {CONTRACT, WRAPPER, SAMPLER, "benchmarks/data_camera_coffee.py",
             "python/skimagexx/_compat.py", "python/skimagexx/data.py", ASSET_ROOT + manifest["rights_notice"]}
    paths.update(ASSET_ROOT + asset["member"] for asset in manifest["assets"].values())
    assert {p: digest(ROOT / p) for p in paths} == go["source_raw_sha256"]
    assert digest(ROOT / SAMPLER) == frozen["sampler"]["sha256"]
    journal = args.output.with_suffix(".batches.jsonl")
    assert not args.output.exists() and not journal.exists()
    return go, manifest, profile, journal


def measure_installed(args, go, manifest, profile, journal):
    sentinel = args.output.resolve().with_suffix(".unused-cache")
    assert not sentinel.exists()
    os.environ["SKIMAGE_DATADIR"] = str(sentinel)

    def deny_network(event, _):
        if event.startswith("socket.") and event not in ("socket.__new__", "socket.gethostname"):
            raise AssertionError(f"network attempt: {event}")

    sys.addaudithook(deny_network)
    import numpy as np
    import skimagexx as xx
    from skimage import data as original
    from skimage.data import _fetchers
    from skimagexx import _core, _data_camera_coffee

    assert go["profile"] in ("windows", "linux-minimum") and _fetchers._image_fetcher is None
    # Without Pooch, data_dir normally aliases the installed distribution.
    _fetchers.data_dir = str(sentinel / "data")
    for name in ("scikit-image", "numpy", "scipy", "pillow", "imageio"):
        assert version(name) == profile["versions"][name]
    candidate = Path(go["candidate_site"]).resolve(strict=True)
    core = Path(_core.__file__).resolve(strict=True)
    assert core == Path(go["candidate_core"]).resolve(strict=True) and core.is_relative_to(candidate)
    assert digest(core) == go["candidate_core_sha256"]
    wrapper = Path(_data_camera_coffee.__file__).resolve(strict=True)
    assert wrapper.is_relative_to(candidate) and not wrapper.is_relative_to(ROOT / "python")
    assert digest(wrapper) == go["source_raw_sha256"][WRAPPER]
    assert Path(xx.data.__file__).resolve(strict=True).is_relative_to(candidate)
    assert digest(Path(xx.data.__file__)) == go["source_raw_sha256"]["python/skimagexx/data.py"]
    assets = files(xx).joinpath("_data_assets")
    notice = assets.joinpath(manifest["rights_notice"])
    assert sha256(notice.read_bytes()).hexdigest() == go["source_raw_sha256"][ASSET_ROOT + notice.name]
    xx.set_num_threads(1)
    xx.set_fallback_enabled(False)
    assert xx.get_num_threads() == 1
    package = distribution("scikit-image")
    pairs = []
    for name, case in zip(manifest["assets"], profile["cases"], strict=True):
        asset = manifest["assets"][name]
        assert case["api"] == f"data.{name}"
        assert case["png_sha256"] == asset["upstream_sha256"]
        assert case["expected"]["decoded_sha256"] == asset["representation"]["decoded_sha256"]
        assert case["expected"]["shape"] == asset["representation"]["shape"]
        assert case["expected"]["strides"] == asset["representation"]["strides"]
        upstream = Path(package.locate_file(case["source_member"])).resolve(strict=True)
        assert upstream == Path(case["source_member_path"]).resolve(strict=True)
        assert Path(_fetchers._fetch(f"data/{name}.png")).resolve(strict=True) == upstream
        assert digest(upstream) == case["png_sha256"] and upstream.stat().st_size == case["png_bytes"]
        resource = assets.joinpath(asset["member"])
        assert sha256(resource.read_bytes()).hexdigest() == case["expected"]["decoded_sha256"]
        reference, target = getattr(original, name), getattr(xx.data, name)
        assert target is getattr(_data_camera_coffee, name)
        assert str(inspect.signature(reference)) == str(inspect.signature(target)) == "()"
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            expected, actual = reference(), target()
        assert not seen
        for result in expected, actual:
            assert type(result) is np.ndarray and result.dtype.str == case["expected"]["dtype"]
            assert list(result.shape) == case["expected"]["shape"]
            assert list(result.strides) == case["expected"]["strides"]
            assert result.base is None and result.flags.owndata and result.flags.writeable
            assert result.flags.c_contiguous and result.flags.aligned
        assert expected.tobytes() == actual.tobytes() and not np.shares_memory(expected, actual)
        assert sha256(actual.tobytes()).hexdigest() == case["expected"]["decoded_sha256"]
        pairs.append((case, reference, target))
    for module, identity in profile["original_loaded_dependencies"].items():
        path = Path(import_module(module).__file__).resolve(strict=True)
        assert path == Path(identity["path"]).resolve(strict=True) and digest(path) == identity["sha256"]
    sampler_spec = spec_from_file_location("_camera_coffee_sampler", ROOT / SAMPLER)
    sampler = module_from_spec(sampler_spec)
    sampler_spec.loader.exec_module(sampler)
    results = []
    for index, (case, reference, target) in enumerate(pairs):
        case_id = case["case_id"]
        assert case_id == go["cases"][index]
        result = sampler.measure(reference, target, 7, 100, journal=journal, case_index=index,
                                 case_id=case_id, pass_index=1, deadline=go["deadline_utc"])
        results.append(dict(case_id=case_id, **result))
    assert not sentinel.exists() and threading.active_count() == 1 and cpu_affinity() == go["cpu"]
    return core, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--go", type=Path, required=True)
    parser.add_argument("--go-sha256", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error("optimized Python disables required validation")
    go, manifest, profile, journal = require_authority(args)
    core, results = measure_installed(args, go, manifest, profile, journal)
    assert source_state(go["git"]) == go["source"]
    assert {p: digest(ROOT / p) for p in go["source_raw_sha256"]} == go["source_raw_sha256"]
    assert digest(core) == go["candidate_core_sha256"]
    assert len(results) == 2 and all(list(map(len, row["samples_ms"])) == [7, 7] for row in results)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(source=go["source"], profile=go["profile"], core=str(core),
                       core_sha256=go["candidate_core_sha256"], cpu=go["cpu"],
                       runtime=sys.version, accepted_samples=28, results=results), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
