"""Fixed, controlled local-file public comparisons for clock and horse."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import distribution, version
from importlib.util import module_from_spec, spec_from_file_location
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading


ROOT = Path(__file__).resolve().parents[1]
LIMITS = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
PATHS = (
    "benchmarks/data_clock_horse.py", "coverage/contracts/data-clock-horse-v1.json",
    "python/skimagexx/_data_clock_horse.py", "python/skimagexx/_compat.py",
    "python/skimagexx/_data_assets/clock.gray", "python/skimagexx/_data_assets/horse.bool",
    "python/skimagexx/_data_assets/clock-horse.RIGHTS.txt",
    "tests/fixtures/clock_horse_original_windows_v1.json",
    "tests/fixtures/clock_horse_original_linux_v1.json", "tools/montage_sustained.py",
)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def load(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def temporary_fixture(path):
    assert path.parent.resolve() == (ROOT / "build").resolve()
    path.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="clock-horse-timing-", dir=path) as child:
        yield Path(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--go", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--threadpoolctl", type=Path, required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    go = json.loads(args.go.read_bytes())
    journal = args.output.with_suffix(".batches.jsonl")
    assert go["authority"] == "clock-horse-two-local-cases"
    assert str(Path(sys.executable)) == go["python"]
    assert datetime.now(timezone.utc) < datetime.fromisoformat(go["deadline_utc"])
    assert not args.output.exists() and not journal.exists()
    assert {os.environ[key] for key in LIMITS} == {"1"} and threading.active_count() == 1
    assert digest(args.threadpoolctl) == go["threadpoolctl_sha256"]
    git_prefix = ["git", "-C", str(ROOT)]
    if os.name != "nt":
        assert sys.flags.isolated and sys.dont_write_bytecode
        assert "PYTHONPATH" not in os.environ and "PYTHONHOME" not in os.environ
        git_prefix = ["/mnt/c/Program Files/Git/cmd/git.exe", "-C", go["windows_checkout"]]

    def git(*command):
        return subprocess.check_output([*git_prefix, *command], text=True).strip()

    assert git("rev-parse", "HEAD") == go["source"] and not git("status", "--porcelain")
    pins = {path:digest(ROOT / path) for path in PATHS}
    assert pins == go["source_raw_sha256"]

    import numpy as np
    from skimage import data as upstream
    from skimage.data import _fetchers

    assert digest(Path(inspect.getsourcefile(upstream.clock))) == go["original_sha256"]
    assert [version(n) for n in ("scikit-image", "numpy", "pillow", "imageio")] == go["versions"]
    manifest = json.loads((ROOT / "coverage/contracts/data-clock-horse-v1.json").read_bytes())
    assert go["cases"] == manifest["performance"]["cases"]
    pool = load("threadpoolctl", args.threadpoolctl)
    sampler = load("_clock_horse_sampler", ROOT / "tools/montage_sustained.py")
    site = distribution("scikit-image")
    results = []
    with temporary_fixture(args.fixture_root) as working, pool.threadpool_limits(limits=1):
        data = working / "legacy/data"
        data.mkdir(parents=True)
        for item in manifest["assets"].values():
            png = Path(site.locate_file(item["upstream_file"]))
            assert digest(png) == item["upstream_sha256"]
            shutil.copyfile(png, data / png.name)
        if args.private:
            stage = working / "skimagexx"
            assets = stage / "_data_assets"
            assets.mkdir(parents=True)
            (stage / "__init__.py").write_bytes(b"")
            for name in ("_data_clock_horse.py", "_compat.py"):
                shutil.copyfile(ROOT / "python/skimagexx" / name, stage / name)
            for name in ("clock.gray", "horse.bool", "clock-horse.RIGHTS.txt"):
                shutil.copyfile(ROOT / "python/skimagexx/_data_assets" / name, assets / name)
            sys.path.insert(0, str(working))
            from skimagexx import _data_clock_horse as candidate, _compat
            _compat.set_fallback_enabled(False)
            scope = "private staged package resources"
        else:
            import skimagexx as xx
            from skimagexx import _data_clock_horse, _compat
            xx.set_num_threads(1)
            assert xx.get_num_threads() == 1
            _compat.set_fallback_enabled(False)
            candidate = xx.data
            assert candidate.clock is _data_clock_horse.clock
            assert candidate.horse is _data_clock_horse.horse
            assert digest(Path(_data_clock_horse.__file__)) == pins["python/skimagexx/_data_clock_horse.py"]
            scope = "installed clock and horse public routes"

        class OfflineProvider:
            abspath = working / "offline-cache"

            def fetch(self, filename):
                raise AssertionError("unexpected remote request: " + filename)

        def forbid_new_cache(**kwargs):
            raise AssertionError("unexpected upstream cache creation")

        prior = (_fetchers._image_fetcher, _fetchers.data_dir,
            _fetchers._DISTRIBUTION_DIR, _fetchers._ensure_cache_dir)
        try:
            _fetchers._image_fetcher = OfflineProvider() if prior[0] is not None else None
            _fetchers.data_dir = str(working / "offline-cache/data")
            _fetchers._DISTRIBUTION_DIR = str(data.parent)
            _fetchers._ensure_cache_dir = forbid_new_cache
            for index, name in enumerate(("clock", "horse")):
                case = name + "-warm-local"
                assert case == go["cases"][index]
                asset = manifest["assets"][name]
                png_name = Path(asset["upstream_file"]).name
                assert Path(_fetchers._fetch("data/" + png_name)) == data / png_name
                original, native = getattr(upstream, name), getattr(candidate, name)
                assert inspect.signature(original) == inspect.signature(native)
                expected, actual = original(), native()
                assert type(expected) is type(actual) is np.ndarray
                row = asset["metadata"]
                assert actual.flags.owndata and actual.flags.writeable and actual.base is None
                assert list(actual.shape) == row["shape"] and list(actual.strides) == row["strides"]
                assert actual.dtype == expected.dtype and actual.dtype.str == row["dtype"]
                assert actual.tobytes() == expected.tobytes()
                checksum = sha256(actual.tobytes()).hexdigest()
                assert checksum == row["decoded_sha256"]
                pools = pool.threadpool_info()
                assert pools and all(p["num_threads"] == 1 for p in pools)
                timing = sampler.measure(original, native, 7, 100, journal=journal,
                    case_index=index, case_id=case, pass_index=1, deadline=go["deadline_utc"])
                assert pool.threadpool_info() == pools
                results.append(dict(case_id=case, decoded_sha256=checksum, pools=pools, **timing))
                print(f"{case}: {timing['speedup']:.6f}x", flush=True)
        finally:
            (_fetchers._image_fetcher, _fetchers.data_dir,
             _fetchers._DISTRIBUTION_DIR, _fetchers._ensure_cache_dir) = prior
    assert not working.exists() and threading.active_count() == 1
    assert git("rev-parse", "HEAD") == go["source"] and not git("status", "--porcelain")
    assert {path:digest(ROOT / path) for path in PATHS} == pins
    assert len(results) == 2 and all(list(map(len, row["samples_ms"])) == [7, 7] for row in results)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(scope=scope, source=go["source"], source_raw_sha256=pins,
            versions=go["versions"], runtime=sys.version, accepted_samples=28, results=results,
            timing_child_retained=False), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
