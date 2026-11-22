"""Sustained one-worker comparisons of the two local Chelsea public paths."""

import argparse
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
    "benchmarks/chelsea.py", "coverage/contracts/chelsea-cat-v1.json",
    "python/skimagexx/_data_chelsea.py", "python/skimagexx/_compat.py",
    "python/skimagexx/_data_assets/chelsea.rgb", "python/skimagexx/_data_assets/chelsea.CC0.txt",
    "tests/fixtures/chelsea_original_windows_v1.json", "tests/fixtures/chelsea_original_linux_v1.json",
    "tools/montage_sustained.py",
)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def load(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--go", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    journal = args.output.with_suffix(".batches.jsonl")
    go = json.loads(args.go.read_bytes())
    assert go["authority"] == "chelsea-two-one-worker-cases"
    assert str(Path(sys.executable)) == go["python"]
    assert datetime.now(timezone.utc) < datetime.fromisoformat(go["deadline_utc"])
    assert not args.output.exists() and not journal.exists()
    assert {os.environ[key] for key in LIMITS} == {"1"} and threading.active_count() == 1
    git_prefix = ["git", "-C", str(ROOT)]
    if os.name != "nt":
        assert sys.flags.isolated and sys.dont_write_bytecode
        assert "PYTHONPATH" not in os.environ and "PYTHONHOME" not in os.environ
        git_prefix = ["/mnt/c/Program Files/Git/cmd/git.exe", "-C", go["windows_checkout"]]

    def git(*git_args):
        return subprocess.check_output([*git_prefix, *git_args], text=True).strip()

    assert git("rev-parse", "HEAD") == go["source"] and not git("status", "--porcelain")
    pins = {path: digest(ROOT / path) for path in PATHS}
    assert pins == go["source_raw_sha256"]
    assert args.fixture_root.resolve().is_relative_to(ROOT / "build")

    import numpy as np
    from skimage import data as upstream
    from skimage.data import _fetchers

    assert upstream.cat is upstream.chelsea
    assert digest(Path(inspect.getsourcefile(upstream.chelsea))) == go["original_sha256"]
    assert [version(name) for name in ("scikit-image", "pillow", "imageio", "numpy")] == go["versions"]
    package = distribution("scikit-image")
    png = Path(package.locate_file("skimage/data/chelsea.png"))
    manifest = json.loads((ROOT / "coverage/contracts/chelsea-cat-v1.json").read_bytes())
    assert digest(png) == manifest["original"]["sha256"]
    sample = load("_chelsea_sampler", ROOT / "tools/montage_sustained.py")
    results = []
    with TemporaryDirectory(prefix="chelsea-timing-", dir=args.fixture_root) as raw:
        working = Path(raw)
        root = working / "original"
        target = root / "data/chelsea.png"
        target.parent.mkdir(parents=True)
        shutil.copyfile(png, target)
        if args.private:
            stage = working / "skimagexx"
            stage.mkdir()
            (stage / "__init__.py").write_bytes(b"")
            for name in ("_data_chelsea.py", "_compat.py"):
                shutil.copyfile(ROOT / "python/skimagexx" / name, stage / name)
            for name in ("chelsea.rgb", "chelsea.CC0.txt"):
                orig = ROOT / "python/skimagexx/_data_assets" / name
                destination = stage / "_data_assets" / name
                destination.parent.mkdir(exist_ok=True)
                shutil.copyfile(orig, destination)
            sys.path.insert(0, raw)
            from skimagexx import _data_chelsea as candidate, _compat
            _compat.set_fallback_enabled(False)
            scope = "private staged Python resource"
        else:
            import skimagexx as xx
            from skimagexx import _data_chelsea, _compat
            xx.set_num_threads(1)
            assert xx.get_num_threads() == 1
            _compat.set_fallback_enabled(False)
            candidate = xx.data
            assert candidate.chelsea is _data_chelsea.chelsea
            assert digest(Path(_data_chelsea.__file__)) == pins["python/skimagexx/_data_chelsea.py"]
            scope = "installed provider-owned public paths"
        assert candidate.cat is candidate.chelsea
        assert inspect.signature(candidate.chelsea) == inspect.signature(upstream.chelsea)

        class OfflineProvider:
            abspath = working / "offline-cache"

            def fetch(self, name):
                raise AssertionError("unexpected Chelsea remote fetch: " + name)

        prior = (_fetchers._image_fetcher, _fetchers.data_dir, _fetchers._DISTRIBUTION_DIR,
                 _fetchers._ensure_cache_dir)
        try:
            _fetchers._image_fetcher = OfflineProvider() if prior[0] is not None else None
            _fetchers.data_dir = str(working / "offline-cache/data")
            _fetchers._DISTRIBUTION_DIR = str(root)

            def forbidden_directory(**kwargs):
                raise AssertionError("unexpected Chelsea cache creation")
            _fetchers._ensure_cache_dir = forbidden_directory
            assert Path(_fetchers._fetch("data/chelsea.png")) == target
            for index, name in enumerate(("chelsea", "cat")):
                case = name + "-warm-local"
                assert case == go["cases"][index]
                original = getattr(upstream, name)
                native = getattr(candidate, name)
                expected, actual = original(), native()
                assert type(actual) is type(expected) is np.ndarray
                assert actual.shape == expected.shape == (300, 451, 3)
                assert actual.strides == expected.strides == (1353, 3, 1)
                assert actual.flags.owndata and actual.flags.writeable and actual.base is None
                assert actual.dtype == expected.dtype == np.uint8
                assert actual.tobytes() == expected.tobytes()
                output_digest = sha256(actual.tobytes()).hexdigest()
                assert output_digest == manifest["resource"]["sha256"]
                timing = sample.measure(original, native, 7, 100, journal=journal,
                    case_index=index, case_id=case, pass_index=1, deadline=go["deadline_utc"])
                assert threading.active_count() == 1
                results.append(dict(case_id=case, decoded_sha256=output_digest, **timing))
                print(f"{case}: {timing['speedup']:.6f}x", flush=True)
        finally:
            (_fetchers._image_fetcher, _fetchers.data_dir, _fetchers._DISTRIBUTION_DIR,
             _fetchers._ensure_cache_dir) = prior
    assert not working.exists()
    assert git("rev-parse", "HEAD") == go["source"] and not git("status", "--porcelain")
    assert {path:digest(ROOT / path) for path in PATHS} == pins
    assert len(results) == 2 and all(list(map(len, row["samples_ms"])) == [7, 7] for row in results)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(scope=scope, source=go["source"], source_raw_sha256=pins,
            versions=go["versions"], python=sys.version, aliases_identical=True, results=results,
            accepted_samples=28, fixture_retained=False), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
