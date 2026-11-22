"""Sustained one-worker measurements for four fixed local texture assets."""

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
ASSETS = ("brick", "grass", "gravel", "text")
RIGHTS = "textures.RIGHTS.txt"
PATHS = (
    "benchmarks/data_textures.py", "coverage/contracts/data-texture-assets-v1.json",
    "python/skimagexx/_data_textures.py", "python/skimagexx/_compat.py",
    "python/skimagexx/_data_assets/" + RIGHTS,
    "tests/fixtures/texture_assets_original_windows_v1.json",
    "tests/fixtures/texture_assets_original_linux_v1.json", "tools/montage_sustained.py",
) + tuple(f"python/skimagexx/_data_assets/{name}.gray" for name in ASSETS)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--go", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    go = json.loads(args.go.read_bytes())
    journal = args.output.with_suffix(".batches.jsonl")
    assert go["authority"] == "four-texture-local-cases"
    assert str(Path(sys.executable)) == go["python"]
    assert datetime.now(timezone.utc) < datetime.fromisoformat(go["deadline_utc"])
    assert not args.output.exists() and not journal.exists()
    assert {os.environ[key] for key in LIMITS} == {"1"} and threading.active_count() == 1
    git_prefix = ["git", "-C", str(ROOT)]
    if os.name != "nt":
        assert sys.flags.isolated and sys.dont_write_bytecode
        assert "PYTHONPATH" not in os.environ and "PYTHONHOME" not in os.environ
        git_prefix = ["/mnt/c/Program Files/Git/cmd/git.exe", "-C", go["windows_checkout"]]

    def git(*items):
        return subprocess.check_output([*git_prefix, *items], text=True).strip()

    assert git("rev-parse", "HEAD") == go["source"] and not git("status", "--porcelain")
    pins = {p: digest(ROOT / p) for p in PATHS}
    assert pins == go["source_raw_sha256"]
    assert args.fixture_root.resolve().is_relative_to(ROOT / "build")

    import numpy as np
    from skimage import data as upstream
    from skimage.data import _fetchers

    origin = Path(inspect.getsourcefile(upstream.brick))
    assert digest(origin) == go["original_sha256"]
    assert [version(x) for x in ("scikit-image", "numpy", "pillow", "imageio")] == go["versions"]
    package = distribution("scikit-image")
    manifest = json.loads((ROOT / "coverage/contracts/data-texture-assets-v1.json").read_bytes())
    assert go["cases"] == manifest["performance"]["cases"]
    spec = spec_from_file_location("_texture_sampler", ROOT / "tools/montage_sustained.py")
    sampler = module_from_spec(spec)
    spec.loader.exec_module(sampler)
    results = []
    with TemporaryDirectory(prefix="texture-timing-", dir=args.fixture_root) as folder:
        working = Path(folder)
        legacy = working / "legacy"
        (legacy / "data").mkdir(parents=True)
        for name, entry in manifest["assets"].items():
            png = Path(package.locate_file(entry["upstream_source"]))
            assert digest(png) == entry["upstream_sha256"]
            shutil.copyfile(png, legacy / "data" / png.name)
        if args.private:
            stage = working / "skimagexx"
            stage.mkdir()
            (stage / "__init__.py").write_bytes(b"")
            for name in ("_data_textures.py", "_compat.py"):
                shutil.copyfile(ROOT / "python/skimagexx" / name, stage / name)
            assets_dir = stage / "_data_assets"
            assets_dir.mkdir()
            for item in (*[name + ".gray" for name in ASSETS], RIGHTS):
                shutil.copyfile(ROOT / "python/skimagexx/_data_assets" / item, assets_dir / item)
            sys.path.insert(0, folder)
            from skimagexx import _data_textures as candidate, _compat
            _compat.set_fallback_enabled(False)
            scope = "private staged Python package assets"
        else:
            import skimagexx as xx
            from skimagexx import _data_textures, _compat
            xx.set_num_threads(1)
            assert xx.get_num_threads() == 1
            _compat.set_fallback_enabled(False)
            candidate = xx.data
            assert all(getattr(candidate, k) is getattr(_data_textures, k) for k in ASSETS)
            assert digest(Path(_data_textures.__file__)) == pins["python/skimagexx/_data_textures.py"]
            scope = "installed public texture assets"

        class OfflineProvider:
            abspath = working / "offline-cache"

            def fetch(self, path):
                raise AssertionError("unexpected remote texture request: " + path)

        def no_cache_creation(**kwargs):
            raise AssertionError("unexpected texture cache creation")

        prior = (_fetchers._image_fetcher, _fetchers.data_dir,
                 _fetchers._DISTRIBUTION_DIR, _fetchers._ensure_cache_dir)
        try:
            _fetchers._image_fetcher = OfflineProvider() if prior[0] is not None else None
            _fetchers.data_dir = str(working / "offline-cache/data")
            _fetchers._DISTRIBUTION_DIR = str(legacy)
            _fetchers._ensure_cache_dir = no_cache_creation
            for index, name in enumerate(ASSETS):
                case = name + "-warm-local"
                assert case == go["cases"][index]
                record = manifest["assets"][name]["representation"]
                assert Path(_fetchers._fetch(f"data/{name}.png")) == legacy / "data" / (name + ".png")
                reference, target = getattr(upstream, name), getattr(candidate, name)
                assert inspect.signature(reference) == inspect.signature(target)
                expected, actual = reference(), target()
                assert type(expected) is type(actual) is np.ndarray
                assert list(actual.shape) == record["shape"] and list(actual.strides) == record["strides"]
                assert actual.flags.owndata and actual.flags.writeable and actual.base is None
                assert actual.dtype == expected.dtype == np.uint8 and actual.tobytes() == expected.tobytes()
                output_hash = sha256(actual.tobytes()).hexdigest()
                assert output_hash == record["decoded_sha256"]
                result = sampler.measure(reference, target, 7, 100, journal=journal, case_index=index,
                    case_id=case, pass_index=1, deadline=go["deadline_utc"])
                results.append(dict(case_id=case, decoded_sha256=output_hash, **result))
                print(f"{case}: {result['speedup']:.6f}x", flush=True)
        finally:
            (_fetchers._image_fetcher, _fetchers.data_dir,
             _fetchers._DISTRIBUTION_DIR, _fetchers._ensure_cache_dir) = prior
    assert not working.exists() and threading.active_count() == 1
    assert git("rev-parse", "HEAD") == go["source"] and not git("status", "--porcelain")
    assert {p: digest(ROOT / p) for p in PATHS} == pins
    assert len(results) == 4 and all(list(map(len, row["samples_ms"])) == [7, 7] for row in results)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(scope=scope, source=go["source"], source_raw_sha256=pins, runtime=sys.version,
            versions=go["versions"], results=results, accepted_samples=56, fixture_retained=False), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
