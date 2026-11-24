"""Isolated sustained comparisons for the local file digest routes."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading


ROOT = Path(__file__).resolve().parents[1]
WINDOWS_GIT = "/mnt/c/Program Files/Git/cmd/git.exe"
LIMITS = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
PATHS = (
    "benchmarks/data_file_hash.py", "python/skimagexx/_data_file_hash.py",
    "coverage/contracts/data-file-hash-v1.json", "tests/fixtures/data_file_hash_reference_v1.py",
    "tests/fixtures/data_file_hash_windows_v1.json", "tests/fixtures/data_file_hash_linux_v1.json",
    "tools/montage_sustained.py",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--go", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    go = json.loads(args.go.read_bytes())
    assert go["authority"] == "data-file-hash-four-one-worker-cases"
    assert datetime.now(timezone.utc) < datetime.fromisoformat(go["deadline_utc"])
    assert not args.output.exists()
    journal = args.output.with_suffix(".batches.jsonl")
    assert not journal.exists()
    assert {os.environ[key] for key in LIMITS} == {"1"}
    assert str(Path(sys.executable)) == go["python"]
    assert threading.active_count() == 1
    if os.name == "nt":
        git_prefix = ["git", "-C", str(ROOT)]
    else:
        assert sys.flags.isolated and sys.dont_write_bytecode
        assert "PYTHONPATH" not in os.environ and "PYTHONHOME" not in os.environ
        git_prefix = [WINDOWS_GIT, "-C", go["windows_checkout"]]

    def git(*git_args):
        return subprocess.check_output([*git_prefix, *git_args], text=True).strip()

    assert git("rev-parse", "HEAD") == go["source"] and not git("status", "--porcelain")
    pins = {path: digest(ROOT / path) for path in PATHS}
    assert pins == go["source_raw_sha256"]

    import skimage
    from skimage.data import file_hash as reference

    assert skimage.__version__ == "0.26.0"
    module_path = ROOT / "python/skimagexx/_data_file_hash.py"
    if args.private:
        candidate = load("_private_file_hash", module_path).file_hash
        scope = "private provider-backed Python route"
    else:
        import skimagexx as xx
        from skimagexx import _data_file_hash
        xx.set_fallback_enabled(False)
        xx.set_num_threads(1)
        assert xx.get_num_threads() == 1
        assert digest(Path(_data_file_hash.__file__)) == digest(module_path)
        candidate = xx.data.file_hash
        assert candidate is _data_file_hash.file_hash
        scope = "installed provider-backed public route"

    identical = candidate is reference
    assert identical is go["pooch_alias"]
    assert inspect.signature(candidate) == inspect.signature(reference)
    assert digest(Path(inspect.getsourcefile(reference))) == go["original_sha256"]
    if identical:
        import pooch
        assert candidate is pooch.file_hash and version("pooch") == go["provider_version"]
    else:
        assert candidate.__module__ != reference.__module__

    factory = load("_data_file_hash_cases", ROOT / "tests/fixtures/data_file_hash_reference_v1.py")
    sampling = load("_data_file_hash_sampler", ROOT / "tools/montage_sustained.py")
    contract = json.loads((ROOT / "coverage/contracts/data-file-hash-v1.json").read_bytes())
    cases = contract["performance"]["case_ids"]
    assert cases == go["cases"]
    results = []
    assert args.fixture_root.resolve().is_relative_to(ROOT / "build")
    with TemporaryDirectory(prefix="file-hash-timing-", dir=args.fixture_root) as raw:
        fixture = Path(raw)
        sizes = contract["performance"]["bytes"]
        last_payload = bytes(range(256)) * (sizes[-1] // 256)
        payloads = (*factory.CONTENTS.values(), last_payload)
        algorithms = ("sha256", "sha256", "sha512", "sha256")
        for index, (case, payload, size, algorithm) in enumerate(zip(
                cases, payloads, sizes, algorithms, strict=True)):
            assert len(payload) == size
            path = fixture / (case + ".bin")
            path.write_bytes(payload)
            filename = (path, str(path), os.fsencode(path), path)[index]
            options = {} if index == 0 else {"alg": algorithm}
            expected = hashlib.new(algorithm, payload).hexdigest()
            original_result = reference(filename, **options)
            actual_result = candidate(filename, **options)
            assert type(original_result) is str and type(actual_result) is str
            assert original_result == actual_result == expected
            timing = sampling.measure(lambda: reference(filename, **options),
                lambda: candidate(filename, **options), 7, 100, journal=journal, case_index=index,
                case_id=case, pass_index=1, deadline=go["deadline_utc"])
            assert threading.active_count() == 1 and path.read_bytes() == payload
            results.append(dict(case_id=case, file_bytes=size,
                                 algorithm=algorithm, hexdigest=expected, **timing))
            print(f"{case}: {timing['speedup']:.6f}x", flush=True)
    assert not fixture.exists()
    assert git("rev-parse", "HEAD") == go["source"] and not git("status", "--porcelain")
    assert {path: digest(ROOT / path) for path in PATHS} == pins
    assert all(list(map(len, row["samples_ms"])) == [7, 7] for row in results)
    result = dict(scope=scope, source=go["source"], source_raw_sha256=pins,
                  original_sha256=go["original_sha256"], identical_callable=identical,
                  supplier="Pooch" if identical else "stdlib hashlib", python=sys.version,
                  results=results, accepted_samples=len(results) * 14, fixture_retained=False)
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
