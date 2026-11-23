"""Bind the frozen circle-peaks stock capture to a prepared Linux CP313.8 host."""
from argparse import ArgumentParser
from base64 import urlsafe_b64encode
import csv
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import sysconfig


def pin(data, digest, size=None):
    assert sha256(data).hexdigest() == digest
    assert size is None or len(data) == size
    return data


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, indent=2)
        output.write("\n")


def payload(args, manifest):
    pinned = {}
    for name, path in (("capture", args.capture), ("prior_entry", args.prior_entry)):
        pinned[name] = pin(path.read_bytes(), manifest["inputs"][name])
    previous = json.loads(pinned["prior_entry"])
    assert (previous["source"], previous["tree"], previous["parent"]) == tuple(manifest["source_identity"])
    assert previous["launcher_sha256"] == manifest["inputs"]["capture"]
    git = [str(args.git), "--no-optional-locks", "--no-replace-objects", "-C", str(args.git_repo)]

    def source(*argv):
        return subprocess.check_output([*git, *argv])

    if not args.check_payload:
        assert source("rev-parse", "HEAD").decode().strip() == os.environ["GITHUB_SHA"]
        assert source("rev-parse", args.tree).decode().strip() == source("rev-parse", "HEAD^{tree}").decode().strip()
    carrier_tree = source("rev-parse", args.tree).decode().strip()
    for name, check in manifest["source_files"].items():
        git_path = ".github/u182-circle-peaks-stock-v1/source/" + name
        _, kind, blob = source("ls-tree", carrier_tree, "--", git_path).split(b"\t", 1)[0].decode().split()
        assert kind == "blob" and blob == check["git_blob"]
        raw = pin(source("cat-file", "blob", f"{carrier_tree}:{git_path}"), check["sha256"], check["bytes"])
        assert raw == (args.source / name).read_bytes().replace(b"\r\n", b"\n")
    fixture = runpy.run_path(str(args.source / "tests/hough_circle_peaks_reference.py"))
    frozen = fixture["reference"]()
    assert set(frozen["profiles"]) == {"windows", "linux-minimum"}
    for rows in frozen["profiles"].values():
        assert [row["id"] for row in rows] == manifest["row_ids"]
    assert list(fixture["FIXED"]) == manifest["fixed"]
    for name in fixture["CIRCLE_IDS"]:
        path = args.prepared / (name + ".npy")
        record = previous["prepared"][name]
        assert record["input"] == frozen["prepared_circle_records"][name]
        pin(path.read_bytes(), record["sha256"], manifest["prepared_sizes"][name])
        record["path"] = str(path)
    return previous, fixture, frozen


def runtime(previous, stock):
    assert sys.platform == "linux" and os.sched_getaffinity(0) == {0}
    assert sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
    assert not sys.flags.optimize
    assert not any(os.environ.get(key) for key in ("PYTHONHOME", "PYTHONPATH", "LD_PRELOAD", "LD_AUDIT",
                                                  "NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE"))
    assert all(os.environ.get(key) == "1" for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                     "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_THREAD_LIMIT"))
    assert not {"numpy", "scipy", "skimage", "skimagexx"} & sys.modules.keys()
    identity = previous["identity"]
    assert sys.version == identity["runtime_version"]
    assert sysconfig.get_config_var("EXT_SUFFIX") == ".cpython-313-x86_64-linux-gnu.so"
    pin(Path(sys.executable).resolve().read_bytes(), identity["runtime_sha256"])
    providers = tuple(metadata.distributions(path=[str(stock)]))
    actual = {}
    for key, group in identity["distributions"].items():
        dist, = [item for item in providers if item.metadata["Name"].lower().replace("-", "_") == key]
        assert dist.version == group["version"]
        fp, = [item for item in dist.files if item.as_posix().endswith(".dist-info/RECORD")]
        record = Path(dist.locate_file(fp)).resolve(strict=True)
        assert record.is_relative_to(stock)
        data = record.read_bytes()
        rows = list(csv.reader(data.decode().splitlines()))
        by_path = {row[0]: row[1:] for row in rows}
        assert len(rows) == len(by_path)
        for name, check in group["files"].items():
            path = Path(dist.locate_file(name)).resolve(strict=True)
            assert path.is_relative_to(stock)
            raw = pin(path.read_bytes(), check["sha256"], check["bytes"])
            value = urlsafe_b64encode(bytes.fromhex(check["sha256"])).decode().rstrip("=")
            assert by_path[name] == ["sha256=" + value, str(len(raw))]
        actual[key] = {"version": dist.version, "record": str(record), "record_sha256": sha256(data).hexdigest(),
                       "members": len(group["files"])}
    assert sum(group["members"] for group in actual.values()) == 18
    assert not (stock / "skimagexx").exists()
    identity["site"] = str(stock)
    return actual


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("manifest", "capture", "prior-entry", "source", "prepared", "git", "git-repo"):
        parser.add_argument("--" + name, required=True, type=lambda value: Path(value).resolve(strict=True))
    parser.add_argument("--tree", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--check-payload", action="store_true")
    parser.add_argument("--stock", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = pin(args.manifest.read_bytes(), args.manifest_sha256)
    manifest = json.loads(raw)
    if not args.check_payload:
        if not os.environ.get("HOUGH_STOCK_AUTHORITY") or not os.environ.get("HOUGH_STOCK_AUTHORITY_SHA256"):
            print("Root authority is required", file=sys.stderr)
            return 2
        assert os.environ["GITHUB_ACTIONS"] == "true" and os.environ["GITHUB_EVENT_NAME"] == "workflow_dispatch"
        assert os.environ["GITHUB_REPOSITORY"] == "bipxx/scikit-imagexx"
        grant = json.loads(pin(os.environ["HOUGH_STOCK_AUTHORITY"].encode(), os.environ["HOUGH_STOCK_AUTHORITY_SHA256"]))
        assert set(grant) == {"purpose", "carrier", "manifest_sha256", "deadline_utc"}
        assert (grant["purpose"], grant["carrier"], grant["manifest_sha256"]) == (
            "circle-peaks-stock-only", os.environ["GITHUB_SHA"], args.manifest_sha256)
        assert 0 < (datetime.fromisoformat(grant["deadline_utc"]) - datetime.now(timezone.utc)).total_seconds() <= 5400
    entry, fixture, frozen = payload(args, manifest)
    if args.check_payload:
        assert args.stock is None and args.output is None
        assert not {"numpy", "skimage", "skimagexx"} & sys.modules.keys()
        print(json.dumps({"rows": len(manifest["row_ids"]), "fields": len(entry["prepared"]), "science_imports": 0}))
        return 0
    stock = args.stock.resolve(strict=True)
    output = args.output.resolve()
    assert not output.exists() and output.parent.is_dir() and output != stock
    assert shutil.disk_usage(output.parent).free >= 30 * 2**30
    actual = runtime(entry, stock)
    capture = runpy.run_path(str(args.capture))["capture"]
    capture.__globals__["HERE"] = output
    output.mkdir()
    entry["launcher_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    entry["hosted"] = {"providers": actual, "scope": grant["purpose"], "authority": grant,
                       "python": str(Path(sys.executable).resolve()), "manifest_sha256": args.manifest_sha256}
    write(output / "entry-v1.json", entry)
    capture(entry, fixture, frozen)
    assert sum(path.stat().st_size for path in output.iterdir()) <= entry["incremental_cap_bytes"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
