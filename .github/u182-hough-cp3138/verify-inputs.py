"""Read only pre-staged Hough source objects or CP313.8 runtime inputs."""

from argparse import ArgumentParser
from base64 import urlsafe_b64encode
import csv
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig


def check(raw, pin):
    assert len(raw) == pin["bytes"] and sha256(raw).hexdigest() == pin["sha256"]


def source_proof(packet, repo, git_path, windows_preflight):
    assert git_path.is_absolute() and git_path.is_file() and repo.is_absolute() and repo.is_dir()

    def git(*args):
        return subprocess.check_output([git_path, "--no-optional-locks", "--no-replace-objects", "-C", repo, *args])

    for name, family in packet["families"].items():
        source = family["source"]
        ref = family["existing_local_ref"] if windows_preflight else family["ref"]
        assert git("rev-parse", "--verify", ref).decode().strip() == source, (name, ref)
        assert git("rev-list", "--parents", "-n", "1", source).decode().split() == [source, family["published_parent"]]
        assert git("rev-parse", f"{source}^{{tree}}").decode().strip() == family["tree"]
        changed = git("diff-tree", "--no-commit-id", "--name-status", "-r", family["published_parent"], source)
        expected = {"A\t" + path for path in family["owned_git_files"]}
        assert set(changed.decode().splitlines()) == expected
        for paths in ("owned_git_files", "verified_shared_git_files"):
            for path, pin in family[paths].items():
                mode, kind, blob = git("ls-tree", source, "--", path).split(b"\t", 1)[0].decode().split()
                assert (mode, kind, blob) == (pin["mode"], "blob", pin["git_blob"]), path
                check(git("cat-file", "blob", f"{source}:{path}"), pin)
    return {"families": len(packet["families"]), "owned_paths": 30, "windows_local_preflight": windows_preflight}


def runtime_proof(packet, stock, pybind, candidate):
    assert sys.platform == "linux" and sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
    assert not sys.flags.optimize and os.sched_getaffinity(0) == {0}
    assert all(os.environ.get(name) == "1" for name in packet["runtime"]["thread_environment"])
    assert not any(os.environ.get(name) for name in ("PYTHONPATH", "PYTHONHOME"))
    assert all(path.is_absolute() and path.is_dir() for path in (stock, pybind, candidate))
    stock, pybind, candidate = (path.resolve() for path in (stock, pybind, candidate))
    assert len({stock, pybind, candidate}) == 3
    runtime = packet["runtime"]
    assert sys.version == runtime["version"] and sysconfig.get_config_var("EXT_SUFFIX") == runtime["ext_suffix"]
    assert sha256(Path(sys.executable).resolve().read_bytes()).hexdigest() == runtime["executable_sha256"]
    include = Path(sysconfig.get_path("include"))
    for name, pin in runtime["cpython_headers"].items():
        check((include / name).read_bytes(), pin)
    for name, pin in runtime["pybind_headers"].items():
        check((pybind / "pybind11" / name).read_bytes(), pin)
    providers = tuple(metadata.distributions(path=[str(stock)]))
    for key, dist in runtime["providers"].items():
        provider, = [p for p in providers if p.metadata["Name"].lower().replace("-", "_") == key]
        assert provider.version == dist["version"]
        rows = list(csv.reader(provider.read_text("RECORD").splitlines()))
        records = {row[0]: row[1:] for row in rows}
        assert len(records) == len(rows)
        for name, pin in dist["files_sha256"].items():
            path = Path(provider.locate_file(name)).resolve(strict=True)
            assert path.is_relative_to(stock), name
            raw = path.read_bytes()
            assert sha256(raw).hexdigest() == pin, name
            token = urlsafe_b64encode(bytes.fromhex(pin)).decode().rstrip("=")
            assert records[name] == ["sha256=" + token, str(len(raw))], name
    core = packet["donor"]["core"]
    check((candidate / core["member"]).read_bytes(), core)
    return {"runtime": sys.version, "selected_provider_members": sum(len(v["files_sha256"]) for v in runtime["providers"].values()),
            "donor_core": core["sha256"], "science_imports": 0}


if __name__ == "__main__":
    if sys.flags.optimize:
        raise SystemExit("Optimization disabled for assertions")
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--packet-sha256", required=True)
    modes = parser.add_subparsers(dest="mode", required=True)
    source = modes.add_parser("source")
    source.add_argument("--repo", required=True, type=Path)
    source.add_argument("--git", required=True, type=Path)
    source.add_argument("--windows-preflight", action="store_true")
    runtime = modes.add_parser("runtime")
    for directory in ("stock", "pybind", "candidate"):
        runtime.add_argument(f"--{directory}", required=True, type=Path)
    args = parser.parse_args()
    raw = args.packet.read_bytes()
    assert sha256(raw).hexdigest() == args.packet_sha256
    packet = json.loads(raw)
    if args.mode == "source":
        verdict = source_proof(packet, args.repo, args.git, args.windows_preflight)
    else:
        verdict = runtime_proof(packet, args.stock, args.pybind, args.candidate)
    print(json.dumps(verdict, sort_keys=True))
