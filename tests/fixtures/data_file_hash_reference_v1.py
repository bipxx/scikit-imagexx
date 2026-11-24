"""Reuse both retained original captures for local file-hash parity checks."""

import inspect
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "coverage/contracts/data-file-hash-v1.json"
CAPTURE_WINDOWS = Path(__file__).with_name("data_file_hash_windows_v1.json")
CAPTURE_LINUX = Path(__file__).with_name("data_file_hash_linux_v1.json")
CONTENTS = {
    "empty": b"",
    "small": b"skimagexx\x00local data\n",
    "chunk-boundary": bytes(range(256)) * 256 + b"\x00",
}


class PathProtocol:
    """Count calls through the file argument's path protocol."""

    def __init__(self, filename):
        self.filename = filename
        self.calls = 0

    def __fspath__(self):
        self.calls += 1
        return self.filename


def outcome(function, args, kwargs):
    try:
        value = function(*args, **kwargs)
    except (OSError, TypeError, ValueError) as error:
        result = dict(type=type(error).__name__, message=str(error))
        if isinstance(error, OSError):
            result["errno"] = error.errno
        return result
    return dict(type=type(value).__name__, hexdigest=value)


def run_contract(directory, candidate, original):
    """Check frozen success values, exceptions, order and exact live parity."""
    assert inspect.signature(candidate) == inspect.signature(original)
    try:
        from pooch import file_hash as provider
    except ModuleNotFoundError:
        provider = None
    if provider is not None:
        assert candidate is original is provider
    else:
        assert candidate is not original

    windows = json.loads(CAPTURE_WINDOWS.read_bytes())
    linux = json.loads(CAPTURE_LINUX.read_bytes())
    frozen_linux = {row["id"]: row["outcome"] for row in linux["outcomes"]}
    paths = {name: directory / (name + ".bin") for name in (*CONTENTS, "missing")}
    paths["directory"] = directory
    for name, content in CONTENTS.items():
        paths[name].write_bytes(content)
    results = []
    for row in windows["outcomes"]:
        path = paths[row["file"]]
        arg = {
            "path": path, "str": str(path), "bytes": os.fsencode(path),
            "invalid": object(), "nul": str(path) + "\x00",
        }[row["path_kind"]]
        algorithm = row["algorithm"]
        kwargs = {} if algorithm is None else {
            "alg": {"__none__": None, "__list__": []}.get(algorithm, algorithm),
        }
        result = outcome(candidate, (arg,), kwargs)
        assert result == outcome(original, (arg,), kwargs), row["id"]
        frozen = row["outcome"] if provider is None else frozen_linux[row["id"]]
        if "hexdigest" in frozen:
            assert result == frozen
        elif (os.name == "nt") == (provider is None):
            assert result["type"] == frozen["type"]
            if "errno" in frozen:
                assert result["errno"] == frozen["errno"]
        results.append(dict(id=row["id"], outcome=result))

    for arg in (str(paths["small"]), os.fsencode(paths["small"])):
        protocol = PathProtocol(arg)
        assert candidate(fname=protocol) == original(arg)
        assert protocol.calls == 1
        protocol.calls = 0
        answer = outcome(candidate, (protocol,), {"alg": "not-a-hash"})
        assert answer["type"] == "ValueError" and protocol.calls == 0
    invalid = PathProtocol(object())
    assert outcome(candidate, (invalid,), {}) == outcome(original, (invalid,), {})
    assert invalid.calls == 2
    assert all(paths[name].read_bytes() == content for name, content in CONTENTS.items())
    return dict(profile="Pooch direct alias" if provider else "stdlib hashlib fallback",
                results=results, protocol_forms=2)
