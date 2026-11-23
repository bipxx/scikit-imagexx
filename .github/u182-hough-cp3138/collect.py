"""Verify changed line code generation, both complete modes and the new ELF."""
from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
NAME = "_hough_line_private.cpython-313-x86_64-linux-gnu.so"


def read(path):
    return json.loads(path.read_bytes())


def seal(data):
    return {"bytes": len(data), "sha256": sha256(data).hexdigest()}


def expected_ids():
    data = read(HERE / "hough_line-stock-0.json")
    rows = data["rows"] + data["future_benchmark_endpoints"]
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids)) == 23
    return [(row["id"], int(row.get("scope", "bounded") == "bounded")) for row in rows]


def collect(reports, transport):
    candidate = read(transport / "line-wide.json")
    staging = read(reports / "authority-capacity.json")
    terminal = read(reports / "terminal.json")
    variant = read(HERE / "wide-line.json")
    source = read(reports / "source-proof.json")
    assert source == {"before": variant["parent"], "wide": variant["source"], "tree": variant["tree"],
                      "file": variant["sole_changed_source"], "other_source_files_unchanged": True,
                      "before_sha256": variant["old_source_sha256"], "wide_sha256": variant["new_source_sha256"]}
    assert {path.name for path in transport.iterdir()} == {NAME, "line-wide.json"}
    assert candidate["authority"] == staging["authority"] and candidate["no_timing"] is True
    assert (candidate["source"], candidate["tree"]) == (variant["source"], variant["tree"])
    assert terminal["exit"] == 0 and terminal["stage"] == "transport" and terminal["purpose"] == variant["purpose"]
    assert len(staging["observed_free_bytes"]) == 2
    assert all(free >= staging["required_bytes"] == 30 * 2**30 for free in staging["observed_free_bytes"].values())
    for stage in ("git-fetch", "baseline-leaf", "hough_line-leaf", "baseline-disassembly", "wide-disassembly", "hough_line-link", "runtime-proof"):
        assert read(reports / (stage + ".json"))["exit"] == 0
    proof = reports / "instruction-proof.json"
    assert seal(proof.read_bytes()) == candidate["instruction_proof"]
    observed = read(proof)
    parse = runpy.run_path(str(HERE / "run.py"))["check_vote_paths"]
    actual = parse((reports / "baseline-disassembly.log").read_text(), (reports / "wide-disassembly.log").read_text())
    assert all(observed[key] == value for key, value in actual.items())
    binary = transport / NAME
    assert not binary.is_symlink() and binary.stat().st_size < 2**20
    data = binary.read_bytes()
    assert data[:6] == bytes((127, 69, 76, 70, 2, 1)) and data[18:20] == bytes((62, 0))
    assert seal(data) == read(reports / "hough_line-elf.json") and candidate["elf"] == {"name": NAME, **seal(data)}
    assert not any(path.name.startswith(("hough_ellipse-", "probabilistic_hough_line-")) or "-sdk" in path.name for path in reports.iterdir())
    runtime = read(reports / "runtime-proof.log")
    assert runtime == read(HERE / "v3-reuse-v1.json")["runtime"]
    compiled = {}
    for mode in ("normal", "scalar"):
        assert read(reports / f"hough_line-{mode}.json")["exit"] == 0
        path = reports / f"hough_line-{mode}-cases.json"
        assert seal(path.read_bytes()) == candidate["modes"][mode]
        witness = read(path)
        assert (witness["mode"], witness["workers"], witness["core_sha256"], witness["private_sha256"]) == (
            mode, 1, runtime["donor_core"], candidate["elf"]["sha256"])
        assert mode != "scalar" or not witness["avx2"]
        names_and_hits = [(row["id"], row["native_hits"]) for row in witness["rows"]]
        assert names_and_hits == expected_ids() and all(row["failure"] is None for row in witness["rows"])
        compiled[mode] = {"cases": len(names_and_hits), "native_hits": sum(hit for _, hit in names_and_hits),
                          "sha256": candidate["modes"][mode]["sha256"]}
    return {"source": variant["source"], "instruction_proof": candidate["instruction_proof"], "modes": compiled,
            "elf": candidate["elf"], "transport": {path.name: seal(path.read_bytes()) for path in transport.iterdir()},
            "timing": False, "sdk_replays": 0}


if __name__ == "__main__":
    reports, transport = map(Path, sys.argv[1:])
    print(json.dumps(collect(reports, transport), indent=2))
