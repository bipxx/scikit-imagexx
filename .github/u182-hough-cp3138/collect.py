"""Check the bounded hosted diagnostic reports without running a component."""
from hashlib import sha256
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
FAMILY = "probabilistic_hough_line"


def read(path):
    return json.loads(path.read_bytes())


def expected_ids(root=HERE):
    line = read(root / "hough_line-stock-0.json")
    ellipse = [json.loads(row) for row in (root / "hough_ellipse-stock-0.jsonl").read_bytes().splitlines()]
    prob = [json.loads(row) for row in (root / "probabilistic_hough_line-stock-0.jsonl").read_bytes().splitlines()]
    prob.extend(read(root / "probabilistic_hough_line-stock-2.json")["rows"])
    cases = {"hough_line": [row["id"] for row in line["rows"] + line["future_benchmark_endpoints"]],
             "hough_ellipse": [row["id"] for row in ellipse],
             "probabilistic_hough_line": [row["id"] for row in prob]}
    assert [len(cases[name]) for name in cases] == [23, 14, 18]
    assert all(len(rows) == len(set(rows)) for rows in cases.values())
    return cases


def check_cases(ids, report, mode, private, core):
    assert report["mode"] == mode and report["workers"] == 1 and (mode != "scalar" or not report["avx2"])
    assert report["private_sha256"] == private and report["core_sha256"] == core
    assert [row["id"] for row in report["rows"]] == ids
    assert all(row["failure"] is None and row["native_hits"] in (0, 1) for row in report["rows"])
    return sum(row["native_hits"] for row in report["rows"])


def collect(folder):
    terminal = read(folder / "terminal.json")
    authority = read(folder / "authority-capacity.json")
    purpose = terminal["purpose"]
    assert terminal["exit"] == 0 and authority["authority"]["purpose"] == purpose == "probabilistic-parity"
    assert len(authority["observed_free_bytes"]) == 2
    assert all(free >= authority["required_bytes"] == 30 * 2**30 for free in authority["observed_free_bytes"].values())
    for stage in ("source-proof", "venv", "stock-install", "pybind-install", "runtime-proof"):
        assert read(folder / (stage + ".json"))["exit"] == 0
    packet = read(HERE / "bootstrap-v1.json")
    core = packet["donor"]["core"]["sha256"]
    previous = read(folder / "historical-v3-admission.json")
    assert previous == read(HERE / "v3-reuse-v1.json")
    assert previous["root_admission_sha256"] == authority["authority"]["prior_root_admission_sha256"]
    assert read(folder / "runtime-proof.log") == previous["runtime"]
    assert read(folder / "source-proof.log") == {"families": 1, "owned_paths": 10, "windows_local_preflight": False}
    assert not any(path.name.startswith(("hough_line-", "hough_ellipse-")) or "-sdk" in path.name for path in folder.iterdir())
    ids = expected_ids()[FAMILY]
    binary = read(folder / (FAMILY + "-elf.json"))["sha256"]
    for stage in ("leaf", "link"):
        assert read(folder / f"{FAMILY}-{stage}.json")["exit"] == 0
    count = {}
    for mode in ("normal", "scalar"):
        assert read(folder / f"{FAMILY}-{mode}.json")["exit"] == 0
        count[mode] = check_cases(ids, read(folder / f"{FAMILY}-{mode}-cases.json"), mode, binary, core)
        assert count[mode] == 17
    assert sorted(path.name for path in folder.glob("*-cases.json")) == [
        f"{FAMILY}-normal-cases.json", f"{FAMILY}-scalar-cases.json"]
    inputs = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix in {".json", ".log"} and path.name != "collection.json":
            assert path.stat().st_size < 2**20
            inputs[path.name] = sha256(path.read_bytes()).hexdigest()
    return {"purpose": purpose, "actual_v4": {FAMILY: {"named_cases_per_mode": len(ids), "native_hits_per_mode": count}},
            "historical_v3": previous, "report_sha256": inputs, "timing": False}


if __name__ == "__main__":
    [path] = sys.argv[1:]
    print(json.dumps(collect(Path(path)), indent=2))
