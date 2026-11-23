"""Validate the bounded 29-case report without importing scientific packages."""
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_bytes())


def ordered_ids(rows, expected):
    assert len(expected) == len(set(expected)) == len(rows) == 29
    assert [row["id"] for row in rows] == expected
    assert all(row["inputs_unchanged"] is True and isinstance(row["warnings"], list) for row in rows)
    assert all(("result" in row) != ("error" in row) for row in rows)


def comparisons(rows, profiles):
    result = {}
    for profile, prior in profiles.items():
        differences = {}
        for row, old in zip(rows, prior, strict=True):
            keys = sorted(key for key in row.keys() | old.keys() if row.get(key) != old.get(key))
            if keys:
                differences[row["id"]] = keys
        result[profile] = {"matches": 29 - len(differences), "differences": differences}
    return result


def check_saved(manifest, profiles):
    windows, minimum = (profiles[name] for name in ("windows", "linux-minimum"))
    for rows in (windows, minimum):
        ordered_ids(rows, manifest["row_ids"])
    delta = comparisons(windows, profiles)
    assert delta["windows"] == {"matches": 29, "differences": {}}
    assert set(delta["linux-minimum"]["differences"]) == {"from-full-border", *manifest["fixed"]}


def collect(out, manifest, profiles):
    assert read(out / "terminal.json") == {"stage": "stock-capture", "exit": 0, "purpose": "circle-peaks-stock-only"}
    assert all(read(out / (stage + ".json"))["exit"] == 0
               for stage in ("family-source", "installer", "stock-install", "stock-capture"))
    capacity = read(out / "authority-capacity.json")
    assert len(capacity["free"]) == 2 and all(value >= capacity["floor"] == 30 * 2**30 for value in capacity["free"].values())
    t = read(out / "donor-transfer.json")
    assert t["artifact"] == 14763663715 and not t["whole_outer_verified"] and t["bytes_read"] <= 160 * 2**20
    bootstrap = read(HERE / "bootstrap.json")
    prefix = read(HERE / "historical-transport.json")["namespace"]
    assert set(t["members"]) == {prefix + name for name in
                                 ("manifest.json", *(pin["full_qualification"] for pin in bootstrap["donor"]["wheel_inputs"].values()))}
    capture = out / "capture"
    terminal = read(capture / "linux-modern-terminal-v1.json")
    entry = read(capture / "entry-v1.json")
    raw = (capture / "linux-modern-rows-v1.jsonl").read_bytes()
    assert len(raw) <= 8 * 2**20 and sha256(raw).hexdigest() == terminal["rows_sha256"]
    assert sha256((capture / "entry-v1.json").read_bytes()).hexdigest() == terminal["entry_sha256"]
    assert [terminal[key] for key in ("stock_peak_calls", "stock_producer_calls", "candidate_calls", "timing_calls")] == [29, 0, 0, 0]
    assert terminal["row_ids"] == manifest["row_ids"] and entry["fixed"] == manifest["fixed"]
    assert sum(group["members"] for group in entry["hosted"]["providers"].values()) == 18
    assert entry["hosted"]["authority"] == capacity["authority"]
    rows = [json.loads(line) for line in raw.splitlines()]
    ordered_ids(rows, manifest["row_ids"])
    assert comparisons(rows, profiles) == terminal["comparisons"]
    return {"rows": 29, "rows_sha256": terminal["rows_sha256"], "comparisons": terminal["comparisons"]}


if __name__ == "__main__":
    [argument] = sys.argv[1:]
    manifest = read(HERE / "family-inputs.json")
    profiles = read(HERE / "source/coverage/contracts/hough-circle-peaks-stock-v1.json")["profiles"]
    check_saved(manifest, profiles)
    result = {"saved_profiles": 2, "rows_each": 29, "science_imports": 0}
    if argument != "--check":
        result = collect(Path(argument), manifest, profiles)
    print(json.dumps(result, indent=2))
