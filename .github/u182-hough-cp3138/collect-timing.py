"""Verify exact case order, provenance and raw sustained batches."""
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import sys

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_bytes())


def verify_row(row, batches, index):
    assert row["sample_labels"] == ["scikit-image", "current"]
    assert row["batch_records"] == batches
    samples = [[], []]
    loops = [[], []]
    warm = [0, 0]
    for batch in batches:
        side = batch["function"]
        assert side in (0, 1) and batch["loops"] > 0 and batch["elapsed_ns"] > 0
        assert batch["complete"] and batch["failure"] is None
        if batch["phase"] == "warmup":
            warm[side] += batch["elapsed_ns"]
        elif batch["accepted"]:
            assert batch["phase"] == "sample" and batch["elapsed_ns"] >= 100_000_000
            samples[side].append(batch["elapsed_ns"] / batch["loops"] / 1e6)
            loops[side].append(batch["loops"])
        else:
            assert batch["phase"] == "underlength-calibration" and batch["elapsed_ns"] < 100_000_000
    assert warm == row["warmup_elapsed_ns"] and min(warm) >= 250_000_000
    assert samples == row["samples_ms"] and loops == row["loops"] and list(map(len, samples)) == [7, 7]
    expected_order = [[0, 1] if (r + index + 1) % 2 == 0 else [1, 0] for r in range(7)]
    assert row["sample_order"] == expected_order
    accepted = [batch for batch in batches if batch["accepted"]]
    assert [(item["round"], item["ordinal"], item["function"]) for item in accepted] == [
        (round_id, ordinal, side) for round_id, order in enumerate(expected_order) for ordinal, side in enumerate(order)]
    medians = [median(side) for side in samples]
    assert medians == [row["reference_ms"], row["cpp_ms"]] and row["speedup"] == medians[0] / medians[1]


def collect(stage, outputs):
    process = read(stage / "timing-process.json")
    grant = read(stage / "quiet.json")
    state = read(stage / "stage-ready.json")
    assert (stage / "stage-process.exit").read_text().strip() == "0"
    assert grant["authority_sha256"] == state["authority_sha256"]
    assert grant["cpu"] == 0 and grant["transfers_and_builds_terminal"] and not grant["active_competing_local_jobs"]
    result = {"process_exit": process["exit"], "authority_sha256": grant["authority_sha256"], "complete": False}
    report, journal = outputs / "report.json", outputs / "journal.jsonl"
    if not (report.is_file() and journal.is_file()):
        assert process["exit"] != 0
        return result
    history = [json.loads(line) for line in journal.read_bytes().splitlines()]
    data = read(report)
    assert data["quiet"] == grant and data["authority_sha256"] == grant["authority_sha256"]
    method = data["method"]
    assert (method["threads"], method["samples_per_side"], method["min_sample_ms"], method["min_warmup_ms"], method["affinity"]) == (1, 7, 100, 250, [0])
    plan = read(HERE / "speed-prep-v1.json")
    prep = read(HERE / "prep-admission.json")
    fixed = [(family, row) for family, spec in plan["families"].items() for row in spec["frozen_cases"]]
    rows = data["results"]
    assert len(rows) <= len(fixed) == 12
    pairs = [(entry, item) for entry, item in enumerate(history) if item["event"] == "paired_exact_before_samples"]
    for index, (row, (family, expected)) in enumerate(zip(rows, fixed)):
        assert (row["family"], row["case_id"]) == (family, expected["id"])
        position, pair = pairs[index]
        assert pair == {"event": "paired_exact_before_samples", "family": family, "case_id": expected["id"],
                        "private_sha256": prep["products"][family]["elf"]["sha256"], "native_hits": 1,
                        "saved_case_sha256": expected["canonical_row_sha256"]}
        batches = [(n, item) for n, item in enumerate(history) if item["event"] == "batch_after" and item["case_index"] == index]
        assert batches and position < batches[0][0]
        verify_row(row, [item for _, item in batches], index)
    if rows:
        entry, = [item for item in history if item["event"] == "quiet_entry_verified"]
        assert entry["cpu"] == method["cpu"] and entry["affinity"] == [0]
        assert entry["pools"] and all(row["num_threads"] == 1 for row in entry["pools"])
    accepted = sum(item["event"] == "batch_after" and item.get("accepted", False) for item in history)
    result.update(cases=len(rows), same_run_pairs=len(pairs), accepted_samples=accepted,
                  report_sha256=sha256(report.read_bytes()).hexdigest(), journal_sha256=sha256(journal.read_bytes()).hexdigest(),
                  ratios=[{"family": row["family"], "case": row["case_id"], "speedup": row["speedup"],
                           "stock_ms": row["reference_ms"], "candidate_ms": row["cpp_ms"]} for row in rows])
    result["complete"] = len(rows) == len(pairs) == 12 and accepted == 168 and not any(row["event"] == "failure" for row in history)
    result["losses"] = [row for row in result["ratios"] if row["speedup"] < 1.0]
    if result["complete"]:
        assert process["exit"] == (1 if result["losses"] else 0)
    else:
        assert process["exit"] != 0
    return result


if __name__ == "__main__":
    stage, outputs = map(Path, sys.argv[1:])
    result = collect(stage, outputs)
    with (stage / "collection.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
