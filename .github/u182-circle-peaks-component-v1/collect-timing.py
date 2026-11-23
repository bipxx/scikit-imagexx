"""Reconstruct four exact medians and 56 accepted batches from the raw journal."""
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


def verify_pairs(plan, report, history, process_exit):
    method = report["method"]
    assert (method["threads"], method["samples_per_side"], method["min_sample_ms"], method["min_warmup_ms"], method["affinity"]) == (
        1, 7, 100, 250, [0])
    assert report["source"] == plan["source"] and report["private_sha256"] == plan["elf"]["sha256"]
    expected = plan["fixed"]
    assert len(expected) == 4
    pairs = [(position, row) for position, row in enumerate(history) if row["event"] == "paired_exact_before_samples"]
    results = report["results"]
    assert len(results) <= len(pairs) <= 4
    for index, (position, pair) in enumerate(pairs):
        pin = expected[index]
        assert pair["case_id"] == pin["id"] and pair["saved_case_sha256"] == pin["row_sha256"]
        assert pair["private_sha256"] == plan["elf"]["sha256"] and pair["native_hits"] == 1 and pair["inputs_unchanged"]
        assert pair["outputs"]["output_sha256"] == pin["outputs"]
        if index < len(results):
            row = results[index]
            assert row["case_id"] == pin["id"]
            batches = [(n, item) for n, item in enumerate(history) if item["event"] == "batch_after" and item["case_index"] == index]
            assert batches and position < batches[0][0]
            verify_row(row, [item for _, item in batches], index)
    if pairs:
        entry, = [row for row in history if row["event"] == "quiet_entry_verified"]
        assert entry["cpu"] == method["cpu"] and entry["affinity"] == [0]
        assert entry["pools"] and all(pin["num_threads"] == 1 for pin in entry["pools"])
        assert sum(pin["members"] for pin in entry["provider_records"].values()) == 18
    accepted = sum(row["event"] == "batch_after" and row.get("accepted", False) for row in history)
    ratios = [{"case": row["case_id"], "ratio": row["speedup"], "stock_ms": row["reference_ms"],
               "native_ms": row["cpp_ms"]} for row in results]
    complete = len(results) == len(pairs) == 4 and accepted == 56 and not any(row["event"] == "failure" for row in history)
    losses = [row for row in ratios if row["ratio"] < 1.0]
    assert process_exit == int(bool(losses)) if complete else process_exit != 0
    return {"complete": complete, "cases": len(results), "accepted": accepted, "paired_correctness": len(pairs), "ratios": ratios, "losses": losses}


def collect(stage, output):
    grant = read(stage / "quiet.json")
    state = read(stage / "stage-ready.json")
    process = read(stage / "timing-process.json")
    assert (stage / "stage-process.exit").read_text().strip() == "0"
    assert grant["authority_sha256"] == state["authority_sha256"] and grant["cpu"] == 0
    assert grant["stage_process_exit"] == 0 and grant["transfers_and_builds_terminal"] and not grant["active_competing_local_jobs"]
    plan = read(HERE / "timing-plan.json")
    transport = read(stage / "prepared-transport.json")
    assert transport["expected"] == plan["transport"] and not transport["archive_retained"]
    assert transport["transfer_bytes"] == plan["transport"]["verified_zip_bytes"]
    report, journal = (output / name for name in ("report.json", "journal.jsonl"))
    if not (report.is_file() and journal.is_file()):
        assert process["exit"] != 0
        return {"complete": False, "process_exit": process["exit"], "authority_sha256": grant["authority_sha256"]}
    raw = journal.read_bytes()
    data = read(report)
    assert data["quiet"] == grant and data["authority_sha256"] == grant["authority_sha256"]
    proof = verify_pairs(plan, data, [json.loads(line) for line in raw.splitlines()], process["exit"])
    return dict(proof, process_exit=process["exit"], authority_sha256=grant["authority_sha256"],
                report_sha256=sha256(report.read_bytes()).hexdigest(), journal_sha256=sha256(raw).hexdigest())


if __name__ == "__main__":
    stage, output = map(Path, sys.argv[1:])
    proof = collect(stage, output)
    with (stage / "collection.json").open("x", encoding="utf8", newline="\n") as stream:
        json.dump(proof, stream, indent=2)
        stream.write("\n")
