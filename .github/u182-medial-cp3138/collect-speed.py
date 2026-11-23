"""Check exact Medial Axis pairs, the sustained raw samples and every loss."""
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import sys

HERE = Path(__file__).resolve().parent
PROFILE = "linux-modern-hosted"
CASE_IDS = ["symmetric-thick", "perforated-distance", "masked-components-distance", "seeded-corridors"]


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()] if path.is_file() else []


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def verify_batches(case, batches, index):
    assert case["sample_labels"] == ["scikit-image", "current"]
    assert case["batch_records"] == batches
    samples, loops, warm = [[], []], [[], []], [0, 0]
    orders = [[0, 1] if (round_id + index + 1) % 2 == 0 else [1, 0] for round_id in range(7)]
    accepted = []
    for item in batches:
        side = item["function"]
        assert side in (0, 1) and item["loops"] > 0 and item["elapsed_ns"] > 0
        assert item["complete"] and item["failure"] is None
        if item["phase"] == "warmup":
            warm[side] += item["elapsed_ns"]
        elif item["accepted"]:
            assert item["phase"] == "sample" and item["elapsed_ns"] >= 100_000_000
            samples[side].append(item["elapsed_ns"] / item["loops"] / 1e6)
            loops[side].append(item["loops"])
            accepted.append((item["round"], item["ordinal"], side))
        else:
            assert item["phase"] == "underlength-calibration" and item["elapsed_ns"] < 100_000_000
    assert list(map(len, samples)) == [7, 7] and min(warm) >= 250_000_000
    assert (samples, loops, warm, orders) == (case["samples_ms"], case["loops"],
                                             case["warmup_elapsed_ns"], case["sample_order"])
    assert accepted == [(i, ordinal, side) for i, order in enumerate(orders)
                        for ordinal, side in enumerate(order)]
    reference, candidate = map(median, samples)
    assert (reference, candidate, reference / candidate) == (case["reference_ms"], case["cpp_ms"], case["speedup"])


def verify_pair(pair, frozen):
    assert pair["case_id"] == frozen["case_id"]
    assert pair["fallback_enabled"] is False and pair["seed"] == int(frozen["rng"])
    assert pair["stock_calls"] == {"public": 1, "label": 1024, "edt": 1}
    assert pair["native_calls"] == ["medial_axis_prepare", "edt", "medial_axis_ordered"]
    assert pair["stock_order"] == frozen["loop"][0]["order"]
    assert all(pair[key] == frozen.get(key) for key in ("input", "mask", "skeleton", "distance"))


def collect(stage, output):
    process = read(stage / "timing-process.json")
    quiet = read(stage / "quiet.json")
    ready = read(stage / "stage-ready.json")
    assert (stage / "stage-process.exit").read_text().strip() == "0"
    assert quiet["stage_exit"] == 0 and quiet["cpu"] == 0 and quiet["active_competing_local_jobs"] == []
    assert quiet["authority_sha256"] == ready["authority_sha256"]
    admission = read(HERE / "scalar-admission.json")
    transfer = read(stage / "elf-transfer.json")
    assert transfer["elf"] == admission["elf"] and transfer["envelope"] == admission["envelope"]
    assert not any(transfer[key] for key in ("archive_retained", "candidate_recompiled", "scalar_replayed"))
    report = output / f"{PROFILE}-result-v1.json"
    journal = output / f"{PROFILE}-raw-batches-v1.jsonl"
    proof = output / f"{PROFILE}-correctness-v1.jsonl"
    history, pairs = rows(journal), rows(proof)
    accepted = sum(row["event"] == "batch_after" and row["accepted"] for row in history)
    if not report.is_file():
        assert process["exit"] != 0 and 0 <= len(pairs) <= 4 and 0 <= accepted <= 56
        return {"complete": False, "process_exit": process["exit"], "pairs": len(pairs), "accepted_samples": accepted}
    result = read(report)
    assert result["accepted_samples"] == accepted == 56 and result["private_component_only"] is True
    assert result["entry_sha256"] == digest(output / "entry.json")
    assert result["correctness_sha256"] == digest(proof) and result["raw_batches_sha256"] == digest(journal)
    assert result["root_go_sha256"] == quiet["authority_sha256"]
    identity = result["identity"]
    assert identity == read(output / "entry.json")
    assert (identity["source"], identity["tree"], identity["cpu"], identity["workers"]) == (
        admission["source"], admission["tree"], 0, 1)
    assert identity["private_core_sha256"] == admission["elf"]["sha256"]
    assert identity["quiet_sha256"] == digest(stage / "quiet.json")
    assert identity["transfer_sha256"] == digest(stage / "elf-transfer.json")
    assert result["pools"] and all(pool["num_threads"] == 1 for pool in result["pools"])
    markers = [(idx, row) for idx, row in enumerate(history) if row["event"] == "paired_exact_before_samples"]
    frozen = rows(HERE / "linux-modern-local-cases-v1.jsonl")[:4]
    cases = result["cases"]
    assert len(cases) == len(pairs) == len(markers) == len(frozen) == 4
    ratios = []
    for index, case_id in enumerate(CASE_IDS):
        case, pair, reference = cases[index], pairs[index], frozen[index]
        assert case["case_id"] == reference["case_id"] == case_id
        verify_pair(pair, reference)
        position, marker = markers[index]
        assert marker == {"event": "paired_exact_before_samples", "case_index": index, "case_id": case_id,
                          "elf": admission["elf"]["sha256"]}
        batches = [(idx, row) for idx, row in enumerate(history) if row["event"] == "batch_after"
                   and row["case_index"] == index and row["case_id"] == case_id]
        assert batches and position < batches[0][0]
        verify_batches(case, [row for _, row in batches], index)
        ratios.append({key: case[key] for key in ("case_id", "speedup", "reference_ms", "cpp_ms")})
    losses = [item for item in ratios if item["speedup"] < 1.]
    assert result["status"] == ("below_parity" if losses else "parity_speed_pass")
    assert process["exit"] == bool(losses)
    return {"complete": True, "process_exit": process["exit"], "pairs": 4, "accepted_samples": accepted,
            "cpu_model": identity["cpu_model"], "elf": admission["elf"], "ratios": ratios, "losses": losses,
            "report_sha256": digest(report), "raw_sha256": digest(journal), "proof_sha256": digest(proof)}


if __name__ == "__main__":
    if sys.flags.optimize:
        raise SystemExit("Optimized Python is unsupported")
    print(json.dumps(collect(*map(Path, sys.argv[1:])), indent=2))
