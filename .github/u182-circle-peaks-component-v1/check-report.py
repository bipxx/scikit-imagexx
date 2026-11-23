"""Validate the scoped modern build and parity records without importing science."""
from hashlib import sha256
import json
from pathlib import Path
import re
import sys


def read(path):
    return json.loads(path.read_bytes())


def rows_gate(rows, plan, observations):
    assert [row["id"] for row in rows] == plan["row_ids"] and len(rows) == 29
    assert all("failure" not in row for row in rows)
    for observed, expected in zip(rows, observations, strict=True):
        assert observed["scope"] == expected["scope"]
        controlled = observed["scope"] in {"compatibility", "error"}
        assert observed["native_returns"] == int(not controlled) and observed["frozen_fallback"] == int(controlled)
        if "error" in expected:
            assert observed["error"] == expected["error"]
        else:
            assert observed["output_sha256"] == [array["sha256"] for array in expected["result"]]
    counts = [sum(row["native_returns"] for row in rows), sum(row["frozen_fallback"] for row in rows)]
    assert counts == [19, 10]
    return counts


def expected_rows(observations):
    for row in observations:
        result = {key: row[key] for key in ("id", "scope")}
        controlled = row["scope"] in {"compatibility", "error"}
        result.update(native_returns=int(not controlled), frozen_fallback=int(controlled))
        if "error" in row:
            result["error"] = row["error"]
        else:
            result["output_sha256"] = [array["sha256"] for array in row["result"]]
        yield result


def collect(folder, plan, observations):
    terminal = read(folder / "terminal.json")
    assert terminal == {"source": plan["source"]["revision"], "status": {"normal": 0, "scalar": 0},
                        "sdk_exit": 0, "stock_calls": 0, "timings": 0, "exit": 0}
    records = {}
    for phase in ("leaf", "bridge", "elf", "sdk-link", "sdk", "normal", "scalar"):
        path = folder / (phase + ".json")
        command = read(path)
        assert command["exit"] == 0
        assert sha256((folder / (phase + ".log")).read_bytes()).hexdigest() == command["log_sha256"]
        records[phase] = sha256(path.read_bytes()).hexdigest()
    entry = read(folder / "entry.json")
    products = read(folder / "products.json")
    assert entry["floor"] == 30 * 2**30 and all(free >= entry["floor"] for free in entry["capacity"].values())
    assert products["scratch_logical_bytes"] <= entry["compiled_cap"] == 150 * 2**20
    assert all(re.fullmatch("[0-9a-f]{64}", products[key]) for key in ("leaf", "private", "sdk"))
    assert sum(pin["members"] for pin in entry["provider_records"].values()) == 18
    modes = {}
    for mode in ("normal", "scalar"):
        p = folder / (mode + "-rows.json")
        doc = read(p)
        assert doc["mode"] == mode and doc["cpu"] == [0] and doc["workers"] == 1 and doc["avx2"] == (mode == "normal")
        assert doc["private_sha256"] == products["private"] and doc["runtime"] == plan["runtime"]["version"]
        assert doc["stock_rows_sha256"] == plan["stock_rows_sha256"] and doc["stock_peak_calls"] == 0
        ck, = [pin["sha256"] for name, pin in plan["runtime"]["provider_files"].items() if name.startswith("scipy/spatial/_ckdtree.")]
        assert doc["genuine_ckdtree_sha256"] == ck
        native, controlled = rows_gate(doc["results"], plan, observations)
        modes[mode] = {"native": native, "controlled_provider": controlled, "rows_sha256": sha256(p.read_bytes()).hexdigest()}
    return {"modes": modes, "products": products, "processes": records, "sdk_exits": [0, 0], "stock_peak_calls": 0, "timings": 0}


def main():
    plan, stock, mode = sys.argv[1:]
    plan = read(Path(plan))
    raw = Path(stock).read_bytes()
    assert sha256(raw).hexdigest() == plan["stock_rows_sha256"]
    observations = [json.loads(line) for line in raw.splitlines()]
    if mode == "--check":
        rows = list(expected_rows(observations))
        assert rows_gate(rows, plan, observations) == [19, 10]
        invalid = (rows[:-1], rows[::-1], [dict(rows[0], native_returns=0), *rows[1:]])
        for values in invalid:
            try:
                rows_gate(values, plan, observations)
            except AssertionError:
                pass
            else:
                raise AssertionError("Expected collector refusal")
        print(json.dumps({"rows": len(rows), "synthetic_negative_cases": len(invalid), "science_imports": 0}))
    else:
        print(json.dumps(collect(Path(mode), plan, observations), indent=2))


if __name__ == "__main__":
    main()
