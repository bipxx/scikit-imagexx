"""Read the compact Windows focus output after authenticated artifact extraction."""

import argparse
from hashlib import sha256
import json
from pathlib import Path, PureWindowsPath
import re
from runpy import run_path


HERE = Path(__file__).resolve().parent
RUN = run_path(str(HERE / "run.py"))
SOURCE = RUN["SOURCE"]
TREE = RUN["TREE"]


def read(folder, name):
    return json.loads((folder / name).read_bytes())


def verify(folder, expected_authority, expected_source_receipt):
    terminal = read(folder, "terminal.json")
    entry = read(folder, "entry.json")
    package = read(folder, "producer-package.json")
    lock = read(folder, "measurement.json")
    sdk = read(folder, "producer-sdk.json")
    modes = ("normal", "scalar")
    checks = dict.fromkeys(modes, 61)
    assert terminal["status"] == "passed_focused_windows" and terminal["profile"] == "windows"
    assert (terminal["source"], terminal["tree"]) == (entry["source"], entry["tree"]) == (SOURCE, TREE)
    assert terminal["complete_source_receipt_sha256"] == entry["source_receipt_sha256"] == expected_source_receipt
    assert terminal["authority_sha256"] == expected_authority
    assert entry["manifest_sha256"] == RUN["MANIFEST_SHA256"]
    assert entry["available_free_bytes"] >= entry["required_free_bytes"] == 44 * 2**30
    assert entry["identity"]["python"].startswith("3.13.16 ") and PureWindowsPath(entry["identity"]["git"]).suffix == ".exe"
    assert terminal["timing_authorized"] is terminal["full_qualification"] is False
    assert terminal["modes"] == terminal["object_provider_modes"] == list(modes)
    assert terminal["public_checks"] == checks and terminal["subtests"] == dict.fromkeys(modes, 18)
    members, core = package["members_sha256"], package["sole_core"]
    [record] = (name for name in members if name.endswith(".dist-info/RECORD"))
    assert package["record_verified"] is True and package["product_scope"] == "current308"
    assert package["source_revision"] == SOURCE and len(members) == 286
    assert len(package["SDK_sha256"]) == 127 and package["SDK_sha256"] == sdk["SDK_sha256"]
    assert package["SDK_sha256"] == {name: digest for name, digest in members.items()
                                      if name.startswith(("include/", "lib/", "share/"))}
    assert terminal["product"] == {"wheel": package["wheel_sha256"], "core": members[core], "record": members[record]}
    assert terminal["sdk_checks"] == sdk["checks"] == 4 and sdk["families"] == list(RUN["FAMILIES"])
    assert lock["source"]["revision"] == SOURCE and lock["runtime"]["sdk_sha256"] == package["SDK_sha256"]
    assert terminal["transport_sha256"] == {name: row["sha256"] for name, row in lock["wheels"].items()}

    cases = None
    nodes = None
    for mode in modes:
        report = read(folder, f"public-{mode}.json")
        witness = read(folder, f"object-provider-{mode}.json")
        assert (report["source"], report["tree"], report["mode"]) == (SOURCE, TREE, mode)
        assert report["product"] == terminal["product"] and report["core"] == core and report["pytest_exit"] == 0
        assert report["family_counts"] == RUN["MODULES"] and report["generic_counts"] == RUN["GENERIC_COUNTS"]
        assert len(report["node_ids"]) == len(set(report["node_ids"])) == 61 and RUN["OBJECT_NODE"] in report["node_ids"]
        assert len(report["subtests"]) == len({row["case"] for row in report["subtests"]}) == 18
        assert all(row["outcome"] == "passed" for row in report["subtests"])
        assert report["original"]["scikit-image"]["version"] == "0.26.0"
        assert tuple(report["original"][name]["version"] for name in ("numpy", "scipy", "pytest")) == (
            "2.5.4", "1.18.1", "9.1.1")
        assert witness["mode"] == mode and witness["calls"] == {"builtin": 3, "stock": 1}
        assert witness["native"] == "tiny-diagonal-1" and witness["strict_refusal_and_replaced_provider"] == "strided"
        assert PureWindowsPath(witness["candidate_core"]).name == PureWindowsPath(core).name
        assert PureWindowsPath(witness["stock_module"]).name == "convex_hull.py"
        RUN["check_suite"](folder / f"public-{mode}.xml", 61, 18)
        RUN["check_suite"](folder / f"sdk-{mode}.xml", 2)
        summary = (folder / f"public-{mode}.log").read_text(encoding="utf-8", errors="replace")
        assert re.search(r"\b61 passed\b[^\n]*\b18 subtests passed\b", summary)
        if nodes is None:
            nodes, cases = report["node_ids"], report["subtests"]
        else:
            assert nodes == report["node_ids"] and cases == report["subtests"]
    return dict(source=SOURCE, tree=TREE, product=terminal["product"], ordinary=checks, subtests=terminal["subtests"],
                sdk=4, authority_sha256=expected_authority,
                evidence_sha256={path.name: sha256(path.read_bytes()).hexdigest()
                                 for path in sorted(folder.iterdir()) if path.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compact_folder", type=Path)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--source-receipt-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.compact_folder, args.authority_sha256, args.source_receipt_sha256), sort_keys=True))


if __name__ == "__main__":
    main()
