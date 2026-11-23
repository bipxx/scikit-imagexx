"""Summarize only an authenticated, complete Medial Axis scalar result."""
from hashlib import sha256
import json
from pathlib import Path
import sys
from xml.etree import ElementTree

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_bytes())


def seal(path):
    content = path.read_bytes()
    return {"bytes": len(content), "sha256": sha256(content).hexdigest()}


def collect(reports, transport, plan=None, expected_nodes=None):
    if plan is None:
        plan = read(HERE / "portable-pins-v1.json")
    if expected_nodes is None:
        expected_nodes = read(HERE / "predecessor.json")["nodes"]
    terminal = read(reports / "terminal.json")
    assert terminal == {"stage": "transport", "exit": 0, "purpose": "medial-axis-modern-scalar-v1"}
    authority = read(reports / "authority-capacity.json")
    assert len(authority["observed_free_bytes"]) == 2
    assert all(free >= authority["floor"] == 30 * 2**30 for free in authority["observed_free_bytes"].values())
    for step in ("git-fetch", "git-checkout", "stock-install", "pybind-install",
                 "verify-portable", "leaf", "link", "scalar"):
        assert read(reports / f"{step}.json")["exit"] == 0, step
    assert not any(path.name.startswith(("normal", "speed", "sdk")) for path in reports.iterdir())
    pins = read(reports / "portable-installed.json")
    assert pins["python"] == plan["runtime"]["version"] and pins["science_imports"] == 0
    assert pins["python_sha256"] == plan["runtime"]["executable_sha256"]
    assert set(pins["distributions"]) == set(plan["stock_distributions"])
    layout = plan["pip_target_layout_v2"]
    installer = read(reports / "installer-layout.json")
    assert all(installer[key] == layout[key] for key in (
        "version", "scheme", "scripts_record_prefix", "bundle_sha256", "source_sha256"))
    for name, served in pins["distributions"].items():
        wanted = plan["stock_distributions"][name]
        assert served["wheel_sha256"] == wanted["wheel_sha256"]
        assert served["retained_uv_record_sha256"] == wanted["retained_uv_record_sha256"]
        assert served["immutable_members"] == wanted["immutable_members"]
        assert served["immutable_digest"] == wanted["retained_immutable_digest"]
        assert len(served["wheel_record_sha256"]) == len(served["host_record_sha256"]) == 64
        wrappers = served["generated_wrappers"]
        assert set(wrappers) == {layout["scripts_record_prefix"] + script
                                for script in wanted["generated_entry_points"]}
        assert all(digest.startswith("sha256=") and len(digest) == 50 and int(size) > 0
                   for digest, size in wrappers.values())
    assert len(pins["member_pins"]) == sum(item["immutable_members"] for item in pins["distributions"].values())
    source = read(reports / "source-proof.json")
    assert source == {"source": plan["source"], "tree": plan["tree"], "paths": len(plan["source_paths"])}
    scalar_path = reports / "scalar-result.json"
    scalar = read(scalar_path)
    assert all(scalar[key] == expected for key, expected in source.items())
    assert (scalar["cases"], scalar["exit"], scalar["cpu"], scalar["workers"], scalar["avx2"]) == (30, 0, 0, 1, False)
    assert all(scalar[key] == 0 for key in ("stock_replay", "normal_replay", "sdk_replay", "timing"))
    assert scalar["bindings"] == ["medial_axis_prepare", "medial_axis_ordered"]
    assert scalar["donor_sha256"] == plan["donor"]["core_sha256"]
    assert scalar["portable"] == seal(reports / "portable-installed.json")
    assert set(plan["required_live_after_scalar"]) <= scalar["loaded"].keys()
    assert all(pins["member_pins"][name][0] == value for name, value in scalar["loaded"].items())
    xml = reports / "scalar.xml"
    assert scalar["xml"] == seal(xml)
    suite, = ElementTree.parse(xml).iter("testsuite")
    nodes = {(case.attrib["classname"], case.attrib["name"]) for case in suite.findall("testcase")}
    assert [int(suite.get(key, 0)) for key in ("tests", "failures", "errors", "skipped")] == [30, 0, 0, 0]
    assert len(nodes) == 30 and all("test_medial_axis_public" in module for module, _ in nodes)
    assert {name for _, name in nodes} == set(expected_nodes)
    envelope = read(transport / "medial-scalar.json")
    name = "_medial_axis_private" + plan["runtime"]["extension_suffix"]
    assert {path.name for path in transport.iterdir()} == {name, "medial-scalar.json"}
    assert envelope["source"] == source["source"] and envelope["tree"] == source["tree"]
    binary = transport / name
    assert binary.stat().st_size < 2**20 and not binary.is_symlink()
    raw = binary.read_bytes()
    assert raw[:6] == b"\x7fELF\x02\x01" and raw[18:20] == b"\x3e\x00"
    assert envelope["elf"] == {"name": name, **seal(binary)}
    assert envelope["elf"]["sha256"] == scalar["private_sha256"]
    assert envelope["scalar"] == seal(scalar_path) and envelope["no_timing"] is True
    assert envelope["authority_sha256"] == authority["authority_sha256"]
    return {"source": source, "scalar": 30, "elf": envelope["elf"],
            "portable_sha256": seal(reports / "portable-installed.json")["sha256"],
            "loaded_members": len(scalar["loaded"]), "generated_wrappers": layout["generated_wrappers"],
            "prior_normal_replay": 0,
            "stock_replay": 0, "sdk_replay": 0, "timing": 0}


if __name__ == "__main__":
    if sys.flags.optimize:
        raise SystemExit("Optimized Python is unsupported")
    print(json.dumps(collect(*map(Path, sys.argv[1:])), indent=2))
