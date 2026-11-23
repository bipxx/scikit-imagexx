"""Compare the private modern overlay against the exact captured peak rows."""
import builtins
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import traceback
import warnings

HERE = Path(__file__).resolve().parent


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def attach(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fallback_oracle(np, row):
    for module, category, text in row["warnings"]:
        assert module == "builtins"
        warnings.warn(text, getattr(builtins, category), stacklevel=2)
    if "error" in row:
        module, category, text = row["error"]
        assert module == "builtins"
        raise getattr(builtins, category)(text)
    values = []
    for output in row["result"]:
        assert not output["owns"] and output["base"] == "ndarray"
        data = np.frombuffer(bytes.fromhex(output["data_hex"]), dtype=output["dtype"])
        assert output["shape"] == [data.size] and output["strides"] == [-data.itemsize if data.size else 0]
        values.append(data[::-1].copy()[::-1])
    return tuple(values)


def main():
    mode, manifest, source, inputs, stock, base, native, report = sys.argv[1:]
    assert mode in {"normal", "scalar"} and os.environ.get("SKIMAGEXX_DISABLE_SIMD") == ("1" if mode == "scalar" else None)
    assert os.sched_getaffinity(0) == {0} and sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
    source, inputs, stock, base, native, report = map(Path, (source, inputs, stock, base, native, report))
    plan = json.loads(Path(manifest).read_bytes())
    assert all(os.environ.get(name) == "1" for name in plan["runtime"]["threads"])
    sys.path[:0] = [str(base), str(stock)]
    import numpy as np
    import skimagexx as xx
    from scipy.spatial import _ckdtree
    from skimage import transform as original
    from skimage.transform import hough_transform
    from skimagexx import _core, transform

    assert np.__version__ == "2.5.4"
    for module in (np, _ckdtree, hough_transform):
        path = Path(module.__file__).resolve()
        member = path.relative_to(stock).as_posix()
        assert digest(path) == plan["runtime"]["provider_files"][member]["sha256"]
    assert hough_transform.cKDTree is _ckdtree.cKDTree
    assert Path(_core.__file__).resolve() == base / plan["base"]["core_member"]
    assert digest(_core.__file__) == plan["base"]["members_sha256"][plan["base"]["core_member"]]
    assert xx.get_num_threads() == 1 and bool(xx.has_avx2()) == (mode == "normal")
    attach("_hough_circle_peaks_private", native).attach_to_core(_core)
    kernel = _core.hough_circle_peak_indices
    assert inspect.isbuiltin(kernel) and kernel.__module__ == "skimagexx._core"
    facade = attach("skimagexx._hough_circle_peaks", source / "python/skimagexx/_hough_circle_peaks.py")
    transform.hough_circle_peaks = facade.hough_circle_peaks
    assert xx.transform.hough_circle_peaks is facade.hough_circle_peaks
    original_peak = original.hough_circle_peaks
    original_code = original_peak.__code__
    assert original_peak is hough_transform.hough_circle_peaks
    fixture = runpy.run_path(str(source / "tests/hough_circle_peaks_reference.py"))
    fields = {}
    for name, pin in plan["fields"].items():
        path = inputs / (name + ".npy")
        assert digest(path) == pin["sha256"]
        array = np.load(path, allow_pickle=False)
        snapshot = fixture["snapshot"]
        initial = snapshot(array)
        delta = {key for key, value in pin["input"].items() if initial[key] != value}
        assert delta <= {"owns", "base"}
        fields[name] = array.copy(order="C") if delta else array
        assert snapshot(fields[name]) == pin["input"]
    raw = inputs / "linux-modern-rows-v1.jsonl"
    assert digest(raw) == plan["stock_rows_sha256"]
    rows = [json.loads(line) for line in raw.read_bytes().splitlines()]
    old_inputs = fixture["cases"](np, fields, "linux-minimum")
    result = []
    for (args, old), row in zip(old_inputs, rows, strict=True):
        assert (row["id"], row["scope"], row["inputs"], row["options"]) == (old["id"], old["scope"], old["inputs"], old["options"])
        native_hits = []
        fallback_hits = []
        compat = row["scope"] in {"compatibility", "error"}

        def watch(frame, event, function):
            if event == "c_return" and function is kernel:
                native_hits.append(True)
            if event == "call" and frame.f_code is original_code:
                raise AssertionError("Stock replay is prohibited")

        def replay_frozen(*received, **options):
            assert len(received) == len(args) and all(a is b for a, b in zip(received, args, strict=True))
            assert options == {"min_xdistance": 1, "min_ydistance": 1, "threshold": None, "num_peaks": np.inf,
                               "total_num_peaks": np.inf, "normalize": False} | row["options"]
            fallback_hits.append(True)
            return fallback_oracle(np, row)

        record = {"id": row["id"], "scope": row["scope"]}
        try:
            original.hough_circle_peaks = replay_frozen
            xx.set_fallback_enabled(False)
            sys.setprofile(watch)
            if compat:
                try:
                    xx.transform.hough_circle_peaks(*args, **row["options"])
                except NotImplementedError as err:
                    assert str(err) == "skimage.transform.hough_circle_peaks: this call is outside native coverage; fallback is disabled"
                else:
                    raise AssertionError("Strict route accepted a compatibility case")
                assert not native_hits and not fallback_hits
                xx.set_fallback_enabled(True)
            record.update(fixture["observe"](xx.transform.hough_circle_peaks, np, args, row, fallback_warning=compat))
            assert native_hits == ([] if compat else [True])
            assert fallback_hits == ([True] if compat else [])
        except Exception:
            record["failure"] = traceback.format_exc()
        finally:
            sys.setprofile(None)
            xx.set_fallback_enabled(False)
            original.hough_circle_peaks = original_peak
        record.update(native_returns=len(native_hits), frozen_fallback=len(fallback_hits))
        result.append(record)
    assert [row["id"] for row in result] == plan["row_ids"]
    document = {"mode": mode, "runtime": sys.version, "cpu": list(os.sched_getaffinity(0)), "workers": xx.get_num_threads(),
                "avx2": bool(xx.has_avx2()), "private_sha256": digest(native), "stock_peak_calls": 0,
                "stock_rows_sha256": digest(raw), "genuine_ckdtree_sha256": digest(_ckdtree.__file__), "results": result}
    with report.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, indent=2)
        stream.write("\n")
    assert len(result) == 29 and all("failure" not in row for row in result)
    assert sum(row["native_returns"] for row in result) == 19 and sum(row["frozen_fallback"] for row in result) == 10


if __name__ == "__main__":
    main()
