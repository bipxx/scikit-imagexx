"""Check the four pinned performance cases on one new private component."""
from hashlib import sha256
from importlib import import_module
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
SYMBOLS = {
    "hough_line": "hough_line_accumulate",
    "hough_ellipse": "hough_ellipse_bounded",
    "probabilistic_hough_line": "probabilistic_hough_line_ordered",
}


def attach(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def raises_disabled(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except NotImplementedError as error:
        assert "fallback is disabled" in str(error)
    else:
        raise AssertionError("Compatibility case reached the strict route")


def line_cases(source, helper, np, xx, function, hits):
    from skimagexx._compat import FallbackWarning

    observation = json.loads((HERE / "hough_line-stock-0.json").read_bytes())
    stable = helper["snapshot"]

    def snapshot(array, expected):
        actual = stable(array)
        if "data_hex" in expected:
            actual["data_hex"] = array.tobytes(order="C").hex()
        assert actual == expected

    def check(image, theta, expected, native):
        args = (image,) if theta is None else (image, theta)
        snapshot(image, expected["image"])
        if isinstance(theta, np.ndarray):
            snapshot(theta, expected["theta"])
        xx.set_fallback_enabled(False)
        if not native:
            raises_disabled(function, *args)
            xx.set_fallback_enabled(True)
        try:
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter("always")
                try:
                    result = function(*args)
                except Exception as error:
                    assert "error" in expected
                    assert dict(type=type(error).__name__, module=type(error).__module__, message=str(error)) == expected["error"]
                else:
                    saved = expected["result"]
                    assert type(result).__name__ == saved["container_type"]
                    for array, section in zip(result, ("accumulator", "angles", "distances"), strict=True):
                        snapshot(array, saved[section])
                    votes, angles, distances = result
                    assert (angles is theta) is saved["theta_is_input"]
                    assert (isinstance(theta, np.ndarray) and np.shares_memory(angles, theta)) is saved["theta_shares_input"]
                    assert np.all(votes.sum(axis=0) == expected["nonzero_count"])
                    assert [float(distances[0]), float(distances[-1])] == saved["distance_endpoints"]
            assert sum(notice.category is FallbackWarning for notice in notices) == int(not native)
            assert [[n.category.__module__, n.category.__name__, str(n.message)]
                    for n in notices if n.category is not FallbackWarning] == expected["warnings"]
            assert len(hits) == int(native) and (not native or hits[0][0] is image)
            if native and "result" in expected:
                assert hits[0][1] is angles
            snapshot(image, expected["image"])
            if isinstance(theta, np.ndarray):
                snapshot(theta, expected["theta"])
        finally:
            xx.set_fallback_enabled(False)

    recipes = runpy.run_path(str(source / "benchmarks/hough_line.py"))["benchmark_cases"](np)
    for case, expected in zip(recipes, observation["future_benchmark_endpoints"], strict=True):
        case_id, image, theta, recipe = case
        assert (case_id, recipe) == (expected["id"], expected["recipe"])
        yield case_id, lambda: check(image, theta, expected, True)


def ellipse_cases(helper, np, function, hits):
    captured = (HERE / "hough_ellipse-stock-0.jsonl").read_bytes()
    rows = helper["frozen_rows"]()
    assert rows == tuple(json.loads(row) for row in captured.splitlines())
    snapshot, restore, result_record = (helper[name] for name in ("snapshot", "restore", "result_record"))

    def check(row):
        image = restore(np, row)
        if row["scope"] in ("compatibility", "error"):
            raises_disabled(function, image, **row["options"])
            assert not hits
        else:
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter("always")
                result = function(image, **row["options"])
            assert [[n.category.__module__, n.category.__name__, str(n.message)] for n in notices] == row["warnings"]
            assert result_record(np, result) == row["result"]
            assert bool(np.shares_memory(image, result)) is row["output_aliases_input"]
            assert len(hits) == 1 and hits[0][0] is image
        assert snapshot(image, content=True) == row["image"]

    for row in rows:
        if row["scope"] != "benchmark":
            continue
        yield row["id"], lambda: check(row)


def probabilistic_cases(helper, np, function, hits):
    raw = (HERE / "probabilistic_hough_line-stock-0.jsonl").read_bytes()
    base = tuple(json.loads(row) for row in raw.splitlines())
    saved = helper["frozen_rows"]()
    intersection = json.loads((HERE / "probabilistic_hough_line-stock-2.json").read_bytes())
    assert saved[:16] == base and len(saved) == 18
    for recorded, row in zip(intersection["rows"], saved[16:], strict=True):
        assert recorded["id"] == row["id"] and recorded["theta"] == row["theta"] and recorded["output"] == row["result"]
    observe = helper["observe"]

    def check(case, row):
        _, image, theta, rng, options, scope = case
        if scope == "compatibility":
            raises_disabled(function, image, theta=theta, rng=rng, **options)
            assert not hits
        else:
            observe(function, np, case, row)
            assert len(hits) == 1 and hits[0][0] is image
            assert sha256(hits[0][3].tobytes()).hexdigest() == row["rng"]["shuffle_sha256"]

    for case, row in zip(helper["cases"](np), saved, strict=True):
        assert case[0] == row["id"]
        if row["scope"] == "benchmark":
            yield row["id"], lambda: check(case, row)


def main():
    name, mode, source, binary, stock, candidate, result = sys.argv[1:]
    assert not sys.flags.optimize and sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
    assert mode == "normal" and not os.environ.get("SKIMAGEXX_DISABLE_SIMD")
    assert os.sched_getaffinity(0) == {0}
    source, binary, stock, candidate, result = map(Path, (source, binary, stock, candidate, result))
    packet = json.loads((HERE / "bootstrap-v1.json").read_bytes())
    prepared = json.loads((HERE / "speed-prep-v1.json").read_bytes())["families"][name]
    recipe = prepared["benchmark_recipe"]
    assert digest(source / recipe["path"]) == recipe["sha256"]
    sys.path[:0] = [str(candidate), str(stock)]
    import numpy as np
    import skimagexx as xx
    from skimagexx import _core, transform

    assert np.__version__ == "2.5.4" and Path(np.__file__).resolve() == stock / "numpy/__init__.py"
    if name == "probabilistic_hough_line":
        random = import_module("numpy.random")
        modules = {key: import_module("numpy.random." + key)
                   for key in ("_pcg64", "_generator", "bit_generator", "_common", "_bounded_integers")}
        for module in (random, *modules.values()):
            path = Path(module.__file__).resolve()
            assert digest(path) == packet["runtime"]["providers"]["numpy"]["files_sha256"][path.relative_to(stock).as_posix()]
        assert np.random is random and random.PCG64 is modules["_pcg64"].PCG64
        assert random.default_rng is modules["_generator"].default_rng
    core = packet["donor"]["core"]
    assert Path(_core.__file__).resolve() == candidate / core["member"] and digest(_core.__file__) == core["sha256"]
    assert xx.get_num_threads() == 1
    attach("_" + name + "_private", binary).attach_to_core(_core)
    symbol = SYMBOLS[name]
    native = getattr(_core, symbol)
    assert inspect.isbuiltin(native) and native.__module__ == "skimagexx._core"
    facade = attach("skimagexx._" + name, source / "python/skimagexx" / ("_" + name + ".py"))
    function = getattr(facade, name)
    setattr(transform, name, function)
    assert getattr(xx.transform, name) is function
    helper = runpy.run_path(str(source / "tests" / (name + "_reference.py")))
    xx.set_fallback_enabled(False)
    hits = []

    def witness(*args):
        hits.append(args)
        return native(*args)

    setattr(_core, symbol, witness)
    if name == "hough_line":
        cases = line_cases(source, helper, np, xx, function, hits)
    elif name == "hough_ellipse":
        cases = ellipse_cases(helper, np, function, hits)
    else:
        cases = probabilistic_cases(helper, np, function, hits)
    rows = []
    for (label, check), expected in zip(cases, prepared["frozen_cases"], strict=True):
        assert label == expected["id"]
        hits.clear()
        try:
            check()
            failure = None
        except Exception:
            failure = traceback.format_exc()
        rows.append({"id": label, "native_hits": len(hits), "failure": failure,
                     "expected_stock_row_sha256": expected["canonical_row_sha256"]})
    setattr(_core, symbol, native)
    with result.open("x", encoding="utf-8", newline="\n") as output:
        json.dump({"rows": rows, "mode": mode, "private_sha256": digest(binary), "core_sha256": core["sha256"],
                   "workers": xx.get_num_threads(), "avx2": bool(xx.has_avx2())}, output, indent=2)
        output.write("\n")
    assert len(rows) == 4 and all(row["native_hits"] == 1 and row["failure"] is None for row in rows)


if __name__ == "__main__":
    main()
