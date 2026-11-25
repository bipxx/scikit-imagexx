"""Fixed one-thread public-call timings against scikit-image 0.26."""

import argparse
from functools import partial
import json
from statistics import median
from time import perf_counter_ns


def fixed_cases(np, graph):
    rng = np.random.default_rng(19260817)
    costs = 0.25 + rng.random((224, 320))
    ties = np.asfortranarray(rng.integers(0, 12, size=(187, 257)).astype(float))
    volume = rng.random((27, 37, 45))
    mask = rng.random((278, 356)) > 0.27

    def find(cls, values, starts, end, **kwargs):
        solver = cls(values, **kwargs)
        result = solver.find_costs(starts)
        return result, solver.traceback(end)

    return {
        "MCP axial tied grid": partial(find, graph.MCP, ties, [(0, 0)], (186, 256),
                                      fully_connected=False),
        "MCP_Geometric sampled grid": partial(find, graph.MCP_Geometric, costs, [(0, 0)],
                                              (223, 319), sampling=(0.75, 1.25)),
        "MCP_Connect two fronts": partial(find, graph.MCP_Connect, costs, [(0, 0), (223, 319)],
                                          (111, 160)),
        "MCP_Flexible default hooks": partial(find, graph.MCP_Flexible, costs, [(0, 0)],
                                              (223, 319)),
        "route_through_array geometric": partial(graph.route_through_array, costs, (0, 0), (207, 285)),
        "route_through_array axial": partial(graph.route_through_array, ties, (186, 0), (0, 256),
                                            False, False),
        "shortest_path 2D": partial(graph.shortest_path, ties, 2),
        "shortest_path 3D axis 1": partial(graph.shortest_path, volume, 1, 1, True),
        "pixel_graph masked boolean": partial(graph.pixel_graph, mask, connectivity=2),
        "pixel_graph float weights": partial(graph.pixel_graph, costs, spacing=(0.7, 1.4),
                                             sparse_type="array"),
    }


def compare_outputs(np, original, native):
    if type(original) is tuple:
        assert type(native) is tuple and len(original) == len(native)
        for a, b in zip(original, native):
            compare_outputs(np, a, b)
    elif hasattr(original, "indptr"):
        assert type(original) is type(native) and original.shape == native.shape
        for name in ("indptr", "indices", "data"):
            compare_outputs(np, getattr(original, name), getattr(native, name))
    elif isinstance(original, np.ndarray):
        assert original.dtype == native.dtype
        np.testing.assert_array_equal(original, native)
    else:
        assert original == native


def elapsed_ms(call):
    start = perf_counter_ns()
    call()
    return (perf_counter_ns() - start) / 1e6


def benchmark(repeats):
    import numpy as np
    import skimage
    from skimage import graph as original
    from skimagexx import get_num_threads, graph, set_fallback_enabled, set_num_threads
    from threadpoolctl import threadpool_info, threadpool_limits

    if skimage.__version__ != "0.26.0":
        raise RuntimeError("Graph paths benchmark requires scikit-image 0.26.0")
    previous = set_fallback_enabled(False)
    previous_threads = get_num_threads()
    set_num_threads(1)
    results = {}
    try:
        with threadpool_limits(limits=1):
            assert all(pool["num_threads"] == 1 for pool in threadpool_info())
            reference = fixed_cases(np, original)
            candidate = fixed_cases(np, graph)
            assert reference.keys() == candidate.keys()
            for name, stock in reference.items():
                native = candidate[name]
                compare_outputs(np, stock(), native())
                samples = ([], [])
                for i in range(repeats):
                    order = (0, 1) if i % 2 == 0 else (1, 0)
                    for slot in order:
                        samples[slot].append(elapsed_ms((stock, native)[slot]))
                baseline, improved = map(median, samples)
                results[name] = {"original_ms": baseline, "native_ms": improved,
                                 "speedup": baseline / improved, "samples_ms": samples}
    finally:
        set_fallback_enabled(previous)
        set_num_threads(previous_threads)
    return {"skimage": skimage.__version__, "threads": 1, "repeats": repeats, "cases": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("at least three repeats are required")
    report = benchmark(args.repeats)
    print(json.dumps(report, indent=2))
    raise SystemExit(any(case["speedup"] < 1.0 for case in report["cases"].values()))
