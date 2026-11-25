"""Public parity for the lattice paths and CSR graph family."""

import inspect
import pickle

import numpy as np
import pytest
from skimage import graph as original


CLASSES = ("MCP", "MCP_Connect", "MCP_Flexible", "MCP_Geometric")
CALLABLES = ("route_through_array", "shortest_path", "pixel_graph")


@pytest.fixture
def api():
    from skimagexx import _core, graph, set_fallback_enabled
    from skimagexx import _graph_paths as family

    for name in (*CLASSES, *CALLABLES):
        assert vars(graph)[name] is getattr(family, name)
    assert inspect.isbuiltin(_core.graph_paths_shortest)
    assert inspect.isbuiltin(_core.graph_paths_pixel)
    previous = set_fallback_enabled(False)
    try:
        yield graph
    finally:
        set_fallback_enabled(previous)


def assert_arrays(left, right):
    np.testing.assert_array_equal(left, right)
    assert left.dtype == right.dtype
    assert left.dtype.metadata == right.dtype.metadata
    assert left.strides == right.strides


@pytest.mark.parametrize("name", (*CLASSES, *CALLABLES))
def test_signatures_and_hierarchy(api, name):
    native = getattr(api, name)
    stock = getattr(original, name)
    if name in CALLABLES:
        assert inspect.signature(native) == inspect.signature(stock)
        return
    assert native.__bases__ == ((object,) if name == "MCP" else (api.MCP,))
    for member in set(dir(stock)) - set(dir(object)) | {"__init__", "__new__", "__reduce__"}:
        if member == "__pyx_vtable__":
            continue
        expected = inspect.getattr_static(stock, member)
        actual = inspect.getattr_static(native, member)
        if member == "offsets":
            assert hasattr(actual, "__get__") and hasattr(actual, "__set__")
        elif callable(expected):
            assert callable(actual)
            try:
                signature = inspect.signature(expected)
            except (TypeError, ValueError):
                continue
            assert inspect.signature(actual) == signature


@pytest.mark.parametrize("name", CLASSES)
@pytest.mark.parametrize("shape", [(13,), (43, 37), (4, 7, 5)])
@pytest.mark.parametrize("fully_connected", [True, False])
def test_mcp_ties_barriers_and_reuse(api, name, shape, fully_connected):
    values = np.random.default_rng(913).integers(0, 10, size=shape).astype(float)
    values.flat[1:4] = [-4, np.inf, 0]
    starts = [tuple(0 for _ in shape), tuple(-1 for _ in shape)]
    goal = tuple(n // 2 for n in shape)
    options = dict(fully_connected=fully_connected, sampling=[0.7 + i for i in range(len(shape))])
    stock = getattr(original, name)(values, **options)
    native = getattr(api, name)(values, **options)
    np.testing.assert_array_equal(np.asarray(stock.offsets), np.asarray(native.offsets))
    for solver in (stock, native):
        with pytest.raises(Exception, match="find_costs"):
            solver.traceback(goal)
    initial_stock = stock.find_costs(starts, [goal])
    initial_native = native.find_costs(starts, [goal])
    for left, right in zip(initial_stock, initial_native):
        assert_arrays(left, right)
    if np.isfinite(initial_stock[0][goal]):
        assert native.traceback(goal) == stock.traceback(goal)
    else:
        with pytest.raises(ValueError, match="no minimum-cost path"):
            native.traceback(goal)
    changed_stock = stock.find_costs(starts[:1], [goal], False, 0.5, max_step_cost=4)
    changed_native = native.find_costs(starts[:1], [goal], False, 0.5, max_step_cost=4)
    for prior, left, right in zip(initial_native, changed_stock, changed_native):
        assert np.shares_memory(prior, right)
        assert_arrays(left, right)
    for solver in (stock, native):
        solver._reset()
        with pytest.raises(Exception, match="find_costs"):
            solver.traceback(goal)


@pytest.mark.parametrize("dtype", [np.longlong, np.intp, np.uintp, ">f8", np.float32,
                                 np.dtype(np.float64, metadata={"unit": "cost"})])
def test_input_dtype_aliases_and_metadata(api, dtype):
    values = np.array([[3, 1, 5, 7], [2, 3, 8, 2], [8, 9, 3, 2]], dtype=dtype)
    for layout in (values, np.asfortranarray(values), values[::-1, ::-1]):
        left = original.MCP(layout).find_costs([(0, 0)])
        right = api.MCP(layout).find_costs([(0, 0)])
        for expected, actual in zip(left, right):
            assert_arrays(expected, actual)
        assert original.route_through_array(layout, (0, 0), (-1, -1)) == api.route_through_array(
            layout, (0, 0), (-1, -1)
        )
        assert_arrays(original.shortest_path(layout)[0], api.shortest_path(layout)[0])


def test_cost_storage_alias(api):
    costs = np.ones((5, 7), order="F")
    left = original.MCP(costs)
    right = api.MCP(costs)
    costs[0, 0] = 20
    for a, b in zip(left.find_costs([(0, 0)]), right.find_costs([(0, 0)])):
        assert_arrays(a, b)


def test_custom_offsets_and_callbacks(api):
    costs = np.arange(30, dtype=float).reshape(5, 6) % 4
    offsets = [(2, 0), (-2, 0), (0, 1), (0, -1)]
    results = []
    for graph in (original, api):
        events = []

        class Flexible(graph.MCP_Flexible):
            def goal_reached(self, index, cumcost):
                events.append(("goal", index, cumcost))
                return 1 if index == 2 else 0

            def travel_cost(self, old_cost, new_cost, offset_length):
                return (old_cost + new_cost) * offset_length

            def examine_neighbor(self, index, new_index, offset_length):
                events.append(("frozen", index, new_index, offset_length))

            def update_node(self, index, new_index, offset_length):
                events.append(("update", index, new_index, offset_length))

        class Connect(graph.MCP_Connect):
            def create_connection(self, id1, id2, tb1, tb2, cost1, cost2):
                events.append(("contact", id1, id2, tb1, tb2, cost1, cost2, type(tb1[0]), type(tb2[0])))

        vals = [cls(costs, offsets=offsets, sampling=(2, 0.5)).find_costs([(0, 0), (4, 5)])
                for cls in (Flexible, Connect)]
        assert {event[0] for event in events} == {"goal", "frozen", "update", "contact"}
        results.append((events, vals))
    assert results[0][0] == results[1][0]
    for left, right in zip(results[0][1], results[1][1]):
        for a, b in zip(left, right):
            assert_arrays(a, b)


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("reach", [0, 1, 2])
@pytest.mark.parametrize("indexlist", [True, False])
def test_shortest_path(api, axis, reach, indexlist):
    values = np.random.default_rng(16).integers(0, 20, size=(11, 7, 8))
    left, left_cost = original.shortest_path(values, reach, axis, indexlist)
    right, right_cost = api.shortest_path(values, reach, axis, indexlist)
    assert type(left) is type(right)
    assert type(left_cost) is type(right_cost)
    if indexlist:
        assert left == right
    else:
        assert_arrays(left, right)
    assert left_cost == right_cost


@pytest.mark.parametrize("dtype", [np.bool_, np.uint8, np.int8, np.float16, np.float64])
@pytest.mark.parametrize("sparse_type", ["array", "matrix"])
@pytest.mark.parametrize("masked", [False, True])
def test_pixel_graph(api, dtype, sparse_type, masked):
    values = np.random.default_rng(26).integers(0, 255, size=(10, 5, 3)).astype(dtype)
    values[::2, ::2, ::2] = 0
    mask = values[::1] != 0 if masked else None
    options = dict(mask=mask, sparse_type=sparse_type, spacing=(0.5, 0.9, 2.3), connectivity=2)
    left, left_nodes = original.pixel_graph(values, **options)
    right, right_nodes = api.pixel_graph(values, **options)
    assert type(left) is type(right)
    assert left.shape == right.shape
    assert_arrays(left_nodes, right_nodes)
    for member in ("data", "indptr", "indices"):
        assert_arrays(getattr(left, member), getattr(right, member))
    assert right.has_sorted_indices and right.has_canonical_format


def test_deprecation_boundaries(api):
    stock = original.MCP(np.ones((5, 7)))
    native = api.MCP(np.ones((5, 7)))
    for cls in (stock, native):
        with pytest.warns(FutureWarning, match="max_cost"):
            result = cls.find_costs([(0, 0)], max_cost=0.5)
        assert np.sum(np.isfinite(result[0])) == 1
        with pytest.warns(FutureWarning, match="max_cost"):
            with pytest.raises(ValueError, match="Both deprecated parameter"):
                cls.find_costs([(0, 0)], max_cost=0.5, max_step_cost=2)


def test_fallback_and_pickle(api):
    from skimagexx import FallbackWarning, set_fallback_enabled

    image = np.arange(9).reshape(3, 3)
    with pytest.raises(NotImplementedError, match="graph.pixel_graph"):
        api.pixel_graph(image, sparse_type="invalid")
    previous = set_fallback_enabled(True)
    try:
        with pytest.warns(FallbackWarning, match="graph.pixel_graph"):
            with pytest.raises(ValueError, match="sparse_type"):
                api.pixel_graph(image, sparse_type="invalid")
        with pytest.warns(FallbackWarning, match="graph.MCP"):
            with pytest.raises(TypeError, match="no default __reduce__"):
                pickle.dumps(api.MCP(image))
    finally:
        set_fallback_enabled(previous)


@pytest.mark.parametrize("spacing", [None, (1.5, 0.5), (0.0, 1.5), np.array([2, 3])])
def test_edge_callback_order(api, spacing):
    values = np.arange(12, dtype=np.intp).reshape(3, 4)
    results = []
    for graph in (original, api):
        observed = []

        def edges(left, right, distance):
            observed.append((left.copy(), right.copy(), distance.copy()))
            return left - right

        csr, nodes = graph.pixel_graph(values, edge_function=edges, connectivity=2, spacing=spacing)
        assert len(observed) == 1
        results.append((*observed[0], csr.indptr, csr.indices, csr.data, nodes))
    for left, right in zip(*results):
        assert_arrays(left, right)


def test_live_cost_mutation_from_hook(api):
    results = []
    for graph in (original, api):
        values = np.ones((4, 4), order="F")

        class Mutating(graph.MCP):
            def goal_reached(self, index, cumcost):
                if index == 0:
                    values[0, 1:] = np.inf
                return 0

        results.append(Mutating(values).find_costs([(0, 0)]))
    for left, right in zip(*results):
        assert_arrays(left, right)


def test_traceback_reads_mutable_offsets(api):
    paths = []
    for graph in (original, api):
        solver = graph.MCP(np.ones((1, 5)), offsets=[(0, 1)])
        solver.find_costs([(0, 0)])
        np.asarray(solver.offsets)[0, 1] = 2
        paths.append(solver.traceback((0, 4)))
    assert paths[0] == paths[1]


def test_hot_swap(api, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("stock graph entry was called")

    for name in (*CLASSES, *CALLABLES):
        monkeypatch.setattr(original, name, unexpected)
    values = np.arange(12.0).reshape(3, 4)
    for name in CLASSES:
        solver = getattr(api, name)(values)
        solver.find_costs([(0, 0)])
        assert solver.traceback((2, 3))[-1] == (2, 3)
    assert api.route_through_array(values, (0, 0), (2, 3))[0][-1] == (2, 3)
    assert len(api.shortest_path(values)[0]) == 4
    assert api.pixel_graph(values)[0].shape == (12, 12)
