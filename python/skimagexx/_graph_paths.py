"""Native lattice paths and CSR assembly with scikit-image graph semantics."""

from functools import lru_cache
from math import inf
from operator import index
from sys import maxsize

import numpy as np

from . import _core
from ._compat import fallback

try:
    from skimage._shared.utils import DEPRECATED, deprecate_parameter
except ImportError:
    class _Deprecated:
        def __repr__(self):
            return "<DEPRECATED>"

    DEPRECATED = _Deprecated()

    def deprecate_parameter(*args, **kwargs):
        return lambda method: method


_MAX_SIZE = np.iinfo(np.int32).max // 2


def _bounded(array):
    try:
        _core.graph_paths_check_storage(array)
    except ValueError:
        return False
    return True


def _shape_supported(array):
    return 1 <= array.ndim <= 9 and 0 < array.size <= _MAX_SIZE


def _cost_buffer(array):
    flat = array.astype(np.float64, copy=False).ravel("F")
    return flat if flat.flags.aligned else flat.copy()


@lru_cache(maxsize=18)
def _default_offsets(ndim, fully_connected):
    return _core.graph_paths_offsets(ndim, ndim if fully_connected else 1)


@lru_cache(maxsize=32)
def _default_grid(shape, fully_connected):
    moves = _default_offsets(len(shape), fully_connected)
    return _core.GraphPathsGrid(shape, moves, [sum(x * x for x in move) ** 0.5 for move in moves], False)


@lru_cache(maxsize=32)
def _pixel_offsets(ndim, connectivity, ordered):
    moves = _core.graph_paths_offsets(ndim, connectivity)
    if ordered:
        moves.sort(key=lambda move: sum(x * x for x in move))
    return moves, [sum(x * x for x in move) ** 0.5 for move in moves]


def _spaced_pixel_offsets(ndim, connectivity, spacing, ordered):
    # Match the provider's spacing dtype and stable distance order, including
    # zero distances tied with the center of the footprint.
    scale = np.asarray(spacing)
    if (scale.ndim > 1 or (scale.ndim == 1 and scale.size not in (1, ndim))
            or scale.dtype.kind not in "biuf" or scale.dtype.itemsize > 8):
        return None
    moves, _ = _pixel_offsets(ndim, connectivity, False)
    vectors = np.array(sorted([*moves, [0] * ndim]), dtype=np.intp)
    distances = np.sqrt(np.sum((vectors * spacing) ** 2, axis=1))
    keep = np.argsort(distances, kind="stable")[1:]
    if not ordered:
        keep.sort()
    return vectors[keep].tolist(), distances[keep].tolist()


def _flat_indices(indices, shape):
    result = []
    for coordinate in indices:
        if len(coordinate) != len(shape):
            return None
        stride = 1
        flat = 0
        for value, extent in zip(coordinate, shape):
            value = int(value)
            if value < 0:
                value += extent
            if not 0 <= value < extent:
                return None
            flat += stride * value
            stride *= extent
        result.append(flat)
    return result


def _override(instance, base, name):
    method = getattr(instance, name)
    return None if getattr(method, "__func__", None) is getattr(base, name) else method


def _index_argument(value):
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("an integer is required")
    result = int(value)
    if not -maxsize - 1 <= result <= maxsize:
        raise OverflowError("int too big to convert")
    return result


def _real_argument(value):
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"must be real number, not {type(value).__name__}")
    return float(value)


def _state_signature(self, __pyx_state):
    raise TypeError("invalid MCP state")


class MCP:
    """Find minimum-cost paths from one or more starts through a lattice."""

    __slots__ = ("_shape", "_costs", "_offsets", "_initial_offsets", "_grid", "_distance",
                 "_predecessor", "_starts", "_ready", "_delegate", "_constructor",
                 "_pending", "_searching")
    _mode = 0
    _original_name = "MCP"

    def __init__(self, costs, offsets=None, fully_connected=True, sampling=None):
        self._constructor = (costs, offsets, fully_connected, sampling)
        self._delegate = None
        self._ready = False
        self._starts = None
        self._pending = []
        self._searching = False
        array = np.asarray(costs)
        if not np.can_cast(array.dtype, np.float64):
            raise TypeError("cannot cast costs array to " + str(np.float64))
        if sampling is not None:
            if not isinstance(sampling, (list, tuple)):
                raise ValueError(f"Invalid type for sampling: {type(sampling)!r}.")
            sampling = np.array(sampling, dtype=np.float64)
            if sampling.ndim != 1 or sampling.size != array.ndim:
                raise ValueError("Need one sampling element per dimension.")
        if not _shape_supported(array) or not _bounded(array):
            self._delegate = self._upstream()
            return
        self._shape = array.shape
        if offsets is None:
            connected = bool(fully_connected)
            moves = np.array(_default_offsets(array.ndim, connected), dtype=np.int8)
        else:
            moves = np.array(offsets, dtype=np.int8)
        if (moves.ndim != 2 or moves.shape[1] != array.ndim or not 0 < len(moves) <= 32767
                or (sampling is not None and not np.isfinite(sampling).all())
                or (offsets is not None and np.any(moves == -128))):
            self._delegate = self._upstream()
            return
        if self._mode == 1 and np.abs(moves).max() > 1:
            raise ValueError("all offset components must be 0, 1, or -1")
        self._offsets = moves
        if offsets is None and sampling is None:
            self._grid = _default_grid(self._shape, connected)
        else:
            spacing = 1.0 if sampling is None else sampling
            lengths = np.sqrt(np.sum((spacing * moves) ** 2, axis=1))
            self._grid = _core.GraphPathsGrid(self._shape, moves.tolist(), lengths.tolist(), False)
        self._initial_offsets = moves.copy()
        self._costs = _cost_buffer(array)
        self._distance = np.empty(array.size, dtype=np.float64)
        self._predecessor = np.empty(array.size, dtype=np.int16)

    @property
    def offsets(self):
        """Signed offsets whose row numbers identify traceback steps."""
        if self._delegate is not None:
            return self._delegate.offsets
        return self._offsets.view() if self._offsets is not None else None

    @offsets.setter
    def offsets(self, value):
        if self._delegate is not None:
            self._delegate.offsets = value
            return
        if value is None:
            self._offsets = None
            return
        view = memoryview(value)
        if view.ndim != 2 or view.format != "b":
            raise ValueError("expected a two-dimensional signed byte buffer")
        self._offsets = np.asarray(view)

    def _upstream(self):
        original = fallback("graph", self._original_name, *self._constructor)
        callbacks = self._callbacks()
        if callbacks:
            names = ("goal_reached", "travel_cost", "examine_neighbor", "update_node",
                     "create_connection")
            overrides = {name: staticmethod(callback) for name, callback in zip(names, callbacks)
                         if callback is not None}
            original = type(self._original_name, (type(original),), overrides)(*self._constructor)
        return original

    def _callbacks(self):
        goal = _override(self, MCP, "goal_reached")
        travel = examine = update = connect = None
        if isinstance(self, MCP_Flexible):
            travel = _override(self, MCP_Flexible, "travel_cost")
            examine = _override(self, MCP_Flexible, "examine_neighbor")
            update = _override(self, MCP_Flexible, "update_node")
        if isinstance(self, MCP_Connect):
            connect = _override(self, MCP_Connect, "create_connection")
        hooks = (goal, travel, examine, update, connect)
        return hooks if any(hook is not None for hook in hooks) else ()

    def _reset(self):
        if self._delegate is not None:
            return self._delegate._reset()
        self._ready = False
        self._distance.fill(inf)
        self._predecessor.fill(-2)
        if self._starts is None:
            raise TypeError("'NoneType' object is not iterable")
        self._predecessor[self._starts] = -1
        if not self._searching:
            self._pending.extend(dict.fromkeys(self._starts))

    def goal_reached(self, index, cumcost):
        """Return 0 to continue, 1 to skip neighbors, or 2 to stop."""
        _index_argument(index)
        _real_argument(cumcost)
        return 0

    @deprecate_parameter("max_cumulative_cost", start_version="0.26", stop_version="0.28",
                         stacklevel=2)
    @deprecate_parameter("max_cost", new_name="max_step_cost", start_version="0.26",
                         stop_version="0.28", stacklevel=3)
    def find_costs(self, starts, ends=None, find_all_ends=True, max_coverage=1.0,
                   max_cumulative_cost=DEPRECATED, max_cost=DEPRECATED, *, max_step_cost=None):
        """Return cumulative costs and predecessor-offset indices in grid shape."""
        if self._delegate is not None:
            return self._delegate.find_costs(starts, ends, find_all_ends, max_coverage,
                                            max_step_cost=max_step_cost)
        all_ends = bool(find_all_ends)
        start_points = list(starts)
        end_points = None if ends is None else list(ends)
        flat_starts = _flat_indices(start_points, self._shape)
        if flat_starts is None:
            raise ValueError("start points must all be within the costs array")
        if not flat_starts:
            raise ValueError("no valid start points to start frontpropagation")
        flat_ends = [] if end_points is None else _flat_indices(end_points, self._shape)
        if flat_ends is None:
            raise ValueError("end points must all be within the costs array")
        visits = int(max_coverage * self._costs.size)
        if not -maxsize - 1 <= visits <= maxsize:
            raise OverflowError("Python int too large to convert to C ssize_t")
        step = inf if max_step_cost is None else _real_argument(max_step_cost)
        self._starts = flat_starts
        self._ready = False
        if type(self)._reset is not MCP._reset:
            self._searching = True
            try:
                self._reset()
            finally:
                self._searching = False
        same_offsets = np.array_equal(self._offsets, self._initial_offsets)
        if same_offsets:
            pending = self._grid.solve(self._costs, flat_starts, flat_ends, self._mode,
                                       all_ends, ends is not None, visits, step, self._distance,
                                       self._predecessor, self._callbacks(), self._pending)
            if pending is not None:
                self._pending = pending
                self._ready = True
                return (self._distance.reshape(self._shape, order="F"),
                        self._predecessor.reshape(self._shape, order="F"))
        if not same_offsets and (self._offsets is None or self._offsets.shape != self._initial_offsets.shape):
            raise ValueError("invalid traceback offsets")
        original = self._upstream()
        if not same_offsets:
            original.offsets = self._offsets
        output = original.find_costs(start_points, end_points, all_ends, max_coverage,
                                     max_step_cost=max_step_cost)
        self._delegate = original
        return output

    def traceback(self, end):
        """Return the recorded path to a given end as coordinate tuples."""
        if self._delegate is not None:
            return self._delegate.traceback(end)
        if not self._ready:
            raise Exception("find_costs() must be run before traceback()")
        flat = _flat_indices([end], self._shape)
        if flat is None:
            raise ValueError("the specified end point must be within the costs array")
        path = self._grid.trace(self._distance, self._predecessor, flat[0])
        if np.array_equal(self._offsets, self._initial_offsets):
            return path
        if self._offsets is None or self._offsets.shape != self._initial_offsets.shape:
            raise ValueError("invalid traceback offsets")
        traceback = self._predecessor.reshape(self._shape, order="F")
        adjusted = [path[-1]]
        for point in reversed(path[1:]):
            move = self._offsets[traceback[point]]
            adjusted.append(tuple(value - int(delta) for value, delta in zip(adjusted[-1], move)))
        return adjusted[::-1]

    def __reduce_cython__(self):
        return (self._delegate if self._delegate is not None else self._upstream()).__reduce_cython__()

    __reduce__ = __reduce_cython__
    __setstate_cython__ = _state_signature
    __setstate__ = _state_signature


class MCP_Connect(MCP):
    """Grow distance-weighted fronts and report when two seeds meet."""

    __slots__ = ()
    _mode = 2
    _original_name = "MCP_Connect"

    def __init__(self, /, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_connection(self, id1, id2, tb1, tb2, cost1, cost2):
        """Handle a contact between the two source regions."""


class MCP_Flexible(MCP):
    """Allow Python hooks to customize traversal and observe node updates."""

    __slots__ = ()
    _original_name = "MCP_Flexible"

    def travel_cost(self, old_cost, new_cost, offset_length):
        """Return the cost of moving to a neighbor."""
        _real_argument(old_cost)
        result = _real_argument(new_cost)
        _real_argument(offset_length)
        return result

    def examine_neighbor(self, index, new_index, offset_length):
        """Observe two settled neighbors."""
        _index_argument(index)
        _index_argument(new_index)
        _real_argument(offset_length)

    def update_node(self, index, new_index, offset_length):
        """Observe an improved path to a neighbor."""
        _index_argument(index)
        _index_argument(new_index)
        _real_argument(offset_length)


class MCP_Geometric(MCP):
    """Integrate pixel costs along each move using its physical length."""

    __slots__ = ()
    _mode = 1
    _original_name = "MCP_Geometric"


def route_through_array(array, start, end, fully_connected=True, geometric=True):
    """Return a minimum-cost path between two coordinates."""
    start, end = tuple(start), tuple(end)
    if (type(array) is np.ndarray and type(fully_connected) is bool and type(geometric) is bool
            and _shape_supported(array) and np.can_cast(array.dtype, np.float64)
            and all(type(i) is int for i in start) and all(type(i) is int for i in end)
            and _bounded(array)):
        shape = array.shape
        endpoints = _flat_indices((start, end), shape)
        if endpoints is not None:
            result = _core.graph_paths_route(
                _default_grid(shape, fully_connected), _cost_buffer(array), *endpoints, geometric
            )
            if result is not None:
                path, cost = result
                return path, np.float64(cost)
    cls = MCP_Geometric if geometric else MCP
    solver = cls(array, fully_connected=fully_connected)
    costs, _ = solver.find_costs([start], [end])
    return solver.traceback(end), costs[end]


def shortest_path(arr, reach=1, axis=-1, output_indexlist=False):
    """Find a minimum-difference path across a selected axis."""
    if (type(arr) is np.ndarray and _shape_supported(arr) and arr.ndim >= 2
            and np.can_cast(arr.dtype, np.float64) and _bounded(arr)):
        try:
            reach_value, axis_value = index(reach), index(axis)
        except TypeError:
            pass
        else:
            if 0 <= reach_value <= 127 and -arr.ndim <= axis_value < arr.ndim:
                result = _core.graph_paths_shortest(
                    _cost_buffer(arr), arr.shape, reach_value, axis_value % arr.ndim, bool(output_indexlist)
                )
                if result is not None:
                    path, cost = result
                    return path, np.float64(cost)
    return fallback("graph", "shortest_path", arr, reach, axis, output_indexlist)


def pixel_graph(image, *, mask=None, edge_function=None, connectivity=1, spacing=None,
                sparse_type="matrix"):
    """Return a CSR graph of neighboring pixels and its C-order node indices."""
    if (type(image) is np.ndarray and _shape_supported(image) and image.dtype.kind in "biuf"
            and _bounded(image)
            and type(sparse_type) is str and sparse_type in ("matrix", "array")
            and (mask is None or (type(mask) is np.ndarray and mask.dtype == bool
                                 and mask.shape == image.shape and _bounded(mask)))):
        try:
            connected = max(1, min(index(connectivity), image.ndim))
        except (TypeError, ValueError, OverflowError):
            pass
        else:
            ordered = edge_function is not None
            geometry = (_pixel_offsets(image.ndim, connected, ordered) if spacing is None
                        else _spaced_pixel_offsets(image.ndim, connected, spacing, ordered))
            if geometry is not None and image.size * len(geometry[0]) <= _MAX_SIZE:
                binary = image.dtype == bool
                selected = image if mask is None and binary else mask
                selection = None if selected is None else selected.ravel()
                nodes, indptr, neighbors, distances = _core.graph_paths_pixel(
                    image.shape, selection, *geometry
                )
                nodes = nodes.astype(np.intp, copy=False)
                from scipy.sparse import csr_array, csr_matrix

                neighbors = neighbors.astype(np.intp, copy=False)
                shape = (nodes.size, nodes.size)
                if ordered:
                    rows = np.repeat(np.arange(nodes.size, dtype=np.intp), np.diff(indptr))
                    values = image.ravel()[nodes]
                    data = edge_function(values[rows], values[neighbors], distances)
                    graph = csr_array((data, (rows, neighbors)), shape=shape)
                else:
                    data = distances
                    if not binary:
                        values = image.ravel()[nodes]
                        sources = np.repeat(values, np.diff(indptr))
                        data = np.abs(sources - values[neighbors]) * distances
                    graph = csr_array((data, neighbors, indptr.astype(np.intp, copy=False)),
                                      shape=shape)
                return (csr_matrix(graph) if sparse_type == "matrix" else graph), nodes
    return fallback("graph", "pixel_graph", image, mask=mask, edge_function=edge_function,
                    connectivity=connectivity, spacing=spacing, sparse_type=sparse_type)
