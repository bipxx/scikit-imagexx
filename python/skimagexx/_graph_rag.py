"""Native region graph kernels with NetworkX, SciPy and Matplotlib adapters."""

import heapq
import math
from functools import wraps
from threading import Lock
from types import FunctionType

import numpy as np

from . import _compat, _core


_rag_lock = Lock()


def _fallback(name, *args, **kwargs):
    return _compat.fallback('graph', name, *args, **kwargs)


def _integer_labels(labels, max_ndim=3):
    return (
        type(labels) is np.ndarray and 0 < labels.ndim <= max_ndim
        and 0 < labels.size <= 2**30 and labels.dtype.kind in 'iu'
    )


def _buffer(array, dtype):
    return np.require(array, dtype=dtype, requirements=('C', 'A'))


def _adjacency_labels(labels, connectivity):
    if not _integer_labels(labels) or type(connectivity) not in (int, bool):
        return None
    _core.graph_rag_check(labels)
    if labels.dtype.itemsize == 8 and (
        int(labels.min()) < -(2**53) or int(labels.max()) > 2**53
    ):
        return None
    return _buffer(labels, np.int64)


def _label_slots(labels):
    """Bound dense tables and preserve NumPy's original max-plus-one behavior."""
    _core.graph_rag_check(labels)
    low, high = int(labels.min()), int(labels.max())
    overflow = high == np.iinfo(labels.dtype).max
    if low < 0 or overflow or high >= max(65536, min(2**20, 4 * labels.size)):
        return None
    return high + 1


def _rag_type():
    rag = globals().get('RAG')
    if rag is not None:
        return rag
    import networkx as nx
    from skimage.graph._rag import min_weight

    with _rag_lock:
        rag = globals().get('RAG')
        if rag is not None:
            return rag

        class RAG(nx.Graph):
            """Region adjacency graph with an optional labeled-image initializer."""

            def __init__(self, label_image=None, connectivity=1, data=None, **attr):
                labels = None if label_image is None else _adjacency_labels(label_image, connectivity)
                if label_image is not None and labels is None:
                    graph = _fallback('RAG', label_image, connectivity=connectivity, data=data, **attr)
                    self.__dict__.update(graph.__dict__)
                    return
                super().__init__(data, **attr)
                self.max_id = max(self) if self.number_of_nodes() else 0
                if labels is None:
                    return
                edges = _core.graph_rag_adjacency(labels, max(1, min(connectivity, labels.ndim)))
                for first, second in edges:
                    first, second = np.intp(first), np.intp(second)
                    if not self.has_edge(first, second):
                        self.add_edge(first, second)

            def merge_nodes(
                self, src, dst, weight_func=min_weight, in_place=True,
                extra_arguments=None, extra_keywords=None,
            ):
                if extra_arguments is None:
                    extra_arguments = []
                if extra_keywords is None:
                    extra_keywords = {}
                neighbors = (set(self.neighbors(src)) | set(self.neighbors(dst))) - {src, dst}
                new = dst
                if not in_place:
                    new = self.next_id()
                    self.add_node(new)
                for neighbor in neighbors:
                    attrs = weight_func(self, src, dst, neighbor, *extra_arguments, **extra_keywords)
                    self.add_edge(neighbor, new, attr_dict=attrs)
                self.nodes[new]['labels'] = self.nodes[src]['labels'] + self.nodes[dst]['labels']
                self.remove_node(src)
                if not in_place:
                    self.remove_node(dst)
                return new

            def add_node(self, n, attr_dict=None, **attr):
                if attr_dict is not None:
                    attr_dict.update(attr)
                    attr = attr_dict
                super().add_node(n, **attr)
                self.max_id = max(n, self.max_id)

            def add_edge(self, u, v, attr_dict=None, **attr):
                if attr_dict is not None:
                    attr_dict.update(attr)
                    attr = attr_dict
                super().add_edge(u, v, **attr)
                self.max_id = max(u, v, self.max_id)

            def copy(self):
                graph = super().copy()
                graph.max_id = self.max_id
                return graph

            def fresh_copy(self):
                return RAG()

            def next_id(self):
                return self.max_id + 1

            def _add_node_silent(self, n):
                super().add_node(n)

        RAG.__qualname__ = 'RAG'
        for member in vars(RAG).values():
            if isinstance(member, FunctionType):
                member.__qualname__ = f'RAG.{member.__name__}'
        globals()['RAG'] = RAG
        return RAG


def __getattr__(name):
    if name == 'RAG':
        return _rag_type()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _is_rag(rag):
    cls = type(rag)
    if cls is globals().get('RAG'):
        return True
    from skimage.graph._rag import RAG as original_rag

    return cls is original_rag


def _require_matplotlib(function):
    @wraps(function)
    def checked(*args, **kwargs):
        from skimage._shared.version_requirements import is_installed

        if not is_installed('matplotlib', '>=3.3'):
            raise ImportError(f'"{function}" in "{__name__}" requires "matplotlib >=3.3"')
        return function(*args, **kwargs)

    return checked


def rag_mean_color(image, labels, connectivity=2, mode='distance', sigma=255.0):
    """Build a region graph weighted by distance or similarity of mean RGB."""
    prepared = _adjacency_labels(labels, connectivity)
    if (
        prepared is None or type(image) is not np.ndarray
        or image.shape != labels.shape + (3,)
        or image.dtype.kind not in 'biuf' or image.itemsize > 8
        or type(mode) is not str or type(sigma) not in (int, float)
    ):
        return _fallback('rag_mean_color', image, labels, connectivity, mode, sigma)
    slots = _label_slots(labels)
    if slots is None:
        return _fallback('rag_mean_color', image, labels, connectivity, mode, sigma)
    _core.graph_rag_check(image)
    graph = _rag_type()(prepared, connectivity)
    rgb = _buffer(image, np.float64)
    # IEEE exceptional cases retain NumPy's warning/error policy via fallback.
    if image.dtype.kind == 'f' and np.max(np.abs(rgb)) > 1e280 / labels.size:
        return _fallback('rag_mean_color', image, labels, connectivity, mode, sigma)
    if image.dtype.kind == 'f' and not np.isfinite(rgb).all():
        return _fallback('rag_mean_color', image, labels, connectivity, mode, sigma)
    totals = _core.graph_rag_colors(prepared, rgb, slots)
    if len(totals) != len(graph):
        return _fallback('rag_mean_color', image, labels, connectivity, mode, sigma)
    for n, attrs in graph.nodes(data=True):
        count, total = totals[n]
        attrs.update(labels=[n], **{'pixel count': count, 'total color': total,
                                    'mean color': total / count})
    for first, second, attrs in graph.edges(data=True):
        distance = np.linalg.norm(graph.nodes[first]['mean color'] - graph.nodes[second]['mean color'])
        if mode == 'similarity':
            attrs['weight'] = math.e ** (-(distance**2) / sigma)
        elif mode == 'distance':
            attrs['weight'] = distance
        else:
            raise ValueError(f"The mode '{mode}' is not recognised")
    return graph


def rag_boundary(labels, edge_map, connectivity=2):
    """Build a region graph weighted by average edge map values on boundaries."""
    if (
        not _integer_labels(labels) or type(connectivity) not in (int, bool)
        or type(edge_map) is not np.ndarray or edge_map.shape != labels.shape
        or edge_map.dtype.kind != 'f' or edge_map.itemsize not in (4, 8)
    ):
        return _fallback('rag_boundary', labels, edge_map, connectivity)
    slots = _label_slots(labels)
    if slots is None:
        return _fallback('rag_boundary', labels, edge_map, connectivity)
    _core.graph_rag_check(edge_map)
    values = _buffer(edge_map, np.float64)
    scalar = edge_map.dtype.type
    prepared = _buffer(labels, np.int64)
    conn = max(1, min(connectivity, labels.ndim))
    index_dtype = np.int64 if labels.itemsize == 8 else np.int32
    # Native summation can use any duplicate order when every partial sum is
    # exactly representable. Otherwise retain the sparse provider's COO order.
    edges = _core.graph_rag_boundary(prepared, values, conn, edge_map.itemsize == 4)
    if edges:
        weighted = (
            (first, index_dtype(second), {'weight': scalar(total / count), 'count': np.intp(count)})
            for first, second, count, total in edges
        )
    elif edges is None:
        low, high, pixels = _core.graph_rag_boundary_samples(prepared, conn)
        if not pixels.size:
            return _fallback('rag_boundary', labels, edge_map, connectivity)
        from scipy import sparse

        coordinates = (low.astype(index_dtype, copy=False), high.astype(index_dtype, copy=False))
        data = edge_map.ravel()[pixels].astype(scalar, copy=False)
        weights = sparse.csr_array((data, coordinates))
        extent = int(high.max()) + 1
        counts = sparse.csr_array((np.ones(pixels.size, dtype=int), coordinates), shape=(extent, extent))
        weights.data /= counts.data
        weighted = (
            (row, weights.indices[i], {'weight': weights.data[i], 'count': counts.data[i]})
            for row in range(weights.shape[0]) for i in range(weights.indptr[row], weights.indptr[row + 1])
        )
    else:
        return _fallback('rag_boundary', labels, edge_map, connectivity)
    graph = _rag_type()()
    graph.add_edges_from(weighted)
    for node, attrs in graph.nodes(data=True):
        attrs['labels'] = [node]
    return graph


def central_pixel(graph, nodes=None, shape=None, partition_size=100):
    """Find the pixel with the smallest nonzero total shortest-path length."""
    from scipy import sparse

    if (
        type(graph) not in (sparse.csr_array, sparse.csr_matrix)
        or graph.shape[0] != graph.shape[1] or not 2 <= graph.shape[0] <= 10000
        or graph.data.dtype.kind not in 'biuf'
        or graph.data.itemsize > 8
        or not (partition_size is None or type(partition_size) is int and partition_size > 0)
        or graph.indices.dtype.kind != 'i' or graph.indptr.dtype.kind != 'i'
    ):
        return _fallback('central_pixel', graph, nodes, shape, partition_size)
    for array in (graph.indices, graph.indptr, graph.data):
        _core.graph_rag_check(array)
    if (
        graph.indices.size and (graph.indices.min() < 0 or graph.indices.max() >= graph.shape[0])
        or not 0 <= graph.indptr[-1] <= min(2**31 - 1, graph.data.size)
    ):
        return _fallback('central_pixel', graph, nodes, shape, partition_size)
    # Preserve the reference's in-place sparse index downcast.
    graph.indices = graph.indices.astype(np.int32, copy=False)
    graph.indptr = graph.indptr.astype(np.int32, copy=False)
    totals = _core.graph_rag_central(
        _buffer(graph.indptr, np.int32), _buffer(graph.indices, np.int32),
        _buffer(graph.data, np.float64),
    )
    if not totals.size:
        return _fallback('central_pixel', graph, nodes, shape, partition_size)
    index = totals.argmin()
    center = index if nodes is None else nodes[index]
    return (center if shape is None else np.unravel_index(center, shape)), totals


def cut_threshold(labels, rag, thresh, in_place=True):
    """Remove edges at or above the threshold and identify components."""
    if (
        not _integer_labels(labels, max_ndim=32) or not _is_rag(rag)
        or type(thresh) not in (int, float, np.float32, np.float64, np.int32, np.int64)
    ):
        return _fallback('cut_threshold', labels, rag, thresh, in_place)
    slots = _label_slots(labels)
    if slots is None:
        return _fallback('cut_threshold', labels, rag, thresh, in_place)
    working = rag if in_place else rag.copy()
    if type(working._adj) is not dict:
        return _fallback('cut_threshold', labels, rag, thresh, in_place)
    result = _core.graph_rag_threshold(
        working._adj, thresh, (np.intp, np.int32, np.uint32, np.uint64, np.float32, np.float64),
    )
    if result is None:
        return _fallback('cut_threshold', labels, rag, thresh, in_place)
    removed, assignments = result
    working.remove_edges_from(removed)
    mapping = np.arange(slots, dtype=labels.dtype)
    # Assign in component order, including graphs with overlapping label lists.
    for node, component in assignments:
        for label in working.nodes[node]['labels']:
            mapping[label] = component
    return mapping[labels]


def _normalized_relabel(rag, thresh, num_cuts, rng):
    import networkx as nx
    from scipy import sparse
    from scipy.sparse import linalg
    from skimage._shared.compat import SCIPY_GE_1_17_0_DEV0
    from skimage.graph._ncut_cy import argmin2

    weights = nx.to_scipy_sparse_array(rag, format='csc')
    entries = weights.sum(axis=0)
    degree = sparse.dia_array((entries, 0), shape=weights.shape).tocsc()
    size = weights.shape[0]
    if size > 2 and (degree != weights).nnz:
        scaling = degree.copy()
        scaling.data = np.reciprocal(np.sqrt(scaling.data, out=scaling.data), out=scaling.data)
        laplacian = scaling @ (degree - weights) @ scaling
        start = rng.random(size)
        kwargs = {'rng': rng} if SCIPY_GE_1_17_0_DEV0 else {}
        values, vectors = linalg.eigsh(laplacian, which='SM', v0=start, k=min(100, size - 2), **kwargs)
        vector = np.real(vectors)[:, argmin2(np.real(values))]
        mask, cost = _core.graph_rag_spectral(
            _buffer(vector, np.float64), _buffer(degree.data, np.float64),
            _buffer(weights.indptr, np.int32), _buffer(weights.indices, np.int32),
            _buffer(weights.data, np.float64), num_cuts,
        )
        if cost < thresh:
            left, right = [], []
            for node, selected in zip(rag, mask):
                (left if selected else right).append(node)
            _normalized_relabel(rag.subgraph(left), thresh, num_cuts, rng)
            _normalized_relabel(rag.subgraph(right), thresh, num_cuts, rng)
            return
    label = rag.nodes[min(rag)]['labels'][0]
    for _, attrs in rag.nodes(data=True):
        attrs['ncut label'] = label


def cut_normalized(labels, rag, thresh=0.001, num_cuts=10, in_place=True, max_edge=1.0, *, rng=None):
    """Recursively label regions using cuts of the normalized Laplacian."""
    if (
        not _integer_labels(labels, max_ndim=32) or not _is_rag(rag) or not rag
        or type(num_cuts) is not int or not 0 <= num_cuts <= 10000
        or type(max_edge) not in (int, float) or not 0 < max_edge <= 1e100
    ):
        return _fallback('cut_normalized', labels, rag, thresh, num_cuts, in_place, max_edge, rng=rng)
    slots = _label_slots(labels)
    if slots is None or any(
        type(attrs.get('weight', 1.0)) not in (float, int, np.float64, np.float32, np.int32, np.int64)
        or not 0 <= attrs.get('weight', 1.0) <= 1e100
        for _, _, attrs in rag.edges(data=True)
    ) or rag.number_of_edges() + len(rag) >= 2**30:
        return _fallback('cut_normalized', labels, rag, thresh, num_cuts, in_place, max_edge, rng=rng)
    generator = np.random.default_rng(rng)
    if not in_place:
        rag = rag.copy()
    for node in rag:
        rag.add_edge(node, node, weight=max_edge)
    _normalized_relabel(rag, thresh, num_cuts, generator)
    mapping = np.zeros(slots, dtype=labels.dtype)
    for _, attrs in rag.nodes(data=True):
        mapping[attrs['labels']] = attrs['ncut label']
    return mapping[labels]


def merge_hierarchical(labels, rag, thresh, rag_copy, in_place_merge, merge_func, weight_func):
    """Merge neighboring regions in weight order using caller-supplied callbacks."""
    if not _integer_labels(labels, max_ndim=32) or not _is_rag(rag):
        return _fallback('merge_hierarchical', labels, rag, thresh, rag_copy, in_place_merge,
                         merge_func, weight_func)
    slots = _label_slots(labels)
    if slots is None:
        return _fallback('merge_hierarchical', labels, rag, thresh, rag_copy, in_place_merge,
                         merge_func, weight_func)
    if rag_copy:
        rag = rag.copy()
    heap = []
    push, pop = heapq.heappush, heapq.heappop
    for first, second, attrs in rag.edges(data=True):
        item = [attrs['weight'], first, second, True]
        push(heap, item)
        attrs['heap item'] = item
    while heap and heap[0][0] < thresh:
        _, src, dst, valid = pop(heap)
        if not valid:
            continue
        for node in (src, dst):
            for attrs in rag.adj[node].values():
                attrs['heap item'][3] = False
        if not in_place_merge:
            new = rag.next_id()
            rag._add_node_silent(new)
            rag.nodes[new].update(rag.nodes[dst])
            for neighbor in rag.neighbors(dst):
                rag.add_edge(neighbor, new, {'weight': rag[dst][neighbor]['weight']})
            rag.remove_node(dst)
            dst = new
        merge_func(rag, src, dst)
        new = rag.merge_nodes(src, dst, weight_func)
        for neighbor, attrs in rag.adj[new].items():
            if 'heap item' in attrs:
                attrs['heap item'][3] = False
            item = [attrs['weight'], new, neighbor, True]
            attrs['heap item'] = item
            push(heap, item)
    mapping = np.arange(slots)
    for index, (_, attrs) in enumerate(rag.nodes(data=True)):
        for label in attrs['labels']:
            mapping[label] = index
    return mapping[labels]


@_require_matplotlib
def show_rag(
    labels, rag, image, border_color='black', edge_width=1.5, edge_cmap='magma',
    img_cmap='bone', in_place=True, ax=None,
):
    """Draw region centroids and weighted edges over an image."""
    if (
        not _integer_labels(labels, max_ndim=2) or labels.ndim != 2
        or not _is_rag(rag) or type(image) is not np.ndarray
        or image.ndim != 3 or image.shape[:2] != labels.shape
        or image.shape[2] not in ((3, 4) if img_cmap is None else (3,))
        or image.dtype.kind not in 'biuf' or image.itemsize > 8
    ):
        return _fallback('show_rag', labels, rag, image, border_color, edge_width, edge_cmap,
                         img_cmap, in_place, ax)
    slots = _label_slots(labels)
    if slots is None or len(rag) >= 2**31 - 1 or labels.size * max(labels.shape) > 2**53:
        return _fallback('show_rag', labels, rag, image, border_color, edge_width, edge_cmap,
                         img_cmap, in_place, ax)
    from matplotlib import colors, pyplot as plt
    from matplotlib.collections import LineCollection
    from skimage import color, segmentation

    _core.graph_rag_check(image)
    if not in_place:
        rag = rag.copy()
    if ax is None:
        _, ax = plt.subplots()
    if img_cmap is None:
        out = image[:, :, :3]
    else:
        out = plt.get_cmap(img_cmap)(color.rgb2gray(image))[:, :, :3]
    edge_cmap = plt.get_cmap(edge_cmap)
    mapping = np.arange(slots, dtype=np.int64)
    for index, (_, attrs) in enumerate(rag.nodes(data=True), 1):
        for label in attrs['labels']:
            mapping[label] = index
    regions = mapping[labels]
    centers = _core.graph_rag_centroids(regions, max(slots, len(rag) + 1))
    for (_, attrs), center in zip(rag.nodes(data=True), centers):
        attrs['centroid'] = center
    if border_color is not None:
        out = segmentation.mark_boundaries(out, regions, color=colors.to_rgb(border_color))
    ax.imshow(out)
    lines, weights = [], []
    for first, second, attrs in rag.edges(data=True):
        lines.append([rag.nodes[first]['centroid'][::-1], rag.nodes[second]['centroid'][::-1]])
        weights.append(attrs['weight'])
    collection = LineCollection(lines, linewidths=edge_width, cmap=edge_cmap)
    collection.set_array(np.array(weights))
    ax.add_collection(collection)
    return collection
