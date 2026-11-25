"""Differential graph contracts for the family wrappers and native binding."""

import inspect

import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse
from skimage import graph as original
from skimage.graph._rag import min_weight

from skimagexx import FallbackWarning, _graph_rag as native, set_fallback_enabled


NAMES = (
    'RAG', 'rag_mean_color', 'rag_boundary', 'central_pixel', 'cut_threshold',
    'cut_normalized', 'merge_hierarchical', 'show_rag',
)


@pytest.fixture(autouse=True)
def strict():
    previous = set_fallback_enabled(False)
    try:
        yield
    finally:
        set_fallback_enabled(previous)


def assert_graph(actual, expected):
    assert list(actual) == list(expected)
    assert list(actual.edges) == list(expected.edges)
    assert [type(node) for node in actual] == [type(node) for node in expected]
    assert actual.max_id == expected.max_id
    for left, right in [(actual.graph, expected.graph), *(
        (actual.nodes[i], expected.nodes[i]) for i in actual
    ), *((actual.edges[i], expected.edges[i]) for i in actual.edges)]:
        assert left.keys() == right.keys()
        for key in left:
            assert type(left[key]) is type(right[key])
            assert_array_equal(left[key], right[key])


def block_image():
    labels = np.arange(12, dtype=np.int16).reshape(3, 4).repeat(3, 0).repeat(4, 1)
    image = np.random.default_rng(12).integers(0, 255, (*labels.shape, 3), dtype=np.uint8)
    return labels, image


@pytest.mark.parametrize('name', NAMES)
def test_signatures(name):
    assert inspect.signature(getattr(native, name)) == inspect.signature(getattr(original, name))


def test_all_static_class_members():
    assert native.RAG.__bases__ == original.RAG.__bases__ == (nx.Graph,)
    assert dir(native.RAG) == dir(original.RAG)
    for name in dir(original.RAG):
        a = inspect.getattr_static(native.RAG, name)
        b = inspect.getattr_static(original.RAG, name)
        assert type(a) is type(b), name
        if isinstance(b, (classmethod, staticmethod, property)):
            a, b = (a.fget, b.fget) if isinstance(b, property) else (a.__func__, b.__func__)
        if inspect.isfunction(b):
            assert inspect.signature(a) == inspect.signature(b), name


@pytest.mark.parametrize('shape', [(14,), (6, 9), (4, 6, 7)])
@pytest.mark.parametrize('connectivity', [1, 2, 9])
def test_construction_order(shape, connectivity):
    labels = np.random.default_rng(17).integers(-3, 8, shape).astype('>i4')[::-1]
    a = native.RAG(labels, connectivity, data=[(12, 13)], note='graph')
    b = original.RAG(labels, connectivity, data=[(12, 13)], note='graph')
    assert_graph(a, b)
    attrs = {'x': 12}
    a.add_node(29, attrs, x=5)
    b.add_node(29, {'x': 12}, x=5)
    assert attrs == {'x': 5}
    a.add_edge(29, 12, {'weight': 3}, color='red')
    b.add_edge(29, 12, {'weight': 3}, color='red')
    a._add_node_silent(50)
    b._add_node_silent(50)
    assert_graph(a.copy(), b.copy())
    assert a.next_id() == b.next_id() == 30
    assert type(a.fresh_copy()) is native.RAG
    assert_graph(a.fresh_copy(), b.fresh_copy())


@pytest.mark.parametrize('in_place', [False, True])
def test_merge_nodes_callback_contract(in_place):
    labels, image = block_image()
    a, b = (mod.rag_mean_color(image, labels) for mod in (native, original))

    def merge(graph):
        observations = []

        def weight(g, src, dst, neighbor, increment, *, tag):
            observations.append((g is graph, src, dst, neighbor, tag))
            return {'weight': min_weight(g, src, dst, neighbor)['weight'] + increment}

        node = graph.merge_nodes(0, 1, weight, in_place, [3], {'tag': 'call'})
        return node, observations

    assert merge(a) == merge(b)
    assert_graph(a, b)


@pytest.mark.parametrize('mode', ['distance', 'similarity'])
@pytest.mark.parametrize('dtype', [np.dtype('>u2'), np.dtype('i2', metadata={'tag': 'labels'})])
def test_rag_mean_color(mode, dtype):
    labels, image = block_image()
    labels = labels.astype(dtype)[:, ::-1]
    image = image[:, ::-1].astype(np.float32) / 11
    assert_graph(native.rag_mean_color(image, labels, mode=mode),
                 original.rag_mean_color(image, labels, mode=mode))


@pytest.mark.parametrize('dtype', ['f4', 'f8', '>f4', np.dtype('f8', metadata={'tag': 'values'})])
def test_rag_boundary(dtype):
    labels, image = block_image()
    values = (image[..., 0] / 32).astype(dtype)
    assert_graph(native.rag_boundary(labels, values), original.rag_boundary(labels, values))


@pytest.mark.parametrize('shape,connectivity', [((41, 31), 1), ((41, 31), 2), ((7, 8, 9), 3)])
@pytest.mark.parametrize('dtype', ['f4', 'f8', '>f4'])
def test_rag_boundary_sparse_duplicate_reduction(dtype, shape, connectivity):
    rng = np.random.default_rng(66)
    labels = rng.integers(0, 7, shape)
    values = rng.random(labels.shape).astype(dtype)
    assert_graph(native.rag_boundary(labels, values, connectivity),
                 original.rag_boundary(labels, values, connectivity))


def test_boundary_limit_matches_reference():
    labels = np.array([[0, 255]], dtype=np.uint8)
    edge_map = np.ones(labels.shape)
    set_fallback_enabled(True)
    with pytest.warns(RuntimeWarning, match='overflow'):
        with pytest.raises(ValueError):
            original.rag_boundary(labels, edge_map)
    with pytest.warns(FallbackWarning):
        with pytest.warns(RuntimeWarning, match='overflow'):
            with pytest.raises(ValueError):
                native.rag_boundary(labels, edge_map)


@pytest.mark.parametrize('constructor', [sparse.csr_array, sparse.csr_matrix])
@pytest.mark.parametrize('weighted', [False, True])
def test_central_pixel(constructor, weighted):
    matrix = np.zeros((15, 15), dtype=np.float32)
    for i in range(14):
        matrix[i, i + 1] = i % 4 + 1 if weighted else 1
    matrix[8, 2] = 3 if weighted else 1
    a, b = constructor(matrix), constructor(matrix)
    a.indices, a.indptr = a.indices.astype(np.int64), a.indptr.astype(np.int64)
    b.indices, b.indptr = b.indices.astype(np.int64), b.indptr.astype(np.int64)
    kwargs = dict(nodes=np.arange(15)[::-1], shape=(3, 5), partition_size=3)
    result, expected = native.central_pixel(a, **kwargs), original.central_pixel(b, **kwargs)
    assert result[0] == expected[0]
    assert_array_equal(result[1], expected[1])
    assert a.indices.dtype == b.indices.dtype == a.indptr.dtype == b.indptr.dtype == np.int32


@pytest.mark.parametrize('in_place', [False, True])
def test_cut_threshold(in_place):
    labels, image = block_image()
    labels = labels.astype(np.dtype('u2', metadata={'tag': 1}))
    a, b = (mod.rag_mean_color(image, labels) for mod in (native, original))
    result = native.cut_threshold(labels, a, 28, in_place)
    expected = original.cut_threshold(labels, b, 28, in_place)
    assert result.dtype.metadata == expected.dtype.metadata
    assert_array_equal(result, expected)
    assert_graph(a, b)


@pytest.mark.parametrize('in_place', [False, True])
@pytest.mark.parametrize('num_cuts', [0, 10, 61])
def test_cut_normalized(in_place, num_cuts):
    labels, image = block_image()
    a, b = (mod.rag_mean_color(image, labels, mode='similarity') for mod in (native, original))
    ra, rb = np.random.default_rng(21), np.random.default_rng(21)
    result = native.cut_normalized(labels, a, 0.45, num_cuts, in_place, rng=ra)
    expected = original.cut_normalized(labels, b, 0.45, num_cuts, in_place, rng=rb)
    assert_array_equal(result, expected)
    assert ra.bit_generator.state == rb.bit_generator.state
    assert_graph(a, b)


def test_spectral_score_uses_reference_reduction_order():
    from skimage.graph._graph_cut import get_min_ncut
    from skimage.graph._ncut import DW_matrices

    rng = np.random.default_rng(132)
    labels = np.arange(144).reshape(12, 12)
    graph = native.rag_mean_color(rng.random((*labels.shape, 3)), labels, mode='similarity')
    graph.add_weighted_edges_from((i, i, 1.0) for i in graph)
    degree, weights = DW_matrices(graph)
    vector = rng.normal(size=len(graph))
    mask, score = native._core.graph_rag_spectral(
        vector, degree.data, weights.indptr.astype(np.int32), weights.indices.astype(np.int32),
        weights.data, 47,
    )
    expected_mask, expected_score = get_min_ncut(vector, degree, weights, 47)
    assert_array_equal(mask, expected_mask)
    assert score == expected_score


@pytest.mark.parametrize('in_place_merge', [False, True])
@pytest.mark.parametrize('rag_copy', [False, True])
def test_merge_hierarchical(in_place_merge, rag_copy):
    labels, image = block_image()
    a, b = (mod.rag_mean_color(image, labels) for mod in (native, original))

    def run(mod, graph):
        calls = []

        def merge(g, src, dst):
            calls.append(('merge', src, dst))
            g.nodes[dst]['pixel count'] += g.nodes[src]['pixel count']

        def weight(g, src, dst, neighbor):
            calls.append(('weight', src, dst, neighbor))
            return min_weight(g, src, dst, neighbor)

        result = mod.merge_hierarchical(labels, graph, 24, rag_copy, in_place_merge, merge, weight)
        return result, calls

    actual, calls = run(native, a)
    expected, reference_calls = run(original, b)
    assert calls and calls == reference_calls
    assert_array_equal(actual, expected)
    assert_graph(a, b)


@pytest.mark.parametrize('img_cmap,border,in_place', [('bone', 'cyan', True), (None, None, False)])
def test_show_rag(img_cmap, border, in_place):
    from matplotlib import pyplot as plt

    labels, image = block_image()
    a, b = (mod.rag_mean_color(image, labels) for mod in (native, original))
    figure, axes = plt.subplots(1, 2)
    try:
        kwargs = dict(border_color=border, img_cmap=img_cmap, edge_width=3.0, in_place=in_place)
        left = native.show_rag(labels, a, image, ax=axes[0], **kwargs)
        right = original.show_rag(labels, b, image, ax=axes[1], **kwargs)
        assert left.axes is axes[0]
        assert_array_equal(left.get_segments(), right.get_segments())
        assert_array_equal(left.get_array(), right.get_array())
        assert_array_equal(left.get_linewidths(), right.get_linewidths())
        assert_array_equal(axes[0].images[0].get_array(), axes[1].images[0].get_array())
        assert_graph(a, b)
    finally:
        plt.close(figure)


def test_fallback_boundary_and_policy():
    labels, image = block_image()
    labels = labels.astype(np.float32)
    with pytest.raises(NotImplementedError, match='fallback is disabled'):
        native.RAG(labels)
    set_fallback_enabled(True)
    with pytest.warns(FallbackWarning, match=r'skimage.graph.RAG'):
        result = native.RAG(labels)
    assert_graph(result, original.RAG(labels))
    with pytest.warns(FallbackWarning, match=r'skimage.graph.rag_mean_color'):
        result = native.rag_mean_color(image, labels.astype(np.int16), connectivity=np.int64(1))
    assert_graph(result, original.rag_mean_color(image, labels.astype(np.int16), connectivity=1))
