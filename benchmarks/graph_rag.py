"""One-thread paired medians for fixed graph RAG public-call cases."""

import argparse
import os
import statistics
import time


def cases(np, sparse):
    labels = np.arange(256, dtype=np.int32).reshape(16, 16).repeat(32, 0).repeat(32, 1)
    rng = np.random.default_rng(140)
    image = rng.integers(0, 256, (*labels.shape, 3), dtype=np.uint8)
    edges = image[..., 0].astype(np.float32) / 256
    yield 'RAG/tiles', lambda api: lambda: api.RAG(labels, connectivity=2), 5
    yield 'rag_mean_color/tiles', lambda api: lambda: api.rag_mean_color(image, labels), 5
    yield 'rag_boundary/tiles', lambda api: lambda: api.rag_boundary(labels, edges), 9

    def center(matrix):
        def setup(api):
            graph = matrix.copy()
            return lambda: api.central_pixel(graph)
        return setup

    path = sparse.diags_array([np.ones(1199)], offsets=[1], shape=(1200, 1200), format='csr')
    yield 'central_pixel/path', center(path), 9
    line = sparse.diags_array([np.ones(24)], offsets=[1], shape=(25, 25), format='csr')
    grid = sparse.kronsum(line, line, format='csr')
    yield 'central_pixel/grid', center(grid), 9

    graph_labels = np.arange(4096, dtype=np.int32).reshape(64, 64)

    def region_grid(api):
        graph = api.RAG()
        graph.add_nodes_from((i, {'labels': [i]}) for i in range(graph_labels.size))
        graph.add_weighted_edges_from(
            (i, i + step, 0.25 if i % 7 else 1.0)
            for i in range(graph_labels.size) for step in (1, 64)
            if i + step < graph_labels.size and (step == 64 or i % 64 != 63)
        )
        return graph

    def threshold(api):
        graph = region_grid(api)
        return lambda: api.cut_threshold(graph_labels, graph, 0.5)

    yield 'cut_threshold/grid', threshold, 15

    def merge(api):
        from skimage.graph._rag import min_weight

        def merged(graph, src, dst):
            graph.nodes[dst]['merged'] = True

        graph = region_grid(api)
        return lambda: api.merge_hierarchical(graph_labels, graph, 0.5, False, True, merged, min_weight)

    yield 'merge_hierarchical/grid', merge, 9

    spectral_labels = np.arange(64, dtype=np.int32).reshape(8, 8)
    spectral_image = rng.random((*spectral_labels.shape, 3))

    def normalized(api):
        graph = api.rag_mean_color(spectral_image, spectral_labels, mode='similarity')
        return lambda: api.cut_normalized(spectral_labels, graph, thresh=0.2, num_cuts=48, rng=7)

    yield 'cut_normalized/grid', normalized, 11

    def draw(api):
        from matplotlib import pyplot as plt

        graph = api.rag_mean_color(image, labels)
        figure, axes = plt.subplots()

        def call():
            return api.show_rag(labels, graph, image, img_cmap=None, ax=axes)

        def close():
            plt.close(figure)

        return call, close

    yield 'show_rag/tiles', draw, 7


def measure(setup, api):
    prepared = setup(api)
    call, close = prepared if isinstance(prepared, tuple) else (prepared, None)
    try:
        start = time.perf_counter_ns()
        value = call()
        elapsed = time.perf_counter_ns() - start
        del value
        return elapsed
    finally:
        if close is not None:
            close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', help='Select full case names for targeted checks')
    args = parser.parse_args()
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[variable] = '1'
    os.environ.setdefault('MPLBACKEND', 'Agg')

    import numpy as np
    from scipy import sparse
    import skimage
    from skimage import graph as original
    from skimagexx import _graph_rag as native, set_fallback_enabled, set_num_threads

    if skimage.__version__ != '0.26.0':
        raise RuntimeError(f'Expected scikit-image 0.26.0, found {skimage.__version__}')
    set_num_threads(1)
    set_fallback_enabled(False)
    failed = []
    seen = set()
    for name, setup, repeats in cases(np, sparse):
        if args.case and name not in args.case:
            continue
        seen.add(name)
        samples = {original: [], native: []}
        for api in samples:
            measure(setup, api)
        for trial in range(repeats):
            for api in ((original, native) if trial % 2 == 0 else (native, original)):
                samples[api].append(measure(setup, api))
        upstream, candidate = (statistics.median(samples[api]) / 1e6 for api in (original, native))
        ratio = upstream / candidate
        print(f'{name:<27} {upstream:10.4f} ms -> {candidate:10.4f} ms ({ratio:.3f}x)', flush=True)
        if ratio < 1.0:
            failed.append(name)
    if args.case and set(args.case) != seen:
        parser.error(f"Unknown cases {sorted(set(args.case) - seen)}")
    return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
