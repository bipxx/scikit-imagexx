"""Native shortest-path and region adjacency graph operations."""
from ._compat import module_attribute
from ._graph_paths import (
    MCP, MCP_Connect, MCP_Flexible, MCP_Geometric,
    pixel_graph, route_through_array, shortest_path,
)

_RAG_EXPORTS = (
    'RAG', 'central_pixel', 'cut_normalized', 'cut_threshold',
    'merge_hierarchical', 'rag_boundary', 'rag_mean_color', 'show_rag',
)

# Public compatibility submodules are resolved by the policy-aware importer.
__path__ = []

__all__ = [
    'MCP', 'MCP_Connect', 'MCP_Flexible', 'MCP_Geometric', 'RAG',
    'central_pixel', 'cut_normalized', 'cut_threshold', 'merge_hierarchical',
    'pixel_graph', 'rag_boundary', 'rag_mean_color', 'route_through_array',
    'shortest_path', 'show_rag',
]


def __getattr__(name):
    if name in _RAG_EXPORTS:
        from . import _graph_rag
        value = getattr(_graph_rag, name)
        globals()[name] = value
        return value
    return module_attribute('graph', name)
