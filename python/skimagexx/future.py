"""Native training and segmentation panels with compatibility dispatch."""
from ._compat import module_attribute as _module_attribute
from ._trainable_segmenter import (
    TrainableSegmenter,
    fit_segmenter,
    manual_lasso_segmentation,
    manual_polygon_segmentation,
    predict_segmenter,
)

# Public compatibility submodules are resolved by the policy-aware importer.
__path__ = []

__all__ = [
    'manual_lasso_segmentation', 'manual_polygon_segmentation',
    'fit_segmenter', 'predict_segmenter', 'TrainableSegmenter',
]


def __getattr__(name):
    return _module_attribute('future', name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
