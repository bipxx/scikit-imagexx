"""Native training-row and polygon processing with estimator and GUI providers."""

import math

import numpy as np

from . import _core, filters
from ._compat import fallback

_ndarray = np.ndarray

if not hasattr(_core, "segment_gather_training"):
    try:
        import _future_filters_core as _core_family
    except ModuleNotFoundError as error:
        if error.name != "_future_filters_core":
            raise
        _core_family = None
else:
    _core_family = _core


# Keep the optional sklearn import and upstream default feature callback lazy.
def _estimator_dependencies():
    from skimage.future import trainable_segmentation

    return trainable_segmentation


def fit_segmenter(labels, features, clf):
    if _core_family is not None and type(labels) is type(features) is np.ndarray:
        try:
            training = _core_family.segment_gather_training(labels, features)
        except ValueError:  # Storage outside the native owner policy.
            training = None
        if training is not None:
            clf.fit(*training)
            return clf
    return fallback("future", "fit_segmenter", labels, features, clf)


def predict_segmenter(features, clf):
    if type(features) is not _ndarray:
        return fallback("future", "predict_segmenter", features, clf)
    shape = features.shape
    if len(shape) > 2:
        features = features.reshape(-1, shape[-1])
    try:
        predicted = clf.predict(features)
    except Exception as error:
        not_fitted = _estimator_dependencies().NotFittedError
        if isinstance(error, not_fitted):
            raise not_fitted(
                "You must train the classifier `clf` first"
                "for example with the `fit_segmenter` function."
            )
        if (isinstance(error, ValueError) and error.args and
                "x must consist of vectors of length" in error.args[0]):
            raise ValueError(
                error.args[0] + "\n"
                "Maybe you did not use the same type of features for training the classifier."
            )
        raise
    return predicted.reshape(shape[:-1])


class TrainableSegmenter:
    """Estimator with a pixel feature callback and a scikit-learn-style classifier."""

    def __init__(self, clf=None, features_func=None):
        if clf is None:
            dependencies = _estimator_dependencies()
            if not dependencies.has_sklearn:
                raise ImportError(
                    "Please install scikit-learn or pass a classifier instance"
                    "to TrainableSegmenter."
                )
            clf = dependencies.RandomForestClassifier(n_estimators=100, n_jobs=-1)
        self.clf = clf
        self.features_func = features_func

    def compute_features(self, image):
        if self.features_func is None:
            self.features_func = _estimator_dependencies().multiscale_basic_features
        self.features = self.features_func(image)

    def fit(self, image, labels):
        self.compute_features(image)
        fit_segmenter(labels, self.features, self.clf)

    def predict(self, image):
        if self.features_func is None:
            self.features_func = _estimator_dependencies().multiscale_basic_features
        return predict_segmenter(self.features_func(image), self.clf)


def try_all_threshold(image, figsize=(8, 5), verbose=True):
    from skimage._shared.version_requirements import is_installed

    if _core_family is None or type(image) is not np.ndarray or not is_installed("matplotlib", ">=3.3"):
        return fallback("filters", "try_all_threshold", image, figsize=figsize, verbose=verbose)
    try:
        hist = _core_family.threshold_panel_histogram(image)
    except ValueError:
        hist = None
    if hist is None:
        return fallback("filters", "try_all_threshold", image, figsize=figsize, verbose=verbose)

    from matplotlib import pyplot as plt

    counts, centers = hist

    def triangle():
        if centers.size == 1:
            return image.flat[0]
        indices, _ = _core.threshold_indices(counts, centers, "triangle", False, True, float("nan"))
        return centers[indices[0]]

    methods = (
        ("Isodata", lambda: filters.threshold_isodata(hist=hist)),
        ("Li", lambda: filters.threshold_li(image)),
        ("Mean", lambda: filters.threshold_mean(image)),
        ("Minimum", lambda: filters.threshold_minimum(hist=hist)),
        ("Otsu", lambda: filters.threshold_otsu(image)),
        ("Triangle", triangle),
        ("Yen", lambda: filters.threshold_yen(hist=hist)),
    )
    fig, axes = plt.subplots(4, 2, figsize=figsize, sharex=True, sharey=True)
    axes = axes.reshape(-1)
    axes[0].imshow(image, cmap=plt.cm.gray)
    axes[0].set_title("Original")
    for axis, (title, threshold) in zip(axes[1:], methods):
        axis.set_title(title)
        try:
            axis.imshow(image > threshold(), cmap=plt.cm.gray)
        except NotImplementedError:
            raise  # Keep strict fallback policy observable.
        except Exception as error:
            axis.text(0.5, 0.5, type(error).__name__, ha="center",
                      va="center", transform=axis.transAxes)
        if verbose:
            print(f"skimage.filters.thresholding.threshold_{title.lower()}")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    return fig, axes


def _stamp(vertices, output, label):
    columns, rows = zip(*vertices)
    height, width = output.shape
    # Match the floor/ceiling and clipping order of draw.polygon.
    first = int(max(0, min(rows))), int(max(0, min(columns)))
    last = min(height - 1, math.ceil(max(rows))), min(width - 1, math.ceil(max(columns)))
    if not _core_family.segment_stamp_polygon(rows, columns, first, last, output, label):
        rr, cc = fallback("draw", "polygon", rows, columns, shape=output.shape)
        output[rr, cc] = label


def _selection_labels(selections, shape, return_all):
    if return_all:
        def masks():
            for label, vertices in enumerate(selections, start=1):
                plane = np.zeros(shape, dtype=int)
                _stamp(vertices, plane, label)
                yield plane

        # Preserve 0.26's stack protocol, including NumPy's generator error.
        return np.stack(masks())
    if not selections:
        return np.broadcast_to(0, shape)
    output = np.zeros(shape, dtype=int)
    for label, vertices in enumerate(selections, start=1):
        _stamp(vertices, output, label)
    return output


def _manual_segmentation(image, alpha, return_all, lasso):
    from skimage._shared.version_requirements import is_installed

    name = "manual_lasso_segmentation" if lasso else "manual_polygon_segmentation"
    if _core_family is None or type(image) is not np.ndarray or not is_installed("matplotlib", ">=3.3"):
        return fallback("future", name, image, alpha=alpha, return_all=return_all)

    from matplotlib import pyplot as plt, widgets
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon

    if image.ndim not in (2, 3):
        raise ValueError("Only 2D grayscale or RGB images are supported.")
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    ax.imshow(image, cmap="gray")
    ax.set_axis_off()
    selections = []
    artists = []

    def draw(vertices, opacity):
        collection = PatchCollection([Polygon(vertices, closed=True)], match_original=True, alpha=opacity)
        artist = ax.add_collection(collection)
        plt.draw()
        return artist

    def undo(event):
        if selections:
            selections.pop()
            artists.pop().remove()
            fig.canvas.draw_idle()

    undo_ax = fig.add_axes([0.85, 0.05, 0.075, 0.075])
    button = widgets.Button(undo_ax, "\u27f2")
    button.on_clicked(undo)

    if lasso:
        def select(vertices):
            if len(vertices) >= 3:
                selections.append(vertices)
                artists.append(draw(vertices, alpha))
                plt.draw()

        selector = widgets.LassoSelector(ax, select)
    else:
        pending = []
        preview = []

        def click(event):
            if event.inaxes is None or event.inaxes is undo_ax or ax.get_navigate_mode():
                return
            if event.button == 1:
                pending.append([event.xdata, event.ydata])
                if preview:
                    preview.pop().remove()
                preview.append(draw(pending, alpha / 1.4))
            elif event.button == 3 and pending:
                selections.append(pending.copy())
                artists.append(draw(pending, alpha))
                preview.pop().remove()
                pending.clear()
                plt.draw()

        fig.canvas.mpl_connect("button_press_event", click)

    plt.show(block=True)
    return _selection_labels(selections, image.shape[:2], return_all)


def manual_polygon_segmentation(image, alpha=0.4, return_all=False):
    return _manual_segmentation(image, alpha, return_all, lasso=False)


def manual_lasso_segmentation(image, alpha=0.4, return_all=False):
    return _manual_segmentation(image, alpha, return_all, lasso=True)
