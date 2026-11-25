import inspect
import sys
from importlib.util import find_spec
from types import SimpleNamespace

import numpy as np
import pytest

from skimage import filters as original_filters, future as original_future
import skimagexx
from skimagexx import filters, future
from fixtures.future_filters_gui import panel_pyplot, selections


@pytest.fixture(autouse=True)
def strict_native():
    previous = skimagexx.set_fallback_enabled(False)
    try:
        yield
    finally:
        skimagexx.set_fallback_enabled(previous)


@pytest.fixture
def pyplot():
    try:
        spec = find_spec("matplotlib")
    except ModuleNotFoundError:
        # An import-masking finder can raise instead of returning no spec.
        spec = None
    if spec is None:
        yield None
        return

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    try:
        yield plt
    finally:
        plt.close("all")


class Recorder:
    def fit(self, data, labels):
        self.data = data
        self.labels = labels

    def predict(self, data):
        return np.asarray(data[:, 0] > 0.4, dtype=np.uint16)


def check_array(actual, expected):
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype
    assert actual.dtype.metadata == expected.dtype.metadata
    assert actual.flags.writeable == expected.flags.writeable


@pytest.mark.parametrize("name", [
    "fit_segmenter", "predict_segmenter", "manual_polygon_segmentation",
    "manual_lasso_segmentation", "TrainableSegmenter",
])
def test_signatures(name):
    assert inspect.signature(getattr(future, name)) == inspect.signature(getattr(original_future, name))


def test_class_members_and_state():
    actual_type = future.TrainableSegmenter
    expected_type = original_future.TrainableSegmenter
    assert actual_type.__bases__ == expected_type.__bases__
    assert dir(actual_type) == dir(expected_type)
    for name in vars(expected_type):
        a = inspect.getattr_static(actual_type, name)
        b = inspect.getattr_static(expected_type, name)
        assert type(a) is type(b)
        if inspect.isfunction(b):
            assert inspect.signature(a) == inspect.signature(b)
    image = np.linspace(0, 1, 30, dtype=np.float32).reshape(5, 6)
    labels = (image > 0.75).astype(np.int32)

    def features_func(values):
        return np.stack((values, values ** 2), axis=-1)

    actual = actual_type(Recorder(), features_func)
    expected = expected_type(Recorder(), features_func)
    assert vars(actual).keys() == vars(expected).keys()
    assert actual.compute_features(image) is expected.compute_features(image) is None
    check_array(actual.features, expected.features)
    assert actual.fit(image, labels) is expected.fit(image, labels) is None
    check_array(actual.clf.data, expected.clf.data)
    check_array(actual.predict(image), expected.predict(image))
    check_array(actual.features, expected.features)


def test_default_feature_and_classifier_contract():
    from skimage.future import trainable_segmentation as provider

    image = np.arange(20, dtype=np.uint8).reshape(4, 5)
    actual = future.TrainableSegmenter(Recorder())
    expected = original_future.TrainableSegmenter(Recorder())
    assert actual.features_func is expected.features_func is None
    actual.compute_features(image)
    expected.compute_features(image)
    assert actual.features_func is expected.features_func
    check_array(actual.features, expected.features)
    if provider.has_sklearn:
        actual_default = future.TrainableSegmenter().clf
        expected_default = original_future.TrainableSegmenter().clf
        assert type(actual_default) is type(expected_default)
        assert actual_default.get_params() == expected_default.get_params()
    else:
        with pytest.raises(ImportError) as expected_error:
            original_future.TrainableSegmenter()
        with pytest.raises(ImportError) as actual_error:
            future.TrainableSegmenter()
        assert str(actual_error.value) == str(expected_error.value)


@pytest.mark.parametrize("alias", [np.bool_, np.byte, np.ubyte, np.short, np.ushort,
                                  np.intc, np.uintc, np.int_, np.uint, np.longlong, np.ulonglong])
def test_fit_dtype_aliases_and_metadata(alias):
    label_dtype = np.dtype(alias, metadata={"role": "label"})
    feature_dtype = np.dtype(">f4", metadata={"role": "feature"})
    labels = np.array([[0, 1, 0], [3, 0, 7]], dtype=label_dtype)
    features = np.arange(24, dtype=feature_dtype).reshape(2, 3, 4)
    expected, actual = Recorder(), Recorder()
    assert original_future.fit_segmenter(labels, features, expected) is expected
    assert future.fit_segmenter(labels, features, actual) is actual
    check_array(actual.data, expected.data)
    check_array(actual.labels, expected.labels)


def test_negative_labels_empty_and_fallback():
    labels = np.array([[-4, 0, -1], [0, 0, 0]], dtype=np.int64)
    features = np.empty((2, 3, 0), dtype=np.float32)
    expected, actual = Recorder(), Recorder()
    original_future.fit_segmenter(labels, features, expected)
    future.fit_segmenter(labels, features, actual)
    check_array(actual.data, expected.data)
    check_array(actual.labels, expected.labels)
    with pytest.raises(NotImplementedError, match="fit_segmenter"):
        future.fit_segmenter(labels[:, ::-1], features[:, ::-1].copy(order="F"), actual)
    previous = skimagexx.set_fallback_enabled(True)
    try:
        with pytest.warns(skimagexx.FallbackWarning, match="fit_segmenter"):
            future.fit_segmenter(labels[:, ::-1], features[:, ::-1].copy(order="F"), actual)
        check_array(actual.data, expected.data)
    finally:
        skimagexx.set_fallback_enabled(previous)


@pytest.mark.parametrize("features", [
    np.arange(24, dtype=np.float32).reshape(2, 3, 4),
    np.arange(24, dtype=np.float32).reshape(2, 3, 4)[:, ::-1],
    np.arange(24, dtype=np.float32).reshape(6, 4),
])
def test_predict(features):
    check_array(future.predict_segmenter(features, Recorder()),
                original_future.predict_segmenter(features, Recorder()))


@pytest.mark.parametrize("kind", ["width", "not_fitted", "other"])
def test_predict_errors(kind):
    from skimage.future import trainable_segmentation as provider

    errors = {"width": ValueError("x must consist of vectors of length 2"),
              "not_fitted": provider.NotFittedError(), "other": ValueError("other")}

    class Failing:
        def predict(self, features):
            raise errors[kind]

    features = np.empty((8, 2))
    with pytest.raises(type(errors[kind])) as expected:
        original_future.predict_segmenter(features, Failing())
    with pytest.raises(type(errors[kind])) as actual:
        future.predict_segmenter(features, Failing())
    assert actual.value.args == expected.value.args


@pytest.mark.parametrize("image", [
    np.repeat(np.array([[5, 50, 180, 239]], dtype=np.uint8), 32, axis=0),
    np.ones((8, 8), dtype=np.uint8),
    np.random.default_rng(22).integers(0, 256, size=(67, 59), dtype=np.uint8),
])
def test_threshold_panel(image, capsys, monkeypatch, pyplot):
    if pyplot is None:
        from skimage.exposure import histogram
        from skimagexx import _trainable_segmenter

        monkeypatch.setitem(sys.modules, "matplotlib",
                            SimpleNamespace(__version__="3.3", pyplot=panel_pyplot()))
        counts, centers = _trainable_segmenter._core_family.threshold_panel_histogram(image)
        expected_counts, expected_centers = histogram(image.ravel(), source_range="image")
        np.testing.assert_array_equal(counts, expected_counts)
        np.testing.assert_array_equal(centers, expected_centers)

    assert inspect.signature(filters.try_all_threshold) == inspect.signature(original_filters.try_all_threshold)
    expected_fig, expected = original_filters.try_all_threshold(image, figsize=(7, 5))
    expected_stdout = capsys.readouterr().out
    actual_fig, actual = filters.try_all_threshold(image, figsize=(7, 5))
    assert capsys.readouterr().out == expected_stdout
    np.testing.assert_array_equal(actual_fig.get_size_inches(), expected_fig.get_size_inches())
    assert [axis.get_title() for axis in actual] == [
        "Original", "Isodata", "Li", "Mean", "Minimum", "Otsu", "Triangle", "Yen",
    ]
    for a, b in zip(actual, expected, strict=True):
        assert a.get_title() == b.get_title()
        assert a.axison == b.axison
        assert [text.get_text() for text in a.texts] == [text.get_text() for text in b.texts]
        assert len(a.images) == len(b.images)
        for a_image, b_image in zip(a.images, b.images):
            check_array(a_image.get_array(), b_image.get_array())


@pytest.mark.parametrize("mode", ["polygon", "lasso"])
@pytest.mark.parametrize("return_all,undo", [(False, False), (True, False), (False, True)])
def test_selection_overlap(mode, return_all, undo, pyplot):
    vertices = [
        [(1, 1), (16, 1), (16, 10), (1, 10)],
        [(5.25, 3), (18, 4.5), (13.75, 15), (8, 13)],
    ]
    name = f"manual_{mode}_segmentation"
    image = np.zeros((17, 20, 3), dtype=np.uint8)
    if pyplot is None:
        from skimage.draw import polygon
        from skimagexx._trainable_segmenter import _selection_labels

        shape = image.shape[:2]
        selected = vertices[:-1] if undo else vertices
        if return_all:
            generator = (i for i in ())
            try:
                np.stack(generator)
            except TypeError:
                with pytest.raises(TypeError, match="sequence"):
                    _selection_labels(selected, shape, return_all=True)
            except ValueError:
                expected_all = []
                for label, region in enumerate(selected, start=1):
                    mask = np.zeros(shape, dtype=int)
                    columns, rows = zip(*region)
                    mask[polygon(rows, columns, shape=shape)] = label
                    expected_all.append(mask)
                check_array(_selection_labels(selected, shape, return_all=True), np.stack(expected_all))
            return
        expected = np.zeros(shape, dtype=int)
        for label, region in enumerate(selected, start=1):
            columns, rows = zip(*region)
            expected[polygon(rows, columns, shape=shape)] = label
        check_array(_selection_labels(selected, shape, return_all=False), expected)
        return

    with selections(mode, vertices, undo=undo):
        if return_all:
            with pytest.raises(TypeError, match="sequence") as expected_error:
                getattr(original_future, name)(image, alpha=0.3, return_all=True)
            with pytest.raises(TypeError, match="sequence") as actual_error:
                getattr(future, name)(image, alpha=0.3, return_all=True)
            assert str(actual_error.value) == str(expected_error.value)
            return
        expected = getattr(original_future, name)(image, alpha=0.3, return_all=return_all)
        actual = getattr(future, name)(image, alpha=0.3, return_all=return_all)
    check_array(actual, expected)
    assert actual.max() == (1 if undo else 2)


@pytest.mark.parametrize("mode", ["polygon", "lasso"])
def test_selection_without_regions(mode, pyplot):
    name = f"manual_{mode}_segmentation"
    image = np.zeros((6, 8), dtype=np.float32)
    if pyplot is None:
        from skimagexx._trainable_segmenter import _selection_labels

        check_array(_selection_labels([], image.shape, return_all=False),
                    np.broadcast_to(0, image.shape))
        try:
            np.stack(i for i in ())
        except TypeError as exc:
            error_type = TypeError
            match = "sequence"
        except ValueError:
            error_type = ValueError
            match = "need at least one array to stack"
        with pytest.raises(error_type, match=match):
            _selection_labels([], image.shape, return_all=True)
        return

    with selections(mode, []):
        check_array(getattr(future, name)(image), getattr(original_future, name)(image))
        with pytest.raises(TypeError, match="sequence") as expected_error:
            getattr(original_future, name)(image, return_all=True)
        with pytest.raises(TypeError, match="sequence") as actual_error:
            getattr(future, name)(image, return_all=True)
        assert str(actual_error.value) == str(expected_error.value)
    with pytest.raises(ValueError, match="Only 2D grayscale or RGB"):
        getattr(future, name)(np.zeros(8))


@pytest.mark.parametrize("namespace,original,name", [
    (filters, original_filters, "try_all_threshold"),
    (future, original_future, "manual_polygon_segmentation"),
    (future, original_future, "manual_lasso_segmentation"),
])
def test_missing_matplotlib(namespace, original, name, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    image = np.zeros((6, 8), dtype=np.uint8)
    function = getattr(namespace, name)
    with pytest.raises(NotImplementedError, match=name):
        function(image)

    with pytest.raises(ImportError) as expected:
        getattr(original, name)(image)
    previous = skimagexx.set_fallback_enabled(True)
    try:
        with pytest.warns(skimagexx.FallbackWarning, match=name):
            with pytest.raises(ImportError) as actual:
                function(image)
        assert type(actual.value) is type(expected.value)
        assert actual.value.args == expected.value.args
    finally:
        skimagexx.set_fallback_enabled(previous)
