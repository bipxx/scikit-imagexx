"""Public Cascade state, input conversion, detections and model validation."""

import importlib
import importlib.util
import inspect
import io
import os
from pathlib import Path
import sys
import types
from xml.etree.ElementTree import ParseError

import numpy as np
import pytest
from skimage import data
from skimage.feature import Cascade as Original


@pytest.fixture(scope="module")
def cascade_class():
    extension = os.environ.get("SKIMAGEXX_CASCADE_PRIVATE")
    if extension:
        name = "_cascade_private"
        package = types.ModuleType(name)
        package.__path__ = [str(Path(__file__).resolve().parents[1] / "python/skimagexx")]
        sys.modules[name] = package
        spec = importlib.util.spec_from_file_location(name + ".cascade_private", extension)
        core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core)
        package._core = core
        return importlib.import_module(name + "._cascade").Cascade

    import skimagexx
    from skimagexx._cascade import Cascade

    assert skimagexx.feature.Cascade is Cascade
    assert skimagexx.feature.__all__.count("Cascade") == 1
    return Cascade


@pytest.fixture(scope="module")
def model_path():
    return Path(data.lbp_frontal_face_cascade_filename())


@pytest.mark.parametrize("source", ["str", "path", "text", "binary", "utf16"])
def test_constructor_metadata_and_mutable_fields(cascade_class, model_path, source):
    def make():
        if source == "str":
            return str(model_path)
        if source == "path":
            return model_path
        if source == "text":
            return io.StringIO(model_path.read_text(encoding="utf-8"))
        if source == "utf16":
            return io.BytesIO(model_path.read_text(encoding="utf-8").encode("utf-16"))
        return io.BytesIO(model_path.read_bytes())

    actual = cascade_class(make(), eps=0.125)
    expected = Original(make(), eps=0.125)
    for attr in ("eps", "window_width", "window_height", "stages_number", "stumps_number", "features_number"):
        assert getattr(actual, attr) == getattr(expected, attr)
        setattr(actual, attr, 3)
        setattr(expected, attr, 3)
        assert getattr(actual, attr) == getattr(expected, attr)
    actual._load_xml(make())
    expected._load_xml(make())
    assert actual.stages_number == expected.stages_number == 20
    assert actual.eps == expected.eps
    assert inspect.signature(actual.detect_multi_scale) == inspect.signature(expected.detect_multi_scale)


@pytest.mark.parametrize("layout", ["rgb", "gray", "strided-f32", "fortran", "rgb-f32", "boolean"])
def test_multiscale_detect_and_helpers(cascade_class, model_path, layout):
    face = data.astronaut()
    variants = {
        "rgb": lambda: face,
        "gray": lambda: face[:, :, 1].copy(),
        "strided-f32": lambda: face[:, ::-2, 0].astype(np.float32)[::-1],
        "fortran": lambda: np.asfortranarray(face[::2, ::2]),
        "rgb-f32": lambda: face[::2, ::2].astype(np.float32) / np.float32(255),
        "boolean": lambda: face[::2, ::2, 0] > 125,
    }
    image = variants[layout]()
    actual = cascade_class(model_path)
    expected = Original(model_path)
    args = (1.2, 1.0, (30, 32), (180, 184), 2, 0.5)
    integral = actual._get_contiguous_integral_image(image)
    np.testing.assert_array_equal(integral, expected._get_contiguous_integral_image(image))
    assert integral.dtype == np.float32 and integral.flags.c_contiguous
    np.testing.assert_array_equal(
        actual._get_valid_scale_factors(args[2], args[3], args[0]),
        expected._get_valid_scale_factors(args[2], args[3], args[0]),
    )
    reference = expected.detect_multi_scale(image, *args)
    if layout == "rgb":
        assert reference
    assert actual.detect_multi_scale(image, *args) == reference


@pytest.mark.parametrize("options", [
    (1.15, 1.5, (24, 24), (150, 150), 1, 0.25),
    (1.2, 1.0, (1000, 1000), (1200, 1200), 4, 0.5),
    (1.25, 1.2, (24, 24), (24, 24), 4, 0.5),
    (1.1, 1000, (24, 24), (48, 48), 0, 0.99),
])
def test_grouping_and_search_windows(cascade_class, model_path, options):
    image = data.astronaut()[::2, ::2, 1]
    assert cascade_class(model_path).detect_multi_scale(image, *options) == (
        Original(model_path).detect_multi_scale(image, *options)
    )


@pytest.mark.parametrize("image, scale, error", [
    (np.zeros(30), 1.2, ValueError),
    (np.zeros((30, 30, 4)), 1.2, ValueError),
    (np.zeros((2, 30, 30, 3)), 1.2, ValueError),
    (np.zeros((30, 30)), 1.0, ZeroDivisionError),
    (np.zeros((30, 30)), "1.2", TypeError),
])
def test_invalid_detection_inputs(cascade_class, model_path, image, scale, error):
    args = (image, scale, 1, (24, 24), (40, 40))
    with pytest.raises(error) as expected:
        Original(model_path).detect_multi_scale(*args)
    with pytest.raises(error) as actual:
        cascade_class(model_path).detect_multi_scale(*args)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("xml, error", [("<foo>", ParseError), ("<foo/>", AttributeError)])
def test_invalid_xml(cascade_class, xml, error):
    with pytest.raises(error) as expected:
        Original(io.StringIO(xml))
    with pytest.raises(error) as actual:
        cascade_class(io.StringIO(xml))
    assert str(actual.value) == str(expected.value)


def test_missing_model(cascade_class, model_path):
    with pytest.raises(FileNotFoundError):
        cascade_class(model_path.with_name("missing-cascade.xml"))


def test_native_haar_stump(cascade_class):
    xml = b"""<opencv_storage><cascade><featureType>HAAR</featureType>
    <height>4</height><width>4</width><stageNum>1</stageNum>
    <stages><_><stageThreshold>0</stageThreshold><maxWeakCount>1</maxWeakCount>
      <weakClassifiers><_><internalNodes>0 -1 0 0</internalNodes>
        <leafValues>-1 1</leafValues></_></weakClassifiers></_></stages>
    <features><_><rects><_>0 0 2 4 -1</_><_>2 0 2 4 1</_></rects>
      <tilted>0</tilted></_></features></cascade></opencv_storage>"""
    model = cascade_class(io.BytesIO(xml))
    image = np.tile(np.arange(5, dtype=np.uint8), (5, 1))
    args = (1.3, 1, (4, 4), (5, 5), 1)
    assert model.detect_multi_scale(image, *args) == [{"r": 0, "c": 0, "width": 4, "height": 4}]
    assert model.detect_multi_scale(image[:, ::-1], *args) == []
    with pytest.raises(ValueError, match="Tilted Haar"):
        cascade_class(io.BytesIO(xml.replace(b"<tilted>0", b"<tilted>1")))
