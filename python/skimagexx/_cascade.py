"""OpenCV cascade detection with native model loading and classification."""

import math
import operator
import sys
from ctypes import c_float
from xml.etree import ElementTree

import numpy as np

from . import _core


def _c_float(value):
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("must be real number, not " + type(value).__name__)
    return c_float(float(value)).value


def _c_ssize(value):
    value = operator.index(value)
    if value < -sys.maxsize - 1 or value > sys.maxsize:
        raise OverflowError("Python int too large to convert to C ssize_t")
    return value


def _model_property(name, convert=_c_ssize):
    return property(
        lambda cascade: getattr(cascade._model, name),
        lambda cascade, value: setattr(cascade._model, name, convert(value)),
    )


def _gray(rgb):
    if rgb.shape[-1] != 3:
        raise ValueError(f"the input array must have size 3 along `channel_axis`, got {rgb.shape}")
    dtype = rgb.dtype
    if dtype.kind in "f":
        rgb = rgb.astype(np.float32 if dtype.itemsize <= 4 else np.float64, copy=False)
    elif dtype.kind in "biu":
        if dtype.kind == "b":
            rgb = rgb.astype(np.float64)
        else:
            rgb = rgb * (1.0 / np.iinfo(dtype).max)
            if dtype.kind == "i":
                np.maximum(rgb, -1.0, out=rgb)
    elif dtype.kind == "c":
        raise ValueError("Cannot convert from complex data types.")
    else:
        raise ValueError(f"Cannot convert from {dtype} to float64.")
    return rgb @ np.array([0.2125, 0.7154, 0.0721], dtype=rgb.dtype)


class Cascade:
    """Detect objects with an OpenCV MB-LBP or axis-aligned Haar stump cascade.

    `xml_file` is a path or readable stream. `eps` adjusts stage thresholds.
    The Haar extension supports the modern OpenCV BOOST XML structure.
    """

    __slots__ = ("_model",)
    eps = _model_property("eps", _c_float)
    stages_number = _model_property("stages_number")
    stumps_number = _model_property("stumps_number")
    features_number = _model_property("features_number")
    window_height = _model_property("window_height")
    window_width = _model_property("window_width")

    def __init__(self, xml_file, eps=1e-5):
        self._load_xml(xml_file, eps)

    def _load_xml(self, xml_file, eps=1e-5):
        if hasattr(xml_file, "read"):
            xml = xml_file.read()
        else:
            with open(xml_file, "rb") as source:
                xml = source.read()
        if isinstance(xml, str):
            xml = xml.encode("utf-8")
        tolerance = _c_float(eps)
        try:
            try:
                model = _core._CascadeModel(xml, tolerance)
            except _core._CascadeXmlError:
                # Normalize valid XML encodings and preserve syntax error locations.
                root = ElementTree.fromstring(xml)
                model = _core._CascadeModel(ElementTree.tostring(root), tolerance)
        except ValueError as error:
            if str(error).startswith("Missing cascade element:"):
                raise AttributeError("'NoneType' object has no attribute 'text'") from error
            raise
        self._model = model

    def _get_valid_scale_factors(self, min_size, max_size, scale_step):
        size = np.array((self.window_height, self.window_width))
        minimum = np.array(min_size, dtype=np.float32)
        maximum = np.array(max_size, dtype=np.float32)
        high = min(math.log(maximum[0] / size[0], scale_step),
                   math.log(maximum[1] / size[1], scale_step))
        low = max(math.log(minimum[0] / size[0], scale_step),
                  math.log(minimum[1] / size[1], scale_step), 0)
        return np.power(scale_step, np.arange(low, high), dtype=np.float32)

    def _get_contiguous_integral_image(self, img):
        if isinstance(img, np.ndarray) and not img.flags.owndata:
            _core._check_cascade_image_storage(img)
        if len(img.shape) > 2:
            img = _gray(img)
        dtype = np.promote_types(img.dtype, np.float64) if img.real.dtype.kind == "f" else None
        for axis in range(img.ndim):
            img = img.cumsum(axis=axis, dtype=dtype)
        return np.ascontiguousarray(img, dtype=np.float32)

    def detect_multi_scale(self, img, scale_factor, step_ratio, min_size, max_size,
                           min_neighbor_number=4, intersection_score_threshold=0.5):
        """Return grouped detections as dictionaries with r, c, width and height."""
        scale_factor = _c_float(scale_factor)
        step_ratio = _c_float(step_ratio)
        integral = self._get_contiguous_integral_image(img)
        if integral.ndim != 2:
            raise ValueError(f"Buffer has wrong number of dimensions (expected 2, got {integral.ndim})")
        scales = self._get_valid_scale_factors(min_size, max_size, scale_factor)
        if isinstance(min_neighbor_number, (str, bytes, bytearray)) or min_neighbor_number is None:
            raise TypeError("an integer is required")
        neighbors = int(min_neighbor_number)
        if not -2**31 <= neighbors < 2**31:
            raise OverflowError("Python int too large to convert to C int")
        return self._model.detect(integral, scales, step_ratio,
                                  neighbors, _c_float(intersection_score_threshold))
