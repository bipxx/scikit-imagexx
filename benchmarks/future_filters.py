"""One-thread wall-clock medians for the six public calls.

Run with the integrated public namespaces and the original scikit-image
installed. The GUI cases script fixed selections and use the Agg backend.
"""

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "1"

from contextlib import nullcontext
from pathlib import Path
from statistics import median
import sys
from time import perf_counter_ns

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from skimage import data, filters as ref_filters, future as ref_future
import skimagexx
from skimagexx import filters, future

sys.path.append(str(Path(__file__).resolve().parents[1] / "tests"))
from fixtures.future_filters_gui import selections


class Classifier:
    """Low-cost estimator to expose the surrounding segmentation overhead."""

    def fit(self, data, labels):
        self.label = labels[0]

    def predict(self, data):
        return np.where(data[:, 0] > 0, 2, 1)


def time_call(function, loops):
    start = perf_counter_ns()
    for _ in range(loops):
        function()
    return (perf_counter_ns() - start) / loops


def measure(name, original, native, loops=1, context=nullcontext):
    original_times, native_times = [], []
    with context():
        original()
        native()
        for index in range(13):
            timed = ((original, original_times), (native, native_times))
            if index % 2:
                timed = timed[::-1]
            for function, times in timed:
                times.append(time_call(function, loops))
    base, candidate = median(original_times), median(native_times)
    ratio = base / candidate
    print(f"{name}: {base / 1000:.3f} -> {candidate / 1000:.3f} us ({ratio:.4f}x)", flush=True)
    return ratio


def main():
    skimagexx.set_num_threads(1)
    skimagexx.set_fallback_enabled(False)
    rng = np.random.default_rng(42)
    features = rng.standard_normal((1024, 1024, 8), dtype=np.float32)
    labels = np.zeros(features.shape[:-1], dtype=np.int32)
    labels[4::32, 7::32] = 2
    original = ref_future.TrainableSegmenter(Classifier(), lambda image: image)
    native = future.TrainableSegmenter(Classifier(), lambda image: image)

    def plot(function, image):
        figure, _ = function(image, verbose=False)
        plt.close(figure)

    camera = np.tile(data.camera(), (4, 4))
    angles = np.arange(12) * (2 * np.pi / 12)
    vertices = [list(zip(650 + 570 * np.cos(angles), 650 + 540 * np.sin(angles)))]
    canvas = np.zeros((1300, 1300), dtype=np.uint8)
    ratios = [
        measure("filters.try_all_threshold", lambda: plot(ref_filters.try_all_threshold, camera),
                lambda: plot(filters.try_all_threshold, camera)),
        measure("future.TrainableSegmenter.fit", lambda: original.fit(features, labels),
                lambda: native.fit(features, labels), loops=4),
        measure("future.fit_segmenter", lambda: ref_future.fit_segmenter(labels, features, original.clf),
                lambda: future.fit_segmenter(labels, features, native.clf), loops=4),
        measure("future.predict_segmenter", lambda: ref_future.predict_segmenter(features, original.clf),
                lambda: future.predict_segmenter(features, native.clf), loops=4),
        measure("future.manual_polygon_segmentation", lambda: ref_future.manual_polygon_segmentation(canvas),
                lambda: future.manual_polygon_segmentation(canvas),
                context=lambda: selections("polygon", vertices)),
        measure("future.manual_lasso_segmentation", lambda: ref_future.manual_lasso_segmentation(canvas),
                lambda: future.manual_lasso_segmentation(canvas),
                context=lambda: selections("lasso", vertices)),
    ]
    return int(min(ratios) < 1)


if __name__ == "__main__":
    raise SystemExit(main())
