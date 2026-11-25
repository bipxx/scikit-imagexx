#include "skimagexx/future_filters.hpp"
#include "skimagexx/draw_polygon.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/stl.h>
#include <limits>

namespace py = pybind11;
namespace {
using skimagexx::SegmentLabel;
using skimagexx::bindings::check_array_storage;

bool supported_arrays(const py::array& labels, const py::array& features) {
    if (labels.ndim() < 1 || features.ndim() != labels.ndim() + 1 ||
        !(labels.flags() & py::array::c_style) || !(features.flags() & py::array::c_style) ||
        !labels.dtype().attr("isnative").cast<bool>())
        return false;
    for (py::ssize_t axis = 0; axis < labels.ndim(); ++axis)
        if (labels.shape(axis) != features.shape(axis)) return false;
    return true;
}

py::object gather_training(const py::array& labels, const py::array& features) {
    if (!supported_arrays(labels, features)) return py::none();
    const auto kind = labels.dtype().kind();
    const auto item = labels.itemsize();
    SegmentLabel type;
    switch (item) {
        case 1: type = kind == 'i' ? SegmentLabel::i8 : SegmentLabel::u8; break;
        case 2: type = kind == 'i' ? SegmentLabel::i16 : SegmentLabel::u16; break;
        case 4: type = kind == 'i' ? SegmentLabel::i32 : SegmentLabel::u32; break;
        case 8: type = kind == 'i' ? SegmentLabel::i64 : SegmentLabel::u64; break;
        default: return py::none();
    }
    if (kind != 'i' && kind != 'u' && kind != 'b') return py::none();
    // Raw copies preserve numeric dtype metadata and byte order. Reference
    // dtypes need NumPy's reference-counted indexing path.
    const auto feature_kind = features.dtype().kind();
    if (feature_kind != 'b' && feature_kind != 'i' && feature_kind != 'u' &&
        feature_kind != 'f' && feature_kind != 'c') return py::none();
    const auto width = features.shape(features.ndim() - 1);
    const auto feature_item = features.itemsize();
    if (width > std::numeric_limits<py::ssize_t>::max() / feature_item) return py::none();
    check_array_storage(labels);
    check_array_storage(features);
    const auto* label_input = labels.data();
    const auto* feature_input = features.data();
    const auto count = static_cast<std::size_t>(labels.size());
    std::vector<std::size_t> selected;
    {
        py::gil_scoped_release release;
        selected = skimagexx::segment_positive_indices(label_input, count, type);
    }
    const auto selected_count = static_cast<py::ssize_t>(selected.size());
    py::array data(features.dtype(), {selected_count, width});
    py::array target(labels.dtype(), std::vector<py::ssize_t>{selected_count});
    auto* data_output = data.mutable_data();
    auto* target_output = target.mutable_data();
    if (!selected.empty()) {
        py::gil_scoped_release release;
        skimagexx::segment_gather_rows(selected, feature_input,
            static_cast<std::size_t>(width * feature_item), label_input,
            static_cast<std::size_t>(item), data_output, target_output);
    }
    return py::make_tuple(data, target);
}

py::object threshold_histogram(const py::array& image) {
    if (image.ndim() != 2 || !image.size() || image.dtype().kind() != 'u' ||
        image.itemsize() != 1 || !(image.flags() & py::array::c_style)) return py::none();
    check_array_storage(image);
    const auto* input = static_cast<const std::uint8_t*>(image.data());
    const auto height = static_cast<std::size_t>(image.shape(0));
    const auto width = static_cast<std::size_t>(image.shape(1));
    std::array<std::uint64_t, 256> histogram;
    {
        py::gil_scoped_release release;
        histogram = skimagexx::histogram_u8(input, height, width);
    }
    std::size_t first = 0, last = histogram.size() - 1;
    while (!histogram[first]) ++first;
    while (!histogram[last]) --last;
    py::array_t<std::int64_t> counts(last - first + 1), centers(last - first + 1);
    for (auto i = first; i <= last; ++i) {
        counts.mutable_data()[i - first] = static_cast<std::int64_t>(histogram[i]);
        centers.mutable_data()[i - first] = static_cast<std::int64_t>(i);
    }
    return py::make_tuple(counts, centers);
}

bool stamp_polygon(const std::vector<double>& rows, const std::vector<double>& columns,
                   const std::array<std::intptr_t, 2>& first,
                   const std::array<std::intptr_t, 2>& last,
                   py::array_t<std::intptr_t, py::array::c_style> output,
                   std::intptr_t label) {
    if (rows.size() != columns.size() || output.ndim() != 2 || !output.writeable())
        throw py::value_error("invalid segment polygon buffers");
    check_array_storage(output);
    auto* destination = output.mutable_data();
    if (reinterpret_cast<std::uintptr_t>(destination) % alignof(std::intptr_t))
        throw py::value_error("unaligned segment polygon output");
    const auto height = static_cast<std::size_t>(output.shape(0));
    const auto width = static_cast<std::size_t>(output.shape(1));
    skimagexx::detail::DrawPolygonPlan plan;
    py::gil_scoped_release release;
    if (!skimagexx::detail::draw_polygon_prepare(
            {rows.data(), columns.data(), rows.size()}, {first, last},
            plan, skimagexx::get_num_threads())) return false;
    skimagexx::segment_stamp_polygon(plan, height, width, label, destination);
    return true;
}
}

void bind_future_filters(py::module_& module) {
    module.def("segment_gather_training", &gather_training,
        py::arg("labels").noconvert(), py::arg("features").noconvert());
    module.def("threshold_panel_histogram", &threshold_histogram, py::arg("image").noconvert());
    module.def("segment_stamp_polygon", &stamp_polygon, py::arg("rows"), py::arg("columns"),
        py::arg("first"), py::arg("last"), py::arg("output").noconvert(), py::arg("label"));
}

#ifdef SKIMAGEXX_FUTURE_FILTERS_STANDALONE
PYBIND11_MODULE(_future_filters_core, module) {
    bind_future_filters(module);
    module.def("set_num_threads", &skimagexx::set_num_threads);
    module.def("has_avx2", &skimagexx::has_avx2);
}
#endif
