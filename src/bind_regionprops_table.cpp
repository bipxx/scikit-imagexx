#include <skimagexx/regionprops_table.hpp>

#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>

namespace py = pybind11;

namespace {

py::object default_table(const py::handle& input) {
    if (Py_TYPE(input.ptr()) != py::detail::npy_api::get().PyArray_Type_) {
        return py::none();
    }
    const auto image = py::reinterpret_borrow<py::array>(input);
    const auto dtype = image.dtype();
    if (image.ndim() != 2 || !(image.flags() & py::array::c_style)
        || image.shape(0) < 1 || image.shape(1) < 1
        || image.shape(0) > py::ssize_t(skimagexx::regionprops_table_max_extent)
        || image.shape(1) > py::ssize_t(skimagexx::regionprops_table_max_extent)
        || dtype.kind() != 'i' || image.itemsize() != sizeof(std::int32_t)
        || !dtype.attr("isnative").cast<bool>() || !dtype.attr("metadata").is_none()) {
        return py::none();
    }
    if (!image.owndata()) {
        try {
            skimagexx::bindings::check_array_storage(image);
        } catch (const py::value_error&) {
            return py::none();
        }
    }

    std::vector<skimagexx::RegionPropsTableRow> rows;
    if (!skimagexx::regionprops_table_bounds(image.data(), std::size_t(image.shape(0)),
                                           std::size_t(image.shape(1)), rows)) {
        return py::none();
    }

    const auto count = py::ssize_t(rows.size());
    const auto storage_rows = std::max(count, py::ssize_t{1});
    py::array_t<std::intptr_t> labels(storage_rows);
    py::array_t<std::intptr_t> boxes(std::vector<py::ssize_t>{storage_rows, 4});
    auto* label_values = labels.mutable_data();
    auto* box_values = boxes.mutable_data();
    for (std::size_t row = 0; row < rows.size(); ++row) {
        label_values[row] = rows[row].label;
        std::copy(rows[row].bbox.begin(), rows[row].bbox.end(), box_values + 4 * row);
    }
    if (rows.empty()) {
        // The upstream table constructs a synthetic one-pixel region before
        // returning empty column slices. Their backing values remain visible.
        label_values[0] = 1;
        std::array<std::intptr_t, 4> dummy{1, 1, 2, 2};
        std::copy(dummy.begin(), dummy.end(), box_values);
    }

    py::dict result;
    const py::slice visible(0, count, 1);
    if (rows.empty()) {
        result["label"] = labels[visible];
    } else {
        result["label"] = labels;
    }
    const std::array<const char*, 4> keys{"bbox-0", "bbox-1", "bbox-2", "bbox-3"};
    for (std::size_t column = 0; column < keys.size(); ++column) {
        result[keys[column]] = boxes[py::make_tuple(visible, py::int_(column))];
    }
    return result;
}

}  // namespace

void bind_regionprops_table(py::module_& module) {
    module.def("regionprops_table_default", &default_table, py::arg("label_image"));
}
