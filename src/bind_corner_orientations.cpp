#include <skimagexx/corner_orientations.hpp>

#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <vector>

namespace py = pybind11;

namespace {

enum class OrientImageKind { none, u8, f32, f64 };

bool plain_c_aligned_2d(const py::array& array) {
    return Py_TYPE(array.ptr()) == py::detail::npy_api::get().PyArray_Type_
        && array.ndim() == 2
        && (array.flags() & py::array::c_style)
        && array.attr("flags").attr("aligned").cast<bool>()
        && array.dtype().attr("metadata").is_none();
}

OrientImageKind image_kind(const py::array& image) {
    if (!plain_c_aligned_2d(image)) {
        return OrientImageKind::none;
    }
    const auto dtype = image.dtype();
    if (dtype.equal(py::dtype::of<std::uint8_t>())) {
        return OrientImageKind::u8;
    }
    if (dtype.equal(py::dtype::of<float>())) {
        return image.writeable() ? OrientImageKind::f32 : OrientImageKind::none;
    }
    if (dtype.equal(py::dtype::of<double>())) {
        return image.writeable() ? OrientImageKind::f64 : OrientImageKind::none;
    }
    return OrientImageKind::none;
}

bool valid_corners(const py::array& corners) {
    return plain_c_aligned_2d(corners)
        && corners.writeable()
        && corners.shape(1) == 2
        && corners.dtype().equal(py::dtype::of<std::intptr_t>());
}

bool valid_mask(const py::array& mask) {
    if (!plain_c_aligned_2d(mask) || mask.shape(0) != mask.shape(1)) {
        return false;
    }
    const auto dtype = mask.dtype();
    return dtype.equal(py::dtype::of<std::uint8_t>())
        || dtype.equal(py::dtype::of<bool>());
}

bool check_backing(const py::array& array) {
    if (!array.owndata()) {
        try {
            skimagexx::bindings::check_array_storage(array);
        } catch (const py::value_error&) {
            return false;
        }
    }
    return true;
}

template <typename OutputT, typename Fn>
py::object run_orientations(const py::array& image, const py::array& corners,
    const py::array& mask, OrientImageKind expected_kind, py::ssize_t rows,
    py::ssize_t columns, py::ssize_t num_corners, py::ssize_t mask_extent,
    Fn&& fn) {
    py::array_t<OutputT> owner(std::vector<py::ssize_t>{num_corners});
    if (image_kind(image) != expected_kind || !valid_corners(corners) || !valid_mask(mask)
        || image.shape(0) != rows || image.shape(1) != columns
        || corners.shape(0) != num_corners || mask.shape(0) != mask_extent) {
        return py::none();
    }
    if (!check_backing(image) || !check_backing(corners) || !check_backing(mask)) {
        return py::none();
    }
    if (!fn(owner.mutable_data())) {
        return py::none();
    }
    return owner;
}

py::object make_orientations(const py::handle& image_handle,
    const py::handle& corners_handle, const py::handle& mask_handle) {
    if (!py::isinstance<py::array>(image_handle)
        || !py::isinstance<py::array>(corners_handle)
        || !py::isinstance<py::array>(mask_handle)) {
        return py::none();
    }
    const auto image = py::reinterpret_borrow<py::array>(image_handle);
    const auto corners = py::reinterpret_borrow<py::array>(corners_handle);
    const auto mask = py::reinterpret_borrow<py::array>(mask_handle);

    const OrientImageKind kind = image_kind(image);
    if (kind == OrientImageKind::none || !valid_corners(corners) || !valid_mask(mask)) {
        return py::none();
    }

    const auto rows = image.shape(0);
    const auto columns = image.shape(1);
    const auto num_corners = corners.shape(0);
    const auto mask_extent = mask.shape(0);
    if (rows < 0 || columns < 0 || num_corners < 0 || mask_extent < 0
        || !skimagexx::corner_orientations_shape_supported(
            static_cast<std::size_t>(rows), static_cast<std::size_t>(columns),
            static_cast<std::size_t>(num_corners), static_cast<std::size_t>(mask_extent))) {
        return py::none();
    }

    const auto r = static_cast<std::size_t>(rows);
    const auto c = static_cast<std::size_t>(columns);
    const auto k = static_cast<std::size_t>(num_corners);
    const auto m = static_cast<std::size_t>(mask_extent);
    const auto* corners_ptr = static_cast<const std::intptr_t*>(corners.data());
    const auto* mask_ptr = static_cast<const std::uint8_t*>(mask.data());

    if (kind == OrientImageKind::u8) {
        return run_orientations<double>(image, corners, mask, kind, rows, columns,
            num_corners, mask_extent, [&](double* out) {
                return skimagexx::corner_orientations_u8_into(
                    static_cast<const std::uint8_t*>(image.data()), r, c,
                    corners_ptr, k, mask_ptr, m, out);
            });
    }
    if (kind == OrientImageKind::f32) {
        return run_orientations<float>(image, corners, mask, kind, rows, columns,
            num_corners, mask_extent, [&](float* out) {
                return skimagexx::corner_orientations_f32_into(
                    static_cast<const float*>(image.data()), r, c,
                    corners_ptr, k, mask_ptr, m, out);
            });
    }
    return run_orientations<double>(image, corners, mask, kind, rows, columns,
        num_corners, mask_extent, [&](double* out) {
            return skimagexx::corner_orientations_f64_into(
                static_cast<const double*>(image.data()), r, c,
                corners_ptr, k, mask_ptr, m, out);
        });
}

}  // namespace

void bind_corner_orientations(py::module_& module) {
    module.def("corner_orientations_apply", &make_orientations,
        py::arg("image"), py::arg("corners"), py::arg("mask"));
}
