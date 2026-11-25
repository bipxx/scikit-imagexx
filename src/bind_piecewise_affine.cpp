#include "skimagexx/piecewise_affine.hpp"
#include "bind_array_storage.hpp"

#include <algorithm>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

template <class T> using Contiguous = py::array_t<T, py::array::c_style>;

template <class T> bool valid(const Contiguous<T>& value) {
    if (reinterpret_cast<std::uintptr_t>(value.data()) % alignof(T)) return false;
    try {
        if (!value.owndata()) skimagexx::bindings::check_array_storage(value);
    } catch (const py::value_error&) {
        return false;
    }
    return true;
}

template <class Row, class T> std::vector<Row> unpack(const T* data, std::size_t count) {
    std::vector<Row> rows(count);
    for (auto& row : rows) {
        std::copy_n(data, row.size(), row.begin());
        data += row.size();
    }
    return rows;
}

} // namespace

void bind_piecewise_affine(py::module_& module) {
    module.def("piecewise_affine_fit", [](const Contiguous<double>& src, const Contiguous<double>& dst,
                                         const Contiguous<std::int32_t>& simplices) -> py::object {
        if (src.ndim() != 2 || dst.ndim() != 2 || src.shape(1) != 2 || dst.shape(1) != 2 ||
            src.shape(0) != dst.shape(0) || simplices.ndim() != 2 || simplices.shape(1) != 3 ||
            simplices.shape(0) > 2 * static_cast<py::ssize_t>(skimagexx::piecewise_affine_max_controls) ||
            !valid(src) || !valid(dst) || !valid(simplices)) return py::none();
        const auto count = static_cast<std::size_t>(simplices.shape(0));
        const auto ordered = unpack<skimagexx::PiecewiseSimplex>(simplices.data(), count);
        skimagexx::PiecewiseAffineMesh mesh;
        if (!skimagexx::piecewise_affine_fit_mesh(src.data(), dst.data(),
                static_cast<std::size_t>(src.shape(0)), ordered.data(), count, mesh)) return py::none();
        py::array_t<double> matrices({simplices.shape(0), py::ssize_t{3}, py::ssize_t{3}});
        auto* data = matrices.mutable_data();
        for (const auto& matrix : mesh.matrices) data = std::copy(matrix.begin(), matrix.end(), data);
        return matrices;
    }, py::arg("src").noconvert(), py::arg("dst").noconvert(), py::arg("simplices").noconvert());
    module.def("piecewise_affine_apply", [](const Contiguous<double>& triangles,
                                            const Contiguous<double>& matrices,
                                            const Contiguous<double>& coords) -> py::object {
        if (coords.ndim() != 2 || coords.shape(1) != 2 || triangles.ndim() != 3 ||
            triangles.shape(1) != 3 || triangles.shape(2) != 2 || matrices.ndim() != 3 ||
            matrices.shape(1) != 3 || matrices.shape(2) != 3 || matrices.shape(0) != triangles.shape(0) ||
            matrices.shape(0) > 2 * static_cast<py::ssize_t>(skimagexx::piecewise_affine_max_controls) ||
            !valid(triangles) || !valid(matrices) || !valid(coords)) return py::none();
        const auto count = static_cast<std::size_t>(triangles.shape(0));
        const auto native_triangles = unpack<skimagexx::PiecewiseTriangle>(triangles.data(), count);
        const auto native_matrices = unpack<skimagexx::PiecewiseMatrix>(matrices.data(), count);
        py::array_t<double> output({coords.shape(0), py::ssize_t{2}});
        bool supported;
        {
            py::gil_scoped_release release;
            supported = skimagexx::piecewise_affine_apply(native_triangles.data(), native_matrices.data(),
                count, coords.data(), static_cast<std::size_t>(coords.shape(0)), output.mutable_data());
        }
        return supported ? py::object(output) : py::none();
    }, py::arg("triangles").noconvert(), py::arg("matrices").noconvert(), py::arg("coords").noconvert());
}
