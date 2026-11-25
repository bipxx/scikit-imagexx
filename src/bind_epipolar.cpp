#include "skimagexx/epipolar.hpp"
#include "bind_array_storage.hpp"

#include <cstdint>
#include <pybind11/numpy.h>

namespace py = pybind11;
namespace {

bool doubles(const py::array& array, py::ssize_t rows, py::ssize_t columns) {
    const int dimensions = columns < 0 ? 1 : 2;
    if (array.ndim() != dimensions || (rows >= 0 && array.shape(0) != rows) ||
        (dimensions == 2 && array.shape(1) != columns) ||
        !(array.flags() & py::array::c_style) ||
        !array.dtype().is(py::dtype::of<double>()) ||
        reinterpret_cast<std::uintptr_t>(array.data()) % alignof(double) != 0) return false;
    try {
        skimagexx::bindings::check_array_storage(array);
    } catch (const py::value_error&) {
        return false;
    }
    return true;
}

const double* data(const py::array& array) {
    return static_cast<const double*>(array.data());
}

py::object lines(const py::array& points, const py::array& matrix) {
    if (!doubles(points, -1, 2) || !doubles(matrix, 3, 3)) return py::none();
    py::array_t<double> output({points.shape(0), py::ssize_t{3}});
    auto* dst = output.mutable_data();
    const auto* src = data(points);
    const auto* coeff = data(matrix);
    const auto count = static_cast<std::size_t>(points.shape(0));
    if (count < 1024) {
        if (!skimagexx::epipolar_lines(src, count, coeff, dst)) return py::none();
    } else {
        bool finite;
        {
            py::gil_scoped_release release;
            finite = skimagexx::epipolar_lines(src, count, coeff, dst);
        }
        if (!finite) return py::none();
    }
    return output;
}

py::object residuals(const py::array& src, const py::array& dst, const py::array& matrix) {
    if (!doubles(src, -1, 2) || !doubles(dst, src.shape(0), 2) ||
        !doubles(matrix, 3, 3)) return py::none();
    py::array_t<double> output(src.shape(0));
    const auto* left = data(src);
    const auto* right = data(dst);
    const auto* coeff = data(matrix);
    auto* values = output.mutable_data();
    const auto count = static_cast<std::size_t>(src.shape(0));
    bool finite;
    {
        py::gil_scoped_release release;
        finite = skimagexx::epipolar_residuals(left, right, count, coeff, values);
    }
    if (!finite) return py::none();
    return output;
}

py::object constraints(const py::array& src, const py::array& dst, unsigned scaling) {
    if (!doubles(src, -1, 2) || src.shape(0) < 8 ||
        !doubles(dst, src.shape(0), 2) || scaling > 2) return py::none();
    py::array_t<double> equations({src.shape(0), py::ssize_t{9}});
    py::array_t<double> left({3, 3}), right({3, 3});
    const auto* src_points = data(src);
    const auto* dst_points = data(dst);
    auto* rows = equations.mutable_data();
    auto* left_matrix = left.mutable_data();
    auto* right_matrix = right.mutable_data();
    const auto count = static_cast<std::size_t>(src.shape(0));
    bool valid;
    {
        py::gil_scoped_release release;
        valid = skimagexx::epipolar_constraints(src_points, dst_points, count,
            static_cast<skimagexx::EpipolarScaling>(scaling), rows, left_matrix, right_matrix);
    }
    if (!valid) return py::none();
    return py::make_tuple(equations, left, right);
}

py::object matrix(const py::array& u, const py::array& values, const py::array& vt,
                  const py::array& src, const py::array& dst, bool essential) {
    if (!doubles(u, 3, 3) || !doubles(values, 3, -1) || !doubles(vt, 3, 3) ||
        !doubles(src, 3, 3) || !doubles(dst, 3, 3)) return py::none();
    py::array_t<double> output({3, 3});
    skimagexx::epipolar_matrix(data(u), data(values), data(vt), data(src), data(dst),
                               essential, output.mutable_data());
    return output;
}

}  // namespace

void bind_epipolar(py::module_& module) {
    using py::arg;
    module.def("epipolar_lines", &lines, arg("points").noconvert(), arg("matrix").noconvert());
    module.def("epipolar_residuals", &residuals, arg("src").noconvert(), arg("dst").noconvert(),
               arg("matrix").noconvert());
    module.def("epipolar_constraints", &constraints, arg("src").noconvert(), arg("dst").noconvert(),
               arg("scaling"));
    module.def("epipolar_matrix", &matrix, arg("u").noconvert(), arg("values").noconvert(),
               arg("vt").noconvert(), arg("src").noconvert(), arg("dst").noconvert(), arg("essential"));
}

#ifdef SKIMAGEXX_EPIPOLAR_PRIVATE_MODULE
PYBIND11_MODULE(_epipolar_native, module) {
    bind_epipolar(module);
}
#endif
