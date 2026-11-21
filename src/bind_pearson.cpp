#include "skimagexx/pearson.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

bool beta_tail(double shape, double x, double& output, void* context) {
    const auto& provider = *static_cast<const py::function*>(context);
    const auto value = provider(shape, x);
    if (value.is_none()) return false;
    output = value.cast<double>();
    return true;
}

bool same_shape(const py::array& left, const py::array& right) {
    if (left.ndim() != right.ndim()) return false;
    for (py::ssize_t axis = 0; axis < left.ndim(); ++axis)
        if (left.shape(axis) != right.shape(axis)) return false;
    return true;
}

} // namespace

void bind_pearson(py::module_& module) {
    module.def("pearson_f64", [](const py::array& left, const py::array& right,
                                 const py::object& mask, py::function provider) -> py::object {
        using Image = py::array_t<double, py::array::c_style>;
        using Mask = py::array_t<bool, py::array::c_style>;
        if (left.ndim() < 1 || left.ndim() > 2 || !same_shape(left, right) ||
            !Image::check_(left) || !Image::check_(right) ||
            left.size() < 2 || left.size() > (1 << 20))
            throw py::value_error("Pearson requires matching bounded native C-order float64 images");

        skimagexx::bindings::check_array_storage(left);
        skimagexx::bindings::check_array_storage(right);
        const std::uint8_t* selection = nullptr;
        if (!mask.is_none()) {
            if (!Mask::check_(mask))
                throw py::type_error("Pearson mask must be a C-order Boolean array");
            const auto array = py::reinterpret_borrow<py::array>(mask);
            if (!same_shape(left, array))
                throw py::value_error("Pearson mask must match image shape");
            skimagexx::bindings::check_array_storage(array);
            selection = static_cast<const std::uint8_t*>(array.data());
        }

        skimagexx::PearsonResult result;
        if (!skimagexx::pearson_corr_coeff(
                static_cast<const double*>(left.data()), static_cast<const double*>(right.data()),
                selection, static_cast<std::size_t>(left.size()), result,
                {beta_tail, &provider}))
            return py::none();
        return py::make_tuple(result.correlation, result.p_value);
    }, py::arg("left").noconvert(), py::arg("right").noconvert(), py::arg("mask"),
       py::arg("beta_tail"));
}
