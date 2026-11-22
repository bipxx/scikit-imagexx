#include "skimagexx/match_descriptors.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

bool descriptor_array(const py::array& array) {
    return Py_TYPE(array.ptr()) == py::detail::npy_api::get().PyArray_Type_ &&
           array.ndim() == 2 && array.shape(0) > 0 && array.shape(1) > 0 &&
           (array.flags() & py::array::c_style) && array.dtype().kind() == 'b' &&
           array.itemsize() == 1;
}

} // namespace

void bind_match_descriptors(py::module_& module) {
    module.def("match_descriptors_bool", [](py::handle left, py::handle right, double max_distance,
                                         bool cross_check, double max_ratio) -> py::object {
        if (!py::isinstance<py::array>(left) || !py::isinstance<py::array>(right))
            return py::none();
        const auto a = py::reinterpret_borrow<py::array>(left);
        const auto b = py::reinterpret_borrow<py::array>(right);
        if (!descriptor_array(a) || !descriptor_array(b) || a.shape(1) != b.shape(1) ||
            a.shape(0) > static_cast<py::ssize_t>(skimagexx::match_descriptors_max_rows) ||
            b.shape(0) > static_cast<py::ssize_t>(skimagexx::match_descriptors_max_rows) ||
            a.shape(1) > static_cast<py::ssize_t>(skimagexx::match_descriptors_max_width))
            return py::none();
        try {
            if (!a.owndata()) skimagexx::bindings::check_array_storage(a);
            if (!b.owndata()) skimagexx::bindings::check_array_storage(b);
        } catch (const py::value_error&) {
            return py::none();
        }
        std::vector<skimagexx::DescriptorMatch> matches;
        if (!skimagexx::match_descriptors_bool(
                static_cast<const std::uint8_t*>(a.data()), static_cast<std::size_t>(a.shape(0)),
                static_cast<const std::uint8_t*>(b.data()), static_cast<std::size_t>(b.shape(0)),
                static_cast<std::size_t>(a.shape(1)), {max_distance, cross_check, max_ratio}, matches))
            return py::none();

        const auto count = static_cast<py::ssize_t>(matches.size());
        const auto stride = count ? static_cast<py::ssize_t>(sizeof(std::intptr_t)) : 0;
        py::array_t<std::intptr_t> output(py::array::ShapeContainer{count, py::ssize_t{2}},
                                         py::array::StridesContainer{2 * stride, stride});
        auto* data = output.mutable_data();
        for (const auto& pair : matches) {
            *data++ = pair[0];
            *data++ = pair[1];
        }
        return output;
    }, py::arg("left"), py::arg("right"), py::arg("max_distance").noconvert(),
       py::arg("cross_check").noconvert(), py::arg("max_ratio").noconvert());
}
