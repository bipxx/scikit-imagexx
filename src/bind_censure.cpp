#include "skimagexx/censure.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_censure(py::module_& module) {
    module.def("censure_detect", [](const py::array& image, int min_scale, int max_scale, int mode,
                                   double non_max_threshold, double line_threshold) -> py::object {
        if (image.ndim() != 2 || (image.flags() & py::array::c_style) == 0
            || !image.attr("flags").attr("aligned").cast<bool>() || mode < 0 || mode > 2) {
            throw py::value_error("invalid CENSURE input");
        }
        const bool single = image.dtype().equal(py::dtype::of<float>());
        if (!single && !image.dtype().equal(py::dtype::of<double>())) {
            throw py::value_error("CENSURE requires a native float32 or float64 image");
        }
        if (!image.owndata()) {
            skimagexx::bindings::check_array_storage(image);
        }
        const auto rows = static_cast<std::size_t>(image.shape(0));
        const auto columns = static_cast<std::size_t>(image.shape(1));
        const skimagexx::CensureOptions options{
            min_scale, max_scale, static_cast<skimagexx::CensureMode>(mode), non_max_threshold, line_threshold,
        };
        const void* input = image.data();
        std::optional<std::vector<skimagexx::CensureKeypoint>> result;
        {
            py::gil_scoped_release release;
            if (single) {
                result = skimagexx::censure_detect(static_cast<const float*>(input), rows, columns, options);
            } else {
                result = skimagexx::censure_detect(static_cast<const double*>(input), rows, columns, options);
            }
        }
        if (!result) {
            return py::none();
        }
        const auto count = static_cast<py::ssize_t>(result->size());
        py::array_t<py::ssize_t> keypoints({count, py::ssize_t{2}});
        py::array_t<py::ssize_t> scales(count);
        auto coordinates = keypoints.mutable_unchecked<2>();
        auto levels = scales.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < count; ++i) {
            const auto& keypoint = (*result)[static_cast<std::size_t>(i)];
            coordinates(i, 0) = static_cast<py::ssize_t>(keypoint.row);
            coordinates(i, 1) = static_cast<py::ssize_t>(keypoint.column);
            levels(i) = keypoint.scale;
        }
        return py::make_tuple(keypoints, scales);
    }, py::arg("image").noconvert(), py::arg("min_scale"), py::arg("max_scale"),
       py::arg("mode"), py::arg("non_max_threshold"), py::arg("line_threshold"));
}

#ifdef SKIMAGEXX_PRIVATE_CENSURE_MODULE
PYBIND11_MODULE(_wave18_censure, module) {
    bind_censure(module);
}
#endif
