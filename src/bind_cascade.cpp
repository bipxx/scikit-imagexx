#include "skimagexx/cascade.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <limits>
#include <string>

namespace py = pybind11;

void bind_cascade(py::module_& module) {
    using skimagexx::Cascade;
    using skimagexx::CascadeDetection;
    py::register_exception<skimagexx::CascadeXmlError>(module, "_CascadeXmlError");
    module.def("_check_cascade_image_storage", &skimagexx::bindings::check_array_storage);
    py::class_<Cascade>(module, "_CascadeModel")
        .def(py::init([](const py::bytes& xml, float eps) {
            return Cascade(static_cast<std::string>(xml), eps);
        }))
        .def_readwrite("eps", &Cascade::eps)
        .def_readwrite("window_height", &Cascade::window_height)
        .def_readwrite("window_width", &Cascade::window_width)
        .def_readwrite("stages_number", &Cascade::stages_number)
        .def_readwrite("stumps_number", &Cascade::stumps_number)
        .def_readwrite("features_number", &Cascade::features_number)
        .def("detect", [](const Cascade& cascade,
                          const py::array_t<float, py::array::c_style>& integral,
                          const py::array_t<float, py::array::c_style>& scales, float step,
                          int neighbors, float threshold) {
            if (integral.ndim() != 2 || scales.ndim() != 1 ||
                integral.shape(0) > std::numeric_limits<int>::max() ||
                integral.shape(1) > std::numeric_limits<int>::max()) {
                throw py::value_error("Invalid cascade integral image or scale array");
            }
            const auto rows = static_cast<int>(integral.shape(0));
            const auto columns = static_cast<int>(integral.shape(1));
            skimagexx::bindings::check_array_storage(integral);
            skimagexx::bindings::check_array_storage(scales);
            const auto* input = integral.data();
            const auto* first = scales.data();
            if (reinterpret_cast<std::uintptr_t>(input) % alignof(float) ||
                reinterpret_cast<std::uintptr_t>(first) % alignof(float)) {
                throw py::value_error("Cascade integral image and scales must be aligned");
            }
            const std::vector<float> multipliers(first, first + scales.size());
            const Cascade snapshot(cascade);
            std::vector<CascadeDetection> boxes;
            {
                py::gil_scoped_release release;
                boxes = snapshot.detect(input, rows, columns, multipliers, step, neighbors, threshold);
            }
            py::list result(boxes.size());
            for (std::size_t i = 0; i < boxes.size(); ++i) {
                const auto& box = boxes[i];
                py::dict item;
                item["r"] = box.r;
                item["c"] = box.c;
                item["width"] = box.width;
                item["height"] = box.height;
                result[i] = std::move(item);
            }
            return result;
        });
}

#ifdef SKIMAGEXX_PRIVATE_CASCADE_MODULE
PYBIND11_MODULE(cascade_private, module) { bind_cascade(module); }
#endif
