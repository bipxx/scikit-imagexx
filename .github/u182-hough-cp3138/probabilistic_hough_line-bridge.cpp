#include <pybind11/pybind11.h>

void bind_probabilistic_hough_line(pybind11::module_&);

PYBIND11_MODULE(_probabilistic_hough_line_private, module) {
    module.def("attach_to_core", [](pybind11::module_ target) {
        if (target.attr("__name__").cast<std::string>() != "skimagexx._core" ||
            pybind11::hasattr(target, "probabilistic_hough_line_ordered"))
            throw pybind11::value_error("expected an unchanged staged skimagexx._core");
        bind_probabilistic_hough_line(target);
    });
}
