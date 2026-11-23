#include <pybind11/pybind11.h>

void bind_hough_ellipse(pybind11::module_&);

PYBIND11_MODULE(_hough_ellipse_private, module) {
    module.def("attach_to_core", [](pybind11::module_ target) {
        if (target.attr("__name__").cast<std::string>() != "skimagexx._core" ||
            pybind11::hasattr(target, "hough_ellipse_bounded"))
            throw pybind11::value_error("expected an unchanged staged skimagexx._core");
        bind_hough_ellipse(target);
    });
}
