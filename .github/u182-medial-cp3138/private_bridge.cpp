#include <pybind11/pybind11.h>

void bind_medial_axis(pybind11::module_&);

PYBIND11_MODULE(_medial_axis_private, module) {
    bind_medial_axis(module);
}
