#include <pybind11/pybind11.h>
void bind_gabor(pybind11::module_&);
void bind_symmetric_eigen(pybind11::module_&);
void bind_structure_tensor(pybind11::module_&);
void bind_corner_response(pybind11::module_&);
void bind_kitchen(pybind11::module_&);

// Keep new binding groups in their own translation units. Adding a kernel
// does not require recompiling the original, large binding implementation.
void bind_extra(pybind11::module_& module) {
    bind_gabor(module);
    bind_symmetric_eigen(module);
    bind_structure_tensor(module);
    bind_corner_response(module);
    bind_kitchen(module);
}
