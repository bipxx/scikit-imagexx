#include <pybind11/pybind11.h>
void bind_gabor(pybind11::module_&);
void bind_symmetric_eigen(pybind11::module_&);
void bind_structure_tensor(pybind11::module_&);
void bind_corner_response(pybind11::module_&);
void bind_kitchen(pybind11::module_&);
void bind_hessian(pybind11::module_&);
void bind_shape_index(pybind11::module_&);
void bind_hessian_det(pybind11::module_&);
void bind_sato(pybind11::module_&);
void bind_frangi(pybind11::module_&);

// Keep new binding groups in their own translation units. Adding a kernel
// does not require recompiling the original, large binding implementation.
void bind_extra(pybind11::module_& module) {
    bind_gabor(module);
    bind_symmetric_eigen(module);
    bind_structure_tensor(module);
    bind_corner_response(module);
    bind_kitchen(module);
    bind_hessian(module);
    bind_shape_index(module);
    bind_hessian_det(module);
    bind_sato(module);
    bind_frangi(module);
}
