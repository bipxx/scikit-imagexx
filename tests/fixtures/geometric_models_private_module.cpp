// Standalone registration used to validate the two leaf families before integration.
#include <pybind11/pybind11.h>

void bind_geometric_models(pybind11::module_&);
void bind_marching_cubes(pybind11::module_&);

PYBIND11_MODULE(_measure_private, module) {
    bind_geometric_models(module);
    bind_marching_cubes(module);
}
