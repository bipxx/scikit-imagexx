#pragma once
#include <pybind11/numpy.h>

namespace skimagexx::bindings {
// Shared existing bounded-ndarray owner policy; no additional guard system.
void check_array_storage(const pybind11::array& value);
}
