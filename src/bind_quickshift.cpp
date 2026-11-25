#include "bind_array_storage.hpp"
#include "skimagexx/quickshift.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace py = pybind11;
namespace {

template<class T>
bool ordinary(const py::array& array, char precision) {
    const auto dtype = array.dtype();
    return Py_TYPE(array.ptr()) == py::detail::npy_api::get().PyArray_Type_ &&
           (array.flags() & py::array::c_style) && array.writeable() &&
           reinterpret_cast<std::uintptr_t>(array.data()) % alignof(T) == 0 &&
           dtype.char_() == precision && dtype.equal(py::dtype::of<T>()) &&
           dtype.attr("metadata").is_none();
}

template<class T>
py::object run(const py::array& image, const py::array& noise, T kernel, T cutoff) {
    if (!std::isfinite(kernel) || kernel < 1 || kernel > 65536 ||
        !std::isfinite(cutoff) || cutoff < 0) {
        throw py::value_error("unsupported quickshift parameters");
    }
    skimagexx::bindings::check_array_storage(image);
    skimagexx::bindings::check_array_storage(noise);
    const auto height = image.shape(0);
    const auto width = image.shape(1);
    const auto channels = image.shape(2);
    const auto* values = static_cast<const T*>(image.data());
    const auto* jitter = static_cast<const T*>(noise.data());
    py::array_t<std::ptrdiff_t> output({height, width});
    auto* labels = output.mutable_data();
    bool finite;
    {
        py::gil_scoped_release release;
        finite = std::all_of(values, values + image.size(), [](T v) { return std::isfinite(v); }) &&
                 std::all_of(jitter, jitter + noise.size(), [](T v) { return std::isfinite(v); });
        if (finite) skimagexx::quickshift_into(values, jitter, labels, height, width, channels, kernel, cutoff);
    }
    if (!finite) return py::none();
    return output;
}

} // namespace

void bind_quickshift(py::module_& module) {
    module.def("quickshift_prepared", [](const py::array& image, const py::array& noise,
                                        double kernel, double cutoff) -> py::object {
        if (image.ndim() != 3 || noise.ndim() != 2 || image.size() == 0 ||
            image.shape(0) > 65536 || image.shape(1) > 65536 ||
            noise.shape(0) != image.shape(0) || noise.shape(1) != image.shape(1)) {
            throw py::value_error("quickshift requires an HWC image and a matching HW noise array");
        }
        if (ordinary<double>(image, 'd') && ordinary<double>(noise, 'd'))
            return run(image, noise, kernel, cutoff);
        if (ordinary<float>(image, 'f') && ordinary<float>(noise, 'f'))
            return run(image, noise, static_cast<float>(kernel), static_cast<float>(cutoff));
        throw py::type_error("quickshift requires writable contiguous float32 or float64 storage");
    }, py::arg("image").noconvert(), py::arg("noise").noconvert(),
       py::arg("kernel_size"), py::arg("max_dist"));
}
