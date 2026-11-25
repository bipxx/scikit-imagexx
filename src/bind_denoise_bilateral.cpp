#include "skimagexx/denoise_bilateral.hpp"
#include "bind_array_storage.hpp"

#include "bind_helpers.hpp"

#include <cfenv>

namespace py = pybind11;

namespace {

bool contiguous_aligned(const py::array& array) {
    return (array.flags() & py::array::c_style) && array.attr("flags").attr("aligned").cast<bool>();
}

template<class T>
py::array bilateral(const py::array& image, double maximum, std::size_t window, double cval,
                    const py::array& color, const py::array& spatial) {
    const py::ssize_t shape[]{image.shape(0), image.shape(1), image.shape(2)};
    auto result = skimagexx::bindings::empty_c_array(image.dtype(), shape, 3);
    const auto* input = static_cast<const T*>(image.data());
    const auto* color_lut = static_cast<const T*>(color.data());
    const auto* spatial_lut = static_cast<const T*>(spatial.data());
    auto* output = static_cast<T*>(result.mutable_data());
    const auto count = static_cast<std::size_t>(image.size());
    const auto bins = static_cast<std::size_t>(color.size());
    const auto spatial_count = static_cast<std::size_t>(spatial.size());
    {
        py::gil_scoped_release release;
        skimagexx::denoise_bilateral_constant_into(
            input, count, static_cast<std::size_t>(shape[0]), static_cast<std::size_t>(shape[1]),
            static_cast<std::size_t>(shape[2]), maximum, window, cval,
            color_lut, bins, spatial_lut, spatial_count, output, count);
    }
    return result;
}

}  // namespace

void bind_denoise_bilateral(py::module_& module) {
    module.def("denoise_bilateral_constant", [](const py::array& image, double maximum,
               std::size_t window, double cval, const py::array& color,
               const py::array& spatial) -> py::object {
        if (image.ndim() != 3 || color.ndim() != 1 || spatial.ndim() != 1 ||
            !contiguous_aligned(image) || !contiguous_aligned(color) || !contiguous_aligned(spatial) ||
            !image.dtype().equal(color.dtype()) || !image.dtype().equal(spatial.dtype()))
            throw py::value_error("bilateral filtering requires matched contiguous prepared arrays");
        if (std::fegetround() != FE_TONEAREST)
            return py::none();
        skimagexx::bindings::check_array_storage(image);
        skimagexx::bindings::check_array_storage(color);
        skimagexx::bindings::check_array_storage(spatial);
        if (image.dtype().equal(py::dtype::of<float>()))
            return bilateral<float>(image, maximum, window, cval, color, spatial);
        if (image.dtype().equal(py::dtype::of<double>()))
            return bilateral<double>(image, maximum, window, cval, color, spatial);
        throw py::type_error("bilateral filtering requires native float32 or float64");
    }, py::arg("image").noconvert(), py::arg("maximum"), py::arg("window"), py::arg("cval"),
       py::arg("color_lut").noconvert(), py::arg("spatial_lut").noconvert());
}
